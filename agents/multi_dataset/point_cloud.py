# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from utils import random_overlay
from svea import SVEAAgent

class OneHotPointNetEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()

        self.hidden_dim = hidden_dim #unused for now

        self.h = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh(),
        )

    def add_one_hot_info(self, points: torch.Tensor, frame_id, total_frames):
        batch_size, num_points, xyz = points.size()
        one_hot = F.one_hot(torch.tensor([frame_id]), total_frames).to(points.device).to(points.dtype)
        one_hot_expanded = one_hot.view(1,1,3).expand(batch_size, num_points, -1)
        points_concat = torch.cat([points, one_hot_expanded], dim=-1)

        return points_concat

    def forward(self, point_cloud):

        """Input size: [b, 3, n, 3]"""

        points1 = self.add_one_hot_info(point_cloud[:,0,:,:],frame_id=0,total_frames=3)
        points2 = self.add_one_hot_info(point_cloud[:,1,:,:],frame_id=1,total_frames=3)
        points3 = self.add_one_hot_info(point_cloud[:,2,:,:],frame_id=2,total_frames=3)

        all_points = torch.cat([points1,points2,points3], dim=1)

        """Size now: [b, n, 6]"""

        x = all_points

        x = torch.permute(x, (0, 2, 1))  # [b,6,n']

        x = self.h(x)  # x -> [b,64,n]

        x = self.mlp2(x)  # x -> [b,128,n]

        x = torch.max(x, dim=2, keepdim=True).values  # x -> [b, 128]

        x = self.mlp3(x)

        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(2)

        return x

class MultiViewPointNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=64, num_views=4):
        super().__init__()
        self.num_views = num_views
        self.encoder = OneHotPointNetEncoder(hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )

    def forward(self, x_views):  
        view_features = []
        for x in x_views:
            encoded = self.encoder(x)  # (B, N, hidden_dim)
            pooled = torch.max(encoded, dim=1)[0]  # (B, hidden_dim)
            view_features.append(pooled)

        # Stack views: (B, num_views, hidden_dim)
        view_features = torch.stack(view_features, dim=1)

        # Pool over views (symmetric over view order)
        global_feature = torch.max(view_features, dim=1)[0]  # (B, hidden_dim)

        # Final MLP
        out = self.fusion(global_feature)  # (B, 256)
        return out


class SymmetricPointCloudEncoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.obs_shape = obs_shape

    def forward(self):
        pass

class PointCloudAgent(SVEAAgent):
    def __init__(self, num_points, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):

        obs_shape = (1, 1, 1) #placeholder

        super().__init__(obs_shape, action_shape, device, lr, feature_dim, 
                         hidden_dim, critic_target_tau, num_expl_steps,
                         update_every_steps, stddev_schedule, stddev_clip, use_tb)

        self.num_points = num_points
        self.encoder = SymmetricPointCloudEncoder(self.num_points).to(self.device)
        self.aug = NoAug()

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        original_obs = obs.clone()
        next_obs = self.aug(next_obs.float())

        # strong augmentation
        # aug_obs = self.encoder(random_overlay(original_obs)) # looks like random_overlay will change the textures
        aug_obs = self.encoder(original_obs) # for now, no augmentation

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics


class NoAug(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
