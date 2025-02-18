import math as m
import os

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms.functional as FT
from torch import autograd
from torch import distributions as torchd
from torch.nn.utils import spectral_norm
from torchvision.models.optical_flow import raft_small
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import rgb_to_grayscale
from torchvision.utils import save_image
from utils_folder import utils
from utils_folder.byol_pytorch import RandomApply, default
from utils_folder.utils_dreamer import Bernoulli
# from agents.multi_dataset.point_cloud_generator import PointCloudGenerator
from point_cloud_generator import PointCloudGenerator
import open3d as o3d

class PointNetHead(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.h = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, point_cloud):

        x = point_cloud

        """Input size: [b, n, 3]"""

        x = torch.permute(x, (0, 2, 1))  # [b,3,n]

        x = self.h(x)  # x -> [b,64,n]

        x = self.mlp2(x)  # x -> [b,128,n]

        x = torch.max(x, dim=2, keepdim=True).values  # x -> [b, 128]

        x = self.mlp3(x)

        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(2)

        return x

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.head1 = PointNetHead(latent_dim)
        self.head2 = PointNetHead(latent_dim)
        self.head3 = PointNetHead(latent_dim)

        self.final = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim), nn.Tanh()
        )


    def forward(self, point_cloud):

        if len(point_cloud.size()) == 3:
            point_cloud = point_cloud.unsqueeze(0)
        """will be in form [b,3,n,3]"""

        """Input size: [b, n, 3]"""


        #first 3 is frame stack

        """for now let's just pretend not batched"""
        #unbatched = point_cloud[0] #[3,n,3]
        unbatched=point_cloud
        #unbatched = torch.permute(unbatched, (0, 2,1))

        pc1 = unbatched[:, 0, :, :]
        pc2 = unbatched[:,1,:,:]
        pc3 = unbatched[:,2,:,:]

        x1 = self.head1(pc1)
        x2 = self.head2(pc2)
        x3 = self.head3(pc3)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.final(x)

        if x.dim() == 3 and x.shape[2] == 1:
            x = x.squeeze(2)

        return x





class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time *= self.scale
        device = time.device
        half_dim = self.dim // 2
        embeddings = m.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        interleaved = torch.empty(time.size(0), self.dim)
        interleaved[:, 0::2] = embeddings.sin()
        interleaved[:, 1::2] = embeddings.cos()
        return interleaved


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=self.in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):

        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = x.shape

        # compute attention
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class CustomAug(nn.Module):
    def __init__(self, aug_function):
        super().__init__()
        self.aug = aug_function

    def forward(self, obs):
        b, c, h, w = obs.size()
        assert h == w

        num_frames = c // 3

        image_aug = []
        for i in range(num_frames):
            frame = obs[:, 3 * i : 3 * i + 3, :, :]
            frame_aug = self.aug(frame)
            image_aug.append(frame_aug)

        image_aug = torch.cat(image_aug, dim=1).float()

        return image_aug


class Grayscale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs):
        b, c, h, w = obs.size()
        assert h == w

        num_frames = c // 3

        image_aug = []
        for i in range(num_frames):
            frame = obs[:, 3 * i : 3 * i + 3, :, :]
            frame_aug = rgb_to_grayscale(frame, num_output_channels=3)
            image_aug.append(frame_aug)

        image_aug = torch.cat(image_aug, dim=1)

        return image_aug


class CL(nn.Module):
    """Contrastive learning for the encoder"""

    def __init__(self, feature_dim):
        super().__init__()
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (feat_dim, B)
        logits = torch.matmul(z_a, Wz)  # (B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, stochastic, log_std_bounds):
        super().__init__()

        assert len(obs_shape) == 3
        self.stochastic = stochastic
        self.log_std_bounds = log_std_bounds

        if obs_shape[-1] == 84:
            self.repr_dim = 32 * 35 * 35
        elif obs_shape[-1] == 64:
            self.repr_dim = 32 * 25 * 25

        self.spatial_dim = 64 * 64
        self.spatial_dim_tuple = (64, 64)
        self.sinusoidal_encodings = SinusoidalPositionalEmbeddings(
            self.spatial_dim, self.spatial_dim // 10
        )
        self.encoding1 = self.sinusoidal_encodings(torch.tensor([1]).float())
        self.encoding2 = self.sinusoidal_encodings(torch.tensor([2]).float())
        self.encoding3 = self.sinusoidal_encodings(torch.tensor([3]).float())

        self.convnet2 = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.additional_dim_optical_flow = -4
        self.initial_conv = nn.Conv3d(
            obs_shape[0] + self.additional_dim_optical_flow,
            32,
            kernel_size=(3, 3, 3),
            stride=2,
        )
        self.relu = nn.ReLU()
        self.convnet = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.attention = AttnBlock(32)
        self.optical_flow_model = raft_small(pretrained=True, progress=False).cuda()
        self.optical_flow_model = self.optical_flow_model.eval()

        if self.stochastic:
            self.trunk = nn.Sequential(
                nn.Linear(self.repr_dim, 2 * feature_dim),
                nn.LayerNorm(2 * feature_dim),
                nn.Tanh(),
            )
        else:
            # self.repr_dim = 100_352
            self.trunk = nn.Sequential(
                nn.Linear(self.repr_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.Tanh(),
            )

        self.apply(utils.weight_init)

    def optical_flow(self, x):
        x = self.min_max_norm(x) * 2 - 1
        x = FT.resize(x, size=(128, 128))
        flow = self.optical_flow_model(x[:, -3:, :, :], x[:, -6:-3, :, :])
        flow = flow[-1]
        flow = FT.resize(flow, size=self.spatial_dim_tuple)

        assert not torch.isnan(flow).any()

        return flow

    def min_max_norm(self, x):
        if torch.min(x) != torch.max(x):
            return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
        else:
            return x

    def forward(self, obs):

        # print(obs.size())

        batch_size, channel_dim, height, width = obs.size()

        # num_frames = channel_dim // 3
        # num_channels = 3

        obs = obs / 255.0 - 0.5

        h = self.convnet2(obs)

        h = h.reshape(h.shape[0], -1)

        z = self.trunk(h)

        if self.stochastic:
            mu, log_std = z.chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
            dist = torchd.Normal(mu, std)
            z = dist.sample()

        return z

        # with torch.no_grad():
        #     flow = self.optical_flow(obs)

        """NEW TEST"""

        # flow = self.min_max_norm(flow)
        # flow = flow - 0.5
        # obs = torch.cat([obs, flow], dim=1)
        # h = self.convnet2(obs)

        # h = h.reshape(h.shape[0], -1)

        # z = self.trunk(h)
        # if self.stochastic:
        # mu, log_std = z.chunk(2, dim=-1)
        # log_std = torch.tanh(log_std)
        # log_std_min, log_std_max = self.log_std_bounds
        # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        # std = log_std.exp()
        # dist = torchd.Normal(mu, std)
        # z = dist.sample()

        # return z

        """"""
        obs = obs.view(batch_size, num_frames, num_channels, height, width)
        obs = obs.permute(0, 2, 1, 3, 4)

        # obs dimensions = (batch, channels, frame, height, width)
        """
        encoding1 = self.encoding1.view(self.spatial_dim_tuple)
        encoding1 = encoding1.unsqueeze(0).unsqueeze(0)
        encoding1 = encoding1.repeat(1, 3, 1, 1)

        encoding2= self.encoding2.view(self.spatial_dim_tuple)
        encoding2 = encoding2.unsqueeze(0).unsqueeze(0)
        encoding2 = encoding2.repeat(1, 3, 1, 1)

        encoding3 = self.encoding3.view(self.spatial_dim_tuple)
        encoding3 = encoding3.unsqueeze(0).unsqueeze(0)
        encoding3 = encoding3.repeat(1, 3, 1, 1)

        obs[:, :, 0, :, :] = encoding1
        obs[:, :, 1, :, :] = encoding2
        obs[:, :, 2, :, :] = encoding3
        """

        flow = self.min_max_norm(flow)

        flow = flow - 0.5

        flow = flow.unsqueeze(2)  # try broadcasting first
        flow = flow.repeat(1, 1, 3, 1, 1)

        obs = torch.cat([obs, flow], dim=1)

        h = self.initial_conv(obs)

        h = torch.squeeze(h)

        if h.dim() == 3:
            h = h.unsqueeze(0)

        h = self.attention(h)

        h = self.convnet(h)

        h = h.reshape(h.shape[0], -1)

        z = self.trunk(h)

        if self.stochastic:
            mu, log_std = z.chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            std = log_std.exp()
            dist = torchd.Normal(mu, std)
            z = dist.sample()

        return z


class Discriminator(nn.Module):
    def __init__(self, input_net_dim, hidden_dim, spectral_norm_bool=False, dist=None):
        super().__init__()

        self.dist = dist
        self._shape = (1,)

        if spectral_norm_bool:
            self.net = nn.Sequential(
                spectral_norm(nn.Linear(input_net_dim, hidden_dim)),
                nn.ReLU(inplace=True),
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(inplace=True),
                spectral_norm(nn.Linear(hidden_dim, 1)),
            )

        else:
            self.net = nn.Sequential(
                nn.Linear(input_net_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )

        self.apply(utils.weight_init)

    def forward(self, transition):
        d = self.net(transition)

        if self.dist == "binary":
            return Bernoulli(
                torchd.independent.Independent(
                    torchd.bernoulli.Bernoulli(logits=d), len(self._shape)
                )
            )
        else:
            return d


class Actor(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_shape[0]),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class LailClAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        stddev_clip,
        use_tb,
        reward_d_coef,
        discriminator_lr,
        spectral_norm_bool,
        check_every_steps,
        log_std_bounds,
        GAN_loss="bce",
        stochastic_encoder=False,
        train_encoder_w_critic=True,
        CL_data_type="all",
        from_dem=False,
        add_aug_anchor_and_positive=False,
        aug_type="full",
        apply_aug="everywhere",  # everywhere, nowhere, CL-only, CL-D
        grayscale=False,
        depth_flag=False,
        segm_flag=False,
        physics=None,
    ):

        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.GAN_loss = GAN_loss
        self.from_dem = from_dem
        self.check_every_steps = check_every_steps
        self.train_encoder_w_critic = train_encoder_w_critic
        self.CL_data_type = CL_data_type
        self.add_aug_anchor_and_positive = add_aug_anchor_and_positive
        self.apply_aug = apply_aug
        self.grayscale = grayscale
        self.grayscale_aug = Grayscale()
        self.point_cloud_generator = PointCloudGenerator(physics)

        # data augmentation
        self.select_aug_type(aug_type, apply_aug, obs_shape)

        if depth_flag or segm_flag:
            self.apply_aug = "nowhere"

        """
        self.encoder = Encoder(
            obs_shape, feature_dim, stochastic_encoder, log_std_bounds
        ).to(device)
        """

        self.encoder = PointNetEncoder(
            feature_dim
        ).to(device)

        self.actor = Actor(action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.CL = CL(feature_dim).to(device)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # added model
        if from_dem:
            if self.GAN_loss == "least-square":
                self.discriminator = Discriminator(
                    feature_dim + action_shape[0], hidden_dim, spectral_norm_bool
                ).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == "bce":
                self.discriminator = Discriminator(
                    feature_dim + action_shape[0],
                    hidden_dim,
                    spectral_norm_bool,
                    dist="binary",
                ).to(device)
            else:
                NotImplementedError

        else:
            if self.GAN_loss == "least-square":
                self.discriminator = Discriminator(
                    2 * feature_dim, hidden_dim, spectral_norm_bool
                ).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == "bce":
                self.discriminator = Discriminator(
                    2 * feature_dim, hidden_dim, spectral_norm_bool, dist="binary"
                ).to(device)
            else:
                NotImplementedError

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=discriminator_lr
        )
        self.CL_opt = torch.optim.Adam(self.CL.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def select_aug_type(self, aug_type, apply_aug, obs_shape):
        # add augmentation for CL
        if aug_type == "brightness":

            if self.CL_data_type == "agent":
                DEFAULT_AUG = torch.nn.Sequential(
                    T.ColorJitter((1, 4), None, None, None)
                )

            else:
                DEFAULT_AUG = torch.nn.Sequential(
                    T.ColorJitter((0, 2), None, None, None)
                )

        elif aug_type == "color":
            DEFAULT_AUG = torch.nn.Sequential(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                T.RandomGrayscale(p=0.2),
                RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.2),
                T.RandomInvert(p=0.2),
                RandomApply(T.RandomChannelPermutation(), p=0.2),
            )

        elif aug_type == "full":
            DEFAULT_AUG = torch.nn.Sequential(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                T.RandomGrayscale(p=0.2),
                T.RandomHorizontalFlip(p=0.1),
                T.RandomVerticalFlip(p=0.1),
                RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p=0.1),
                T.RandomInvert(p=0.2),
                T.RandomResizedCrop(
                    (obs_shape[-1], obs_shape[-1]), scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ),
            )

        else:
            NotImplementedError

        if apply_aug == "everywhere":
            self.augment1 = default(None, DEFAULT_AUG)
            self.augment2 = default(None, self.augment1)
            self.aug_D = CustomAug(self.augment1)
            self.aug_Q = CustomAug(self.augment1)

        elif apply_aug == "nowhere":
            self.augment1 = RandomShiftsAug(pad=4)
            self.augment2 = RandomShiftsAug(pad=4)
            self.aug_D = RandomShiftsAug(pad=4)
            self.aug_Q = RandomShiftsAug(pad=4)

        elif apply_aug == "CL-only":
            self.augment1 = default(None, DEFAULT_AUG)
            self.augment2 = default(None, self.augment1)
            self.aug_D = RandomShiftsAug(pad=4)
            self.aug_Q = RandomShiftsAug(pad=4)

        elif apply_aug == "CL-D":
            self.augment1 = default(None, DEFAULT_AUG)
            self.augment2 = default(None, self.augment1)
            self.aug_D = CustomAug(self.augment1)
            self.aug_Q = RandomShiftsAug(pad=4)

        elif apply_aug == "CL-Q":
            self.augment1 = default(None, DEFAULT_AUG)
            self.augment2 = default(None, self.augment1)
            self.aug_D = RandomShiftsAug(pad=4)
            self.aug_Q = CustomAug(self.augment1)

        else:
            NotImplementedError

    def augment(self, obs):
        b, c, h, w = obs.size()
        assert h == w

        num_frames = c // 3

        if self.add_aug_anchor_and_positive:
            image_one = []
            image_two = []
            for i in range(num_frames):
                frame = obs[:, 3 * i : 3 * i + 3, :, :]
                frame_one = self.augment1(frame)
                frame_two = self.augment2(frame)
                image_one.append(frame_one)
                image_two.append(frame_two)

            image_one = torch.cat(image_one, dim=1).float()
            image_two = torch.cat(image_two, dim=1).float()

        else:
            image_two = []
            for i in range(num_frames):
                frame = obs[:, 3 * i : 3 * i + 3, :, :]
                frame_two = self.augment2(frame)
                image_two.append(frame_two)

            image_two = torch.cat(image_two, dim=1).float()
            image_one = obs.float()

        return image_one, image_two

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)
        self.CL.train(training)

    def act(self, obs, step, eval_mode, from_buffer):
        if from_buffer == True:
            pass
            #it should already be good
        if from_buffer==False:
            """I believe it is (192, 64)"""
            #hard coding for now
            obs_reshaped = obs.reshape(3, 64, 64)
            point_clouds = []
            max_obs_length = -1
            for i in range(3):
                obs = obs_reshaped[i, :, :]
                obs = self.point_cloud_generator.depthImageToPointCloud(obs, 0)
                max_obs_length = max(obs.shape[0], max_obs_length)
                #obs = torch.as_tensor(obs, device=self.device).float()
                point_clouds.append(obs)

            tensors = [torch.tensor(arr,dtype=torch.float32, device=self.device) for arr in point_clouds]
            padded_tensors = pad_sequence(tensors,batch_first=True,padding_value=0)

            point_clouds = padded_tensors.unsqueeze(0)


        if self.grayscale:
            obs = self.grayscale_aug(obs.unsqueeze(0))
            obs = self.encoder(obs)
        else:
            #obs = self.encoder(obs.unsqueeze(0))
            obs = self.encoder(point_clouds)

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        if self.train_encoder_w_critic:
            # optimize encoder and critic
            self.encoder_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()
            self.encoder_opt.step()
        else:
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self.critic_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def compute_reward(self, obs_a, next_a):
        metrics = dict()

        # augment
        if self.from_dem:
            obs_a = self.aug_D(obs_a)
        else:
            obs_a = self.aug_D(obs_a)
            next_a = self.aug_D(next_a)

        # encode
        with torch.no_grad():
            if self.from_dem:
                obs_a = self.encoder(obs_a)
            else:
                obs_a = self.encoder(obs_a)
                next_a = self.encoder(next_a)

            self.discriminator.eval()
            transition_a = torch.cat([obs_a, next_a], dim=-1)

            d = self.discriminator(transition_a)

            if self.GAN_loss == "least-square":
                reward_d = self.reward_d_coef * torch.clamp(
                    1 - (1 / 4) * torch.square(d - 1), min=0
                )

            elif self.GAN_loss == "bce":
                reward_d = d.mode()

            reward = reward_d

            if self.use_tb:
                metrics["reward_d"] = reward_d.mean().item()

            self.discriminator.train()

        return reward, metrics

    def compute_discriminator_grad_penalty_LS(self, obs_e, next_e, lambda_=10):

        expert_data = torch.cat([obs_e, next_e], dim=-1)
        expert_data.requires_grad = True

        d = self.discriminator(expert_data)

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(
            outputs=d,
            inputs=expert_data,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_discriminator_grad_penalty_bce(
        self, obs_a, next_a, obs_e, next_e, lambda_=10
    ):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha * agent_feat + (1 - alpha) * expert_data

        disc_penalty_input.requires_grad = True

        d = self.discriminator(disc_penalty_input).mode()

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(
            outputs=d,
            inputs=disc_penalty_input,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update_discriminator(self, obs_a, next_a, obs_e, next_e):
        metrics = dict()

        transition_a = torch.cat([obs_a, next_a], dim=-1)
        transition_e = torch.cat([obs_e, next_e], dim=-1)

        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == "least-square":
            expert_labels = 1.0
            agent_labels = -1.0

            expert_loss = F.mse_loss(
                expert_d,
                expert_labels * torch.ones(expert_d.size(), device=self.device),
            )
            agent_loss = F.mse_loss(
                agent_d, agent_labels * torch.ones(agent_d.size(), device=self.device)
            )
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(
                obs_e.detach(), next_e.detach()
            )
            loss = 0.5 * (expert_loss + agent_loss) + grad_pen_loss

        elif self.GAN_loss == "bce":
            expert_loss = (
                expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))
            ).mean()
            agent_loss = (
                agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))
            ).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(
                obs_a.detach(), next_a.detach(), obs_e.detach(), next_e.detach()
            )
            loss = -(expert_loss + agent_loss) + grad_pen_loss

        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()

        if self.use_tb:
            metrics["discriminator_expert_loss"] = expert_loss.item()
            metrics["discriminator_agent_loss"] = agent_loss.item()
            metrics["discriminator_loss"] = loss.item()
            metrics["discriminator_grad_pen"] = grad_pen_loss.item()

        return metrics

    def update_CL(
        self, obs, obs_e_raw, obs_random, obs_e_raw_random, check_every_steps, step
    ):
        metrics = dict()

        if self.CL_data_type == "agent":
            anchor_to_aug = obs

        elif self.CL_data_type == "expert":
            anchor_to_aug = torch.cat([obs, obs_e_raw], dim=0)

        elif self.CL_data_type == "all":
            anchor_to_aug = torch.cat([obs, obs_e_raw], dim=0)
            anchors = torch.cat([obs_random, obs_e_raw_random], dim=0)
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat(
                [obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0
            )

        elif self.CL_data_type == "agent-random":
            anchor_to_aug = obs
            anchors = obs_random
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat(
                [obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0
            )

        elif self.CL_data_type == "random-only":
            anchors = torch.cat([obs_random, obs_e_raw_random], dim=0)
            rand_idx = torch.randperm(obs_random.shape[0])
            positives = torch.cat(
                [obs_e_raw_random[rand_idx], obs_random[rand_idx]], dim=0
            )

        else:
            NotImplementedError

        if self.CL_data_type == "all" or self.CL_data_type == "agent-random":
            image_one, image_two = self.augment(anchor_to_aug)
            anchors = torch.cat([anchors.float(), image_one], dim=0)
            positives = torch.cat([positives.float(), image_two], dim=0)

            if step % check_every_steps == 0:
                self.check_aug_CL(image_one, image_two, step)

        elif self.CL_data_type == "random-only":
            anchors = anchors.float()
            positives = positives.float()

        else:
            image_one, image_two = self.augment(anchor_to_aug)
            anchors = image_one
            positives = image_two

            if step % check_every_steps == 0:
                self.check_aug_CL(image_one, image_two, step)

        z_anchor = self.encoder(anchors)
        with torch.no_grad():
            z_pos = self.encoder(positives)

        logits = self.CL.compute_logits(z_anchor, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        self.encoder_opt.zero_grad()
        self.CL_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        self.CL_opt.step()

        if self.use_tb:
            metrics["CL_loss"] = loss.item()

        return metrics

    def rotate_aug(self, data):
        batch_size = data.size()[0]

        x_theta = torch.rand((batch_size), dtype=torch.float32) * 2 * m.pi
        y_theta = torch.rand((batch_size), dtype=torch.float32) * 2 * m.pi
        z_theta = torch.rand((batch_size), dtype=torch.float32) * 2 * m.pi

        cos_x, sin_x = torch.cos(x_theta), torch.sin(x_theta)
        cos_y, sin_y = torch.cos(y_theta), torch.sin(y_theta)
        cos_z, sin_z = torch.cos(z_theta), torch.sin(z_theta)

        x_matrix = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        x_matrix[:, 0, 0] = 1.0
        x_matrix[:, 1, 1] = cos_x
        x_matrix[:, 1, 2] = -sin_x
        x_matrix[:, 2, 1] = sin_x
        x_matrix[:, 2, 2] = cos_x

        y_matrix = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        y_matrix[:, 1, 1] = 1.0
        y_matrix[:, 0, 0] = cos_y
        y_matrix[:, 0, 2] = sin_y
        y_matrix[:, 2, 0] = -sin_y
        y_matrix[:, 2, 2] = cos_y

        z_matrix = torch.zeros((batch_size, 3, 3), device=self.device, dtype=torch.float32)
        z_matrix[:, 2, 2] = 1.0
        z_matrix[:, 0, 0] = cos_z
        z_matrix[:, 0, 1] = -sin_z
        z_matrix[:, 1, 0] = sin_z
        z_matrix[:, 1, 1] = cos_z

        z_matrix = z_matrix[:, None, :, :]
        y_matrix = y_matrix[:, None, :, :]
        x_matrix = x_matrix[:, None, :, :]

        data = torch.matmul(data, z_matrix.transpose(-1, -2))
        data = torch.matmul(data, y_matrix.transpose(-1, -2))
        data = torch.matmul(data, x_matrix.transpose(-1, -2))

        return data

    def update(
        self,
        replay_iter,
        replay_iter_expert,
        replay_iter_random,
        replay_iter_expert_random,
        step,
    ):
        """Get rid of expert stuff + Disc"""
        """Experiment with mismatch between train and eval"""
        metrics = dict()
        breakpoint()

        if step % self.update_every_steps != 0:
            return metrics
        batch = next(replay_iter)
        obs, action, reward_a, discount, next_obs = utils.to_torch(batch, self.device) #reward_a unused

        sample_cloud = obs[42, 0, :, :].detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_cloud)
        o3d.io.write_point_cloud("point_cloud.ply", pcd)
        exit()


        
        #[b, cx3, h,w]
        #[b, 3, d, 3]

        # batch_expert = next(replay_iter_expert)
        # obs_e_raw, action_e, _, _, next_obs_e_raw = utils.to_torch(
        #     batch_expert, self.device
        # )

        # sample random data
        batch_agent_random = next(replay_iter_random)
        obs_random, _, _, _, next_obs_random = utils.to_torch(
            batch_agent_random, self.device
        )

        batch_expert_random = next(replay_iter_expert_random)
        obs_e_raw_random, _, _, _, next_obs_e_raw_random = utils.to_torch(
            batch_expert_random, self.device
        )

        #usually off, but would shift all to gray 
        # if self.grayscale:
        #     obs = self.grayscale_aug(obs)
        #     next_obs = self.grayscale_aug(next_obs)
        #     obs_e_raw = self.grayscale_aug(obs_e_raw)
        #     next_obs_e_raw = self.grayscale_aug(next_obs_e_raw)
        #     obs_random = self.grayscale_aug(obs_random)
        #     next_obs_random = self.grayscale_aug(next_obs_random)
        #     obs_e_raw_random = self.grayscale_aug(obs_e_raw_random)
        #     next_obs_e_raw_random = self.grayscale_aug(next_obs_e_raw_random)

        # if step % self.check_every_steps == 0 and False:
        #     self.check_aug(
        #         obs_random.float(),
        #         next_obs_random.float(),
        #         obs_e_raw_random.float(),
        #         next_obs_e_raw_random.float(),
        #         "random_buffer",
        #         step,
        #     )

        """
        metrics.update(
            self.update_CL(
                obs,
                obs_e_raw,
                obs_random,
                obs_e_raw_random,
                self.check_every_steps,
                step,
            )
        )
        """

        #CHANGE THIS TO BE RANDOM ROTATION
        """
        obs_e = self.aug_D(obs_e_raw)
        next_obs_e = self.aug_D(next_obs_e_raw)
        obs_a = self.aug_D(obs)
        next_obs_a = self.aug_D(next_obs)
        """

        obs_a = self.rotate_aug(obs.float())
        next_obs_a = self.rotate_aug(next_obs.float())

        #ADDED THIS
        # obs_e = obs_e_raw
        # next_obs_e = next_obs_e_raw

        # if step % self.check_every_steps == 0 and False:
        #     self.check_aug(
        #         obs_a, next_obs_a, obs_e, next_obs_e, "learning_buffer", step
        #     )

        with torch.no_grad():
            # z_e = self.encoder(obs_e.float())
            # next_z_e = self.encoder(next_obs_e.float())
            z_a = self.encoder(obs_a.float())
            next_z_a = self.encoder(next_obs_a.float())

        """
        # update critic
        if self.from_dem:
            metrics.update(self.update_discriminator(z_a, action, z_e, action_e))
            reward, metrics_r = self.compute_reward(obs, action) #this function uses disc to compute reward
        else:
            metrics.update(self.update_discriminator(z_a, next_z_a, z_e, next_z_e))
            reward, metrics_r = self.compute_reward(obs, next_obs)
        """

        # metrics.update(metrics_r)

        # augment
        # obs = self.aug_Q(obs)
        # next_obs = self.aug_Q(next_obs)

        # encode
        obs = self.encoder(obs.float())
        with torch.no_grad():
            next_obs = self.encoder(next_obs.float())

        if self.use_tb:
            metrics["batch_reward"] = reward_a.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward_a, discount, next_obs, step) #replaced reward with reward_a
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def check_aug(self, obs, next_obs, obs_e, next_obs_e, type, step):

        if not os.path.exists(f"checkimages_{type}"):
            os.makedirs(f"checkimages_{type}")

        obs = obs / 255
        next_obs = next_obs / 255
        obs_e = obs_e / 255
        next_obs_e = next_obs_e / 255

        obs = torch.cat([obs, next_obs], dim=0)
        obs_e = torch.cat([obs_e, next_obs_e])
        rand_idx = torch.randperm(obs.shape[0])
        imgs1 = obs[rand_idx[:9]]
        imgs2 = obs[rand_idx[-9:]]
        imgs3 = obs_e[rand_idx[9:18]]
        imgs4 = obs_e[rand_idx[-18:-9]]

        saved_imgs = torch.cat(
            [
                imgs1[:, :3, :, :],
                imgs2[:, :3, :, :],
                imgs3[:, :3, :, :],
                imgs4[:, :3, :, :],
            ],
            dim=0,
        )
        save_image(saved_imgs, f"./checkimages_{type}/{step}.png", nrow=9)

    def check_aug_CL(self, img1, img2, step):

        if not os.path.exists("checkaugs"):
            os.makedirs("checkaugs")

        img1 = img1 / 255  # -> [0, 1]
        img2 = img2 / 255  # -> [0, 1]

        rand_idx = torch.randperm(img1.shape[0])

        imgs1 = img1[rand_idx[:9]]
        imgs2 = img2[rand_idx[:9]]

        imgs3 = img1[rand_idx[-9:]]
        imgs4 = img2[rand_idx[-9:]]

        saved_imgs = torch.cat(
            [
                imgs1[:, :3, :, :],
                imgs2[:, :3, :, :],
                imgs3[:, :3, :, :],
                imgs4[:, :3, :, :],
            ],
            dim=0,
        )
        save_image(saved_imgs, "./checkaugs/%d.png" % (step), nrow=9)
