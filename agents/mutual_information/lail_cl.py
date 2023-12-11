import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd
from torch import autograd
from torch.nn.utils import spectral_norm 
from torchvision.utils import save_image
from torchvision import transforms as T

from utils_folder import utils
from utils_folder.utils_dreamer import Bernoulli
from utils_folder.byol_pytorch import default, RandomApply

import os

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
    
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
        Wz = torch.matmul(self.W, z_pos.T) #(feat_dim, B)
        logits = torch.matmul(z_a, Wz) #(B, B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, stochastic, log_std_bounds):
        super().__init__()

        assert len(obs_shape) == 3
        self.stochastic = stochastic
        self.log_std_bounds = log_std_bounds

        if obs_shape[-1]==84:
            self.repr_dim = 32 * 35 * 35
        elif obs_shape[-1]==64:
            self.repr_dim = 32 * 25 * 25

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        if self.stochastic:
            self.trunk = nn.Sequential(nn.Linear(self.repr_dim, 2 * feature_dim),
                                    nn.LayerNorm(2 * feature_dim), nn.Tanh())
        else:
            self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
            
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
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
            self.net = nn.Sequential(spectral_norm(nn.Linear(input_net_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                                    nn.ReLU(inplace=True),
                                    spectral_norm(nn.Linear(hidden_dim, 1)))  

        else:
            self.net = nn.Sequential(nn.Linear(input_net_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, 1))  
        
        self.apply(utils.weight_init)

    def forward(self, transition):
        d = self.net(transition)

        if self.dist == 'binary':
            return Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=d), len(self._shape)))
        else:
            return d 
        
class MIEstimator(nn.Module):
    def __init__(self, feature_dim, hidden_dim, labels_dim = 1):
        super().__init__()

        self.M1 = nn.Sequential(
            nn.Linear(feature_dim+labels_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.M2 = nn.Sequential(
            nn.Linear(feature_dim+labels_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, z):
        m1 = self.M1(z)
        m2 = self.M2(z)

        return m1, m2

class Actor(nn.Module):
    def __init__(self, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

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
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2

class LailClAgent:
    def __init__(self, 
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
                 dual_mi_constant, # lagrangian variable used for the prior data MI loss
                 dual_max_mi, # max MI for the prior data MI loss (\approx 0)
                 dual_min_mi_constant, # minimal value of the lagrangian variable for the prior data MI loss
                 max_mi, # max value of MI to update the lagrangian variable for the actual data MI loss 
                 min_mi, # min value of MI to update the lagrangian variable for the actual data MI loss
                 min_mi_constant, # min value of the lagrangian variable for the actual data MI loss
                 max_mi_constant, # max value of the lagrangian variable for the actual data MI loss
                 mi_constant, # initial value of the lagrangian variable for the actual data MI loss
                 unbiased_mi_decay, # rate of decay for the unbiased MI estimator loss
                 reward_d_coef, 
                 discriminator_lr, 
                 spectral_norm_bool, 
                 check_every_steps, 
                 log_std_bounds,
                 GAN_loss='bce', 
                 stochastic_encoder=False, 
                 from_dem=False, 
                 add_aug=False, 
                 depth_flag=False, 
                 segm_flag=False):
        
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
        self.add_aug = add_aug

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        # add augmentation for CL
        DEFAULT_AUG = torch.nn.Sequential(
            RandomApply(T.ColorJitter(0.8, 0.8, 0.8, 0.2), p = 0.3),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(T.GaussianBlur((3, 3), (1.0, 2.0)), p = 0.2),
            T.RandomResizedCrop((obs_shape[-1], obs_shape[-1]), scale=(0.9, 1.0), ratio=(0.9, 1.1))
        )

        self.augment1 = default(None, DEFAULT_AUG)
        self.augment2 = default(None, self.augment1)

        if depth_flag or segm_flag:
            self.add_aug = False

        self.encoder = Encoder(obs_shape, feature_dim, stochastic_encoder, log_std_bounds).to(device)

        self.actor = Actor(action_shape, feature_dim, hidden_dim).to(device)
        self.critic = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.CL = CL(feature_dim).to(device) 
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.mi_estimator = MIEstimator(feature_dim, hidden_dim, labels_dim=1).to(device)
        self.log_mi_constant = torch.tensor(np.log(dual_mi_constant)).to(device)
        self.log_mi_constant.requires_grad = True
        self.log_min_mi_constant = torch.tensor(np.log(dual_min_mi_constant)).to(device)
        self.dual_max_mi = dual_max_mi
        self.max_mi = max_mi
        self.min_mi = min_mi 
        self.max_mi_constant = max_mi_constant
        self.min_mi_constant = min_mi_constant
        self.adaptive_penalty = torch.tensor(mi_constant).to(device)
        self.unbiased_mi_decay = unbiased_mi_decay
        self.unbiased_mi_ma1 = torch.tensor(1.0).to(device) # initial value of the MI for the unbiased estimator loss 
        self.unbiased_mi_ma2 = torch.tensor(1.0).to(device) # initial value of the MI for the unbiased estimator loss
        
        # added model
        if from_dem:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(feature_dim+action_shape[0], hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        else:
            if self.GAN_loss == 'least-square':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool).to(device)
                self.reward_d_coef = reward_d_coef

            elif self.GAN_loss == 'bce':
                self.discriminator = Discriminator(2*feature_dim, hidden_dim, spectral_norm_bool, dist='binary').to(device)
            else:
                NotImplementedError

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discriminator_opt = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        self.CL_opt = torch.optim.Adam(self.CL.parameters(), lr=lr)
        self.mi_estimator_opt = torch.optim.Adam(self.mi_estimator.parameters(), lr=lr)
        self.log_mi_constant_opt = torch.optim.Adam([self.log_mi_constant], lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.discriminator.train(training)
        self.CL.train(training)
        self.mi_estimator.train(training)

    @property
    def mi_constant(self):
        return torch.maximum(self.log_mi_constant, self.log_min_mi_constant).exp()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
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
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

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
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def compute_reward(self, obs_a, next_a):
        metrics = dict()

        # augment
        if self.from_dem:
            obs_a = self.aug(obs_a.float())
        else:
            obs_a = self.aug(obs_a.float())
            next_a = self.aug(next_a.float())
        
        # encode
        with torch.no_grad():
            if self.from_dem:
                obs_a = self.encoder(obs_a)
            else:
                obs_a = self.encoder(obs_a)
                next_a = self.encoder(next_a)
        
            self.discriminator.eval()
            transition_a = torch.cat([obs_a, next_a], dim = -1)

            d = self.discriminator(transition_a)

            if self.GAN_loss == 'least-square':
                reward_d = self.reward_d_coef * torch.clamp(1 - (1/4) * torch.square(d - 1), min=0)

            elif self.GAN_loss == 'bce':
                reward_d = d.mode()
            
            reward = reward_d

            if self.use_tb:
                metrics['reward_d'] = reward_d.mean().item()
    
            self.discriminator.train()
            
        return reward, metrics
    
    def compute_discriminator_grad_penalty_LS(self, obs_e, next_e, lambda_=10):
        
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        expert_data.requires_grad = True
        
        d = self.discriminator(expert_data)

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=expert_data, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
        return grad_pen

    def compute_discriminator_grad_penalty_bce(self, obs_a, next_a, obs_e, next_e, lambda_=10):

        agent_feat = torch.cat([obs_a, next_a], dim=-1)
        alpha = torch.rand(agent_feat.shape[:1]).unsqueeze(-1).to(self.device)
        expert_data = torch.cat([obs_e, next_e], dim=-1)
        disc_penalty_input = alpha*agent_feat + (1-alpha)*expert_data

        disc_penalty_input.requires_grad = True

        d = self.discriminator(disc_penalty_input).mode()

        ones = torch.ones(d.size(), device=self.device)
        grad = autograd.grad(outputs=d, inputs=disc_penalty_input, grad_outputs=ones, create_graph=True,
                             retain_graph=True, only_inputs=True)[0]
        
        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen
        
    def update_discriminator(self, obs_a, next_a, obs_e, next_e):
        metrics = dict()

        transition_a = torch.cat([obs_a, next_a], dim=-1)
        transition_e = torch.cat([obs_e, next_e], dim=-1)
        
        agent_d = self.discriminator(transition_a)
        expert_d = self.discriminator(transition_e)

        if self.GAN_loss == 'least-square':
            expert_labels = 1.0
            agent_labels = -1.0

            expert_loss = F.mse_loss(expert_d, expert_labels*torch.ones(expert_d.size(), device=self.device))
            agent_loss = F.mse_loss(agent_d, agent_labels*torch.ones(agent_d.size(), device=self.device))
            grad_pen_loss = self.compute_discriminator_grad_penalty_LS(obs_e.detach(), next_e.detach())
            loss = 0.5*(expert_loss + agent_loss) + grad_pen_loss
        
        elif self.GAN_loss == 'bce':
            expert_loss = (expert_d.log_prob(torch.ones_like(expert_d.mode()).to(self.device))).mean()
            agent_loss = (agent_d.log_prob(torch.zeros_like(agent_d.mode()).to(self.device))).mean()
            grad_pen_loss = self.compute_discriminator_grad_penalty_bce(obs_a.detach(), next_a.detach(), obs_e.detach(), next_e.detach())
            loss = -(expert_loss+agent_loss) + grad_pen_loss

        self.discriminator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_opt.step()
        
        if self.use_tb:
            metrics['discriminator_expert_loss'] = expert_loss.item()
            metrics['discriminator_agent_loss'] = agent_loss.item()
            metrics['discriminator_loss'] = loss.item()
            metrics['discriminator_grad_pen'] = grad_pen_loss.item()
        
        return metrics    
    
    def update_adaptive_penalty(self, mi_loss):
        metrics = dict()

        if mi_loss > self.max_mi:
            self.adaptive_penalty = self.adaptive_penalty * 1.5
        elif mi_loss < self.min_mi:
            self.adaptive_penalty = self.adaptive_penalty / 1.5

        self.adaptive_penalty = torch.clip(self.adaptive_penalty, self.min_mi_constant, self.max_mi_constant)

        if self.use_tb:
            metrics['adaptive_penalty'] = self.adaptive_penalty

        return metrics

    def update_dual_penalty(self, mi_loss):
        metrics = dict()

        mi_diff = self.dual_max_mi - mi_loss
        self.log_mi_constant_opt.zero_grad(set_to_none=True)
        mi_dual_loss = self.log_mi_constant * mi_diff.detach()
        mi_dual_loss.backward()
        self.log_mi_constant_opt.step()

        if self.use_tb:
            metrics['mi_dual_loss'] = mi_dual_loss.item()
            metrics['mi_dual_constant_value'] = self.mi_constant

        return metrics  

    def augment(self, obs):
        b, c, h, w = obs.size()
        assert h == w 

        num_frames = c // 3
        image_one = []
        image_two = []
        for i in range(num_frames):
            frame = obs[:, 3*i:3*i+3, :, :]
            frame_one = self.augment1(frame)
            frame_two = self.augment2(frame)
            image_one.append(frame_one)
            image_two.append(frame_two)

        image_one = torch.cat(image_one, dim=1)
        image_two = torch.cat(image_two, dim=1)

        return image_one, image_two
    
    def update_CL(self, obs, z_a, next_z_a, z_e, next_z_e, z_a_random, next_z_a_random, z_e_random, next_z_e_random, check_every_steps, step):
        metrics = dict()

        if self.add_aug:
            image_one, image_two = self.augment(obs)

            if step % check_every_steps == 0:
                self.check_aug_CL(image_one.float(), image_two.float(), step)

            z_anchor = self.encoder(image_one.float())
            with torch.no_grad():
                z_pos = self.encoder(image_two.float())

        else:
            z_anchor = self.encoder(self.aug(obs.float()))
            with torch.no_grad():
                z_pos = self.encoder(self.aug(obs.float()))

        logits = self.CL.compute_logits(z_anchor, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        loss = self.cross_entropy_loss(logits, labels)

        _, _, m1_loss, m2_loss = self.compute_mi_loss(z_a, next_z_a, z_e, next_z_e)
        mi_est = torch.maximum(m1_loss, m2_loss)
        mi_loss = torch.clamp(mi_est, min=0.0)

        _, _, m1_random_loss, m2_random_loss = self.compute_mi_loss(z_a_random, next_z_a_random, z_e_random, next_z_e_random)
        mi_random_est = torch.maximum(m1_random_loss, m2_random_loss)
        mi_random_loss = torch.clamp(mi_random_est, min=0.0)

        metrics.update(self.update_adaptive_penalty(torch.clamp(mi_est, min=0.0, max=1.0))) # refers to the penalty for mi_loss
        metrics.update(self.update_dual_penalty(torch.clamp(mi_random_est, min=0.0, max=1.0))) # refers to the penalty for mi_random_loss

        loss = loss + self.adaptive_penalty * mi_loss + self.mi_constant * mi_random_loss

        self.encoder_opt.zero_grad()
        self.CL_opt.zero_grad()
        loss.backward()
        self.encoder_opt.step()
        self.CL_opt.step()

        if self.use_tb:
            metrics['CL_loss'] = loss.item()

        return metrics    
    
    def compute_mi_loss(self, z_a, next_z_a, z_e, next_z_e):

        agent_z = torch.cat([z_a, next_z_a], dim=0)
        expert_z = torch.cat([z_e, next_z_e], dim=0)
        data_dim = agent_z.shape[0]
        domain_labels = torch.cat([torch.zeros([data_dim, 1]), torch.ones([data_dim, 1])], dim=0).to(self.device)
        data = torch.cat([agent_z, expert_z], dim=0)
        rand_idx = torch.randperm(domain_labels.shape[0])
        shuffled_domain_labels = domain_labels[rand_idx]
        positive = torch.cat([data, domain_labels], dim=1)
        negative = torch.cat([data, shuffled_domain_labels], dim=1)

        m1_positive, m2_positive = self.mi_estimator(positive)
        m1_negative, m2_negative = self.mi_estimator(negative) 

        self.unbiased_mi_ma1 = (self.unbiased_mi_decay*self.unbiased_mi_ma1 + (1 - self.unbiased_mi_decay)*torch.mean(torch.exp(m1_negative))).detach()
        self.unbiased_mi_ma2 = (self.unbiased_mi_decay*self.unbiased_mi_ma2 + (1 - self.unbiased_mi_decay)*torch.mean(torch.exp(m2_negative))).detach()

        m1_loss = torch.mean(m1_positive) - torch.log(torch.mean(torch.exp(m1_negative)))
        m2_loss = torch.mean(m2_positive) - torch.log(torch.mean(torch.exp(m2_negative))) 

        unbiased_m1_loss = torch.mean(m1_positive) - torch.mean(torch.exp(m1_negative))/self.unbiased_mi_ma1
        unbiased_m2_loss = torch.mean(m2_positive) - torch.mean(torch.exp(m2_negative))/self.unbiased_mi_ma2

        return unbiased_m1_loss, unbiased_m2_loss, m1_loss, m2_loss    

    def update_mi_estimator(self, z_a, next_z_a, z_e, next_z_e):
        metrics = dict()

        unbiased_m1_loss, unbiased_m2_loss, _, _ = self.compute_mi_loss(z_a, next_z_a, z_e, next_z_e)
        loss = -1*(unbiased_m1_loss + unbiased_m2_loss)

        self.mi_estimator_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.mi_estimator_opt.step()

        if self.use_tb:
            metrics['mi_estimator_loss'] = loss.item()
            metrics['unbiased_MI_MA1'] = self.unbiased_mi_ma1.item()
            metrics['unbiased_MI_MA2'] = self.unbiased_mi_ma2.item()

        return metrics

    def update(self, replay_iter, replay_iter_expert, replay_iter_random, replay_iter_expert_random, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, reward_a, discount, next_obs = utils.to_torch(batch, self.device)
        
        batch_expert = next(replay_iter_expert)
        obs_e_raw, action_e, _, _, next_obs_e_raw = utils.to_torch(batch_expert, self.device)
        
        obs_e = self.aug(obs_e_raw.float())
        next_obs_e = self.aug(next_obs_e_raw.float())
        obs_a = self.aug(obs.float())
        next_obs_a = self.aug(next_obs.float())

        if step % self.check_every_steps == 0:
            self.check_aug(obs_a, next_obs_a, obs_e, next_obs_e, "learning_buffer", step)

        # sample random data
        batch_agent_random = next(replay_iter_random)
        obs_random, _, _, _, next_obs_random = utils.to_torch(batch_agent_random, self.device)

        batch_expert_random = next(replay_iter_expert_random)
        obs_e_raw_random, _, _, _, next_obs_e_raw_random = utils.to_torch(batch_expert_random, self.device)

        obs_e_random = self.aug(obs_e_raw_random.float())
        next_obs_e_random = self.aug(next_obs_e_raw_random.float())
        obs_a_random = self.aug(obs_random.float())
        next_obs_a_random = self.aug(next_obs_random.float())

        if step % self.check_every_steps == 0:
            self.check_aug(obs_a_random, next_obs_a_random, obs_e_random, next_obs_e_random, "random_buffer", step)

        z_e = self.encoder(obs_e)
        next_z_e = self.encoder(next_obs_e)

        z_a = self.encoder(obs_a)
        next_z_a = self.encoder(next_obs_a)

        z_e_random = self.encoder(obs_e_random)
        next_z_e_random = self.encoder(next_obs_e_random)

        z_a_random = self.encoder(obs_a_random)
        next_z_a_random = self.encoder(next_obs_a_random)

        metrics.update(self.update_CL(obs,
                                      z_a, 
                                      next_z_a, 
                                      z_e, 
                                      next_z_e, 
                                      z_a_random, 
                                      next_z_a_random, 
                                      z_e_random, 
                                      next_z_e_random, 
                                      self.check_every_steps, 
                                      step))

        with torch.no_grad():
            z_e = self.encoder(obs_e)
            next_z_e = self.encoder(next_obs_e)
            z_a = self.encoder(obs_a)
            next_z_a = self.encoder(next_obs_a)

        metrics.update(self.update_mi_estimator(z_a, next_z_a, z_e, next_z_e)) #Mutual information estimator update estimator

        # update critic
        if self.from_dem:
            metrics.update(self.update_discriminator(z_a, action, z_e, action_e))
            reward, metrics_r = self.compute_reward(obs, action)
        else:
            metrics.update(self.update_discriminator(z_a, next_z_a, z_e, next_z_e))
            reward, metrics_r = self.compute_reward(obs, next_obs)

        metrics.update(metrics_r)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward_a.mean().item()

        # update critic
        metrics.update(self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target, self.critic_target_tau)

        return metrics
    
    def check_aug(self, obs, next_obs, obs_e, next_obs_e, type, step):

        if not os.path.exists(f'checkimages_{type}'):
            os.makedirs(f"checkimages_{type}")

        obs = obs/255
        next_obs = next_obs/255
        obs_e = obs_e/255
        next_obs_e = next_obs_e/255

        obs = torch.cat([obs, next_obs], dim=0)
        obs_e = torch.cat([obs_e, next_obs_e])
        rand_idx = torch.randperm(obs.shape[0])
        imgs1 = obs[rand_idx[:9]]
        imgs2 = obs[rand_idx[-9:]]
        imgs3 = obs_e[rand_idx[9:18]]
        imgs4 = obs_e[rand_idx[-18:-9]]
                
        saved_imgs = torch.cat([imgs1[:,:3,:,:], imgs2[:,:3,:,:], imgs3[:,:3,:,:], imgs4[:,:3,:,:]], dim=0)
        save_image(saved_imgs, f"./checkimages_{type}/{step}.png", nrow=9)

    def check_aug_CL(self, img1, img2, step):

        if not os.path.exists('checkaugs'):
            os.makedirs("checkaugs")

        img1 = img1/255 # -> [0, 1]
        img2 = img2/255 # -> [0, 1]

        rand_idx = torch.randperm(img1.shape[0])

        imgs1 = img1[rand_idx[:9]]
        imgs2 = img2[rand_idx[:9]]

        imgs3 = img1[rand_idx[-9:]]
        imgs4 = img2[rand_idx[-9:]]
                
        saved_imgs = torch.cat([imgs1[:,:3,:,:], imgs2[:,:3,:,:], imgs3[:,:3,:,:], imgs4[:,:3,:,:]], dim=0)
        save_image(saved_imgs, "./checkaugs/%d.png" % (step), nrow=9)
