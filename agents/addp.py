from __future__ import annotations

import random
from copy import deepcopy
from typing import Optional

import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from torchvision.utils import save_image

from dda.operations import *

from utils_folder import utils
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

import os

class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 temperature: float,
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            return (torch.stack([op(input) for op in self.operations]) * self.weights.view(-1, 1, 1, 1, 1)).sum(0)
        else:
            return self.operations[Categorical(self.weights).sample()](input)

    @property
    def weights(self):
        return self._weights.div(self.temperature).softmax(0)

class SubPolicy(nn.Module):
    def __init__(self,
                 sub_policy_stage: SubPolicyStage,
                 operation_count: int,
                 ):
        super(SubPolicy, self).__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) for _ in range(operation_count)])

    def forward(self, input: Tensor) -> Tensor:
        for stage in self.stages:
            input = stage(input)
        return input

class Policy(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 num_sub_policies: int,
                 temperature: float = 0.05,
                 operation_count: int = 2,
                 mean: float = 0.5,
                 std: float = 0.5,
                 ):
        super(Policy, self).__init__()

        self.num_sub_policies = num_sub_policies

        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations, temperature), operation_count) for _ in range(self.num_sub_policies)])
        self.temperature = temperature
        self.operation_count = operation_count
        self._mean = mean
        self._std = std

        for p in self.parameters():
            nn.init.uniform_(p, 0, 1)

    def forward(self, input: Tensor) -> Tensor:
        # [0, 1]
        x = self._forward(input)
        return x # [0, 1]

    def _forward(self, input: Tensor) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](input)

    def normalize_(self,
                   input: Tensor
                   ) -> Tensor:
        # [0, 1] -> [-1, 1]
        return (input - self._mean)/self._std

    def denormalize_(self,
                     input: Tensor
                     ) -> Tensor:
        # [-1, 1] -> [0, 1]
        return input*self._std + self._mean

    @staticmethod
    def dda_operations(subset = 'all'):

        operations_dict = dict()

        operations_dict['all'] = [ShearX(),
                                ShearY(),
                                TranslateY(),
                                TranslateY(),
                                Rotate(),
                                HorizontalFlip(),
                                Invert(),
                                Solarize(),
                                Posterize(),
                                Gray(),
                                Hue(),
                                Contrast(),
                                Saturate(),
                                Brightness(),
                                Sharpness(),
                                AutoContrast(),
                                Equalize(),
                            ]
        
        operations_dict['brightness'] = [Brightness()]

        operations_dict['color'] = [Invert(),
                                    Solarize(),
                                    Posterize(),
                                    Gray(),
                                    Hue(),
                                    Contrast(),
                                    Saturate(),
                                    Brightness(),
                                    Sharpness(),
                                    AutoContrast(),
                                    Equalize(),
                                ]

        return operations_dict[subset]

    @staticmethod
    def faster_auto_augment_policy(num_sub_policies: int,
                                temperature: float,
                                operation_count: int,
                                operations_subset: str,
                                mean: Optional[float] = None,
                                std: Optional[float] = None,
                                ) -> Policy:
        
        if mean is None or std is None:
            mean = 0.5
            std = 0.5

        operations_types = Policy.dda_operations(operations_subset)

        return Policy(nn.ModuleList(operations_types), num_sub_policies, temperature, operation_count, mean=mean, std=std)
    
class ADDPAgent:
    def __init__(self,
                 device,
                 num_sub_policies: int,
                 temperature: float,
                 operation_count: int,
                 operations_subset: str,
                 aug_policy_lr: float,
                 check_every_steps: int,
                 mean = None,
                 std = None
                 ):
        
        self.device = device
        self.check_every_steps = check_every_steps

        self.ssim_loss = SSIM(data_range = 1.0)

        self.aug_policy = Policy.faster_auto_augment_policy(num_sub_policies,
                                                            temperature,
                                                            operation_count,
                                                            operations_subset,
                                                            mean,
                                                            std).to(device)
        
        self.aug_policy_opt = torch.optim.Adam(self.aug_policy.parameters(), 
                                               lr=aug_policy_lr)
        
        self.train()

    def train(self, training=True):
        self.aug_policy.train(training)

    def normalize(self, obs):
        return obs / 255

    def zerocenter(self, obs):
        obs = self.normalize(obs) #[0, 1]
        obs = (obs - 0.5)/0.5 # [0, 1] -> [-1, 1]
        return obs
    
    def update(self, replay_iter_random, replay_iter_expert_random, step):

        # sample random data
        obs_agent_random = replay_iter_random.gather_images()
        obs_random = torch.as_tensor(obs_agent_random, device=self.device)

        obs_expert_random = replay_iter_expert_random.gather_images()
        obs_e_raw_random = torch.as_tensor(obs_expert_random, device=self.device)

        obs_a = self.normalize(obs_random.float()) # [0, 1]
        obs_e = self.aug_policy(self.normalize(obs_e_raw_random.float())) #[0, 1]

        if step % self.check_every_steps == 0:
            self.check_aug(obs_e, obs_a, step)

        self.aug_policy_opt.zero_grad()
        _ssim_loss = 1 - self.ssim_loss(obs_a, obs_e)
        _ssim_loss.backward()
        self.aug_policy_opt.step()

        ssim_value = ssim(obs_a, obs_e)

        return ssim_value.item()

    def check_aug(self, obs_e, obs_a, step):

        if not os.path.exists('checkimages_aug_policy'):
            os.makedirs("checkimages_aug_policy")

        rand_idx_e = torch.randperm(obs_e.shape[0])
        rand_idx_a = torch.randperm(obs_a.shape[0])
        imgs1 = obs_e[rand_idx_e[:9]]
        imgs2 = obs_e[rand_idx_e[-9:]]
        imgs3 = obs_a[rand_idx_a[:9]]
        imgs4 = obs_a[rand_idx_a[-9:]]

                
        saved_imgs = torch.cat([imgs1[:,:3,:,:], imgs2[:,:3,:,:], imgs3[:,:3,:,:], imgs4[:,:3,:,:]], dim=0)
        save_image(saved_imgs, "./checkimages_aug_policy/%d.png" % (step), nrow=9)






    
