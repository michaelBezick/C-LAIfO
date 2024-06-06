#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

from DeepMind_control import dmc, dmc_expert
from utils_folder import utils
from logger_folder.logger import Logger
from video import TrainVideoRecorder, VideoRecorder, VideoRecorder_bio_expert

torch.backends.cudnn.benchmark = True

def make_env_expert(cfg):
    """Helper function to create dm_control environment"""
    domain, task = cfg.task_name_expert.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = dmc_expert.make_remastered_states_only(domain_name=domain,
                                                 task_name=task,
                                                 seed=cfg.seed,
                                                 visual_seed=cfg.visual_seed_source,
                                                 delta = cfg.delta_source,
                                                 height=cfg.image_height,
                                                 width=cfg.image_width,
                                                 camera_id=0,
                                                 frame_skip=1,
                                                 num_frames = cfg.frame_stack,
                                                 vary = cfg.vary,
                                                 depth_flag = cfg.depth_flag,
                                                 segm_flag = cfg.segm_flag,
                                                )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env

def make_env_agent(cfg):
    """Helper function to create dm_control environment"""
    domain, task = cfg.task_name_expert.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)

    env = dmc_expert.make_remastered_states_only(domain_name=domain,
                                                 task_name=task,
                                                 seed=cfg.seed,
                                                 visual_seed=cfg.visual_seed_target,
                                                 delta = cfg.delta_target,
                                                 height=cfg.image_height,
                                                 width=cfg.image_width,
                                                 camera_id=0,
                                                 frame_skip=1,
                                                 num_frames = cfg.frame_stack,
                                                 vary = cfg.vary,
                                                 depth_flag = cfg.depth_flag,
                                                 segm_flag = cfg.segm_flag,
                                                )
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1
    return env

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
              
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        # create target envs and agent                                  
        self.eval_env = make_env_agent(self.cfg)

        # create replay buffer
        data_specs = (self.eval_env.observation_spec(),
                      self.eval_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_buffer = hydra.utils.instantiate(self.cfg.replay_buffer, data_specs=data_specs)
        self.replay_buffer_random = hydra.utils.instantiate(self.cfg.replay_buffer_expert)

        #create source envs and agent
        self.expert_env = make_env_expert(self.cfg)
        self.cfg.expert.obs_dim = self.expert_env.observation_space.shape[0]
        self.cfg.expert.action_dim = self.expert_env.action_space.shape[0]
        self.cfg.expert.action_range = [float(self.expert_env.action_space.low.min()),
                                        float(self.expert_env.action_space.high.max())]
        
        self.expert = hydra.utils.instantiate(self.cfg.expert)
        self.replay_buffer_expert = hydra.utils.instantiate(self.cfg.replay_buffer_expert)
        self.replay_buffer_random_expert = hydra.utils.instantiate(self.cfg.replay_buffer_expert)

        self.video_recorder = VideoRecorder(self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(self.work_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat
    
    def store_expert_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        
        while eval_until_episode(episode):
            obs, time_step = self.expert_env.reset()
            self.expert.reset()
            self.video_recorder.init(self.expert_env, enabled=(episode == 0))
            
            extended_time_step = self.expert_env.step_learn_from_pixels(time_step)
            self.replay_buffer_expert.add(extended_time_step)
            
            done = False
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.global_step, eval_mode=True)
                obs, reward, done, _, time_step = self.expert_env.step(action)    
                
                extended_time_step = self.expert_env.step_learn_from_pixels(time_step, action)
                self.replay_buffer_expert.add(extended_time_step)
                self.video_recorder.record(self.expert_env)
                
                total_reward += extended_time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save('expert.mp4')

        print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')

    def store_random_expert_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        self.expert.num_expl_steps = 1.0
        
        while eval_until_episode(episode):
            obs, time_step = self.expert_env.reset()
            self.expert.reset()
            self.video_recorder.init(self.expert_env, enabled=(episode == 0))
            
            extended_time_step = self.expert_env.step_learn_from_pixels(time_step)
            self.replay_buffer_random_expert.add(extended_time_step)
            
            done = False
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.global_step, eval_mode=False)
                obs, reward, done, _, time_step = self.expert_env.step(action)    
                
                extended_time_step = self.expert_env.step_learn_from_pixels(time_step, action)
                self.replay_buffer_random_expert.add(extended_time_step)
                self.video_recorder.record(self.expert_env)
                
                total_reward += extended_time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save('random_expert.mp4')

        print(f'Average random expert reward: {total_reward / episode}, Total number of samples: {step}')

    def store_agent_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        
        while eval_until_episode(episode):
            obs, time_step = self.eval_env.reset()
            self.expert.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            
            extended_time_step = self.eval_env.step_learn_from_pixels(time_step)
            self.replay_buffer.add(extended_time_step)
            
            done = False
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.global_step, eval_mode=True)

                obs, reward, done, _, time_step = self.eval_env.step(action)    
                
                extended_time_step = self.eval_env.step_learn_from_pixels(time_step, action)
                self.replay_buffer.add(extended_time_step)
                self.video_recorder.record(self.eval_env)
                
                total_reward += extended_time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save('agent.mp4')

        print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')

    def store_random_agent_transitions(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
        self.expert.num_expl_steps = 1.0
        
        while eval_until_episode(episode):
            obs, time_step = self.eval_env.reset()
            self.expert.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            
            extended_time_step = self.eval_env.step_learn_from_pixels(time_step)
            self.replay_buffer_random.add(extended_time_step)
            
            done = False
            
            while not done:
                with torch.no_grad(), utils.eval_mode(self.expert):
                    action = self.expert.act(obs, self.global_step, eval_mode=False)

                obs, reward, done, _, time_step = self.eval_env.step(action)    
                
                extended_time_step = self.eval_env.step_learn_from_pixels(time_step, action)
                self.replay_buffer_random.add(extended_time_step)
                self.video_recorder.record(self.eval_env)
                
                total_reward += extended_time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save('random_agent.mp4')

        print(f'Average random expert reward: {total_reward / episode}, Total number of samples: {step}')
        

    def save_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name_agent}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def save_buffer(self):
        snapshot = self.work_dir / f'replay_buffer_{self.cfg.task_name_agent}.pt'
        keys_to_save = ['replay_buffer']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def save_buffer_random(self):
        snapshot = self.work_dir / f'replay_buffer_random_{self.cfg.task_name_agent}.pt'
        keys_to_save = ['replay_buffer_random']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def save_expert_buffer(self):
        snapshot = self.work_dir / f'replay_buffer_expert_{self.cfg.task_name_expert}.pt'
        keys_to_save = ['replay_buffer_expert']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def save_expert_buffer_random(self):
        snapshot = self.work_dir / f'replay_buffer_expert_random_{self.cfg.task_name_expert}.pt'
        keys_to_save = ['replay_buffer_random_expert']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f, pickle_protocol=4)

    def load_snapshot(self):
        snapshot = self.work_dir / f'snapshot_{self.cfg.task_name_agent}.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v
        
    def load_expert(self, snapshot):
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        self.expert = payload['agent']

@hydra.main(config_path='config_folder/POMDP', config_name='generate_data')
def main(cfg):
    from generate_data import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    parent_dir = root_dir.parents[3]
    snapshot = parent_dir / f'expert_policies/snapshot_{cfg.task_name_expert}_frame_skip_{cfg.frame_skip}.pt'
    assert snapshot.exists()
    print(f'loading expert target: {snapshot}')
    workspace.load_expert(snapshot)
    
    workspace.store_expert_transitions()
    workspace.store_random_expert_transitions()
    workspace.store_agent_transitions()
    workspace.store_random_agent_transitions()

    workspace.save_expert_buffer()
    workspace.save_expert_buffer_random()
    workspace.save_buffer()
    workspace.save_buffer_random()


if __name__ == '__main__':
    main()