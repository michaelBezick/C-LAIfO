#!/usr/bin/env python3
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

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


def make_agent(obs_spec, action_spec, env, cfg, physics):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg, physics=physics)

def make_env_expert(cfg, expert_physics):
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
                                                 physics=expert_physics
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

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.train_env,
                                self.cfg.agent,
                                self.train_env.physics)
              
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create target envs and agent
        """By changing visual seed targets, will change """
        self.train_env = dmc.make_remastered(self.cfg.task_name_agent, self.cfg.frame_stack,
                                            self.cfg.action_repeat, self.cfg.seed, self.cfg.visual_seed_target,
                                            self.cfg.vary, self.cfg.delta_target, self.cfg.image_height, self.cfg.image_width,
                                            self.cfg.depth_flag, self.cfg.segm_flag)
                                                
        self.eval_env = dmc.make_remastered(self.cfg.task_name_agent, self.cfg.frame_stack,
                                            self.cfg.action_repeat, self.cfg.seed, self.cfg.visual_seed_target,
                                            self.cfg.vary, self.cfg.delta_target, self.cfg.image_height, self.cfg.image_width,
                                            self.cfg.depth_flag, self.cfg.segm_flag)

        """ACTUALLY PHYSICS ARE THE SAME, WHAT CHANGES IS CAMERA ID"""
        """ONLY THE ENVIRONMENTS NEED POINT CLOUD GENERATOR NOT REPLAY BUFFER"""
        self.train_physics = self.train_env.physics
        self.eval_physics = self.eval_env.physics
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      )

        self.replay_buffer = hydra.utils.instantiate(self.cfg.replay_buffer, data_specs=data_specs, physics=self.train_physics)
        # self.replay_buffer_random = hydra.utils.instantiate(self.cfg.replay_buffer_expert, physics=self.train_physics) #don't need

        #create source envs and agent
        self.expert_env = make_env_expert(self.cfg, self.train_env.physics)
        self.expert_physics = self.expert_env.physics
        self.cfg.expert.obs_dim = self.expert_env.observation_space.shape[0]
        self.cfg.expert.action_dim = self.expert_env.action_space.shape[0]
        self.cfg.expert.action_range = [float(self.expert_env.action_space.low.min()),
                                        float(self.expert_env.action_space.high.max())]
        
        self.expert = hydra.utils.instantiate(self.cfg.expert)
        # self.replay_buffer_expert = hydra.utils.instantiate(self.cfg.replay_buffer_expert, physics=self.train_physics)
        # self.replay_buffer_random_expert = hydra.utils.instantiate(self.cfg.replay_buffer_expert, physics=self.train_physics)#don't need

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
    
    # def store_expert_transitions(self):
    #     step, episode, total_reward = 0, 0, 0
    #     eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
    #     #eval_until_episode = utils.Until(1)
    #     
    #     while eval_until_episode(episode):
    #         obs, time_step = self.expert_env.reset()
    #         self.expert.reset()
    #         self.video_recorder.init(self.expert_env, enabled=(episode == 0))
    #         
    #         extended_time_step = self.expert_env.step_learn_from_pixels(time_step)
    #         self.replay_buffer_expert.add(extended_time_step)
    #         
    #         done = False
    #         
    #         while not done:
    #             with torch.no_grad(), utils.eval_mode(self.expert):
    #                 action = self.expert.act(obs, self.global_step, eval_mode=True)
    #             obs, reward, done, _, time_step = self.expert_env.step(action)    
    #             
    #             extended_time_step = self.expert_env.step_learn_from_pixels(time_step, action)
    #             self.replay_buffer_expert.add(extended_time_step)
    #             self.video_recorder.record(self.expert_env)
    #             
    #             total_reward += extended_time_step.reward
    #             step += 1
    #
    #         episode += 1
    #         self.video_recorder.save('expert.mp4')
    #
    #     print(f'Average expert reward: {total_reward / episode}, Total number of samples: {step}')

    # def store_random_expert_transitions(self):
    #     step, episode, total_reward = 0, 0, 0
    #     eval_until_episode = utils.Until(self.cfg.num_expert_episodes)
    #     # eval_until_episode = utils.Until(1)
    #     self.expert.num_expl_steps = 1.0
    #     
    #     while eval_until_episode(episode):
    #         obs, time_step = self.expert_env.reset()
    #         self.expert.reset()
    #         self.video_recorder.init(self.expert_env, enabled=(episode == 0))
    #         
    #         extended_time_step = self.expert_env.step_learn_from_pixels(time_step)
    #         self.replay_buffer_random_expert.add(extended_time_step)
    #         
    #         done = False
    #         
    #         while not done:
    #             with torch.no_grad(), utils.eval_mode(self.expert):
    #                 action = self.expert.act(obs, self.global_step, eval_mode=False)
    #             obs, reward, done, _, time_step = self.expert_env.step(action)    
    #             
    #             extended_time_step = self.expert_env.step_learn_from_pixels(time_step, action)
    #             self.replay_buffer_random_expert.add(extended_time_step)
    #             self.video_recorder.record(self.expert_env)
    #             
    #             total_reward += extended_time_step.reward
    #             step += 1
    #
    #         episode += 1
    #         self.video_recorder.save('random_expert.mp4')
    #
    #     print(f'Average random expert reward: {total_reward / episode}, Total number of samples: {step}')
        
    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        # eval_until_episode = utils.Until(1)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True, from_buffer=False)
                
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            print(episode)
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        """
        OKAY I believe that the replay buffer adding point cloud thing is always going to be from depth image
        This means that I can keep the dequeue, and use it as such to only extract the final point cloud
        """
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        
        random_until_step = utils.Until(self.cfg.num_expl_frames,
                                        self.cfg.action_repeat)
        
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()

        self.replay_buffer.add(time_step, point_cloud=False)
        # self.replay_buffer_random.add(time_step, point_cloud=False)

        self.train_video_recorder.init(time_step.observation)
        metrics = None
        """
        ISSUE, self.env.step_learn_from_pixels(time_step, action) is where expert point cloud is made
        """
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_buffer))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step, point_cloud=False)

                # if random_until_step(self.global_step):
                #     self.replay_buffer_random.add(time_step, point_cloud=False)

                self.train_video_recorder.init(time_step.observation)
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

                if self.cfg.save_replay_buffers:
                    self.save_buffer_random()
                    self.save_buffer()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False, from_buffer=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_buffer, 
                                            replay_iter_expert=None, 
                                            replay_iter_random=None, 
                                            replay_iter_expert_random=None, 
                                            step=self.global_step)
                
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_buffer.add(time_step, point_cloud=False)

            # if random_until_step(self.global_step):
            #     self.replay_buffer_random.add(time_step, point_cloud=False)

            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

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

@hydra.main(config_path='config_folder/POMDP', config_name='debug_config_lail_MI')
def main(cfg):
    from train_LAIL_MI import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    parent_dir = root_dir.parents[3]
    # snapshot = parent_dir / f'expert_policies/snapshot_{cfg.task_name_expert}_frame_skip_{cfg.frame_skip}.pt'
    # assert snapshot.exists()
    # print(f'loading expert target: {snapshot}')
    # workspace.load_expert(snapshot)
    # workspace.store_expert_transitions()
    # workspace.store_random_expert_transitions()

    # if cfg.save_replay_buffers:
    #     workspace.save_expert_buffer()
    #     workspace.save_expert_buffer_random()

    workspace.train()

if __name__ == '__main__':
    main()
