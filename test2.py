import numpy as np
import torch
from dm_control import suite

from buffers.np_replay_buffer import EfficientReplayBuffer

env = suite.load(domain_name="walker", task_name="walk")
physics = env.physics
buffer = EfficientReplayBuffer(
    buffer_size=1000,
    batch_size=100,
    nstep=3,
    discount=0.99,
    frame_stack=3,
    physics=physics,
)

depth = physics.render(
    width=64,
    height=64,
    camera_id=physics.cam_names[0],
    depth=True,
)
