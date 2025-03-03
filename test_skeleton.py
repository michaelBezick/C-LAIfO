import open3d as o3d
import time
import numpy as np
import sys
import os

# point_cloud = o3d.io.read_point_cloud("./point_cloud_images/filtered_point_cloud.ply")

import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
from dm_control.suite.walker import stand
from dm_control import mujoco
from pc_skeletor import LBC

# Load the walker environment
env = suite.load(domain_name="walker", task_name="walk")

# Access physics engine
physics = env.physics

# Step the environment a few times
action_spec = env.action_spec()
for _ in range(10):
    action = np.random.uniform(action_spec.minimum, action_spec.maximum)
    env.step(action)

# Render a depth image
depth = physics.render(
    height=64,
    width=64,
    camera_id=0,  # Default camera
    depth=True
)

from point_cloud_generator import PointCloudGenerator

pcg = PointCloudGenerator(sim=physics)

point_cloud = pcg.depthImageToPointCloud(depth, 0, down_sample_voxel_size=-1)

new_point_cloud = o3d.geometry.PointCloud()
new_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
pcg.save_point_cloud(new_point_cloud, is_point_cloud=True, output_file="random_point_cloud_unfiltered.ply")
point_cloud = new_point_cloud

points = np.asarray(point_cloud.points)
x_min, y_min, z_min = points.min(axis=0)
x_max, y_max, z_max = points.max(axis=0)


valid_indices = np.where(points[:, 2] >= -1.2)[0]
pcd_filtered = point_cloud.select_by_index(valid_indices)

pcg.save_point_cloud(pcd_filtered, is_point_cloud=True, output_file="random_point_cloud_filtered.ply")

point_cloud = pcd_filtered

time1 = time.time()
lbc = LBC(point_cloud=point_cloud, down_sample=-1)
lbc.extract_skeleton()
lbc.extract_topology()
time2 = time.time()

print("TIME TIME TIME TIME: ", time2 - time1)
lbc.visualize()
lbc.export_results('./output')
lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            steps=300,
            output='./output')
