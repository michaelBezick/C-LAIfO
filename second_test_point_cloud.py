from dm_control import suite
import open3d as o3d
from matplotlib.figure import projections
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

env = suite.load(domain_name = "walker", task_name = "walk")

time_step = env.reset()

physics = env.physics

walker_positions = physics.named.data.geom_xpos['walker/torso']

print("Torso Position:", walker_positions)

all_positions = physics.data.geom_xpos
print("All positions (point cloud):", all_positions)

point_cloud = np.array(all_positions)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
o3d.open3d.visualization.draw([pcd])
