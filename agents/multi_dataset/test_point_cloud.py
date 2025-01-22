import numpy as np
from dm_control import suite
from dm_control import mujoco
import math

# 1) Create a dm_control environment
env = suite.load(domain_name="cartpole", task_name="swingup")
physics = env.physics

# 2) Render a depth image
camera_id = 0
width, height = 320, 240

rgbd = physics.render(
    camera_id=camera_id,
    width=width,
    height=height,
    depth=True
)
rgb = rgbd[..., :3]
depth_buf = rgbd[..., 3]

# 3) Convert from z-buffer to actual depth Z
model = physics.model
near = model.cam_near[camera_id]
far = model.cam_far[camera_id]
Z = (near * far) / (far - (far - near) * depth_buf)

# 4) Compute intrinsics
fovy_deg = model.cam_fovy[camera_id]
fovy_rad = math.radians(fovy_deg)
f_y = (height / 2.) / math.tan(fovy_rad / 2.)
f_x = f_y
c_x = width / 2.
c_y = height / 2.

# 5) Create pixel coordinate grid
us = np.arange(width)
vs = np.arange(height)
u_grid, v_grid = np.meshgrid(us, vs)
u_grid = u_grid.astype(np.float32)
v_grid = v_grid.astype(np.float32)

# 6) Re-project to camera coordinates
X = (u_grid - c_x) / f_x * Z
Y = (v_grid - c_y) / f_y * Z
points_cam = np.stack([X, Y, Z], axis=-1)  # shape (H, W, 3)

# 7) Optional: Flatten + remove invalid depths
points_cam = points_cam.reshape(-1, 3)
mask = np.isfinite(points_cam[:, 2]) & (points_cam[:, 2] > 0)
points_cam = points_cam[mask]

# 8) Optional: Transform to world frame
cam_quat = model.cam_quat[camera_id]  # [w, x, y, z]
cam_pos = model.cam_pos[camera_id]    # [x, y, z]

# We'll use dm_control's built-in transform utilities:
R = mujoco.mjlib.mju_quat2Mat(cam_quat)
R = np.array(R).reshape(3, 3)  # 3x3 rotation matrix
points_world = (R @ points_cam.T).T + cam_pos

# Now 'points_world' is an Nx3 array in world coordinates

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_world)
o3d.visualization.draw_geometries([pcd])
