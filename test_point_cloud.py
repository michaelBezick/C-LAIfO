import math
import numpy as np
from dm_control import suite
from dm_control import mujoco
import open3d as o3d

# 1) Create a dm_control environment
env = suite.load(domain_name="cartpole", task_name="swingup")
physics = env.physics

# 2) Render a depth image from the default (free) camera
camera_id = 0   # free camera ID in dm_control is usually 0
width, height = 320, 240
rgbd = physics.render(
    camera_id=camera_id,
    width=width,
    height=height,
    depth=True
)
#rgb = rgbd[..., :3]
#depth_buf = rgbd[..., 3]
depth_buf = rgbd

# 3) Convert from z-buffer to actual depth Z
model = physics.model
near = model.vis.map_.znear
far = model.vis.map_.zfar
z_buffer = depth_buf  # in [0,1]
Z = (near * far) / (far - (far - near) * z_buffer)

# 4) Approximate the intrinsics
# If you haven't overridden the free camera FOV, you might guess something like 45 deg
# Or you can read model.vis.global_.fovy if it's set
fovy_deg = model.vis.global_.fovy  # often the default is 45
fovy_rad = math.radians(fovy_deg)
f_y = (height / 2.) / math.tan(fovy_rad / 2.)
f_x = f_y
c_x = width / 2.
c_y = height / 2.

# 5) Create the pixel coordinate grid
us = np.arange(width, dtype=np.float32)
vs = np.arange(height, dtype=np.float32)
u_grid, v_grid = np.meshgrid(us, vs)

# 6) Re-project into CAMERA coordinates
X = (u_grid - c_x) / f_x * Z
Y = (v_grid - c_y) / f_y * Z
points_cam = np.stack([X, Y, Z], axis=-1)  # shape (H, W, 3)

# 7) Flatten + remove invalid/inf depths
points_cam = points_cam.reshape(-1, 3)
mask = np.isfinite(points_cam[:, 2]) & (points_cam[:, 2] > 0)
points_cam = points_cam[mask]

# 8) Get the free camera pose from physics.data (not from model.cam_quat)
#   data.cam_xpos[camera_id] is the camera position
#   data.cam_xmat[camera_id] is a 3x3 rotation matrix in row-major format
cam_pos = physics.data.cam_xpos[camera_id]
cam_mat = physics.data.cam_xmat[camera_id].reshape(3, 3)

# Transform from camera frame to world frame
points_world = (cam_mat @ points_cam.T).T + cam_pos

# 9) Visualize with Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_world)
o3d.visualization.draw_geometries([pcd])

