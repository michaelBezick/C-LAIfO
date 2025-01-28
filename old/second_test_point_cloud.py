import open3d as o3d
import numpy as np
from dm_control import suite

# Load the walker environment
env = suite.load(domain_name="walker", task_name="walk")
time_step = env.reset()
physics = env.physics

# Extract all positions (point cloud data)
all_positions = physics.data.geom_xpos
point_cloud = np.array(all_positions)

# Create a PointCloud object in Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)

# Set up an offscreen renderer
width = 800
height = 800
renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

# Configure the scene
renderer.scene.add_geometry("point_cloud", pcd, o3d.visualization.rendering.MaterialRecord())
renderer.scene.camera.look_at([0, 0, 0], [0, 0, 1.5], [0, 1, 0])  # Adjust the viewpoint
renderer.scene.set_background([0, 0, 0, 1])  # Black background

# Render and save the image
image = renderer.render_to_image()
o3d.io.write_image("point_cloud_output.png", image)

print("Point cloud visualization saved as point_cloud_output.png")

