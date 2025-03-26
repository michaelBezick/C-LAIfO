import numpy as np
import matplotlib.pyplot as plt
from dm_control import suite
from point_cloud_generator import PointCloudGenerator

# Create the Walker Walk environment
env = suite.load(domain_name="walker", task_name="walk")

pcg = PointCloudGenerator(env.physics)

# Number of random frames to generate
num_frames = 10  

# Camera name (default camera)
camera_name = "side"

for i in range(num_frames):
    # Reset environment
    time_step = env.reset()

    # Take a random action
    action_spec = env.action_spec()
    random_action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    
    # Step environment
    time_step = env.step(random_action)

    # Render depth image (returns float values in [0,1])
    depth_image = env.physics.render(height=128, width=128, camera_id=camera_name, depth=True)

    numpy_point_cloud = pcg.depthImageToPointCloud(depth_image, 0)
    pcg.save_point_cloud(numpy_point_cloud, is_point_cloud=False, output_file=f"pca_point_cloud_{i}.ply")
