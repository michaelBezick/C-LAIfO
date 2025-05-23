import numpy as np
import sys
import matplotlib.pyplot as plt
from dm_control import suite
import matplotlib.pyplot as plt

# Load the walker environment
env = suite.load(domain_name="walker", task_name="walk")
time_step = env.reset()
physics = env.physics

depth_map = physics.render(height=480, width=640, depth=True, camera_id=0)
rgb_image = physics.render(height=480, width=640, camera_id=0)


print(np.shape(depth_map))
print(np.max(depth_map))
print(np.min(depth_map))
print(np.mean(depth_map))

min_depth, max_depth = 1, 6
clipped_depth = np.clip(depth_map, min_depth, max_depth)
normalized_depth = (clipped_depth - min_depth) / (max_depth - min_depth)
depth_map = normalized_depth

data = depth_map.flatten()
counts, bins = np.histogram(data, bins=100)
plt.figure()
plt.stairs(counts, bins)

plt.title("depth")

plt.savefig("./depth_test/hist.png", dpi=300)


# Display the RGB image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title("RGB Image")

# Display the depth map
plt.subplot(1, 2, 2)
plt.imshow(depth_map, cmap="viridis")
plt.colorbar(label="Depth")
plt.title("Depth Map")

plt.savefig("./depth_test/3rd.png", dpi=300)
