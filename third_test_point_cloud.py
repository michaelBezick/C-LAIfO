import numpy as np
import sys
import matplotlib.pyplot as plt
from dm_control import suite

# Load the walker environment
env = suite.load(domain_name="walker", task_name="walk")
time_step = env.reset()
physics = env.physics

depth_map = physics.render(height=480, width=640, depth=True, camera_id=0)
rgb_image = physics.render(height=480, width=640, camera_id=0)


np.set_printoptions(threshold=sys.maxsize)
print(np.shape(depth_map))
print(np.max(depth_map))
print(np.min(depth_map))
print(np.mean(depth_map))

data = depth_map.flatten()
counts, bins = np.histogram(data, bins=100)
plt.stairs(counts, bins)
plt.savefig("hist.png", dpi=300)
exit()


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

plt.savefig("third.png", dpi=300)
