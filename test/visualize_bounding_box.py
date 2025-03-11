import matplotlib.pyplot as plt
import numpy as np
# Generate points strictly on the edge of the circle with non-uniform density
num_points_high_density = 300
num_points_low_density = 3

angles_high = np.random.uniform(0, np.pi, num_points_high_density)  # More points in upper half
angles_low = np.random.uniform(np.pi, 2 * np.pi, num_points_low_density)  # Fewer points in lower half

x_high = np.cos(angles_high)  # Radius is 1 (circle edge)
y_high = np.sin(angles_high)

x_low = np.cos(angles_low)
y_low = np.sin(angles_low)

x = np.concatenate([x_high, x_low])
y = np.concatenate([y_high, y_low])

# Define the bounding box
bounding_box_x = [-1, 1, 1, -1, -1]
bounding_box_y = [-1, -1, 1, 1, -1]

# Center point
center_x, center_y = 0, 0

# Plot
plt.figure(figsize=(6,6))
plt.scatter(x, y, color='blue', s=10, label="Points on Circle Edge")
plt.plot(bounding_box_x, bounding_box_y, 'k-', linewidth=2, label="Bounding Box")
plt.scatter(center_x, center_y, color='red', s=50, label="Center Point")

# Circle boundary (for visualization, though all points are on it)
# circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2, linestyle='dashed', label="Circle")
# plt.gca().add_patch(circle)

plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

