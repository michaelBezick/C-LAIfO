
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 🔹 Load point cloud from file
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)  # Load point cloud
    points = np.asarray(pcd.points)  # Convert to NumPy array
    return points, pcd

# 🔹 Perform PCA and get principal components
def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    components = pca.components_  # Principal components (3x3 matrix)
    explained_variance = pca.explained_variance_ratio_  # Variance explained by each PC
    return pca, components, explained_variance

# 🔹 Transform points into PCA basis
def transform_to_pca_basis(points, pca):
    return pca.transform(points)  # Project points onto PCA basis

# 🔹 Rotate point cloud 180° about the Z-axis
def rotate_point_cloud_180_z(points):
    R_z_180 = np.array([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0,  1]
    ])
    return points @ R_z_180.T  # Rotate points

# 🔹 Compute axis limits to keep plots consistent
def get_axis_limits(*point_sets):
    """Compute min and max limits across all provided point clouds."""
    all_points = np.vstack(point_sets)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    return min_vals, max_vals

# 🔹 Plot point cloud and PCA vectors
def plot_pca_with_point_cloud(points, components, title, axis_limits, scale_factor=1.0):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)

    # Define PCA vectors (scaled for visualization)
    origin = np.mean(points, axis=0)  # Use centroid as the reference point
    scale = np.max(axis_limits[1] - axis_limits[0]) * scale_factor  # Ensure PCA axes are large

    # Plot principal components as bidirectional arrows
    for i in range(3):
        vector = components[i] * scale
        ax.plot([origin[0] - vector[0], origin[0] + vector[0]],
                [origin[1] - vector[1], origin[1] + vector[1]],
                [origin[2] - vector[2], origin[2] + vector[2]],
                linewidth=2, label=f'PC {i+1}')

    # Apply consistent axis limits
    min_vals, max_vals = axis_limits
    ax.set_box_aspect([1,1,1])  # Ensures equal aspect ratio in 3D
    ax.set_xlim([min_vals[0], max_vals[0]])
    ax.set_ylim([min_vals[1], max_vals[1]])
    ax.set_zlim([min_vals[2], max_vals[2]])

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    ax.legend()
    plt.show(block=False)

# 🔹 Main Execution
file_path = "./pca_point_cloud_0.ply"  # Change to your file path
points, pcd = load_point_cloud(file_path)

# Compute PCA for original point cloud
pca, components, variance = compute_pca(points)
print("Original Principal Components:\n", components)
print("Original Explained Variance Ratios:\n", variance)

# 🔹 Rotate the same point cloud 180° around Z-axis
rotated_points = rotate_point_cloud_180_z(points)

# Compute PCA separately for the rotated point cloud
rotated_pca, rotated_components, rotated_variance = compute_pca(rotated_points)
print("Rotated Principal Components:\n", rotated_components)
print("Rotated Explained Variance Ratios:\n", rotated_variance)

# 🔹 Transform original points into its PCA basis
transformed_points = transform_to_pca_basis(points, pca)

# 🔹 Compute consistent axis limits for all plots
axis_limits = get_axis_limits(points, rotated_points, transformed_points)

# 🔹 Plot original point cloud with its own PCA
plot_pca_with_point_cloud(points, components, title="Original Point Cloud", axis_limits=axis_limits, scale_factor=0.5)

# 🔹 Plot rotated point cloud with newly computed PCA
plot_pca_with_point_cloud(rotated_points, rotated_components, title="Rotated 180° Around Z-Axis", axis_limits=axis_limits, scale_factor=0.5)

# 🔹 Plot transformed point cloud in PCA basis
plot_pca_with_point_cloud(transformed_points, np.eye(3), title="Transformed to PCA Basis", axis_limits=axis_limits, scale_factor=0.5)

plt.show()  # Keep all figures open

