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
    return components, explained_variance

# 🔹 Plot point cloud and PCA vectors
def plot_pca_with_point_cloud(points, components):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original point cloud
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, alpha=0.5)

    # Define PCA vectors (scaled for visualization)
    origin = np.mean(points, axis=0)  # Use centroid as the reference point
    scale = np.max(points) * 0.5  # Adjust vector length for visibility

    # Plot principal components as bidirectional arrows
    for i in range(3):
        vector = components[i] * scale
        ax.plot([origin[0] - vector[0], origin[0] + vector[0]],
                [origin[1] - vector[1], origin[1] + vector[1]],
                [origin[2] - vector[2], origin[2] + vector[2]],
                linewidth=2, label=f'PC {i+1}')

    # Labels and view
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Point Cloud with PCA Principal Components")
    ax.legend()
    plt.show()

# 🔹 Main Execution
file_path = "./pca_point_cloud_0.ply"  # Change to your file path
points, pcd = load_point_cloud(file_path)
components, variance = compute_pca(points)

print("Principal Components:\n", components)
print("Explained Variance Ratios:\n", variance)

plot_pca_with_point_cloud(points, components)

file_path = "./pca_point_cloud_1.ply"  # Change to your file path
points, pcd = load_point_cloud(file_path)
components, variance = compute_pca(points)

print("Principal Components:\n", components)
print("Explained Variance Ratios:\n", variance)

plot_pca_with_point_cloud(points, components)
