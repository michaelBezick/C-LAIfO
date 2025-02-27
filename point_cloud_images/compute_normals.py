import open3d as o3d
import numpy as np

# Load .ply file
pcd = o3d.io.read_point_cloud("./view1.ply")

# Compute normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=10)  # Adjust K as needed
)

# Convert points and normals to numpy arrays
points = np.asarray(pcd.points, dtype=np.float32)  # Nx3
normals = np.asarray(pcd.normals, dtype=np.float32)  # Nx3

# Save as .npy files
np.save("points.npy", points)
np.save("normals.npy", normals)
