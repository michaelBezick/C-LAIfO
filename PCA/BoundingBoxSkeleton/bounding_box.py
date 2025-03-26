import time

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
from scipy.spatial import KDTree

# Load the point cloud
voxel_size = 0.17
point_cloud = o3d.io.read_point_cloud("./random_point_cloud_filtered.ply")
point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points))

# Downsample the point cloud
down_sampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
down_sampled_point_cloud_points = torch.from_numpy(
    np.asarray(down_sampled_point_cloud.points)
)

# KD-Tree for fast neighbor search
kdtree = KDTree(point_cloud_points)
radius = 0.08
neighbors_list = kdtree.query_ball_point(
    down_sampled_point_cloud_points.numpy(), radius
)

time1 = time.time()

# Convert neighbors list into tensor index format
max_neighbors = max(len(n) for n in neighbors_list)  # Find max neighborhood size
num_centers = len(neighbors_list)

# Create an index matrix with padding
neighbors_idx = torch.full(
    (num_centers, max_neighbors), -1, dtype=torch.long
)  # Fill with -1 initially

for i, neighbors in enumerate(neighbors_list):
    neighbors_idx[i, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)

# Mask out invalid entries
valid_mask = neighbors_idx != -1

# Gather all neighborhood points
neighbors_points = point_cloud_points[
    neighbors_idx
]  # Shape: [num_centers, max_neighbors, 3]

# Apply the mask
valid_neighbors_points = torch.where(
    valid_mask.unsqueeze(-1), neighbors_points, float("inf")
)  # Min
min_xyz = torch.amin(valid_neighbors_points, dim=1)  # Min along the neighbor axis
valid_neighbors_points = torch.where(
    valid_mask.unsqueeze(-1), neighbors_points, float("-inf")
)  # Max
max_xyz = torch.amax(valid_neighbors_points, dim=1)  # Max along the neighbor axis

# Compute bounding box centers
final_points = (min_xyz + max_xyz) / 2

time2 = time.time()
print(f"Time taken: {time2 - time1:.4f} seconds")


def visualize_point_clouds_new(pc1, pc2):
    """
    Overlay two Open3D point clouds and visualize with Plotly.

    Args:
        pc1 (o3d.geometry.PointCloud): First point cloud (Blue).
        pc2 (o3d.geometry.PointCloud): Second point cloud (Red).
    """
    # Convert Open3D point clouds to NumPy arrays
    points1 = np.asarray(pc1.points)  # (N1, 3)
    points2 = np.asarray(pc2.points)  # (N2, 3)

    points3 = np.copy(points2)

    points3[:, 1] = points2[:, 1] - 0.5

    # Concatenate both point clouds to find global min/max
    all_points = np.vstack((points1, points2))
    min_range = np.min(all_points)
    max_range = np.max(all_points)

    # Create Plotly scatter plots
    trace1 = go.Scatter3d(
        x=points1[:, 0],
        y=points1[:, 1],
        z=points1[:, 2],
        mode="markers",
        marker=dict(size=3, color="blue"),  # Blue
    )

    trace2 = go.Scatter3d(
        x=points2[:, 0],
        y=points2[:, 1],
        z=points2[:, 2],
        mode="markers",
        marker=dict(size=3, color="red"),  # Red
    )

    trace3 = go.Scatter3d(
        x=points3[:, 0],
        y=points3[:, 1],
        z=points3[:, 2],
        mode="markers",
        marker=dict(size=3, color="orange"),  # Red
    )

    # Plot using Plotly
    fig = go.Figure(data=[trace1, trace2, trace3])

    # Force equal aspect ratio by setting identical axis ranges
    fig.update_layout(
        title="Overlayed Point Clouds",
        scene=dict(
            xaxis=dict(title="X", range=[min_range, max_range]),
            yaxis=dict(title="Y", range=[min_range, max_range]),
            zaxis=dict(title="Z", range=[min_range, max_range]),
            aspectmode="cube",  # Ensures equal scaling of all axes
        ),
    )
    fig.show()


final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(final_points.cpu().numpy())
visualize_point_clouds_new(point_cloud, final_pcd)
