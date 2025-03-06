import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch
from scipy.spatial import KDTree


def bounding_box_centers(point_cloud, radius=0.08, voxel_size=0.17):
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

    return final_points.numpy()
