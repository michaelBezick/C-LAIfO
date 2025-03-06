import torch
import numpy as np
from scipy.spatial import KDTree
import open3d as o3d

def bounding_box_centers(point_cloud, radius=0.08, voxel_size=0.17):
    point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points))

    # Downsample the point cloud
    down_sampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    down_sampled_point_cloud_points = torch.from_numpy(
        np.asarray(down_sampled_point_cloud.points)
    )

    # KD-Tree for fast neighbor search
    kdtree = KDTree(point_cloud_points.numpy())
    neighbors_list = kdtree.query_ball_point(down_sampled_point_cloud_points.numpy(), radius)

    # Remove empty neighborhoods
    valid_indices = [i for i, neighbors in enumerate(neighbors_list) if len(neighbors) > 0]
    filtered_neighbors_list = [neighbors for neighbors in neighbors_list if len(neighbors) > 0]

    if len(valid_indices) == 0:
        print("Warning: No valid neighborhoods found! Returning empty array.")
        return np.empty((0, 3), dtype=np.float32)

    # Convert neighbors list into tensor index format
    max_neighbors = max(len(n) for n in filtered_neighbors_list)
    num_centers = len(filtered_neighbors_list)

    # Create an index matrix with padding
    neighbors_idx = torch.full((num_centers, max_neighbors), -1, dtype=torch.long)

    for i, neighbors in enumerate(filtered_neighbors_list):
        neighbors_idx[i, : len(neighbors)] = torch.tensor(neighbors, dtype=torch.long)

    # Mask out invalid entries
    valid_mask = neighbors_idx != -1

    # Gather all neighborhood points
    neighbors_points = point_cloud_points[neighbors_idx]  # Shape: [num_centers, max_neighbors, 3]

    # Apply the mask (only consider valid points)
    min_xyz = torch.amin(neighbors_points, dim=1)
    max_xyz = torch.amax(neighbors_points, dim=1)

    # Compute bounding box centers
    final_points = (min_xyz + max_xyz) / 2

    return final_points.numpy().astype(np.float32)
