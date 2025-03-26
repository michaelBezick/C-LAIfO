import math as m

import torch
import time
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

from functions import *

"""
Steps:
1. sample subset of points randomly
2. iteratively project and redistribute point to center of local neighborhood
3. size of neighborhood is gradually increased to handle structures to different levels of details
"""

"""
Need to label skeleton branch points
Initial points are all considered non-branch
Need to compute sigma_i for non-branch points x_i and smooth them within K-nearest neighborhood (K=5 by default)
"""

"""KEEP ON CPU"""

I = 50
K = 1
num_neighborhoods = 1
voxel_downsample = True
voxel_size = 0.17

point_cloud = o3d.io.read_point_cloud("./random_point_cloud_filtered.ply")

time1 = time.time()

if voxel_downsample:
    down_sampled_point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    down_sampled_point_cloud_points = torch.from_numpy(np.asarray(down_sampled_point_cloud.points))
    original_point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points)) 
    J = original_point_cloud_points.shape[0]
    Q = original_point_cloud_points
    X = down_sampled_point_cloud_points
    I = X.shape[0] #overwritten
else:
    point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points))
    J = point_cloud_points.shape[0]
    Q = point_cloud_points
    indices = torch.randint(0, J, (I,))
    indices = torch.randperm(J)[:I]
    X = Q[indices]

mu = 0.35
print("num points: ", I)

#compute bounding box
min_xyz, _ = torch.min(Q, dim=0)
max_xyz, _ = torch.max(Q, dim=0)
d_bb = torch.norm(max_xyz - min_xyz, p=2)

h_0 = (1) * d_bb / (J ** (1/3))
h = h_0
del_h = h_0 / 2

# for neighborhood_idx in range(num_neighborhoods):
#     for _ in range(K):
#
#         C = compute_covariance_matrix_optimized(X, h)
#
#         sigma, eigvals, eigvecs = calc_directionality_degree(C)
#
#         max_sigma = torch.max(sigma)
#
#         # assert (torch.max(mu * sigma) < 0.5), f"{torch.max(mu * sigma)}"
#         # assert (torch.min(mu * sigma) > 0), f"{torch.min(mu * sigma)}"
#
#         Alpha, Beta = compute_Alpha_Beta(X, Q, h)
#         gamma = calc_gamma(sigma, Alpha, Beta, mu)
#
#         R = compute_repulsion_force_optimized(X, sigma, gamma, h)
#
#         next_X = compute_next_X(X, sigma, Q, Alpha, mu, Beta)
#
#     X = next_X
#     h += del_h
#     # non_branch_points, branch_points = find_branch_points(X, sigma, eigvals, eigvecs, k=5) #potentially optimize later instead of recalculating eigenvectors

# for _ in range(K):

"""BEGINNING"""

C = compute_covariance_matrix_optimized(X, h)

sigma, eigvals, eigvecs = calc_directionality_degree(C)

max_sigma = torch.max(sigma)

Alpha, Beta = compute_Alpha_Beta(X, Q, h)
gamma = calc_gamma(sigma, Alpha, Beta, mu)

R = compute_repulsion_force_optimized(X, sigma, gamma, h)

next_X = compute_next_X(X, sigma, Q, Alpha, mu, Beta)
X = next_X

"""END"""

time2 = time.time()
print(time2 - time1)

X = X.numpy()
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(X)

visualize_point_clouds_new(point_cloud, final_pcd)
