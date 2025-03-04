import math as m

import torch
import time
import open3d as o3d
import numpy as np
import plotly.graph_objects as go

"""
Steps:
1. sample subset of points randomly
2. iteratively project and redistribute point to center of local neighborhood
3. size of neighborhood is gradually increased to handle structures to different levels of details
"""

def visualize_point_clouds(pc1, pc2):
    """
    Overlay two Open3D point clouds and visualize with Plotly.
    
    Args:
        pc1 (o3d.geometry.PointCloud): First point cloud (Blue).
        pc2 (o3d.geometry.PointCloud): Second point cloud (Red).
    """
    # Convert Open3D point clouds to NumPy arrays
    points1 = np.asarray(pc1.points)  # (N1, 3)
    points2 = np.asarray(pc2.points)  # (N2, 3)

    # Create color arrays
    color1 = np.tile(np.array([[0, 0, 1]]), (points1.shape[0], 1))  # Blue
    color2 = np.tile(np.array([[1, 0, 0]]), (points2.shape[0], 1))  # Red

    # Create Plotly scatter plots
    trace1 = go.Scatter3d(
        x=points1[:, 0], y=points1[:, 1], z=points1[:, 2],
        mode='markers',
        marker=dict(size=3, color=['rgb(0, 0, 255)'] * points1.shape[0])  # Blue
    )

    trace2 = go.Scatter3d(
        x=points2[:, 0], y=points2[:, 1], z=points2[:, 2],
        mode='markers',
        marker=dict(size=3, color=['rgb(255, 0, 0)'] * points2.shape[0])  # Red
    )

    # Plot using Plotly
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(title="Overlayed Point Clouds", scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    ))
    fig.show()


def calc_gamma(sigma, Alpha, Beta, mu):

    sigma = sigma.unsqueeze(1)
    alpha_summed = torch.sum(Alpha, dim=1)
    alpha_summed = alpha_summed.unsqueeze(1)

    denom = sigma * alpha_summed

    lhs = denom * mu

    beta_summed = torch.sum(Beta, dim=1) #ensure diagonal is 0
    beta_summed = beta_summed.unsqueeze(1)

    gamma = lhs / beta_summed

    gamma = torch.squeeze(gamma)

    return gamma


def compute_next_X(X_k, sigma_k, Q, Alpha, mu, Beta):
    I, D = X_k.shape
    J, D = Q.shape

    #first term
    row_Q = Q.unsqueeze(0) #(1, J, 3)
    row_Q = row_Q.expand(I, J, 3) #(I, J, 3)
    #Alpha is (I, Q)
    Alpha_expanded = Alpha.unsqueeze(-1) #(I,J,1)


    num = torch.sum(row_Q * Alpha_expanded, dim=1) #(I,3)
    denom = torch.sum(Alpha, dim=1) #(I)
    denom = denom.unsqueeze(-1)
    first_term = num / denom #(I, 3)

    #second term
    diff_X_X = X_k.unsqueeze(0) - X_k.unsqueeze(1) #(X,X), diagonal shouldn't matter
    Beta_expanded = Beta.unsqueeze(-1)
    diff_X_X = diff_X_X * Beta_expanded
    num = torch.sum(diff_X_X, dim=1) #ensure diagonal is 0
    denom = torch.sum(Beta_expanded,dim=1) #ensure diagonal is 0
    sigma_k = sigma_k.unsqueeze(-1)
    second_term = mu * sigma_k * (num / denom)

    next_X = first_term + second_term

    return next_X


def compute_Alpha_Beta(X:torch.Tensor, Q:torch.Tensor):
    I, D = X.shape
    J, D = Q.shape

    diff_X_Q = X.unsqueeze(1) - Q.unsqueeze(0) #(I,J,D)
    diff_X_X = X.unsqueeze(1) - X.unsqueeze(0) #(I,I,D)
    l2_norm_diff = torch.norm(diff_X_X, dim=2)

    mask = torch.eye(I, device=X.device).bool()
    l2_norm_diff = l2_norm_diff.masked_fill(mask, float('inf'))

    Beta = theta(l2_norm_diff, h) / (l2_norm_diff * l2_norm_diff) # diagonal is zero

    l2_norm_diff = torch.norm(diff_X_Q, dim=2)

    Alpha = theta(l2_norm_diff, h) / (l2_norm_diff + epsilon)

    return Alpha, Beta
    

def theta(r, h):
    return m.e ** (-(r**2) / ((h / 2) ** 2))

def calc_directionality_degree(C: torch.Tensor):

    sigma = torch.linalg.eig(C)[0].real

    sigma = sigma[:, 2] / (sigma[:,0] + sigma[:,1] + sigma[:,2])

    return sigma

def compute_repulsion_force(points, sigma, gamma, h):
    I, D = points.shape

    outer_sum = 0
    for i in range(I):
        inner_sum = 0
        for i_prime in range(I):
            if i == i_prime:
                continue

            x = l2_norm(points[i] - points[i_prime])
            inner_sum += theta(x, h) / (sigma[i] * x)

        inner_sum *= gamma[i]
        outer_sum += inner_sum

    return outer_sum


def compute_repulsion_force_optimized(points:torch.Tensor, sigma:torch.Tensor, gamma:torch.Tensor, h):
    I, D = points.shape
    
    # Compute pairwise differences: Shape (I, I, D)
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (I, I, D)
    
    # Compute the L1 norm of the differences: Shape (I, I)
    l2_norm_diff = torch.norm(diff, dim=2)

    mask = torch.eye(I, device=points.device).bool()
    
    # Apply the theta function to the L1 norms: Shape (I, I)
    kernel = theta(l2_norm_diff, h)

    l2_norm_diff = l2_norm_diff.masked_fill(mask, float('inf'))
    
    # Compute the repulsion force for each pair (i, j)
    repulsion_contrib = kernel / (sigma.unsqueeze(1) * l2_norm_diff)  # Shape (I, I)
    
    # Scale by gamma: Shape (I, I)
    repulsion_contrib *= gamma.unsqueeze(1)  # Broadcast gamma across all pairs
    
    # Sum over all pairs, excluding the diagonal (i == i)
    repulsion_force = repulsion_contrib.masked_fill(mask,0).sum()  # Scalar value
    
    return repulsion_force


def l1_norm(x):
    return torch.sum(torch.abs(x))


def l2_norm(x):
    return torch.norm(x, p=2)

def compute_covariance_matrix_optimized(points, h):
    I, D = points.shape  # Number of points, Dimensionality (should be 3)
    
    # Initialize covariance matrix for each point
    C = torch.zeros((I, D, D), device=points.device)

    for i in range(I):
        x_i = points[i]  # Select reference point
        diffs = points - x_i  # Compute differences from all other points
        
        # Compute Euclidean distances
        distances = torch.norm(diffs, dim=1)  # (N,)

        # Exclude self-distance
        mask = torch.ones(I, dtype=torch.bool, device=points.device)
        mask[i] = False
        distances = distances[mask]
        diffs = diffs[mask]  # Remove self from the set

        # Compute weights using theta function
        weights = theta(distances, h)  # (N-1,)

        # Reshape for matrix multiplication
        diffs_col = diffs.unsqueeze(2)  # (N-1, 3, 1)
        diffs_row = diffs.unsqueeze(1)  # (N-1, 1, 3)

        # Compute weighted covariance sum
        weighted_outer_products = weights[:, None, None] * torch.bmm(diffs_col, diffs_row)

        # Sum over all neighbors to get the covariance matrix for this point
        C[i] = weighted_outer_products.sum(dim=0)

    return C


"""KEEP ON CPU"""

h = 5
epsilon = 1e-8
I = 100
K = 10

point_cloud = o3d.io.read_point_cloud("./random_point_cloud_filtered.ply")

point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points))

J = point_cloud_points.shape[0]

Q = point_cloud_points

indices = torch.randint(0, J, (I,))
indices = torch.randperm(J)[:I]
X = Q[indices]

time1 = time.time()

mu = 0.35

for _ in range(K):

    C = compute_covariance_matrix_optimized(X, h)
    sigma = calc_directionality_degree(C)

    max_sigma = torch.max(sigma)

    mu = 0.48 / max_sigma #authors use 0.35 as default setting
    assert (torch.max(mu * sigma) < 0.5) and (torch.min(mu * sigma) > 0)

    Alpha, Beta = compute_Alpha_Beta(X, Q)
    gamma = calc_gamma(sigma, Alpha, Beta, mu)

    R = compute_repulsion_force_optimized(X, sigma, gamma, h)

    next_X = compute_next_X(X, sigma, Q, Alpha, mu, Beta)

    X = next_X

time2 = time.time()
print(time2 - time1)

"""
What depends on k? alpha, beta, sigma, x, covariance
"""

X = X.numpy()
final_pcd = o3d.geometry.PointCloud()
final_pcd.points = o3d.utility.Vector3dVector(X)

visualize_point_clouds(point_cloud, final_pcd)
