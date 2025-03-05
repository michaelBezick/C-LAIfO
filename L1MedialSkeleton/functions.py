import math as m
import time

import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import torch

epsilon = 1e-8


def l1_medial_skeleton(point_cloud, down_sample_voxel_size=0.17, mu=0.35):

    down_sampled_point_cloud = point_cloud.voxel_down_sample(
        voxel_size=down_sample_voxel_size
    )
    down_sampled_point_cloud_points = torch.from_numpy(
        np.asarray(down_sampled_point_cloud.points)
    )
    original_point_cloud_points = torch.from_numpy(np.asarray(point_cloud.points))
    J = original_point_cloud_points.shape[0]
    Q = original_point_cloud_points
    X = down_sampled_point_cloud_points
    I = X.shape[0]
    min_xyz, _ = torch.min(Q, dim=0)
    max_xyz, _ = torch.max(Q, dim=0)
    d_bb = torch.norm(max_xyz - min_xyz, p=2)

    h = (1) * d_bb / (J ** (1 / 3))

    """BEGINNING"""

    C = compute_covariance_matrix_optimized(X, h)

    sigma, eigvals, eigvecs = calc_directionality_degree(C)

    # max_sigma = torch.max(sigma)

    Alpha, Beta = compute_Alpha_Beta(X, Q, h)
    gamma = calc_gamma(sigma, Alpha, Beta, mu)

    # R = compute_repulsion_force_optimized(X, sigma, gamma, h)

    next_X = compute_next_X(X, sigma, Q, Alpha, mu, Beta)
    X = next_X

    X = X.numpy()

    return X


def knn(points, k):
    """
    Computes K-Nearest Neighbors using PyTorch (vectorized for efficiency).

    Args:
        points (torch.Tensor): Tensor of shape (N, D) where N = number of points, D = dimensions.
        k (int): Number of nearest neighbors to find.

    Returns:
        indices (torch.Tensor): Tensor of shape (N, k) containing the indices of nearest neighbors.
    """
    # Compute pairwise squared Euclidean distances
    dist_matrix = torch.cdist(points, points, p=2)  # Shape: (N, N)

    # Get the indices of the k smallest distances (excluding itself)
    knn_indices = dist_matrix.topk(k=k + 1, largest=False)[1][
        :, 1:
    ]  # Exclude self (first column)

    return knn_indices


def proj(u: torch.Tensor, v: torch.Tensor):
    return (torch.sum(u * v, dim=1) / (torch.norm(v, p=2, dim=1) ** 2)).unsqueeze(1) * v


def find_branch_points(
    non_branch_points: torch.Tensor,
    sigma: torch.Tensor,
    eigvals: torch.Tensor,
    eigvecs: torch.Tensor,
    k,
):
    """
    Finds branch points.

    Returns:
        non_branch points, branch points
    """

    I = non_branch_points.shape[0]

    knn_indices = knn(non_branch_points, k)  # (I, K)

    sigma_expanded = sigma.unsqueeze(0).expand(I, I)

    knn_neighbors = torch.gather(sigma_expanded, dim=1, index=knn_indices)

    smoothed_sigma = torch.mean(knn_neighbors, dim=1)  # should be (I)

    # mask = smoothed_sigma > 0.9

    """CHANGE THIS BACK EVENTUALLY"""
    mask = smoothed_sigma > 0.7  # for debugging

    """MAKE SURE TO HANDLE CASE WHERE MASK < 5"""

    # can be empty, checking for all false
    if torch.all(~mask):
        return non_branch_points, None

    candidates_sigma = smoothed_sigma[mask]

    if candidates_sigma.shape[0] < 5:
        return non_branch_points, None

    candidates_points = non_branch_points[mask]
    dominate_directions = eigvecs[mask][:, 2, :]  # sorted in ascending order

    # need to find largest sigma within candidates to get x_0

    index = torch.argmax(candidates_sigma)

    x_0 = candidates_points[index]
    dominate_direction_x_0 = dominate_directions[index].double()

    # we only need to trace to nearby candidates

    # goal, project every point onto dominant direction. then get signed distances

    x_0_to_candidates_vectors = candidates_points - x_0

    dominate_direction_x_0_expanded = dominate_direction_x_0.unsqueeze(0).expand(
        x_0_to_candidates_vectors.shape[0], 3
    )
    proj_onto_dominant_dir = proj(
        x_0_to_candidates_vectors, dominate_direction_x_0_expanded
    )

    proj_onto_dominant_dir_distances = torch.norm(proj_onto_dominant_dir, p=2, dim=1)

    # for now lets get 5 points
    points = []
    points.append(x_0)
    """
    IDEA: for each point, see if it's infront or behind x_0, and append or prepend accordingly
    """

    def direction(u, v):
        return torch.dot(u, v)

    x_0_index = 0
    for i in range(
        len(proj_onto_dominant_dir_distances) - 1
    ):  # verify if this range -1 is correct
        _, indices = torch.topk(proj_onto_dominant_dir_distances, i + 2, largest=False)
        ith_smallest_index = indices[-1]  # the ith smallest index not including x_0

        point_of_interest = candidates_points[ith_smallest_index]

        dir = direction(point_of_interest, dominate_direction_x_0)

        if dir >= 0:
            # append
            points.append(point_of_interest)
        else:
            x_0_index += 1
            # prepend
            points.insert(0, point_of_interest)

    """
    Now we have a list of points in order, we can compare adjacent pairs to see if their angles meet the condition
    """

    def is_sufficiently_linear(u, v):
        cosine_theta = torch.dot(u, v) / (torch.norm(u, p=2) * torch.norm(v, p=2))

        if cosine_theta <= -0.9:
            return True
        else:
            return False

    beginning = x_0_index - 1
    end = x_0_index + 1
    num_points = len(points)

    if x_0_index == 0:
        end += 1
    elif x_0_index == num_points - 1:
        beginning -= 1

    expand_beginning = True
    expand_end = True

    while expand_beginning or expand_end:

        if beginning < 0 and end > num_points - 1:
            # end of tracing
            break

        if beginning >= 0 and expand_beginning:
            # can trace beginning
            prev = points[beginning + 1]
            prev_prev = points[beginning + 2]
            cur = points[beginning]
            vec = cur - prev
            vec_prev = prev - prev_prev

            if not is_sufficiently_linear(vec, vec_prev):

                # end tracing this direction

                expand_beginning = False
                beginning += 1
            else:

                # extend tracing
                beginning -= 1
        else:
            expand_beginning = False
            if beginning < 0:
                beginning = 0

        if end <= num_points - 1 and expand_end:
            # can trace end

            cur = points[end]
            prev_point = points[end - 1]
            prev_prev = points[end - 2]

            vec = cur - prev_point
            vec_prev = prev_point - prev_prev

            if not is_sufficiently_linear(vec, vec_prev):

                # end tracing in this direction

                expand_end = False
                end -= 1
            else:
                end += 1
        else:
            expand_end = False
            if end > num_points - 1:
                end = num_points - 1

    """
    At this point, I hope that beginning points to the beginning of a branch and end points to the end
    """

    if (end - beginning + 1) < 5:
        # less than points
        return non_branch_points, None

    # candidate_distances = torch.cdist(candidates_points, candidates_points, p=2) #calculates pairwise distances
    # candidate_distances.fill_diagonal_(float('inf'))

    return None, None


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


def calc_gamma(sigma, Alpha, Beta, mu):

    sigma = sigma.unsqueeze(1)
    alpha_summed = torch.sum(Alpha, dim=1)
    alpha_summed = alpha_summed.unsqueeze(1)

    denom = sigma * alpha_summed

    lhs = denom * mu

    beta_summed = torch.sum(Beta, dim=1)  # ensure diagonal is 0
    beta_summed = beta_summed.unsqueeze(1)

    gamma = lhs / beta_summed

    gamma = torch.squeeze(gamma)

    return gamma


def compute_next_X(X_k, sigma_k, Q, Alpha, mu, Beta):
    I, D = X_k.shape
    J, D = Q.shape

    # first term
    row_Q = Q.unsqueeze(0)  # (1, J, 3)
    row_Q = row_Q.expand(I, J, 3)  # (I, J, 3)
    # Alpha is (I, Q)
    Alpha_expanded = Alpha.unsqueeze(-1)  # (I,J,1)

    num = torch.sum(row_Q * Alpha_expanded, dim=1)  # (I,3)
    denom = torch.sum(Alpha, dim=1)  # (I)
    denom = denom.unsqueeze(-1)
    first_term = num / denom  # (I, 3)

    # second term
    diff_X_X = X_k.unsqueeze(0) - X_k.unsqueeze(1)  # (X,X), diagonal shouldn't matter
    Beta_expanded = Beta.unsqueeze(-1)
    diff_X_X = diff_X_X * Beta_expanded
    num = torch.sum(diff_X_X, dim=1)  # ensure diagonal is 0
    denom = torch.sum(Beta_expanded, dim=1)  # ensure diagonal is 0
    sigma_k = sigma_k.unsqueeze(-1)
    second_term = mu * sigma_k * (num / denom)

    next_X = first_term + second_term

    return next_X


def compute_Alpha_Beta(X: torch.Tensor, Q: torch.Tensor, h):

    I, D = X.shape
    J, D = Q.shape

    diff_X_Q = X.unsqueeze(1) - Q.unsqueeze(0)  # (I,J,D)
    diff_X_X = X.unsqueeze(1) - X.unsqueeze(0)  # (I,I,D)
    l2_norm_diff = torch.norm(diff_X_X, dim=2)

    mask = torch.eye(I, device=X.device).bool()
    l2_norm_diff = l2_norm_diff.masked_fill(mask, float("inf"))

    Beta = theta(l2_norm_diff, h) / (l2_norm_diff * l2_norm_diff + epsilon)

    l2_norm_diff = torch.norm(diff_X_Q, dim=2)

    Alpha = theta(l2_norm_diff, h) / (l2_norm_diff + epsilon)

    return Alpha, Beta


def theta(r, h):
    return m.e ** (-(r**2) / ((h / 2) ** 2))


def calc_directionality_degree(C: torch.Tensor):

    eigvals, eigvecs = torch.linalg.eigh(C)

    max_eigenvalues = torch.max(eigvals, dim=1)[0]

    sigma = max_eigenvalues / (eigvals.sum(dim=1))

    return sigma, eigvals, eigvecs


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


def compute_repulsion_force_optimized(
    points: torch.Tensor, sigma: torch.Tensor, gamma: torch.Tensor, h
):
    I, D = points.shape

    # Compute pairwise differences: Shape (I, I, D)
    diff = points.unsqueeze(0) - points.unsqueeze(1)  # (I, I, D)

    # Compute the L1 norm of the differences: Shape (I, I)
    l2_norm_diff = torch.norm(diff, dim=2)

    mask = torch.eye(I, device=points.device).bool()

    # Apply the theta function to the L1 norms: Shape (I, I)
    kernel = theta(l2_norm_diff, h)

    l2_norm_diff = l2_norm_diff.masked_fill(mask, float("inf"))

    # Compute the repulsion force for each pair (i, j)
    repulsion_contrib = kernel / (sigma.unsqueeze(1) * l2_norm_diff)  # Shape (I, I)

    # Scale by gamma: Shape (I, I)
    repulsion_contrib *= gamma.unsqueeze(1)  # Broadcast gamma across all pairs

    # Sum over all pairs, excluding the diagonal (i == i)
    repulsion_force = repulsion_contrib.masked_fill(mask, 0).sum()  # Scalar value

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
        weighted_outer_products = weights[:, None, None] * torch.bmm(
            diffs_col, diffs_row
        )

        # Sum over all neighbors to get the covariance matrix for this point
        C[i] = weighted_outer_products.sum(dim=0)

    return C
