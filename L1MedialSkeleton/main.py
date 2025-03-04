"""
Steps:
1. sample subset of points randomly
2. iteratively project and redistribute point to center of local neighborhood
3. size of neighborhood is gradually increased to handle structures to different levels of details
"""

import math as m

import torch
import time



h = 5
epsilon = 1e-8

def compute_next_X(X_k, sigma_k, Q, Alpha, mu, Beta):
    I, D = X_k.shape
    J, D = Q.shape

    #first term
    row_Q = Q.unsqueeze(0) #(1, Q)
    num = torch.sum(Alpha * row_Q,dim=1) #(X,1)
    denom = torch.sum(Alpha, dim=1) #(X,1)
    first_term = num / denom

    #second term
    diff_X_X = X_k.unsqueeze(0) - X_k.unsqueeze(1) #(X,X), diagonal shouldn't matter





    pass

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
    C = torch.zeros((I, D, D))

    for i in range(I):
        x_i = points[i]  # Select reference point
        diffs = points - x_i  # Compute differences from all other points
        
        # Compute Euclidean distances
        distances = torch.norm(diffs, dim=1)  # (N,)

        # Exclude self-distance
        mask = torch.ones(I, dtype=torch.bool)
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

def compute_covariance_matrix(points, h):
    # each point is a row vector

    I = points.size()[0]

    C = torch.zeros((I, 3, 3), dtype=torch.float64)

    for i in range(I):
        sum = torch.zeros((3, 3), dtype=torch.float64)
        for j in range(I):

            if i == j:
                continue

            x_i = points[i]
            x_i_prime = points[j]

            thing3 = x_i - x_i_prime

            thing = theta(l2_norm(thing3), h)

            thing2 = thing3.unsqueeze(1)
            thing3 = thing3.unsqueeze(0)

            thing2 = thing2 * thing

            all = torch.matmul(thing2, thing3)

            sum += all

        C[i, :] = sum

    return C

J = 10
I = 5
Q = torch.randn((J, 3), dtype=torch.float64)

indices = torch.randint(0, 10, (5,))
indices = torch.randperm(J)[:I]
X = Q[indices]

C = compute_covariance_matrix_optimized(Q, h)
sigma = calc_directionality_degree(C)
gamma = torch.ones_like(sigma)
R = compute_repulsion_force_optimized(Q, sigma, gamma, h)
Alpha, Beta = compute_Alpha_Beta(X, Q)

compute_Alpha_Beta(X, Q)

exit()

python_time = time2 - time1

time1 = time.time()
C2 = cpp.build.l1MedialSkeleton.computeCovarianceMatrix(points, h)

time2 = time.time()

print(C - C2)

c_time = time2 - time1

print("Python: ", python_time)
print("C: ", c_time)
