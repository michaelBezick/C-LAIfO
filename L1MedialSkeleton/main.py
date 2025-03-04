"""
Steps:
1. sample subset of points randomly
2. iteratively project and redistribute point to center of local neighborhood
3. size of neighborhood is gradually increased to handle structures to different levels of details
"""

import math as m

import torch
import cpp.build.l1MedialSkeleton
import numpy as np
import time

points = torch.randn((100, 3), dtype=torch.float64)

h = 5


def theta(r, h):
    return m.e ** (-(r**2) / ((h / 2) ** 2))


def l1_norm(x):
    return torch.sum(torch.abs(x))


def l2_norm(x):
    return torch.norm(x, p=2)

def optimized_compute_covariance_matrix(points, h):
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


time1 = time.time()
C = optimized_compute_covariance_matrix(points, h)
time2 = time.time()

python_time = time2 - time1

time1 = time.time()
C2 = cpp.build.l1MedialSkeleton.computeCovarianceMatrix(points, h)

time2 = time.time()

print(C - C2)

c_time = time2 - time1

print("Python: ", python_time)
print("C: ", c_time)
