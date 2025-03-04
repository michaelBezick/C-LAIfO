import torch

# Generate random 3D points
points = torch.randn((100, 3))  # (N, 3)

h = 5  # Bandwidth parameter

# Weight function theta using squared Euclidean distance
def theta(r, h):
    return torch.exp(- (r ** 2) / ((h / 2) ** 2))

def compute_covariance_matrix(points, h):
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

# Compute covariance matrices
covariance_matrices = compute_covariance_matrix(points, h)

# Print one example covariance matrix
print(covariance_matrices[0])
