import numpy as np
import torch
from sklearn.decomposition import PCA


def transform_to_pca_basis(points, pca):
    return pca.transform(points)  # Project points onto PCA basis


def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    components = pca.components_  # Principal components (3x3 matrix)
    explained_variance = pca.explained_variance_ratio_  # Variance explained by each PC
    return pca, components, explained_variance


def form_canonical_bases(pca: PCA):
    bases = []
    components = pca.components_
    components = torch.tensor(components)

    signs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]

    for sign in signs:
        sign1, sign2 = sign
        basis = torch.zeros(3, 3)
        basis[0] = components[0] * sign1
        basis[1] = components[1] * sign2
        basis[2] = components[2]

        if torch.det(basis) <= 0:
            basis[2] = basis[2] * -1
            bases.append(basis)
        else:
            bases.append(basis)

    return bases


points = torch.randn((100, 3)).numpy()

pca, _, _ = compute_pca(points)
bases = form_canonical_bases(pca)
