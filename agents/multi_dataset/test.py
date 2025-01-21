import math as m
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.colors import BoundaryNorm


class SinusoidalPositionalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = m.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        interleaved = torch.empty(time.size(0), self.dim)
        interleaved[:, 0::2] = embeddings.sin()
        interleaved[:, 1::2] = embeddings.cos()
        return interleaved


repr_dim = 32 * 35 * 35
positions = torch.linspace(1, 3, 1000)
positions *= repr_dim / 35

encoding_size = 32 * 35 * 35
batch_size = 1000
encoder = SinusoidalPositionalEmbeddings(encoding_size)
torch.set_printoptions(sci_mode=False)
encoded = encoder(positions)
print(encoded.size())
print(torch.min(encoded))
print(torch.max(encoded))
print(encoded)

boundaries = np.linspace(torch.min(encoded).item(), torch.max(encoded).item(), 10)
cmap = matplotlib.cm.get_cmap("viridis")  # Use matplotlib.cm.get_cmap()

norm = BoundaryNorm(boundaries, cmap.N, clip=True)


fig, ax = plt.subplots()
new_vectors = torch.zeros((encoding_size, batch_size))
for i in range(batch_size):
    new_vectors[:, i] = encoded[i, :]

im = ax.imshow(new_vectors, cmap=cmap, norm=norm, origin="lower")
# cbar = fig.colorbar(im, ax=ax, boundaries=boundaries, ticks=boundaries)
# cbar.set_label('Value Interval')

# Add labels and title
# ax.set_title('Sinusoidal Positional Encodings')
# ax.set_xlabel('FOM')
# ax.set_ylabel('Positional Encoding')

ax.set_yticklabels([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_xticks([])

# Show the plot
plt.savefig("Sinusoidal Positional Encodings.pdf", bbox_inches="tight")
