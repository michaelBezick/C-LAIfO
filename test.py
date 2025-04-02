import math as m

import torch
import torch.nn.functional as F

batch, frames, views, d = 2, 2, 2, 1

tensor = torch.randn((batch, frames, views, d, 3))

indices = torch.randint(views, size=(batch, frames))
batch_idx = torch.arange(batch).view(-1, 1).expand(-1, frames)
frame_idx = torch.arange(frames).view(1, -1).expand(batch, -1)



# print(indices.shape)
# print(tensor[batch_idx, frame_idx, indices].shape)
print(tensor.shape)
print(tensor)
print(indices)
print(tensor[batch_idx, frame_idx, indices])
