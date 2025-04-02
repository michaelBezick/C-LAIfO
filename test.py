import math as m

import torch
import torch.nn.functional as F

tensor = torch.randn((100, 3, 4, 50, 3))

indices = torch.randint(0, 4, size=(100, 3, 1))

print(tensor[indices].shape)
