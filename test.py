import torch
from torch.nn.utils.rnn import pad_sequence
pc1 = torch.randn((1231,3)).numpy()
pc2 = torch.randn((1200,3)).numpy()
pc3 = torch.randn((1215,3)).numpy()

array = [pc1, pc2, pc3]

tensors = [torch.tensor(arr,dtype=torch.float32) for arr in array]
padded_tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
