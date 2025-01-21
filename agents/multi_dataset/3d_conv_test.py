import torch
import torch.nn as nn

batch_size = 100
channel_dim = 3
height = 64
width = 64
time_steps = 10

video_data = torch.randn((batch_size, channel_dim, time_steps, height, width))

print(video_data.size())

layer = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=(3, 3, 3))

out = layer(video_data)

print(out.size())

batch_size = 1
channel_dim = 3
height = 2
width = 2
frame1 = torch.ones((1, 3, 2, 2))
frame2 = torch.ones((1, 3, 2, 2)) * 2
frame3 = torch.ones((1, 3, 2, 2)) * 3
all = torch.cat([frame1, frame2, frame3], dim = 1)



all = all.view(batch_size, 3, 3, 2, 2)
all = all.permute(0, 2, 1, 3, 4)
print(all[:, :, 0, :, :])


