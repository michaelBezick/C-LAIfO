import math as m

import torch
import torch.nn.functional as F

def add_one_hot_info(points: torch.Tensor, frame_id, total_frames):
    batch_size, num_points, xyz = points.size()
    one_hot = F.one_hot(torch.tensor([frame_id]), total_frames)
    one_hot_expanded = one_hot.view(1,1,3).expand(batch_size, num_points, -1)
    points_concat = torch.cat([points, one_hot_expanded], dim=-1)

    return points_concat




batch_size=100
num_points = 1500

points = torch.randn((batch_size, 3, num_points, 3))

points1 = add_one_hot_info(points[:,0,:,:],frame_id=0,total_frames=3)
points2 = add_one_hot_info(points[:,1,:,:],frame_id=1,total_frames=3)
points3 = add_one_hot_info(points[:,2,:,:],frame_id=2,total_frames=3)

all_points = torch.cat([points1,points2,points3],dim=1)
print(all_points.size())
