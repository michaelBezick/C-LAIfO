import torch.nn as nn
import torch
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.h = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.BatchNorm1d(128),
        )

        self.mlp3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, latent_dim, kernel_size=1),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh()
        )

    def forward(self, point_cloud: torch.Tensor):

        x = point_cloud

        """Input size: [b, n, 3]"""


        x = torch.permute(x, (0, 2, 1))  # [b,3,n]

        x = self.h(x)  # x -> [b,64,n]

        x = self.mlp2(x)  # x -> [b,128,n]

        x = torch.max(x, dim=2, keepdim=True).values  # x -> [b, 128]

        x = self.mlp3(x)

        return x

if __name__ == "__main__":
    net = PointNetEncoder(64)
    batch=10
    num_points=1500
    pc = torch.randn((batch, num_points, 3))

    x = net(pc)
    print(x.size())

    num_points=1800
    pc = torch.randn((batch, num_points, 3))
    x = net(pc)
    print(x.size())
