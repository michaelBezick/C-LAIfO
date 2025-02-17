import math as m

import torch

points = torch.randn((100, 3, 1500, 3))


def rotate_aug(data):
    batch_size = data.size()[0]

    x_theta = torch.rand((batch_size)) * 2 * m.pi
    y_theta = torch.rand((batch_size)) * 2 * m.pi
    z_theta = torch.rand((batch_size)) * 2 * m.pi

    cos_x, sin_x = torch.cos(x_theta), torch.sin(x_theta)
    cos_y, sin_y = torch.cos(y_theta), torch.sin(y_theta)
    cos_z, sin_z = torch.cos(z_theta), torch.sin(z_theta)

    x_matrix = torch.zeros((batch_size, 3, 3), device=x_theta.device)
    x_matrix[:, 0, 0] = 1.0
    x_matrix[:, 1, 1] = cos_x
    x_matrix[:, 1, 2] = -sin_x
    x_matrix[:, 2, 1] = sin_x
    x_matrix[:, 2, 2] = cos_x

    y_matrix = torch.zeros((batch_size, 3, 3), device=y_theta.device)
    y_matrix[:, 1, 1] = 1.0
    y_matrix[:, 0, 0] = cos_y
    y_matrix[:, 0, 2] = sin_y
    y_matrix[:, 2, 0] = -sin_y
    y_matrix[:, 2, 2] = cos_y

    z_matrix = torch.zeros((batch_size, 3, 3), device=z_theta.device)
    z_matrix[:, 2, 2] = 1.0
    z_matrix[:, 0, 0] = cos_z
    z_matrix[:, 0, 1] = -sin_z
    z_matrix[:, 1, 0] = sin_z
    z_matrix[:, 1, 1] = cos_z

    z_matrix = z_matrix[:, None, :, :]
    y_matrix = y_matrix[:, None, :, :]
    x_matrix = x_matrix[:, None, :, :]

    data = torch.matmul(data, z_matrix.transpose(-1, -2))
    data = torch.matmul(data, y_matrix.transpose(-1, -2))
    data = torch.matmul(data, x_matrix.transpose(-1, -2))

    return data


print(points)
rotate_aug(points)
print(points)
