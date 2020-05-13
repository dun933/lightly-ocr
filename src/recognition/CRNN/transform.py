import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ported from https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/transformation.py

class TPS_STN(nn.Module):
    # RARE backbone: thine plate spline STN
    def __init__(self, F, I_size, I_r_size, num_channels=1):
        # I_size : image.shape, I_r_size : rectified iamge I_r, num_channels: number of channels of image -> returns batch_I_r: rectified image
        super(TPS_STN, self).__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size
        self.num_channels = num_channels
        self.locnet = LocalizationNetwork(self.F, self.num_channels)
        self.gridgen = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.locnet(batch_I)
        build_P_prime = self.gridgen.build_P_prime(batch_C_prime)
        reshape_P_prime = build_P_prime.reshape([build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2])
        batch_I_r = F.grid_sample(batch_I, reshape_P_prime, padding_mode='border', align_corner=True)
        return batch_I_r


class LocalizationNetwork(nn.Module):
    def __init__(self, F, num_channels):
        self.F = F
        self.num_channels = num_channels
        self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 64,kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)
                nn.MaxPool2d(2,2),
                nn.Conv2d(64, 128,kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True)
                nn.MaxPool2d(2,2),
                nn.Conv2d(128, 256,kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True)
                nn.MaxPool2d(2,2),
                nn.Conv2d(256, 512,kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True)
                nn.AdaptiveAvgPool2d(1)
                )
        self.loc_fc1 = nn.Sequential(nn.Linear(512,256), nn.ReLU(True))
        self.loc_fc2 = nn.Linear(256, self.F*2)
        # init fc2
        self.loc_fc2.weight.data.fill_(0)
        self.loc_fc2.bias.data = torch.from_numpy(
                np.concatenate([
                    np.stack([np.linspace(-1.,1., int(F/2)), np.linspace(0.,-1., int(F/2))], axis=1),
                    np.stack([np.linspace(-1.,1., int(F/2)), np.linspace(1.,0., int(F/2))], axis=1)
                    ], axis=0)
                ).float().view(-1)

    def forward(self, batch_I):
        # batch_I : batch input image: [b, num_channels, I_height, I_width]
        # batch_C_prime: predicted coordinates of fiducial points [b, F, 2]
        batch_size = batch_I.size(0)
        feats = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.loc_fc2(self.loc_fc1(feats)).view(batch_size, self.F, 2)
        return batch_C_prime

class GridGenerator(nn.Module):
    # grid generator for RARE, retuns P` by T*P
    def __init__(self, F, I_r_size):
        super(GridGenerator, self).__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)
        self.inv_delta_C = torch.Tensor(self._build_inv_delta_C(self.F, self.C)).float().cuda()  # F+3 x F+3
        self.P_hat = torch.Tensor(self._build_P_hat(self.F, self.C, self.P)).float().cuda()  # n x F+3

    def _build_C(self, F):
        # return coordinates of fiducial points in I_r; C
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        # return inv_delta_C which is needed to calculate T
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1)  # 1 x F+3
            ],
            axis=0
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (np.arange(-I_r_width, I_r_width, 2) + 1.0) / I_r_width  # self.I_r_width
        I_r_grid_y = (np.arange(-I_r_height, I_r_height, 2) + 1.0) / I_r_height  # self.I_r_height
        P = np.stack(np.meshgrid(I_r_grid_x, I_r_grid_y),axis=2) # self.I_r_width x self.I_r_height x 2
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(np.expand_dims(P, axis=1), (1, F, 1))  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        # generate Grid from batch_C_prime [batch_size x F x 2]
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat((batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1)  # batch_size x F+3 x 2
        batch_T = torch.bmm(batch_inv_delta_C, batch_C_prime_with_zeros)  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2
