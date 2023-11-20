import torch
from torch import nn as nn


# class CorrTorch_unfold(nn.Module):
#     """
#     3D correlation layer unfold version
#     dosen't work
#     """
#
#     def __init__(self, pad_size=1):
#         super().__init__()
#         self.padlayer = nn.ConstantPad3d(pad_size, 0)
#         self.activate = nn.LeakyReLU(0.2)
#         self.unfold = nn.Unfold(kernel_size=3)
#
#     def forward(self, x, y):
#         y_pad = self.padlayer(y)
#
#         C, D, H, W = x.shape[1], x.shape[2], x.shape[3], x.shape[4]
#
#         similarity = []
#
#         for i in range(D):
#             # slide window
#             unfold_y = self.unfold(y_pad[:, :, i, :, :]).reshape(3 * 3, H, W, 1, C)  # [1, C*9, H*W]
#             sim = torch.matmul(unfold_y,
#                                x[:, :, i, :, :].permute(2, 3, 1, 0)).sum(4).permute(3, 0, 1,
#                                                                                     2)  # d^2,H,W,1,1->1,d^2,H,W
#
#             similarity.append(sim)
#
#         # stack in 'D'
#         similarity = torch.stack(similarity, dim=2)  # torch.Size([1, d^2, D, H, W])
#
#         return self.activate(similarity)


class CorrTorch(nn.Module):
    def __init__(self, pad_size=1, max_displacement=1, stride1=1, stride2=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad3d(pad_size, 0)
        # self.activate = nn.LeakyReLU(0.2)

    def forward(self, in1, in2):
        b, c, depth, hei, wid = in1.shape
        conv = nn.Conv3d(kernel_size=1, in_channels=b*c*27, out_channels=27, stride=1, padding=0).cuda()
        in1 = in1.view(-1, depth, hei, wid)
        in2 = in2.view(-1, depth, hei, wid)

        in2_pad = self.padlayer(in2)
        offsetz, offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1)])


        sum = []
        for dz, dx, dy in zip(offsetz.reshape(-1), offsetx.reshape(-1), offsety.reshape(-1)):
            sum.append(in1 * in2_pad[:, dz:dz + depth, dy:dy + hei, dx:dx + wid] / torch.sqrt(torch.tensor(c).float()))
        output = torch.cat(sum, 0).unsqueeze(0)

        return conv(output)


# 2D
# class CorrBlock:
#     def __init__(
#             self,
#             fmap1: torch.Tensor,
#             fmap2: torch.Tensor,
#             num_levels: Optional[int] = 4,
#             radius: Optional[int] = 3
#     ):
#         """ Large-memory correlation-volume implementation
#         Args:
#             fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
#             fmap2: Correlation features of shape (B, N, F, H, W)
#             num_levels: Number of correlation-volume levels (Each level l has dimensions (H/2**l, W/2**l, H, W))
#             radius: Volume sampling radius
#         """
#         self.num_levels = num_levels
#         self.radius = radius
#         self.corr_pyramid = []
#
#         # all pairs correlation
#         corr = CorrBlock.corr(fmap1, fmap2)
#
#         batch, num, h1, w1, h2, w2 = corr.shape
#         corr = corr.reshape(batch * num * h1 * w1, 1, h2, w2)
#
#         for i in range(self.num_levels):
#             self.corr_pyramid.append(
#                 corr.view(batch * num, h1, w1, h2 // 2 ** i, w2 // 2 ** i))
#             corr = F.avg_pool2d(corr, 2, stride=2)
#
#     def __call__(self, coords: torch.Tensor, *args) -> torch.Tensor:
#         """ Sample from correlation volume
#         Args:
#             coords: Sampling coordinates of shape (B, N, H, W, 2)
#         Returns:
#             Sampled features of shape (B, N, num_levels * (radius + 1)**2, H, W)
#         """
#         out_pyramid = []
#         batch, num, ht, wd, _ = coords.shape
#         coords = coords.permute(0, 1, 4, 2, 3)
#         coords = coords.contiguous().view(batch * num, 2, ht, wd)
#
#         for i in range(self.num_levels):
#             corr = CorrSampler.apply(self.corr_pyramid[i], coords / 2 ** i, self.radius)
#             out_pyramid.append(corr.view(batch, num, -1, ht, wd))
#
#         return torch.cat(out_pyramid, dim=2)
#
#     def cat(self, other: Self) -> Self:
#         """ Concatenates self with given corr-volume
#         Args:
#             other: Other correlation-volume
#         Returns:
#             self
#         """
#         for i in range(self.num_levels):
#             self.corr_pyramid[i] = torch.cat([self.corr_pyramid[i], other.corr_pyramid[i]], 0)
#         return self
#
#     def __getitem__(self, index: torch.Tensor) -> Self:
#         """ Index correlation volume
#         Args:
#             index: Mask or indices of volumes to keep
#         Returns:
#             Indexed instance
#         """
#         for i in range(self.num_levels):
#             self.corr_pyramid[i] = self.corr_pyramid[i][index]
#         return self
#
#     @staticmethod
#     def corr(fmap1: torch.Tensor, fmap2: torch.Tensor) -> torch.Tensor:
#         """ Calculates all-pairs correlation
#         Args:
#             fmap1: Correlation features of shape (B, N, F, H, W) - F features per pixel
#             fmap2: Correlation features of shape (B, N, F, H, W)
#         Returns:
#             Correlation volume of shape (B, N, H, W, H; W)
#         """
#         batch, num, dim, ht, wd = fmap1.shape
#         fmap1 = fmap1.reshape(batch * num, dim, ht * wd) / 4.0
#         fmap2 = fmap2.reshape(batch * num, dim, ht * wd) / 4.0
#
#         corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
#         return corr.view(batch, num, ht, wd, ht, wd)


if __name__ == '__main__':
    a = torch.randn((1, 3, 96, 144, 144))
    b = torch.randn((1, 3, 96, 144, 144))

    corr2 = CorrTorch()

    d = corr2(a, b)

    print(d.shape)
