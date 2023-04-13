import torch
from torch import nn as nn


class CorrTorch_unfold(nn.Module):
    """
    3D correlation layer unfold version
    dosen't work
    """

    def __init__(self, pad_size=1):
        super().__init__()
        self.padlayer = nn.ConstantPad3d(pad_size, 0)
        self.activate = nn.LeakyReLU(0.2)
        self.unfold = nn.Unfold(kernel_size=3)

    def forward(self, x, y):
        y_pad = self.padlayer(y)

        C, D, H, W = x.shape[1], x.shape[2], x.shape[3], x.shape[4]

        similarity = []

        for i in range(D):
            # slide window
            unfold_y = self.unfold(y_pad[:, :, i, :, :]).reshape(3 * 3, H, W, 1, C)  # [1, C*9, H*W]
            sim = torch.matmul(unfold_y,
                               x[:, :, i, :, :].permute(2, 3, 1, 0)).sum(4).permute(3, 0, 1,
                                                                                    2)  # d^2,H,W,1,1->1,d^2,H,W

            similarity.append(sim)

        # stack in 'D'
        similarity = torch.stack(similarity, dim=2)  # torch.Size([1, d^2, D, H, W])

        return self.activate(similarity)


class CorrTorch(nn.Module):
    def __init__(self, pad_size=1, max_displacement=1, stride1=1, stride2=1):
        assert pad_size == max_displacement
        assert stride1 == stride2 == 1
        super().__init__()
        self.max_hdisp = max_displacement
        self.padlayer = nn.ConstantPad3d(pad_size, 0)
        self.activate = nn.LeakyReLU(0.2)
        self.conv = nn.Conv3d(in_channels=27, out_channels=3, kernel_size=(1, 1, 1), stride=1)

    def forward(self, in1, in2):
        in2_pad = self.padlayer(in2)
        offsetz, offsety, offsetx = torch.meshgrid([torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1),
                                                    torch.arange(0, 2 * self.max_hdisp + 1)])

        depth, hei, wid = in1.shape[2], in1.shape[3], in1.shape[4]

        sum = []
        for dz, dx, dy in zip(offsetz.reshape(-1), offsetx.reshape(-1), offsety.reshape(-1)):
            sum.append(torch.mean(in1 * in2_pad[:, :, dz:dz + depth, dy:dy + hei, dx:dx + wid], 1, keepdim=True))

        output = torch.cat(sum, 1)
        output = self.conv(output)

        return self.activate(output)


if __name__ == '__main__':
    a = torch.randn((1, 2, 3, 4, 5))
    b = torch.randn((1, 2, 3, 4, 5))

    corr2 = CorrTorch()

    d = corr2(a, b)

    print(d)
