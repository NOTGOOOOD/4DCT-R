import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# basic conv block
class BasicConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, active=True):
        super(BasicConv3D, self).__init__()
        self.active = active
        # self.bn = nn.BatchNorm1d(in_channels)
        if self.active == True:
            self.activation = Mish()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, bias=False)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x = self.bn(x)
        if self.active == True:
            x = self.activation(x)
        x = self.conv(x)

        return x


class Resblock3D(nn.Module):
    def __init__(self, channels, out_channels, residual_activation=nn.Identity()):
        super(Resblock3D, self).__init__()

        self.channels = channels
        self.out_channels = out_channels
        if self.channels != self.out_channels:
            self.res_conv = BasicConv3D(channels, out_channels, 1)

        self.activation = Mish()
        self.block = nn.Sequential(
            BasicConv3D(channels, out_channels // 2, 1),
            BasicConv3D(out_channels // 2, out_channels, 1, active=False)
        )

    def forward(self, x):
        residual = x
        if self.channels != self.out_channels:
            residual = self.res_conv(x)
        return self.activation(residual + self.block(x))


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim):
        super(Self_Attn, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 查询卷积
        self.query_conv = BasicConv3D(in_dim, out_dim)

        self.value_conv = nn.Sequential(
            Resblock3D(in_dim, out_dim),
            Resblock3D(out_dim, out_dim)
        )

        # if in_dim != out_dim:
        #    self.short_conv = BasicConv1D(in_dim, out_dim)

        # self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X N)  32, 1024, 64
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        proj_query = self.query_conv(x).permute(0, 2, 3, 4, 1)  # B,C,D,H,W-> B,D,H,W,C
        proj_key = proj_query.permute(0, 1, 2, 4, 3)  # B,D,H,W,C-> B,D,H,C,W

        energy = torch.matmul(proj_query, proj_key)  # transpose check    B D H W W

        attention = self.softmax(energy)  # B D H W W

        proj_value = self.value_conv(x)  # proj_key# #B, C, D, H, W

        out_x = torch.matmul(proj_value.permute(0, 2, 3, 1, 4), attention).permute(0, 3, 1, 2,
                                                                                   4)  # B,D,H,C,W -> B,C,D,H,W

        out = self.beta * out_x + proj_value

        return out


# cross
class Cross_attention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, out_dim):
        super(Cross_attention, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.query_conv = BasicConv3D(in_dim, out_dim)

        self.beta = nn.Parameter(torch.zeros(1))
        self.activate = nn.LeakyReLU(0.2)

    def forward(self, x, y):  # B, C, D, H, W
        """
            inputs :
                x : current level feature maps ( B, C, D, H, W )
                y : i-1 level feature maps
            returns :
                out : B, C, D, H, W
        """
        proj_query_x = self.query_conv(x).permute(0, 2, 3, 1, 4)  # B D H C W

        proj_key_y = self.query_conv(y).permute(0, 2, 3, 4, 1)  # B D H W C

        energy_xy = torch.matmul(proj_query_x, proj_key_y)  # xi 对 y所有点的注意力得分   B D H C C

        attention_xy = self.activate(energy_xy)  # B D H C C
        attention_yx = self.activate(energy_xy.permute(0, 1, 2, 4, 3))  # BDHCC

        proj_value_x = proj_query_x  # self.value_conv_x(x) # [B, out_dim, 64]  B D H C W
        proj_value_y = proj_key_y.permute(0, 1, 2, 4,
                                          3)  # self.value_conv_x(y) # [B, out_dim, 64]  B D H W C -> B D H C W

        out_x = torch.matmul(attention_xy, proj_value_x)  # [B, out_dim, D, H, W]  B D H C W
        out_x = self.beta * out_x + proj_value_x  # self.kama*

        out_y = torch.matmul(attention_yx, proj_value_y)  # [B, out_dim, D, H, W] B D H C W
        out_y = self.beta * out_y + proj_value_y  # self.kama *

        # # At last, a mutiplication between out_x and out_y then follow an active function, instead of concat
        # return self.activate(torch.mul(out_x, out_y)).permute(0, 3, 1, 2, 4)  # B D H C W -> B,C,D,H,W
        return torch.cat([out_x, out_y], dim=3).permute(0, 3, 1, 2, 4)  # B D H C W -> B,C,D,H,W


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)
