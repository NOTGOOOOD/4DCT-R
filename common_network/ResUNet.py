import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet18
from torch.distributions.normal import Normal
from typing import Optional, Union, Type, List, Tuple, Dict
from collections import OrderedDict
from utils.Functions import SpatialTransformer
from resnet18 import BasicBlock, Bottleneck


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """

    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        self.conv = nn.Conv3d(int(in_channels), int(out_channels), kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv(x)
        return out


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class ResNetEncoder(nn.Module):
    """ Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers: Optional[int] = 18, pretrained: Optional[bool] = True):
        super(ResNetEncoder, self).__init__()

        assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
        blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
        block_type = {18: BasicBlock, 50: Bottleneck}[num_layers]

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv3d(2, self.num_ch_enc[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.dilation = 1
        self.inplanes = 64
        self.layer1 = self._make_layer(block_type, self.num_ch_enc[1], blocks[0])
        self.layer2 = self._make_layer(block_type, self.num_ch_enc[2], blocks[1], stride=2)
        self.layer3 = self._make_layer(block_type, self.num_ch_enc[3], blocks[2], stride=2)
        self.layer4 = self._make_layer(block_type, self.num_ch_enc[4], blocks[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load ImageNet pretrained weights for available layers
        # import pdb;pdb.set_trace()
        # if pretrained:
        #     model_state = self.state_dict()
        #     loaded_state = model_zoo.load_url(resnet.model_urls['resnet{}'.format(num_layers)])
        #     for loaded_key in loaded_state:
        #         if loaded_key in model_state:
        #             if model_state[loaded_key].shape != loaded_state[loaded_key].shape:
        #                 model_state[loaded_key][:, :loaded_state[loaded_key].shape[1]].copy_(loaded_state[loaded_key])
        #                 print("{}: model_state_shape: {}, loaded_state_shape: {}".format(
        #                     loaded_key, model_state[loaded_key].shape, loaded_state[loaded_key].shape))
        #             else:
        #                 model_state[loaded_key].copy_(loaded_state[loaded_key])
        #         else:
        #             print("{}: In checkpoint but not in model".format(loaded_key))

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: Optional[int] = 1,
            dilate: Optional[bool] = False
    ) -> nn.Sequential:
        """ Adapted from torchvision/models/resnet.py
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet18.conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, nn.BatchNorm3d)]

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=nn.BatchNorm3d))

        return nn.Sequential(*layers)

    def forward(self, src_img: torch.Tensor, trt_img: torch.Tensor) -> List[torch.Tensor]:
        features = []
        # x = (input_image - 0.45) / 0.225
        x = torch.cat((src_img, trt_img), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        features.append(self.relu(x))
        features.append(self.layer1(self.maxpool(features[-1])))
        features.append(self.layer2(features[-1]))
        features.append(self.layer3(features[-1]))
        features.append(self.layer4(features[-1]))

        return features


class UnetDecoder(nn.Module):
    """ Depth completion decoder stage
    """

    def __init__(
            self,
            scales: Optional[List[int]] = None,
            num_output_channels: Optional[int] = 3,
            use_skips: Optional[bool] = True,
            dim: int = 3,
    ):
        super(UnetDecoder, self).__init__()

        self.scales = scales if scales is not None else [0, 1, 2, 3]
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.dim = dim
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            # init flow layer with small weights and bias
            self.convs[("dispconv", s)].weight = nn.Parameter(Normal(0, 1e-5).sample(self.convs[("dispconv", s)].conv.weight.shape))
            self.convs[("dispconv", s)].bias = nn.Parameter(torch.zeros(self.convs[("dispconv", s)].conv.bias.shape))

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.activate = nn.LeakyReLU(0.2)

    def upsample(self, x, trt_size=None):
        """Upsample input tensor by a factor of 2
        """
        if trt_size is None:
            return F.interpolate(x, scale_factor=2, mode='bilinear' if self.dim == 2 else 'trilinear')
        else:
            return F.interpolate(x, size=trt_size, mode='bilinear' if self.dim == 2 else 'trilinear')

    def forward(self, input_features: List[torch.Tensor]) -> Dict[Tuple, torch.Tensor]:
        outputs = {}
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [self.upsample(x, input_features[i - 1].shape[2:] if i > 0 else None)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                outputs[("disp", i)] = self.activate(self.convs[("dispconv", i)](x))

        return outputs


class ResUnetModel(nn.Module):
    def __init__(self):
        super(ResUnetModel, self).__init__()

        self.encoder = ResNetEncoder(num_layers=18)
        self.decoder = UnetDecoder(num_output_channels=3, dim=3)
        self.stn = SpatialTransformer()

    def forward(self, src_img, trt_img):
        res = {}
        features = self.encoder(src_img, trt_img)
        output = self.decoder(features)
        # TODO 利用多尺度dvf计算loss

        warped_img = self.stn(src_img, output[("disp", 0)])

        res['warped_img'] = warped_img
        res['disp'] = output[("disp", 0)]
        return res
