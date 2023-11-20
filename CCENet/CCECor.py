import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.Functions import SpatialTransformer
from utils.correlation_layer import CorrTorch as Correlation

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class UNet_Encoder(nn.Module):
    def __init__(self, in_channel, enc_features, max_pool=2, ndims=3):
        super(UNet_Encoder, self).__init__()
        nb_levels = len(enc_features) + 1
        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = in_channel
        self.encoder_nfs = [in_channel]
        self.encoder = nn.ModuleList()
        for level in range(nb_levels - 1):
            convs = nn.ModuleList()
            nf = enc_features[level]
            convs.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
            self.encoder.append(convs)
            self.encoder_nfs.append(prev_nf)

    def forward(self, x):
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)

            x_history.append(x)
            x = self.pooling[level](x)
            # x = F.interpolate(x, scale_factor=0.5, mode='trilinear',
            #                   align_corners=True, recompute_scale_factor=False)

        return x, x_history


class UNet_Decoder(nn.Module):
    def __init__(self, enc_features, dec_features, final_convs, max_pool=2, ndims=3):
        super(UNet_Decoder, self).__init__()

        # enc_nf, dec_nf = nb_features
        nb_levels = len(enc_features) + 1

        # if isinstance(max_pool, int):
        #     max_pool = [max_pool] * nb_levels
        # cache downsampling / upsampling operations
        # self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        prev_nf = enc_features[-1]
        encoder_nfs = np.flip(enc_features)
        self.decoder = nn.ModuleList()
        for level in range(nb_levels - 1):
            convs = nn.ModuleList()
            nf = dec_features[level]
            convs.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
            self.decoder.append(convs)
            prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, fea, x_history):
        # decoder forward pass with upsampling and concatenation
        fea_list = []
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                fea = conv(fea)

            # fea = self.upsampling[level](fea)
            if fea.shape[2:] != x_history[-1].shape[2:]:
                fea = F.interpolate(fea, x_history[-1].shape[2:], mode='trilinear',
                                  align_corners=True)

            fea_list.append(fea)
            fea = torch.cat([fea, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            fea = conv(fea)

        return fea, fea_list

class Dual_Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 dim=None,
                 infeats=None,
                 nb_features=None,
                 max_pool=2):

        super().__init__()

        # ensure correct dimensionality
        ndims = dim
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            nf = enc_nf[level]
            convs.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            nf = dec_nf[level]
            convs.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf
            self.decoder.append(convs)
            prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)
            # x = F.interpolate(x, scale_factor=0.5, mode='trilinear',
            #                   align_corners=True, recompute_scale_factor=False)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)

            x = self.upsampling[level](x)
            if x.shape[2:] != x_history[-1].shape[2:]:
                x = F.interpolate(x, x_history[-1].shape[2:], mode='trilinear',
                                  align_corners=True)
            x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class FlowNet(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super(FlowNet, self).__init__()
        med_channels = int(in_channels/2)
        # med_channels = 27
        # self.corr_layer = Correlation()

        self.conv_layer = nn.Sequential(
                # nn.Conv3d(27+in_channels, med_channels, kernel_size=3, padding=1),
                nn.Conv3d(in_channels, med_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )
        self.resblock_layer_1 = nn.Sequential(
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv3d(med_channels, med_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
            )

        self.offset_layer = nn.Conv3d(med_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x1, x2):
        # x3 = self.corr_layer(x1, x2)
        # x = torch.cat([x1, x2, x3], dim=1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_layer(x)
        x = x + self.resblock_layer_1(x)
        x = self.offset_layer(x)
        return x

class CCECoNet(nn.Module):
    def __init__(self, dim):
        super(CCECoNet, self).__init__()

        self.enc_nf = [16, 32, 32, 32]
        self.dec_nf = [32, 32, 32, 32]
        self.final_nf = [32, 16, 16]

        self.unet_encoder = UNet_Encoder(in_channel=2, enc_features=self.enc_nf)
        self.unet_decoder = UNet_Decoder(enc_features=self.enc_nf, dec_features=self.dec_nf, final_convs=self.final_nf)
        # nb_unet_features = [enc_nf, dec_nf]
        # self.dual_stream_model = Dual_Unet(
        #     dim=dim,
        #     infeats=(2),
        #     nb_features=nb_unet_features
        # )

        self.flow = nn.Conv3d(self.unet_decoder.final_nf, dim, kernel_size=3, padding=1)
        self.transform = SpatialTransformer(dim=dim)

    def forward(self, source, target, istrain=False):
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        fea, x_history = self.unet_encoder(x)
        x, flow_list = self.unet_decoder(fea, x_history)

        # transform into flow field
        flow_field = self.flow(x)

        if source.shape[2:] != flow_field.shape[2:]:
            print("warning: source dosent consistence with pos_flow")
            flow_field = F.interpolate(flow_field, source.shape[2:], mode='trilinear',
                                     align_corners=True)

        # warp image with flow field
        y_source = self.transform(source, flow_field)

        # return non-integrated flow field if training
        return {'warped_img': y_source, 'flow': flow_field}

class CCECoNetDual(CCECoNet):
    def __init__(self,dim):
        super(CCECoNetDual, self).__init__(dim=dim)
        self.conv_size = [16, 16, 16, 16]
        self.offset = []
        for i in range(len(self.enc_nf)):
            # fusion
            offset_layer = FlowNet(self.conv_size[-i-1]*2)
            # cascade
            self.offset.append(offset_layer)
        self.offset = nn.ModuleList(self.offset)

    def forward(self, source, target, istrain=False):
        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)

        fea, x_history = self.unet_encoder(x)
        x, fea_list = self.unet_decoder(fea, x_history)

        fea_moving = []
        fea_fix = []
        shape_list = []
        delta_list = []
        self.keep_delta = True

        for i, feature in enumerate(fea_list):
            fea_moving.append(feature[:, : int(self.dec_nf[-1-i]/2), ...])
            fea_fix.append(feature[:,  int(self.dec_nf[-1-i]/2):, ...])
            shape_list.append(feature.shape[2:])

        last_flow = None
        for lvl, (x_warp, x_fix) in enumerate(zip(fea_moving, fea_fix)):
            # apply flow
            if last_flow is not None:
                x_warp = self.transform(x_warp, last_flow.detach())

            # fusion
            flow = self.offset[lvl](x_warp, x_fix)
            # cascade
            if self.keep_delta:
                delta_list.append(flow)

            if last_flow is not None:
                flow += last_flow

            if lvl < len(fea_list)-1:
                # last_flow = F.interpolate(flow, scale_factor=2, mode=self.interp_mode)
                last_flow = F.interpolate(flow, size=shape_list[lvl+1], mode='trilinear', align_corners=True)
            else:
                last_flow = flow

        # fusion
        flow_field = self.flow(x)
        flow_field += last_flow

        # transform into flow field
        if source.shape[2:] != flow_field.shape[2:]:
            print("warning: source dosent consistence with pos_flow")
            flow_field = F.interpolate(flow_field, source.shape[2:], mode='trilinear',
                                     align_corners=True)

        # warp image with flow field
        y_source = self.transform(source, flow_field)

        # return non-integrated flow field if training
        return {'warped_img': y_source, 'flow': flow_field}
