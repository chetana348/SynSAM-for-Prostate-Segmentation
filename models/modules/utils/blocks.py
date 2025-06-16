import torch
import torch.nn as nn
from models.modules.utils.conv import *
from monai.networks.layers.factories import Act, Norm, split_args, Conv

class ResBlock(nn.Module):
    def __init__(self, dims, in_ch, out_ch, kernel_size, stride, norm):
        super().__init__()
        self.conv1 = ConvBlock(dims, in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvBlock(dims, out_ch, out_ch, kernel_size=kernel_size, stride=1)
        self.conv3 = ConvBlock(dims, in_ch, out_ch, kernel_size=1, stride=stride,
        )
        self.lrelu = Act["leakyrelu"](inplace=True, negative_slope=0.01)
        self.norm1 = Norm[norm, dims](out_ch, affine=True)
        self.norm2 = Norm[norm, dims](out_ch, affine=True)
        self.norm3 = Norm[norm, dims](out_ch, affine=True)
        self.downsample = in_ch != out_ch
        stride_np = np.atleast_1d(stride)
        if not np.all(stride_np == 1):
            self.downsample = True

    def forward(self, inp):
        residual = inp
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.lrelu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        if self.downsample:
            residual = self.conv3(residual)
            residual = self.norm3(residual)
        out += residual
        out = self.lrelu(out)
        return out

