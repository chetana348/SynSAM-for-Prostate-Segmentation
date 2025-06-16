import torch
import torch.nn as nn
from models.modules.utils.conv import *
from models.modules.utils.blocks import *

class ResInBlock(nn.Module):

    def __init__(self, dims, in_ch, out_ch, kernel_size, stride, norm) -> None:

        super().__init__()

        self.layer = ResBlock(dims=dims, in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, norm=norm)

    def forward(self, inp):
        return self.layer(inp)



class ResUNetSkipEncoder(nn.Module):
   def __init__(self, dims, in_ch, out_ch, num_layer, kernel_size,stride, upsample_kernel_size, norm) -> None:

        super().__init__()

        upsample_stride = upsample_kernel_size
        self.transp_conv_init = ConvBlock(dims, in_ch, out_ch, kernel_size=upsample_kernel_size, stride=upsample_stride, transpose=True)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBlock(dims, out_ch, out_ch, kernel_size=upsample_kernel_size, stride=upsample_stride, transpose=True),
                    ResBlock(dims=dims, in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, norm=norm))
                for i in range(num_layer)
            ]
        )


   def forward(self, x):
        x = self.transp_conv_init(x)
        for blk in self.blocks:
            x = blk(x)
        return x


class ResUNetSkipDecoder(nn.Module):

    def __init__(self, dims, in_ch, out_ch, kernel_size, upsample_kernel_size, norm) -> None:
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = ConvBlock(dims, in_ch, out_ch, kernel_size=upsample_kernel_size, stride=upsample_stride, transpose=True)

        self.conv_block = ResBlock(dims, out_ch + out_ch, out_ch, kernel_size=kernel_size, stride=1, norm=norm)

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out


class ConvOut(nn.Module):
    def __init__(self, dims, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(dims, in_ch, out_ch, kernel_size=1, stride=1, bias=True)

    def forward(self, inp):
        out = self.conv(inp)
        return out
