from __future__ import annotations
import numpy as np
from typing import Union, Sequence, Tuple, Optional
from monai.networks.layers.factories import Act, Norm, split_args, Conv
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv


class ConvLayer(nn.Sequential):
    def __init__(
        self, dims, in_ch, out_ch, strides=1, kernel_size=3, dilation=1,
        groups=1, bias=True, transpose=False, padding=None, output_padding=None
    ) -> None:
        super().__init__()
        self.dims = dims
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.transpose = transpose

        # If padding isn't specified, calculate symmetric padding based on kernel and dilation
        if padding is None:
            kernel_size = [kernel_size] * dims if isinstance(kernel_size, int) else kernel_size
            dilation = [dilation] * dims if isinstance(dilation, int) else dilation
            padding = [((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation)]

        # Choose the appropriate convolution class
        conv_cls = Conv[Conv.CONVTRANS if transpose else Conv.CONV, dims]

        # Build convolution or transposed convolution layer
        if transpose:
            if output_padding is None:
                output_padding = strides - 1 if isinstance(strides, int) else [s - 1 for s in strides]

            layer = conv_cls(
                in_ch, out_ch, kernel_size=kernel_size, stride=strides, padding=padding,
                output_padding=output_padding, dilation=dilation, groups=groups, bias=bias
            )
        else:
            layer = conv_cls(
                in_ch, out_ch, kernel_size=kernel_size, stride=strides, padding=padding,
                dilation=dilation, groups=groups, bias=bias
            )

        self.add_module("conv", layer)


def ConvBlock(dims, in_ch, out_ch, kernel_size=3, stride=1, norm=Norm.INSTANCE, bias=False, transpose=False):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_vals = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_vals)
    padding = padding[0] if len(padding) == 1 else padding

    output_padding = None
    if transpose:
        padding_np = np.atleast_1d(padding)
        out_pad_calc = 2 * padding_np + stride_np - kernel_size_np
        output_padding = tuple(int(p) for p in out_pad_calc)
        output_padding = output_padding[0] if len(output_padding) == 1 else output_padding

    return ConvLayer(
        dims=dims, in_ch=in_ch, out_ch=out_ch, strides=stride, kernel_size=kernel_size,
        bias=bias, transpose=transpose, padding=padding, output_padding=output_padding
    )
