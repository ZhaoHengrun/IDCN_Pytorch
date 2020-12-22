import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class ImplicitTrans(nn.Module):
    def __init__(self, q, in_channels):
        self.q = q
        super(ImplicitTrans, self).__init__()
        conv_shape = (64, 64, 1, 1)
        kernel = np.zeros(conv_shape, dtype='float32')

        r1 = math.sqrt(1.0 / 8)
        r2 = math.sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = self.q[u, v] * math.cos(_u * u * math.pi / 16) * math.cos(_v * v * math.pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index, index2, 0, 0] = t
        self.kernel = torch.from_numpy(kernel)
        self.kernel = self.kernel.cuda()

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.kernel, stride=1)
        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(ConvRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return output


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, padding=1, use_bias=True, dilation_rate=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=1, padding=padding,
                              bias=use_bias,
                              dilation=dilation_rate)

    def forward(self, x):
        output = self.conv(x)
        return output


class PyramidCell(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(PyramidCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation_rates = dilation_rates
        self.dilation_rate = 0
        # (3, 2, 1, 1, 1, 1)
        self.conv_relu_1 = ConvRelu(in_channels=self.in_channels, out_channels=self.out_channels,
                                    kernel=3, padding=3,
                                    dilation_rate=dilation_rates[0])
        self.conv_relu_2 = ConvRelu(in_channels=self.in_channels * 2, out_channels=self.out_channels,
                                    kernel=3, padding=2,
                                    dilation_rate=dilation_rates[1])
        self.conv_relu_3 = ConvRelu(in_channels=self.in_channels * 3, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_4 = ConvRelu(in_channels=self.in_channels * 4, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_5 = ConvRelu(in_channels=self.in_channels * 5, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])
        self.conv_relu_6 = ConvRelu(in_channels=self.in_channels * 6, out_channels=self.out_channels,
                                    kernel=3, padding=1,
                                    dilation_rate=dilation_rates[2])

    def forward(self, x):
        t = self.conv_relu_1(x)  # 64
        _t = torch.cat([x, t], dim=1)  # 128

        t = self.conv_relu_2(_t)
        _t = torch.cat([_t, t], dim=1)  #

        t = self.conv_relu_3(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_4(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_5(_t)
        _t = torch.cat([_t, t], dim=1)

        t = self.conv_relu_6(_t)
        _t = torch.cat([_t, t], dim=1)
        return _t


class DualDomainBlock(nn.Module):
    def __init__(self, n_channels, n_pyramid_cells, n_pyramid_channels, qy, qc):
        super(DualDomainBlock, self).__init__()
        self.pyramid = PyramidCell(in_channels=n_channels, out_channels=n_pyramid_channels,
                                   dilation_rates=n_pyramid_cells)
        self.conv_1 = Conv(in_channels=n_channels * 7, out_channels=n_channels, kernel=3, padding=1)
        self.conv_2 = Conv(in_channels=n_channels * 7, out_channels=n_channels, kernel=3,
                           padding=2, dilation_rate=2)
        self.implicit_trans_1 = ImplicitTrans(q=qy, in_channels=n_channels)
        self.implicit_trans_2 = ImplicitTrans(q=qc, in_channels=n_channels)
        self.conv_3 = Conv(in_channels=n_channels * 7, out_channels=n_channels, kernel=3, padding=1)
        self.conv_4 = Conv(in_channels=n_channels * 2, out_channels=n_channels, kernel=3, padding=1)

    def forward(self, x):
        _t = self.pyramid(x)
        _ty = self.conv_1(_t)
        _tc = self.conv_2(_t)
        _ty = torch.clamp(_ty, -0.5, 0.5)
        _ty = self.implicit_trans_1(_ty)
        _tc = self.implicit_trans_2(_tc)
        _tp = self.conv_3(_t)
        _td = torch.cat([_ty, _tc], dim=1)
        _td = self.conv_4(_td)
        y = torch.add(_td, _tp)
        y = y.mul(0.1)
        y = torch.add(x, y)
        return y


class IDCN(nn.Module):
    def __init__(self, n_channels, n_pyramids, n_pyramid_cells, n_pyramid_channels, qy, qc):
        super(IDCN, self).__init__()
        self.n_channels = n_channels
        self.n_pyramids = n_pyramids
        self.n_pyramid_cells = n_pyramid_cells
        self.n_pyramid_channels = n_pyramid_channels
        self.qy = qy
        self.qc = qc

        self.conv_relu_1 = ConvRelu(in_channels=4, out_channels=n_channels, kernel=5, padding=2)
        self.conv_relu_2 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_relu_3 = ConvRelu(in_channels=n_channels, out_channels=n_channels, kernel=3, padding=1)
        self.conv_relu_4 = ConvRelu(in_channels=n_channels, out_channels=3, kernel=5, padding=2)

        self.dual_domain_blocks = self.make_layer(
            block=DualDomainBlock,
            num_of_layer=self.n_pyramids)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(n_channels=self.n_channels, n_pyramid_cells=self.n_pyramid_cells,
                                n_pyramid_channels=self.n_pyramid_channels, qy=self.qy, qc=self.qc))
        return nn.Sequential(*layers)

    def forward(self, x):
        t = self.conv_relu_1(x)
        t = self.conv_relu_2(t)
        t = self.dual_domain_blocks(t)
        t = self.conv_relu_3(t)
        y = self.conv_relu_4(t)
        return y
