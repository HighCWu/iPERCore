# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import paddle
import paddle.nn as nn


class ResidualBlock(nn.Layer):
    """Residual Block."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2D(dim_in, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(dim_out, weight_attr=False, bias_attr=False),
            nn.ReLU(),
            nn.Conv2D(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2D(dim_out, weight_attr=False, bias_attr=False),
        )

    def forward(self, x):
        return x + self.main(x)


class ResNetInpaintor(nn.Layer):
    """Generator. Encoder-Decoder Architecture."""

    def __init__(self, c_dim=4, num_filters=(64, 128, 256, 512), n_res_block=6):
        super(ResNetInpaintor, self).__init__()
        self._name = 'ResNetInpaintor'

        layers = list()
        layers.append(nn.Conv2D(c_dim, num_filters[0], kernel_size=7, stride=1, padding=3, bias_attr=True))
        layers.append(nn.InstanceNorm2D(num_filters[0], weight_attr=False, bias_attr=False))
        layers.append(nn.ReLU())

        # Down-Sampling
        n_down = len(num_filters) - 1
        for i in range(1, n_down + 1):
            layers.append(nn.Conv2D(num_filters[i - 1], num_filters[i], kernel_size=3, stride=2, padding=1, bias_attr=True))
            layers.append(nn.InstanceNorm2D(num_filters[i], weight_attr=False, bias_attr=False))
            layers.append(nn.ReLU())

        # Bottleneck
        for i in range(n_res_block):
            layers.append(ResidualBlock(dim_in=num_filters[-1], dim_out=num_filters[-1]))

        # Up-Sampling
        for i in range(n_down, 0, -1):
            layers.append(nn.Conv2DTranspose(num_filters[i], num_filters[i - 1], kernel_size=4,
                                             stride=2, padding=1, bias_attr=False))
            layers.append(nn.InstanceNorm2D(num_filters[i - 1], weight_attr=False, bias_attr=False))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2D(num_filters[0], 3, kernel_size=7, stride=1, padding=3, bias_attr=False))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
