# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import paddle
import paddle.nn as nn
import functools


class PatchDiscriminator(nn.Layer):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=32, n_layers=3, max_nf_mult=8,
                 norm_type="batch", use_sigmoid=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchDiscriminator, self).__init__()

        norm_layer = self._get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2D has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2D
        else:
            use_bias = norm_layer != nn.BatchNorm2D

        kw = 4
        padw = 1
        sequence = [nn.Conv2D(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, max_nf_mult)
            sequence += [
                nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias_attr=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, max_nf_mult)
        sequence += [
            nn.Conv2D(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias_attr=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2D(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.model = nn.Sequential(*sequence)

    def _get_norm_layer(self, norm_type="batch"):
        if norm_type == "batch":
            norm_layer = functools.partial(nn.BatchNorm2D)
        elif norm_type == "instance":
            norm_layer = functools.partial(nn.InstanceNorm2D)
        elif norm_type == "batchnorm2d":
            norm_layer = nn.BatchNorm2D
        else:
            raise NotImplementedError(f"normalization layer [{norm_type}] is not found")

        return norm_layer

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
