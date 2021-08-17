# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import paddle
import paddle.nn as nn


class TVLoss(nn.Layer):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, mat):
        return paddle.mean(paddle.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               paddle.mean(paddle.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))


class TemporalSmoothLoss(nn.Layer):
    def __init__(self):
        super(TemporalSmoothLoss, self).__init__()

    def forward(self, mat):
        return paddle.mean(paddle.abs(mat[:, 1:] - mat[:, 0:-1]))

