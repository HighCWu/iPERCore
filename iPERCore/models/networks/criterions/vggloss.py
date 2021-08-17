# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision import models
from typing import Union


class VGG19(nn.Layer):
    """
    Sequential(
          (0): Conv2D(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (1): ReLU()
          (2): Conv2D(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU()
          (4): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2D(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (6): ReLU()
          (7): Conv2D(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU()
          (9): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2D(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (11): ReLU()
          (12): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU()
          (14): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU()
          (16): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (17): ReLU()
          (18): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (19): Conv2D(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (20): ReLU()
          (21): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU()
          (23): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (24): ReLU()
          (25): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (26): ReLU()
          (27): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (28): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (29): ReLU()
          xxxx(30): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(31): ReLU()
          xxxx(32): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(33): ReLU()
          xxxx(34): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(35): ReLU()
          xxxx(36): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    """

    def __init__(self, ckpt_path: Union[str, bool] = "./assets/pretrains/vgg19-dcbb9e9d.pth",
                 requires_grad=False, before_relu=False):
        super(VGG19, self).__init__()

        if ckpt_path:
            vgg = models.vgg19(pretrained=False)
            ckpt = paddle.load(ckpt_path)
            vgg.load_state_dict(ckpt, strict=False)
            vgg_pretrained_features = vgg.features
        else:
            vgg_pretrained_features = models.vgg19(pretrained=True).features

        print(f"Loading vgg19 from {ckpt_path}...")

        if before_relu:
            slice_ids = [1, 6, 11, 20, 29]
        else:
            slice_ids = [2, 7, 12, 21, 30]

        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out


class VGG16(nn.Layer):
    """
        Sequential(
          (0): Conv2D(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (1): ReLU()
          (2): Conv2D(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (3): ReLU()
          (4): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (5): Conv2D(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (6): ReLU()
          (7): Conv2D(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (8): ReLU()
          (9): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (10): Conv2D(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (11): ReLU()
          (12): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (13): ReLU()
          (14): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (15): ReLU()
          (16): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (17): Conv2D(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (18): ReLU()
          (19): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (20): ReLU()
          (21): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (22): ReLU()
          (23): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          (24): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

          (25): ReLU()
          xxxx(26): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(27): ReLU()
          xxxx(28): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          xxxx(29): ReLU()
          xxxx(30): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )
    """

    def __init__(self, ckpt_path=False, requires_grad=False, before_relu=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        print("loading vgg16 ...")

        if before_relu:
            slice_ids = [1, 6, 11, 18, 25]
        else:
            slice_ids = [2, 7, 12, 19, 26]

        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out


class VGG11(nn.Layer):
    """
    Sequential(
      (0): Conv2D(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (1): ReLU()
      (2): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (3): Conv2D(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (4): ReLU()
      (5): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (6): Conv2D(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (7): ReLU()
      (8): Conv2D(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (9): ReLU()
      (10): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (11): Conv2D(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (12): ReLU()
      (13): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (14): ReLU()
      (15): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (16): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

      (17): ReLU()
      (18): Conv2D(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      ###(19): ReLU()
      ###(20): MaxPool2D(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    """
    def __init__(self, ckpt_path=False, requires_grad=False, before_relu=False):
        super(VGG11, self).__init__()
        vgg_pretrained_features = models.vgg11(pretrained=True).features
        print("loading vgg11 ...")

        if before_relu:
            slice_ids = [1, 4, 7, 12, 17]
        else:
            slice_ids = [2, 5, 8, 13, 18]

        self.slice1 = paddle.nn.Sequential()
        self.slice2 = paddle.nn.Sequential()
        self.slice3 = paddle.nn.Sequential()
        self.slice4 = paddle.nn.Sequential()
        self.slice5 = paddle.nn.Sequential()
        for x in range(slice_ids[0]):
            self.slice1.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[0], slice_ids[1]):
            self.slice2.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[1], slice_ids[2]):
            self.slice3.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[2], slice_ids[3]):
            self.slice4.add_sublayer(str(x), vgg_pretrained_features[x])
        for x in range(slice_ids[3], slice_ids[4]):
            self.slice5.add_sublayer(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.stop_gradient = True

    def forward(self, X):
        h_out1 = self.slice1(X)
        h_out2 = self.slice2(h_out1)
        h_out3 = self.slice3(h_out2)
        h_out4 = self.slice4(h_out3)
        h_out5 = self.slice5(h_out4)
        out = [h_out1, h_out2, h_out3, h_out4, h_out5]
        return out


class VGGLoss(nn.Layer):
    def __init__(self, 
                 before_relu=False, slice_ids=(0, 1, 2, 3, 4), vgg_type="VGG19", 
                 ckpt_path=False, resize=False):
        super(VGGLoss, self).__init__()

        if vgg_type == "VGG19":
            self.vgg = VGG19(ckpt_path=ckpt_path, before_relu=before_relu)
        elif vgg_type == "VGG16":
            self.vgg = VGG16(ckpt_path=ckpt_path, before_relu=before_relu)
        else:
            self.vgg = VGG11(ckpt_path=ckpt_path, before_relu=before_relu)

        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        # self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.slice_ids = slice_ids

        self.resize = resize

    def forward(self, x, y):
        if self.resize:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)
            y = F.interpolate(y, size=(224, 224), mode="bilinear", align_corners=True)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0

        for i in self.slice_ids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss
