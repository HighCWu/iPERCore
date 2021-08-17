# Copyright (c) 2020-2021 impersonator.org authors (Wen Liu and Zhixin Piao). All rights reserved.

import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F

__all__ = ["SENet", "Sphere20a", "senet50", "FaceLoss"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


# This SEModule is not used.
class SEModule(nn.Layer):

    def __init__(self, planes, compress_rate):
        super(SEModule, self).__init__()
        self.conv1 = nn.Conv2D(planes, planes // compress_rate, kernel_size=1, stride=1, bias_attr=True)
        self.conv2 = nn.Conv2D(planes // compress_rate, planes, kernel_size=1, stride=1, bias_attr=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = F.avg_pool2d(module_input, kernel_size=module_input.shape[2])
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, stride=stride, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, planes * 4, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        # SENet
        compress_rate = 16
        # self.se_block = SEModule(planes * 4, compress_rate)  # this is not used.
        self.conv4 = nn.Conv2D(planes * 4, planes * 4 // compress_rate, kernel_size=1, stride=1, bias_attr=True)
        self.conv5 = nn.Conv2D(planes * 4 // compress_rate, planes * 4, kernel_size=1, stride=1, bias_attr=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        ## senet
        out2 = F.avg_pool2d(out, kernel_size=out.size(2))
        out2 = self.conv4(out2)
        out2 = self.relu(out2)
        out2 = self.conv5(out2)
        out2 = self.sigmoid(out2)
        # out2 = self.se_block.forward(out)  # not used

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out2 * out + residual
        # out = out2 + residual  # not used
        out = self.relu(out)
        return out


class SENet(nn.Layer):

    def __init__(self, block, layers, num_classes=8631, include_top=True):
        self.inplanes = 64
        super(SENet, self).__init__()
        self.include_top = include_top

        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2D(7, stride=1)

        if self.include_top:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.assign(paddle.normal(0, math.sqrt(2. / n), m.weight.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.assign(paddle.ones_like(m.weight))
                m.bias.assign(paddle.zeros_like(m.bias))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, get_feat=True):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_avg = self.avgpool(x4)

        if not self.include_top:
            if get_feat:
                return [x0, x1, x2, x3, x4]
            else:
                return x_avg

        else:
            x_fc = x_avg.reshape((x_avg.shape[0], -1))
            x_fc = self.fc(x_fc)

            if get_feat:
                return [x0, x1, x2, x3, x4]
            else:
                return x_fc


def senet50(**kwargs):
    """Constructs a SENet-50 model.
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


class Sphere20a(nn.Layer):
    def __init__(self, classnum=10574, feature=False):
        super(Sphere20a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        # input = B*3*112*96
        self.conv1_1 = nn.Conv2D(3, 64, 3, 2, 1)  # =>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2D(64, 64, 3, 1, 1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2D(64, 64, 3, 1, 1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2D(64, 128, 3, 2, 1)  # =>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2D(128, 128, 3, 1, 1)  # =>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2D(128, 128, 3, 1, 1)
        self.relu2_5 = nn.PReLU(128)

        self.conv3_1 = nn.Conv2D(128, 256, 3, 2, 1)  # =>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2D(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2D(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2D(256, 256, 3, 1, 1)  # =>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2D(256, 256, 3, 1, 1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2D(256, 512, 3, 2, 1)  # =>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2D(512, 512, 3, 1, 1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512 * 7 * 6, 512)

    def forward(self, x):
        feat_outs = []
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))
        feat_outs.append(x)

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))
        feat_outs.append(x)

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))
        feat_outs.append(x)

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
        feat_outs.append(x)

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        feat_outs.append(x)

        return feat_outs


class FaceLoss(nn.Layer):
    BASE_FACT_SIZE = 256

    def __init__(self, pretrained_path="asset/spretrains/sphere20a_20171020.pth", factor=1):
        super(FaceLoss, self).__init__()
        if "senet" in pretrained_path:
            self.net = senet50(include_top=False)
            self.load_senet_model(pretrained_path)
            self.height, self.width = 224, 224
        else:
            self.net = Sphere20a()
            self.load_sphere_model(pretrained_path)
            self.height, self.width = 112, 96

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        self.height = int(self.height * factor)
        self.width = int(self.width * factor)

        self.net.eval()
        self.criterion = nn.L1Loss()

        for param in self.net.parameters():
            param.stop_gradient = True

        # from utils.visualizers.visdom_visualizer import VisdomVisualizer
        # self._visualizer = VisdomVisualizer("debug", ip="http://10.10.10.100", port=31102)

    def forward(self, imgs1, imgs2, kps1=None, kps2=None, bbox1=None, bbox2=None):
        """
        Args:
            imgs1:
            imgs2:
            kps1:
            kps2:
            bbox1:
            bbox2:

        Returns:

        """
        if kps1 is not None:
            head_imgs1 = self.crop_head_kps(imgs1, kps1)
        elif bbox1 is not None:
            head_imgs1 = self.crop_head_bbox(imgs1, bbox1)
        elif self.check_need_resize(imgs1):
            head_imgs1 = F.interpolate(imgs1, size=(self.height, self.width), mode="bilinear", align_corners=True)
        else:
            head_imgs1 = imgs1

        if kps2 is not None:
            head_imgs2 = self.crop_head_kps(imgs2, kps2)
        elif bbox2 is not None:
            head_imgs2 = self.crop_head_bbox(imgs2, bbox2)
        elif self.check_need_resize(imgs2):
            head_imgs2 = F.interpolate(imgs1, size=(self.height, self.width), mode="bilinear", align_corners=True)
        else:
            head_imgs2 = imgs2

        if len(head_imgs1) == 0 or len(head_imgs2) == 0:
            loss = 0.0
        else:
            loss = self.compute_loss(head_imgs1, head_imgs2)

        # self._visualizer.vis_named_img("img2", imgs2)
        # self._visualizer.vis_named_img("head imgs2", head_imgs2)
        #
        # self._visualizer.vis_named_img("img1", imgs1)
        # self._visualizer.vis_named_img("head imgs1", head_imgs1)
        # import ipdb
        # ipdb.set_trace()

        return loss

    def compute_loss(self, img1, img2):
        """

        Args:
            img1 (paddle.Tensor): (bs, 3, 112, 92)
            img2 (paddle.Tensor): (bs, 3, 112, 92)

        Returns:

        """

        f1, f2 = self.net(img1), self.net(img2)

        loss = 0.0
        for i in range(len(f1)):
            loss += self.weights[i] * self.criterion(f1[i], f2[i].detach())

        return loss

    def check_need_resize(self, img):
        return img.shape[2] != self.height or img.shape[3] != self.width

    def crop_head_bbox(self, imgs, bboxs):
        """
        Args:
            bboxs: (N, 4), 4 = [min_x, max_x, min_y, max_y]

        Returns:
            resize_image:
        """
        bs, _, ori_h, ori_w = imgs.shape

        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = bboxs[i]
            if (min_x != max_x) and (min_y != max_y):
                head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h", w")
                head = F.interpolate(head, size=(self.height, self.width), mode="bilinear", align_corners=True)
                head_imgs.append(head)

        if len(head_imgs) != 0:
            head_imgs = paddle.concat(head_imgs, axis=0)

        return head_imgs

    def crop_head_kps(self, imgs, kps):
        """

        Args:
            imgs (paddle.Tensor): (N, C, H, W)
            kps (paddle.Tensor): (N, 19, 2)

        Returns:

        """
        
        bs, _, ori_h, ori_w = imgs.shape

        rects = self.find_head_rect(kps, ori_h, ori_w)
        head_imgs = []

        for i in range(bs):
            min_x, max_x, min_y, max_y = rects[i]

            if (min_x != max_x) and (min_y != max_y):
                head = imgs[i:i + 1, :, min_y:max_y, min_x:max_x]  # (1, c, h", w")
                head = F.interpolate(head, size=(self.height, self.width), mode="bilinear", align_corners=True)
                head_imgs.append(head)

        if len(head_imgs) != 0:
            head_imgs = paddle.concat(head_imgs, axis=0)

        return head_imgs

    @staticmethod
    @paddle.no_grad()
    def find_head_rect(kps, height, width):
        NECK_IDS = 12

        kps = (kps + 1) / 2.0

        necks = kps[:, NECK_IDS, 0]
        zeros = paddle.zeros_like(necks)
        ones = paddle.ones_like(necks)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_x, _ = paddle.min(kps[:, NECK_IDS:, 0] - 0.05, axis=1)
        min_x = paddle.max(min_x, zeros)

        max_x, _ = paddle.max(kps[:, NECK_IDS:, 0] + 0.05, axis=1)
        max_x = paddle.min(max_x, ones)

        # min_x = int(max(0.0, np.min(kps[HEAD_IDS:, 0]) - 0.1) * image_size)
        min_y, _ = paddle.min(kps[:, NECK_IDS:, 1] - 0.05, axis=1)
        min_y = paddle.max(min_y, zeros)

        max_y, _ = paddle.max(kps[:, NECK_IDS:, 1], axis=1)
        max_y = paddle.min(max_y, ones)

        min_x = (min_x * width).astype(paddle.int64)  # (T, 1)
        max_x = (max_x * width).astype(paddle.int64)  # (T, 1)
        min_y = (min_y * height).astype(paddle.int64)  # (T, 1)
        max_y = (max_y * height).astype(paddle.int64)  # (T, 1)

        # print(min_x.shape, max_x.shape, min_y.shape, max_y.shape)
        rects = paddle.stack((min_x, max_x, min_y, max_y), axis=1)

        # import ipdb
        # ipdb.set_trace()

        return rects

    def load_senet_model(self, pretrain_model):
        saved_data = paddle.load(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith("fc"):
                continue
            save_weights_dict[key] = paddle.to_tensor(val)

        self.net.load_state_dict(save_weights_dict)

        print(f"Loading face model from {pretrain_model}")

    def load_sphere_model(self, pretrain_model):
        saved_data = paddle.load(pretrain_model)
        save_weights_dict = dict()

        for key, val in saved_data.items():
            if key.startswith("fc6"):
                continue
            save_weights_dict[key] = val

        self.net.load_state_dict(save_weights_dict)

        print(f"Loading face model from {pretrain_model}")
