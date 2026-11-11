# -*- coding: utf-8 -*-
"""骨干网络与特征金字塔.

论文 4.2 节：保留步长 4 的最高分辨率层 P2，供小目标检测使用；
P2 只接检测分支与分割分支的融合上采样，避免显存爆炸。
"""

import torch
import torch.nn as nn
import torchvision


class HighResFPN(nn.Module):
    """ResNet + FPN，含 stride-4 层.

    返回: [P2(1/4), P3(1/8), P4(1/16), P5(1/32)]
    """

    def __init__(self, out_channels=96, depth=50):
        super().__init__()
        resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # 1/4
        self.layer2 = resnet.layer2   # 1/8
        self.layer3 = resnet.layer3   # 1/16
        self.layer4 = resnet.layer4   # 1/32

        self.lateral = nn.ModuleList([
            nn.Conv2d(256, out_channels, 1),
            nn.Conv2d(512, out_channels, 1),
            nn.Conv2d(1024, out_channels, 1),
            nn.Conv2d(2048, out_channels, 1)])
        self.smooth = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(4)])

        for m in self.lateral.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
        for m in self.smooth.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

    def forward(self, x):
        c2 = self.layer1(self.stem(x))    # 1/4
        c3 = self.layer2(c2)              # 1/8
        c4 = self.layer3(c3)              # 1/16
        c5 = self.layer4(c4)              # 1/32

        p5 = self.lateral[3](c5)
        p4 = self.smooth[2](self.lateral[2](c4)
                            + nn.functional.interpolate(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.smooth[1](self.lateral[1](c3)
                            + nn.functional.interpolate(p4, size=c3.shape[-2:], mode="nearest"))
        p2 = self.smooth[0](self.lateral[0](c2)
                            + nn.functional.interpolate(p3, size=c2.shape[-2:], mode="nearest"))
        return [p2, p3, p4, p5]


def anchors_for_shape(img_shape, strides=(4, 8, 16, 32),
                      octave=(3.0, 4.0, 5.0, 6.0), device="cpu"):
    """按层生成锚点框（单尺度方形锚，xyxy 像素坐标）.

    锚尺寸取 2^octave 像素，小层用小锚。
    """
    anchors = []
    h, w = img_shape
    for s, o in zip(strides, octave):
        gh, gw = h // s, w // s
        size = 2.0 ** o
        ys = (torch.arange(gh, device=device) + 0.5) * s
        xs = (torch.arange(gw, device=device) + 0.5) * s
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        half = size / 2.0
        a = torch.stack([xx - half, yy - half, xx + half, yy + half], dim=-1)
        anchors.append(a.reshape(-1, 4))
    return anchors
