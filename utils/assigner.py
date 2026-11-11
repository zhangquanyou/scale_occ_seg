# -*- coding: utf-8 -*-
"""高斯感受野距离样本分配器（Gaussian Receptive Field Distance Assigner）.

对应论文 4.2 节：小目标场景下 IoU 对框的微小偏移极其敏感，
改用先验框感受野与真实框中心的高斯距离做正样本分配（RFLA 思路），
并以距离阈值代替 IoU 阈值，保证极小目标至少分到一个正样本。
"""

import torch
import torch.nn.functional as F


def box_center(boxes):
    """boxes: (N, 4) xyxy -> (N, 2)"""
    return torch.stack([(boxes[..., 0] + boxes[..., 2]) * 0.5,
                        (boxes[..., 1] + boxes[..., 3]) * 0.5], dim=-1)


def box_wh(boxes):
    """boxes: (N, 4) xyxy -> (N, 2) w,h (下限 1e-4，防止除零)"""
    wh = torch.stack([boxes[..., 2] - boxes[..., 0],
                      boxes[..., 3] - boxes[..., 1]], dim=-1)
    return wh.clamp(min=1e-4)


def gaussian_rf_distance(anchor_c, anchor_wh, gt_c, gt_wh, stride, norm=4.0):
    """计算锚点感受野中心到 GT 框中心的归一化 KLD 距离近似.

    anchor_c: (A, 2) 锚点中心（像素坐标）
    anchor_wh: (A, 2) 锚点尺寸
    gt_c: (G, 2) GT 中心
    gt_wh: (G, 2) GT 尺寸
    stride: 当前特征层步长
    返回: (G, A) 距离矩阵，越小越匹配
    """
    # 感受野等效标准差（RFLA 的近似：sqrt(w*h)/stride 尺度归一）
    rf_std_a = torch.sqrt(anchor_wh[..., 0] * anchor_wh[..., 1]) / stride
    rf_std_g = torch.sqrt(gt_wh[..., 0] * gt_wh[..., 1]) / stride

    d = gt_c[:, None, :] - anchor_c[None, :, :]          # (G, A, 2)
    d2 = d[..., 0] ** 2 + d[..., 1] ** 2                  # (G, A)

    # 双向对称 KLD 近似，再取指数转相似度
    log_term = torch.log((rf_std_g[:, None] / (rf_std_a[None, :] + 1e-8)).clamp(min=1e-8))
    kld = 0.5 * (d2 / (4.0 * rf_std_a[None, :] ** 2 + 1e-8)
                 + d2 / (4.0 * rf_std_g[:, None] ** 2 + 1e-8)
                 + log_term)
    return kld.clamp(min=0)


class GaussianRFAssigner:
    """逐层高斯感受野距离分配.

    topk: 每层每 GT 强制取距离最小的 topk 个锚为正，保证小目标有监督。
    thr:  相似度阈值（exp(-kld) >= thr 视为候选）。
    """

    def __init__(self, strides=(4, 8, 16, 32), topk=4, thr=0.5):
        self.strides = strides
        self.topk = topk
        self.thr = thr

    @torch.no_grad()
    def assign(self, anchors_per_level, gt_boxes, img_shape):
        """返回每层正样本索引列表.

        anchors_per_level: list of (A_l, 4) xyxy 像素坐标
        gt_boxes: (G, 4) xyxy，允许 G=0
        """
        num_levels = len(anchors_per_level)
        assigned = [torch.full((a.shape[0],), -1, dtype=torch.long,
                               device=gt_boxes.device) for a in anchors_per_level]

        if gt_boxes.numel() == 0:
            # 无 GT：全部置负（背景）
            return assigned

        gt_c, gt_wh = box_center(gt_boxes), box_wh(gt_boxes)
        # GT 尺度决定主负责层：s <= 2*stride 视为该层管辖范围
        for l, anchors in enumerate(anchors_per_level):
            stride = self.strides[min(l, len(self.strides) - 1)]
            a_c, a_wh = box_center(anchors), box_wh(anchors)
            dist = gaussian_rf_distance(a_c, a_wh, gt_c, gt_wh, stride)
            sim = torch.exp(-dist)                        # (G, A)

            valid = sim >= self.thr
            # topk 强制保底
            topk = min(self.topk, anchors.shape[0])
            topk_sim, topk_idx = sim.topk(topk, dim=1)    # (G, topk)

            for g in range(gt_boxes.shape[0]):
                for k in range(topk):
                    a = topk_idx[g, k]
                    if valid[g, a] or k == 0:             # k=0 无条件保底
                        if assigned[l][a] < 0:
                            assigned[l][a] = g
        return assigned


def scale_inverse_weight(gt_wh, s_min=8.0, s_max=64.0, gamma=1.0):
    """尺度反比损失权重：目标越小权重越大，归一到均值 1.

    gt_wh: (N, 2) 或 (N,) 面积的平方根
    """
    if gt_wh.dim() == 2:
        s = torch.sqrt(gt_wh[..., 0] * gt_wh[..., 1])
    else:
        s = gt_wh
    w = (s_max / s.clamp(min=s_min)).pow(gamma)
    w = w / w.mean().clamp(min=1e-6)
    return w.clamp(max=8.0)
