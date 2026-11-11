# -*- coding: utf-8 -*-
"""完整网络：尺度感知检测分支 + 遮挡感知双头分割 + 高光分支.

对应论文第 4 节整体架构：
  1. 检测分支（4.2）：共享 FPN 特征，尺度解耦回归头，
     尺度反比损失加权，小目标为主。
  2. 分割双头（4.3）：可见掩膜头 + 遮挡区头，
     遮挡区面积占比作为遮挡率证据可靠性输出，不做 amodal 补全。
  3. 高光分支（4.4）：检测镜面高光像素并标记不可信，
     下游几何量计算只用可信像素。

输出统一打包成 dict，训练与推理共用前向，推理时额外输出
遮挡率、高光占比与逐类可辨识边界标志。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import HighResFPN, anchors_for_shape
from utils.assigner import GaussianRFAssigner
from utils.losses import detection_loss, segmentation_loss


class ScaleDecoupledHead(nn.Module):
    """尺度解耦检测头.

    P2/P3 与 P4/P5 用独立卷积组，避免小目标响应被大目标梯度淹没。
    """

    def __init__(self, ch=96, num_classes=8):
        super().__init__()
        self.small_tower = nn.Sequential(*[
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(16, ch), nn.ReLU(inplace=True)
        ] * 3)
        self.large_tower = nn.Sequential(*[
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(16, ch), nn.ReLU(inplace=True)
        ] * 3)
        self.cls = nn.Conv2d(ch, num_classes, 3, padding=1)
        self.reg = nn.Conv2d(ch, 4, 3, padding=1)
        self.iou = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, feats):
        """feats: [P2, P3, P4, P5]
        返回三个列表，每层输出拉平为 (A_l, C)。
        """
        logits, boxes, ious = [], [], []
        for l, f in enumerate(feats):
            tower = self.small_tower(f) if l < 2 else self.large_tower(f)
            logits.append(tower.new_zeros(0))  # 占位，见下方
            logits[-1] = self.cls(tower)
            ious.append(self.iou(tower))
            boxes.append(self.reg(tower))
        return logits, boxes, ious


class DualHeadSegmentor(nn.Module):
    """可见掩膜头 + 遮挡区头 + 高光头.

    输入 P2 与 P3 融合的高分辨率特征；三个头独立，
    遮挡头与可见头之间无权重共享（可见掩膜不应决定遮挡区形状）。
    """

    def __init__(self, ch=96, num_classes=8):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1), nn.GroupNorm(16, ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(16, ch), nn.ReLU(inplace=True))
        self.head = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(ch, ch, 3, padding=1), nn.GroupNorm(16, ch),
                nn.ReLU(inplace=True)] * 3)
            for _ in range(3)])
        self.out_vis = nn.Conv2d(ch, num_classes, 1)
        self.out_occ = nn.Conv2d(ch, 1, 1)   # 遮挡区与类别无关，单通道
        self.out_hl = nn.Conv2d(ch, 1, 1)

    def forward(self, p2, p3):
        size = p2.shape[-2:]
        f = self.fuse(torch.cat(
            [p2, F.interpolate(p3, size=size, mode="bilinear", align_corners=False)], 1))
        vis = self.out_vis(self.head[0](f))
        occ = self.out_occ(self.head[1](f))
        hl = self.out_hl(self.head[2](f))
        return vis, occ, hl


class ScaleOccSegNet(nn.Module):
    """完整模型.

    num_classes 同时覆盖检测类别与分割类别。
    """

    def __init__(self, num_classes=8, strides=(4, 8, 16, 32),
                 assign_topk=4, assign_thr=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.backbone = HighResFPN()
        self.det_head = ScaleDecoupledHead(self.backbone.smooth[0].out_channels,
                                           num_classes)
        self.seg = DualHeadSegmentor(self.backbone.smooth[0].out_channels,
                                     num_classes)
        self.assigner = GaussianRFAssigner(strides, topk=assign_topk,
                                           thr=assign_thr)
        # 回归解码尺度（分层可学习 scale，GFL 风格）
        self.scales = nn.Parameter(torch.ones(len(strides)))

    # ------------------------------------------------------ 训练前向
    def forward_train(self, images, gt_boxes=None, gt_labels=None,
                      vis_masks=None, occ_masks=None, hl_masks=None):
        """images: (B, 3, H, W)，归一化后的张量
        其余均为 per-image 列表或 None。
        """
        feats = self.backbone(images)
        logits, boxes_raw, ious = self.det_head(feats)

        loss = {}
        B = images.shape[0]
        # 各层输出拉平为 (B, A_l, C)，与分配器/损失的对齐要求一致
        logits = [l.permute(0, 2, 3, 1).reshape(B, -1, l.shape[1]) for l in logits]
        ious = [i.permute(0, 2, 3, 1).reshape(B, -1, 1) for i in ious]
        img_hw = (images.shape[2], images.shape[3])
        anchors_per_level = [
            anchors_for_shape(img_hw, (s,), (3.0 + 2 * l,), device=images.device)[0]
            for l, s in enumerate(self.strides)]
        anchors = torch.cat(anchors_per_level, dim=0)
        # 逐图分配并累计检测损失
        cls_losses, box_losses, iou_losses = [], [], []
        for b in range(B):
            decoded = self._decode(boxes_raw, anchors, img_hw)
            assigned = self.assigner.assign(
                anchors_per_level, gt_boxes[b] if gt_boxes else torch.empty(0, 4),
                img_hw)
            d = detection_loss([x[b] for x in logits],
                               [x[b] for x in decoded],
                               [x[b] for x in ious],
                               anchors_per_level, assigned,
                               gt_boxes[b] if gt_boxes else torch.empty(0, 4),
                               img_hw)
            cls_losses.append(d["cls"]); box_losses.append(d["box"])
            iou_losses.append(d["iou"])
        loss["det_cls"] = sum(cls_losses) / B
        loss["det_box"] = sum(box_losses) / B
        loss["det_iou"] = sum(iou_losses) / B

        # 分割三头
        vis_logits, occ_logits, hl_logits = self.seg(feats[0], feats[1])
        vis_up = F.interpolate(vis_logits, size=img_hw, mode="bilinear",
                               align_corners=False)
        occ_up = F.interpolate(occ_logits, size=img_hw, mode="bilinear",
                               align_corners=False)
        hl_up = F.interpolate(hl_logits, size=img_hw, mode="bilinear",
                              align_corners=False)
        for b in range(B):
            if vis_masks is None:
                continue
            # 逐实例掩膜按类别合成 (C, H, W) 目标（同类实例取并集）
            vis_t = torch.zeros(self.num_classes, *img_hw, device=vis_up.device)
            for i in range(vis_masks[b].shape[0]):
                c = int(gt_labels[b][i])
                vis_t[c] = torch.maximum(vis_t[c], vis_masks[b][i])
            s = segmentation_loss(vis_up[b], occ_up[b], hl_up[b],
                                  vis_t, occ_masks[b], hl_masks[b])
            for k, v in s.items():
                loss[f"seg_{k}"] = loss.get(f"seg_{k}", 0) + v / B
        return loss

    # ------------------------------------------------------ 推理前向
    @torch.no_grad()
    def forward_infer(self, images, ident_boundary=None,
                      score_thr=0.3, nms_iou=0.6, max_det=300):
        """推理输出：检测结果 + 可靠性量.

        ident_boundary: dict {class_id: (s_min, rho_max, h_max)}，
        论文 3 节可辨识性分析的逐类边界；超边界的实例被标记
        ident_ok=False，由下游转人工复核。
        """
        self.eval()
        feats = self.backbone(images)
        logits, boxes_raw, ious = self.det_head(feats)
        img_hw = (images.shape[2], images.shape[3])
        anchors_per_level = [
            anchors_for_shape(img_hw, (s,), (3.0 + 2 * l,), device=images.device)[0]
            for l, s in enumerate(self.strides)]
        anchors = torch.cat(anchors_per_level, dim=0)
        boxes = self._decode(boxes_raw, anchors, img_hw)

        scores = torch.cat([l.sigmoid().flatten(2).permute(0, 2, 1) for l in logits], 1)
        quality = torch.cat([i.sigmoid().flatten(2).permute(0, 2, 1) for i in ious], 1)
        all_boxes = torch.cat(boxes, dim=1)   # 各层 (B, A_l, 4) 直接拼接

        # 分割与可靠性
        vis_logits, occ_logits, hl_logits = self.seg(feats[0], feats[1])
        vis_prob = F.interpolate(vis_logits.sigmoid(), size=img_hw,
                                 mode="bilinear", align_corners=False)
        occ_prob = F.interpolate(occ_logits.sigmoid(), size=img_hw,
                                 mode="bilinear", align_corners=False)
        hl_prob = F.interpolate(hl_logits.sigmoid(), size=img_hw,
                                mode="bilinear", align_corners=False)

        results = []
        for b in range(images.shape[0]):
            s = scores[b].max(dim=-1)                 # (A,) 分数与类别
            cls_ids, cls_scores = s.indices, s.values
            joint = cls_scores * quality[b].squeeze(-1)
            keep = joint > score_thr
            boxes_b = all_boxes[b][keep]
            ids_b, scores_b = cls_ids[keep], joint[keep]

            # 简易 NMS
            if boxes_b.numel():
                order = scores_b.argsort(descending=True)[:max_det]
                boxes_b, ids_b, scores_b = boxes_b[order], ids_b[order], scores_b[order]
                keep_n = self._nms(boxes_b, scores_b, nms_iou)
                boxes_b, ids_b, scores_b = boxes_b[keep_n], ids_b[keep_n], scores_b[keep_n]

            # 每个实例的可靠性量：遮挡率、高光占比、可辨识标志
            occlusion_ratio, highlight_ratio, ident_ok = [], [], []
            for i in range(boxes_b.shape[0]):
                x1, y1, x2, y2 = boxes_b[i].long()
                m_vis = vis_prob[b, ids_b[i], y1:y2, x1:x2] > 0.5
                m_occ = occ_prob[b, 0, y1:y2, x1:x2]
                m_hl = hl_prob[b, 0, y1:y2, x1:x2] > 0.5
                area = max(int((y2 - y1) * (x2 - x1)), 1)
                occ_r = float((m_occ > 0.5).sum()) / area
                hl_r = float(m_hl.sum()) / area
                occlusion_ratio.append(occ_r)
                highlight_ratio.append(hl_r)
                ok = True
                if ident_boundary is not None and int(ids_b[i]) in ident_boundary:
                    s_min, rho_max, h_max = ident_boundary[int(ids_b[i])]
                    scale = float(torch.sqrt((x2 - x1) * (y2 - y1)))
                    ok = scale >= s_min and occ_r <= rho_max and hl_r <= h_max
                ident_ok.append(ok)

            results.append({
                "boxes": boxes_b, "labels": ids_b, "scores": scores_b,
                "occlusion_ratio": torch.tensor(occlusion_ratio, device=images.device),
                "highlight_ratio": torch.tensor(highlight_ratio, device=images.device),
                "ident_ok": torch.tensor(ident_ok, device=images.device)
                            if ident_ok else torch.empty(0, device=images.device),
                "vis_prob": vis_prob[b], "occ_prob": occ_prob[b],
                "hl_prob": hl_prob[b],
            })
        return results

    # ------------------------------------------------------ 内部工具
    def _level_sizes(self, img_hw, device):
        h, w = img_hw
        return [(h // s) * (w // s) for s in self.strides]

    def _decode(self, boxes_raw, anchors, img_hw):
        """分层回归解码：deltas -> xyxy."""
        decoded = []
        sizes = self._level_sizes(img_hw, boxes_raw[0].device)
        start = 0
        B = boxes_raw[0].shape[0]
        for l, raw in enumerate(boxes_raw):
            a = anchors[start:start + sizes[l]]
            start += sizes[l]
            stride = self.strides[min(l, len(self.strides) - 1)]
            ac_x = (a[:, 0] + a[:, 2]) / 2
            ac_y = (a[:, 1] + a[:, 3]) / 2
            B = raw.shape[0]
            flat = raw.reshape(B, 4, -1).permute(0, 2, 1)   # (B, A_l, 4)
            dx = flat[..., 0:1].exp() * stride * self.scales[l]
            dy = flat[..., 1:2].exp() * stride * self.scales[l]
            dw = flat[..., 2:3].exp() * stride * self.scales[l]
            dh = flat[..., 3:4].exp() * stride * self.scales[l]
            x1 = ac_x[None, :, None] - dx / 2
            y1 = ac_y[None, :, None] - dy / 2
            x2 = ac_x[None, :, None] + dx / 2
            y2 = ac_y[None, :, None] + dy / 2
            decoded.append(torch.cat([x1, y1, x2, y2], dim=-1).clamp(0))
        return decoded

    @staticmethod
    def _nms(boxes, scores, iou_thr):
        from torchvision.ops import nms
        return nms(boxes, scores, iou_thr)
