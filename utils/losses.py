# -*- coding: utf-8 -*-
"""损失函数.

对应论文 4.2（尺度反比加权检测损失）与 4.3（双头分割损失、边界损失）。
"""

import torch
import torch.nn.functional as F

from .assigner import scale_inverse_weight


# ---------------------------------------------------------------- 检测损失

def giou_loss(pred_boxes, target_boxes):
    """pred/target: (N, 4) xyxy"""
    xa = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    ya = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    xb = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    yb = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (xb - xa).clamp(min=0) * (yb - ya).clamp(min=0)
    a1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * \
         (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
    a2 = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=0) * \
         (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=0)
    union = a1 + a2 - inter + 1e-8
    iou = inter / union
    # 最小包围框
    cx1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    cy1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    cx2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    cy2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-8
    return 1.0 - (iou - (c - union) / c).mean()


def quality_focal_loss(pred_logits, pred_iou, gt_labels, iou_targets,
                       num_pos, alpha=0.25, gamma=2.0):
    """QFL 融合分类与 IoU 质量（GFL 风格），小目标数少时按 num_pos 归一."""
    score = (pred_iou * (gt_labels > 0).float()).detach().clamp(max=1.0)
    scale = (pred_logits - score).abs().pow(gamma)
    loss = alpha * (gt_labels > 0).float() * scale * pred_logits.sigmoid() \
         + (1 - alpha) * scale * (1 - pred_logits.sigmoid())
    return loss.sum() / max(num_pos, 1)


def detection_loss(pred_logits, pred_boxes, pred_ious, anchors_per_level,
                   assigned_per_level, gt_boxes_per_image, img_shape,
                   gamma_scale=1.0):
    """检测分支总损失.

    pred_*_per_level: 各层检测头输出（已拉平）
    assigned_per_level: assigner 输出的正样本索引
    """
    cls_targets, iou_targets, box_preds, box_targets, weights = [], [], [], [], []
    num_pos = 0
    H, W = img_shape
    for l, assign in enumerate(assigned_per_level):
        pos = assign >= 0
        num_pos += int(pos.sum())
        cls_targets.append(torch.zeros_like(pred_logits[l]))
        iou_targets.append(torch.zeros_like(pred_ious[l]))
        cls_targets[-1][pos] = 1.0
        if pos.any():
            g = assign[pos]
            box_preds.append(pred_boxes[l][pos])
            box_targets.append(gt_boxes_per_image[g])
            iou_targets[-1][pos] = 1.0
            # 尺度反比权重
            wh = torch.stack([gt_boxes_per_image[g, 2] - gt_boxes_per_image[g, 0],
                              gt_boxes_per_image[g, 3] - gt_boxes_per_image[g, 1]], -1)
            weights.append(scale_inverse_weight(wh, gamma=gamma_scale))

    if num_pos == 0:
        zero = pred_logits[0].sum() * 0.0
        return {"cls": zero, "box": zero, "iou": zero}

    loss_cls = quality_focal_loss(
        torch.cat([p for p in pred_logits]), torch.cat([p for p in pred_ious]),
        torch.cat(cls_targets), torch.cat(iou_targets), num_pos)
    box_pred = torch.cat(box_preds)
    box_t = torch.cat(box_targets)
    w = torch.cat(weights)
    loss_box = (giou_loss(box_pred, box_t) * w).sum() / w.sum()
    loss_iou = F.binary_cross_entropy_with_logits(
        torch.cat(pred_ious), torch.cat(iou_targets), reduction="sum") / max(num_pos, 1)
    return {"cls": loss_cls, "box": loss_box, "iou": loss_iou}


# ---------------------------------------------------------------- 分割损失

def dice_loss(pred, target, eps=1e-5):
    """pred/target: (N, H, W) logits / binary"""
    pred = pred.sigmoid()
    inter = (pred * target).sum(dim=(1, 2))
    union = pred.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1 - (2 * inter + eps) / (union + eps)).mean()


def boundary_loss(pred_logits, target, kernel=3):
    """边界带损失：仅对 GT 边界附近像素施加 BCE，强化玻璃器皿的开口边界.

    pred/target: (N, H, W) logits / binary，逐通道计算。
    """
    pred = pred_logits.sigmoid()
    # 用 max-pool 近似形态学膨胀求边界带
    tgt_dil = F.max_pool2d(target[:, None], kernel, 1, kernel // 2)[:, 0]
    tgt_ero = -F.max_pool2d(-target[:, None], kernel, 1, kernel // 2)[:, 0]
    band = ((tgt_dil - tgt_ero) > 0) & (target > 0.5)
    if band.sum() == 0:
        return pred_logits.sum() * 0.0
    return F.binary_cross_entropy(
        pred[band].clamp(1e-6, 1 - 1e-6), target[band].float())


def segmentation_loss(vis_logits, occ_logits, hl_logits,
                      vis_masks, occ_masks, hl_masks):
    """双头分割 + 高光分支的总损失.

    vis/occ/hl 目标维度自动对齐：occ/hl 目标为 (H, W) 时扩成 (1, H, W)。
    返回 dict：可见掩膜、遮挡区、边界、高光四项。
    """
    if occ_masks.dim() == occ_logits.dim() - 1:
        occ_masks = occ_masks.unsqueeze(0)
    if hl_masks.dim() == hl_logits.dim() - 1:
        hl_masks = hl_masks.unsqueeze(0)
    loss = {
        "vis": F.binary_cross_entropy_with_logits(vis_logits, vis_masks)
             + dice_loss(vis_logits, vis_masks),
        "occ": F.binary_cross_entropy_with_logits(occ_logits, occ_masks)
             + dice_loss(occ_logits, occ_masks),
        "bound": boundary_loss(vis_logits, vis_masks),
        "hl": F.binary_cross_entropy_with_logits(hl_logits, hl_masks),
    }
    return loss
