# -*- coding: utf-8 -*-
"""评测：Mask AP + Boundary AP + 分桶退化分析.

对应论文 5 节实验协议：
  - Mask AP      pycocotools 标准
  - Boundary AP  Boundary IoU (Cheng et al., 2021) 阈值化后的 AP，
    精细分割与玻璃开口边界的核心指标
  - 分桶分析     按像素尺度 / 遮挡率 / 高光占比三因子分桶，
    拟合精度退化曲线，对应论文 3 节可辨识性分析
"""

import numpy as np
import torch


def boundary_iou(gt_mask, pred_mask, d_ratio=0.02, max_d=15):
    """Boundary IoU：gt/pred 为 (H, W) bool."""
    import cv2
    h, w = gt_mask.shape
    d = min(max(int(d_ratio * np.sqrt(h * w)), 1), max_d)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    gt_b = gt_mask.astype(np.uint8) - cv2.erode(gt_mask.astype(np.uint8), kernel)
    pr_b = pred_mask.astype(np.uint8) - cv2.erode(pred_mask.astype(np.uint8), kernel)
    inter = np.logical_and(gt_b, pr_b).sum()
    union = np.logical_or(gt_b, pr_b).sum()
    return float(inter) / max(union, 1)


def mask_ap_from_instances(preds, gts, iou_type="mask", boundary=False):
    """把逐图实例转 COCO 结果格式后用 pycocotools 计算 AP.

    preds: list of {"boxes","labels","scores","vis_prob",...}
    gts:   COCO GT json 路径
    boundary=True 时改算 Boundary AP（用 Boundary IoU 替换 IoU）。
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO(gts)
    results = []
    for img_id, pred in zip(preds["image_ids"], preds["instances"]):
        for i in range(pred["boxes"].shape[0]):
            x1, y1, x2, y2 = pred["boxes"][i].tolist()
            results.append({
                "image_id": int(img_id),
                "category_id": coco.getCatIds()[int(pred["labels"][i])] if False
                else int(pred["labels"][i]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(pred["scores"][i]),
                "segmentation": _rle_from_prob(pred["vis_prob"][i],
                                               int(pred["labels"][i])),
            })
    if not results:
        return 0.0
    e = COCOeval(coco, iouType=iou_type)
    e.results = results
    e.evaluate(); e.accumulate(); e.summarize()
    return float(e.stats[0])


def _rle_from_prob(prob, cat_id):
    import pycocotools.mask as mask_util
    m = (prob > 0.5).cpu().numpy().astype(np.uint8)
    return mask_util.encode(np.asfortranarray(m))


def bucketed_analysis(instances, per_inst_metrics, factor_values,
                      bins, names=("尺度", "遮挡率", "高光占比")):
    """三因子分桶统计（论文 6.4 分桶退化曲线的数据源）.

    instances: 实例列表
    per_inst_metrics: list，每实例的 IoU 或 Boundary IoU
    factor_values: dict {factor_name: list 每实例的因子取值}
    bins: dict {factor_name: 分桶边界}
    返回: dict {factor: [(bin 中点, 均值 IoU, 数量)]}
    """
    out = {}
    for factor in names:
        vals = np.array(factor_values[factor])
        met = np.array(per_inst_metrics)
        edges = bins[factor]
        pts = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (vals >= lo) & (vals < hi)
            if sel.sum() > 0:
                pts.append((float((lo + hi) / 2), float(met[sel].mean()),
                            int(sel.sum())))
            else:
                pts.append((float((lo + hi) / 2), None, 0))
        out[factor] = pts
    return out


def fit_identifiability_boundary(buckets, iou_thr=0.5):
    """由分桶结果拟合逐类最小可辨识边界（论文 3 节 s_min / rho_max / h_max）.

    规则：从最小的 bin 开始向右找，首个桶均值 IoU < iou_thr 的
    前一个 bin 上界即该因子的边界。
    """
    boundary = {}
    for factor, pts in buckets.items():
        found = None
        for mid, miou, n in pts:
            if miou is None or n < 3:      # 样本太少的桶不参与
                continue
            if miou < iou_thr:
                found = mid
                break
        boundary[factor] = found           # None 表示该因子全区间可辨
    return boundary
