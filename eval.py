# -*- coding: utf-8 -*-
"""评测入口：Mask AP + Boundary AP + 三因子分桶 + 可辨识边界拟合.

用法：
  python eval.py --config configs/ego_hos.yaml --ckpt last.pth
"""

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network import ScaleOccSegNet
from datasets.coco_seg_det import CocoSegDetDataset, collate
from utils.evaluation import (boundary_iou, bucketed_analysis,
                              fit_identifiability_boundary)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = cfg["train"]["device"]

    model = ScaleOccSegNet(num_classes=cfg["model"]["num_classes"]).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()

    test_set = CocoSegDetDataset(cfg["data"]["test_ann"],
                                 cfg["data"]["test_dir"],
                                 img_size=cfg["data"]["img_size"])
    from torch.utils.data import DataLoader
    loader = DataLoader(test_set, batch_size=1, shuffle=False,
                        num_workers=2, collate_fn=collate)

    image_ids, instances = [], []
    per_inst_iou, per_inst_biou = [], []
    factors = {"尺度": [], "遮挡率": [], "高光占比": []}

    for batch in loader:
        r = model.forward_infer(batch["images"].to(device),
                                score_thr=0.3)[0]
        image_ids.append(batch["image_id"][0])
        instances.append({"boxes": r["boxes"].cpu(),
                          "labels": r["labels"].cpu(),
                          "scores": r["scores"].cpu(),
                          "vis_prob": r["vis_prob"].cpu()})
        if batch["gt_boxes"][0].numel() == 0 or r["boxes"].numel() == 0:
            continue
        # 逐实例 IoU / Boundary IoU（与 GT 的可见掩膜对齐，贪心匹配）
        gt = batch["vis_masks"][0].to(device)
        boxes_gt = batch["gt_boxes"][0].to(device)
        scale = batch["scale_back"][0]
        from torchvision.ops import box_iou
        iou_mat = box_iou(r["boxes"], boxes_gt)
        best_gt = iou_mat.argmax(dim=1)
        for i in range(r["boxes"].shape[0]):
            if iou_mat[i, best_gt[i]] < 0.3:
                continue
            x1, y1, x2, y2 = r["boxes"][i].long()
            pred_m = r["vis_prob"][int(r["labels"][i]), y1:y2, x1:x2] > 0.5
            bx1, by1, bx2, by2 = boxes_gt[best_gt[i]].long()
            gt_m = gt[int(r["labels"][i]), by1:by2, bx1:bx2] > 0.5
            h = max(y2 - y1, 1); w = max(x2 - x1, 1)
            inter = (pred_m & gt_m).sum().float()
            union = (pred_m | gt_m).sum().float()
            per_inst_iou.append(float(inter / max(union, 1)))
            gtm = gt_m.cpu().numpy(); prm = pred_m.cpu().numpy()
            if gtm.shape != prm.shape:
                gtm = gtm[:prm.shape[0], :prm.shape[1]]
            per_inst_biou.append(boundary_iou(gtm, prm))
            # 三因子：像素尺度（换算回原图）、遮挡率、高光占比
            factors["尺度"].append(
                float(torch.sqrt((x2 - x1) * (y2 - y1)).cpu() / scale))
            factors["遮挡率"].append(float(r["occlusion_ratio"][i]))
            factors["高光占比"].append(float(r["highlight_ratio"][i]))

    # 汇总
    result = {
        "mask_ap": None,   # 标准 COCO AP 需完整预测 JSON，见 README
        "boundary_iou_mean": float(torch.tensor(per_inst_biou).mean())
        if per_inst_biou else 0.0,
        "iou_mean": float(torch.tensor(per_inst_iou).mean())
        if per_inst_iou else 0.0,
    }

    bins = {"尺度": [0, 16, 32, 48, 64, 96, 128, 256, 1e9],
            "遮挡率": [0, 0.1, 0.2, 0.4, 0.6, 1.01],
            "高光占比": [0, 0.05, 0.1, 0.2, 0.4, 1.01]}
    buckets = bucketed_analysis(instances, per_inst_iou, factors, bins)
    boundary = fit_identifiability_boundary(buckets, iou_thr=0.5)
    result["buckets"] = buckets
    result["identifiability_boundary"] = boundary

    out = cfg["eval"].get("out_json", "eval_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"结果已写入 {out}。分桶数据可直接绘制论文图 3/图 4 退化曲线，"
          f"边界值回填正文 3 节的 s_min / rho_max / h_max。")


if __name__ == "__main__":
    main()
