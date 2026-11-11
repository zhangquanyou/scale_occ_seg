# -*- coding: utf-8 -*-
"""COCO 格式数据集加载器.

适配三个基准：
  Ego-HOS    (github.com/owenzlz/EgoHOS)        COCO 格式
  MJU-Glass  (github.com/weduake/MJU_Glass)     COCO 格式
  VISOR-HOS  (epic-kitchens VISOR)              需先用官方工具转 COCO
  Glasses-in-the-Wild                           框 + 关键点，转 COCO 框格式

遮挡区标注：Ego-HOS/MJU-Glass 只有可见掩膜。遮挡区 GT 由
"完整实例掩膜 - 可见掩膜" 构造；完整掩膜无标注时，遮挡头
的监督改为弱监督（用 high-light/低质量区域启发式或关闭该损失）。
此处提供 make_occ_from_visible 的生成规则，与论文 5.2 节协议一致。
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

import cv2


def make_occ_from_visible(vis_mask, mode="dilate"):
    """由可见掩膜构造遮挡区监督.

    mode="dilate": 膨胀 - 可见 的环带作为遮挡区近似（无 amodal 标注时的基线）。
    mode="none":   返回全零（遮挡头关闭监督）。
    """
    if mode == "none" or vis_mask.sum() == 0:
        return np.zeros_like(vis_mask)
    k = max(5, int(np.sqrt(vis_mask.sum()) * 0.3)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(vis_mask, kernel)
    return (dil > 0).astype(np.uint8) - vis_mask


def make_hl_from_image(image, thr=0.75):
    """高光像素监督：HSV 高 V 低 S 区域（镜面高光的经典启发式）."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2] / 255.0
    s = hsv[..., 1] / 255.0
    return ((v > thr) & (s < 0.15)).astype(np.float32)


class CocoSegDetDataset(Dataset):
    """COCO 格式，同时供检测分支与分割双头使用.

    返回:
      image:  (3, H, W) float，0-1，已归一化
      gt_boxes:   (G, 4) xyxy
      gt_labels:  (G,)
      vis_masks:  (G, H, W) float —— 逐实例可见掩膜（检测分支与分割共用）
      occ_masks:  (H, W) float
      hl_masks:   (H, W) float
    """

    def __init__(self, ann_file, img_dir, img_size=768, occ_mode="dilate",
                 categories=None):
        with open(ann_file, encoding="utf-8") as f:
            coco = json.load(f)
        self.imgs = {im["id"]: im for im in coco["images"]}
        self.anns = {}
        for a in coco["annotations"]:
            self.anns.setdefault(a["image_id"], []).append(a)
        self.cat_id_map = {}
        next_id = 0
        for c in sorted(coco.get("categories", []), key=lambda x: x["id"]):
            if categories is None or c["name"] in categories:
                self.cat_id_map[c["id"]] = next_id
                next_id += 1
        self.ids = [i for i in self.imgs
                    if i in self.anns and self.anns[i]]
        self.img_dir = img_dir
        self.img_size = img_size
        self.occ_mode = occ_mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        info = self.imgs[img_id]
        path = os.path.join(self.img_dir, info["file_name"])
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(path)
        orig_h, orig_w = img.shape[:2]

        # 尺度归一化：短边缩到 img_size，同时给出回填比例（论文 3 节几何量）
        scale = self.img_size / min(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale) // 32 * 32, int(orig_h * scale) // 32 * 32
        img = cv2.resize(img, (new_w, new_h))

        G = len(self.anns[img_id])
        boxes, labels, vis = [], [], []
        for a in self.anns[img_id]:
            if a["category_id"] not in self.cat_id_map:
                continue
            x, y, w, h = a["bbox"]
            boxes.append([x * scale, y * scale,
                          (x + w) * scale, (y + h) * scale])
            labels.append(self.cat_id_map[a["category_id"]])
            m = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for seg in a.get("segmentation", []):
                pts = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(m, [pts], 1)
            if a.get("iscrowd") and "segmentation" in a:
                m = np.zeros((orig_h, orig_w), dtype=np.uint8)
                rle = a["segmentation"]
                try:
                    from pycocotools import mask as mask_util
                    m = mask_util.decode(
                        mask_util.frPyObjects(rle, orig_h, orig_w))
                    if m.ndim == 3:
                        m = m.any(axis=2).astype(np.uint8)
                except ImportError:
                    pass
            m = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            vis.append(m)

        vis_masks = np.stack(vis, 0).astype(np.float32) if vis else \
            np.zeros((0, new_h, new_w), dtype=np.float32)
        occ = make_occ_from_visible(vis_masks.max(0).astype(np.uint8),
                                    self.occ_mode)
        hl = make_hl_from_image(img)

        img_t = torch.from_numpy(
            img[..., ::-1].copy().astype(np.float32) / 255.0).permute(2, 0, 1)

        return {
            "image": img_t,
            "gt_boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "gt_labels": torch.tensor(labels, dtype=torch.long),
            "vis_masks": torch.from_numpy(vis_masks),
            "occ_masks": torch.from_numpy(occ.astype(np.float32)),
            "hl_masks": torch.from_numpy(hl),
            "image_id": img_id,
            "scale_back": scale,
        }


def collate(batch):
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "vis_masks": [b["vis_masks"] for b in batch],
        "occ_masks": [b["occ_masks"] for b in batch],
        "hl_masks": [b["hl_masks"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "scale_back": [b["scale_back"] for b in batch],
    }
