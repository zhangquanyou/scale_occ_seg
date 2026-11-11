# -*- coding: utf-8 -*-
"""训练入口.

用法：
  python train.py --config configs/ego_hos.yaml
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.network import ScaleOccSegNet
from datasets.coco_seg_det import CocoSegDetDataset, collate


def build_from_config(cfg):
    model = ScaleOccSegNet(
        num_classes=cfg["model"]["num_classes"],
        assign_topk=cfg["model"].get("assign_topk", 4),
        assign_thr=cfg["model"].get("assign_thr", 0.5))
    train_set = CocoSegDetDataset(cfg["data"]["train_ann"],
                                  cfg["data"]["train_dir"],
                                  img_size=cfg["data"]["img_size"],
                                  occ_mode=cfg["data"].get("occ_mode", "dilate"))
    return model, train_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ego_hos.yaml", help="配置文件路径")
    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg["train"]["device"]
    model, train_set = build_from_config(cfg)
    model.to(device).train()

    loader = DataLoader(train_set,
                        batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=cfg["train"]["workers"],
                        collate_fn=collate, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(),
                            lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    total = cfg["train"]["epochs"] * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg["train"]["lr"], total_steps=total,
        pct_start=0.06)

    w = cfg["train"].get("loss_weight", {})
    os.makedirs(cfg["train"]["out_dir"], exist_ok=True)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        t0 = time.time()
        for batch in loader:
            images = batch["images"].to(device)
            loss_dict = model.forward_train(
                images,
                gt_boxes=[b.to(device) for b in batch["gt_boxes"]],
                vis_masks=[m.to(device) for m in batch["vis_masks"]],
                occ_masks=[m.to(device) for m in batch["occ_masks"]],
                hl_masks=[m.to(device) for m in batch["hl_masks"]])

            loss = (w.get("det_cls", 1.0) * loss_dict["det_cls"]
                    + w.get("det_box", 2.0) * loss_dict["det_box"]
                    + w.get("det_iou", 1.0) * loss_dict["det_iou"])
            for k in ("seg_vis", "seg_occ", "seg_bound", "seg_hl"):
                if k in loss_dict and torch.is_tensor(loss_dict[k]):
                    loss = loss + w.get(k, 1.0) * loss_dict[k]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                parts = ", ".join(f"{k}={float(v):.3f}"
                                  for k, v in loss_dict.items()
                                  if torch.is_tensor(v))
                print(f"ep{epoch} step{step}/{total} loss={float(loss):.3f} "
                      f"({parts}) lr={sched.get_last_lr()[0]:.2e}")
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "epoch": epoch},
                   os.path.join(cfg["train"]["out_dir"], "last.pth"))


if __name__ == "__main__":
    main()
