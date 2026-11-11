# scale_occ_seg

论文《面向实验室违规识别的小目标检测与精细分割方法》的模型代码。
三分支结构与论文第 4 节一一对应：

| 论文章节 | 代码位置 | 内容 |
|---|---|---|
| 4.2 尺度感知检测 | `utils/assigner.py` `utils/losses.py::detection_loss` | 高斯感受野距离分配、尺度反比损失加权 |
| 4.2 高分辨率层 | `models/backbone.py::HighResFPN` | 保留 stride-4 的 P2 层 |
| 4.3 遮挡感知双头分割 | `models/network.py::DualHeadSegmentor` | 可见掩膜头 + 遮挡区头，遮挡率作可靠性输出 |
| 4.4 高光分支 | `datasets/coco_seg_det.py::make_hl_from_image` `network.py` | 高光像素检测、不可信像素标记 |
| 3 可辨识性分析 | `utils/evaluation.py::fit_identifiability_boundary` | 三因子分桶、s_min / rho_max / h_max 拟合 |

## 数据准备（全部公开可得）

| 基准 | 地址 | 格式 |
|---|---|---|
| Ego-HOS | github.com/owenzlz/EgoHOS | COCO，直接下载 |
| VISOR | epic-kitchens.github.io/VISOR，HOS 基线 github.com/epic-kitchens/VISOR-HOS | 需官方工具转 COCO |
| MJU-Glass | github.com/weduake/MJU_Glass | COCO |
| Glasses-in-the-Wild | zenodo.org/records/17288503 | 框+关键点，转 COCO |
| KINS | github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset | 遮挡区真值对照 |
| ENIGMA-51 | iplab.dmi.unict.it/ENIGMA-51（申请） | COCO |

按 `configs/*.yaml` 中的路径放置数据即可。

## 训练与评测

```bash
pip install -r requirements.txt
python train.py --config configs/ego_hos.yaml
python eval.py  --config configs/ego_hos.yaml --ckpt outputs/egohos/last.pth
```

评测输出三样东西：

1. 逐实例 IoU 与 Boundary IoU 均值；
2. 按像素尺度 / 遮挡率 / 高光占比的分桶退化数据（直接绘制论文图 3、图 4）；
3. 拟合的逐类可辨识边界（回填正文 3 节的 s_min / rho_max / h_max）。

标准 Mask AP 建议把预测导出 COCO JSON 后跑 pycocotools 的完整 COCOeval，
`utils/evaluation.py::mask_ap_from_instances` 已留接口。

## 实现说明与已知限制

- **遮挡区监督是弱监督**。Ego-HOS / MJU-Glass 只有可见掩膜，遮挡头用
  膨胀环带近似（`occ_mode: dilate`），与 KINS 的真值对照实验才能给出
  遮挡区的硬数字。这对应论文 5 节的对照设计，不是实现偷懒。
- **高光监督是启发式**。HSV 高 V 低 S 规则生成弱标签；达姆施塔特合成
  玻璃数据（BlenderProc，7000 张）自带真实高光标注，可做预训练。
- **分配器 topk 保底**保证极小目标每层至少一个正样本，`k=0` 无条件
  分配，这是小目标不漏监督的关键，改动前先想清楚。
- RefineHOS 无公开代码，对照表 2 中其行引用论文数字，或按其模块自行复现。
- 推理输出的 `occlusion_ratio / highlight_ratio / ident_ok` 即论文的
  证据可靠性三元组，下游判定层直接消费，不做 amodal 补全。

## 首个验证动作（对应实验计划）

```bash
# 1. 下载 Ego-HOS
# 2. 只训 2 个 epoch 验证管线
python train.py --config configs/ego_hos.yaml   # epochs 临时改 2
# 3. 对照 RefineHOS 论文中 VISOR/Ego-HOS 的量级，确认边界 AP 可复现
```

管线跑通后再上完整训练与全部基准。
