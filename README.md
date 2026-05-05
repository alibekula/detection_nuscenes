# PointBEV v0

Camera-conditioned point-based BEV detector for `nuScenes`.

## Overview

PointBEV v0 is a compact baseline for BEV-2D detection from LiDAR points and
multi-view camera images. The model keeps one prediction per retained LiDAR
point, fuses point-wise image features with lightweight LiDAR BEV context, and
predicts objectness, class, center offset, BEV size, and yaw.

This repository intentionally contains only the clean v0 training baseline. The
older polar / cached Stage 2 experiments and clustering-only experiments are
not included.

## Architecture

```text
nuScenes frame
    |
    |-- LiDAR points
    |     |-- raw point attributes: x, y, z, intensity, range, sin(theta), cos(theta)
    |     |-- BEV stats map: count, occupied, z_max, height_range
    |     `-- TinyLidarFeatureMap -> fine/context LiDAR features
    |
    |-- six camera images
    |     `-- ResNet + FPN -> C2, P2, P3, P4
    |
    |-- cheap gate
    |     `-- low-channel P2 center + local LiDAR context
    |
    `-- point head
          |-- obj
          |-- cls
          |-- dx, dy
          |-- dlog_w, dlog_l
          `-- sin_yaw, cos_yaw
```

## Qualitative Example

The example below uses a qualitative checkpoint/render with cheap-gate
threshold `-0.05`. It is a deployment-style visual sanity check, not a
controlled ablation across gate thresholds.

![PointBEV qualitative example](docs/assets/point_bev_gate_m005.png)

## Training

Edit the config block at the top of `train.py`, then run:

```bash
python train.py
```

The default paths are Colab-friendly:

```text
DATAROOT = /content/data/nuscenes
VERSION = v1.0-trainval
SAVE_DIR = /content/drive/MyDrive/point_bev
```

## Requirements

```text
torch
torchvision
nuscenes-devkit
numpy
Pillow
tqdm
pyquaternion
```

## Project Structure

```text
point_bev_nuscenes/
|-- point_bev_v0/
|   |-- __init__.py
|   |-- dataset.py
|   |-- model.py
|   `-- losses.py
|-- docs/
|   `-- assets/
|       `-- point_bev_gate_m005.png
|-- .gitignore
|-- README.md
`-- train.py
```
