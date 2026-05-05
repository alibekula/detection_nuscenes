from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PointLossConfig:
    gate_alpha: float = 0.25
    gate_gamma: float = 2.0
    obj_pos_weight: float = 20.0
    class_weights: Sequence[float] | None = None
    lambda_gate: float = 1.0
    lambda_obj: float = 1.0
    lambda_cls: float = 1.0
    lambda_xy: float = 1.0
    lambda_size: float = 1.0
    lambda_yaw: float = 0.5


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0

    logits = logits.float()
    targets = targets.to(dtype=logits.dtype)
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    if alpha >= 0.0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        ce = ce * alpha_t
    return (ce * modulating).mean()


def weighted_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: float,
) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.sum() * 0.0

    logits = logits.float()
    pw = torch.tensor([pos_weight], device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(
        logits,
        targets.to(dtype=logits.dtype),
        pos_weight=pw,
    )


def assign_points_to_boxes(
    points_xyz: torch.Tensor,
    gt_boxes: torch.Tensor,
) -> torch.Tensor:
    """Assign each point to one GT 3D box or -1 if background.

    gt_boxes layout: [cx, cy, cz, w, l, h, yaw]
    points_xyz layout: [x, y, z] in lidar frame
    """
    num_points = int(points_xyz.shape[0])
    if num_points == 0 or gt_boxes.numel() == 0:
        return torch.full((num_points,), -1, device=points_xyz.device, dtype=torch.long)

    px = points_xyz[:, 0:1]
    py = points_xyz[:, 1:2]
    pz = points_xyz[:, 2:3]

    cx = gt_boxes[:, 0].unsqueeze(0)
    cy = gt_boxes[:, 1].unsqueeze(0)
    cz = gt_boxes[:, 2].unsqueeze(0)
    w = gt_boxes[:, 3].unsqueeze(0)
    l = gt_boxes[:, 4].unsqueeze(0)
    h = gt_boxes[:, 5].unsqueeze(0)
    yaw = gt_boxes[:, 6].unsqueeze(0)

    dx = px - cx
    dy = py - cy
    dz = pz - cz

    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    local_long = cos_yaw * dx + sin_yaw * dy
    local_width = -sin_yaw * dx + cos_yaw * dy

    inside = (
        (local_long.abs() <= (l * 0.5))
        & (local_width.abs() <= (w * 0.5))
        & (dz.abs() <= (h * 0.5))
    )

    dist2 = dx.square() + dy.square()
    dist2 = dist2.masked_fill(~inside, float("inf"))

    best_dist2, owner_idx = dist2.min(dim=1)
    owner_idx = owner_idx.to(dtype=torch.long)
    owner_idx[~torch.isfinite(best_dist2)] = -1
    return owner_idx


def estimate_class_mean_sizes_from_samples(
    gt_boxes_list: Sequence[torch.Tensor],
    gt_labels_list: Sequence[torch.Tensor],
    *,
    num_classes: int,
) -> torch.Tensor:
    sums = torch.zeros((num_classes, 2), dtype=torch.float64)
    counts = torch.zeros((num_classes,), dtype=torch.float64)

    for boxes, labels in zip(gt_boxes_list, gt_labels_list):
        if boxes.numel() == 0:
            continue
        for cls_idx in range(num_classes):
            mask = labels == cls_idx
            if not mask.any():
                continue
            sums[cls_idx] += boxes[mask][:, 3:5].double().sum(dim=0)
            counts[cls_idx] += float(mask.sum().item())

    means = torch.ones((num_classes, 2), dtype=torch.float32)
    valid = counts > 0
    if valid.any():
        means[valid] = (sums[valid] / counts[valid].unsqueeze(1)).float()
    return means


def estimate_class_weights_from_samples(
    gt_labels_list: Sequence[torch.Tensor],
    *,
    num_classes: int,
) -> torch.Tensor:
    counts = torch.zeros((num_classes,), dtype=torch.float64)
    for labels in gt_labels_list:
        if labels.numel() == 0:
            continue
        counts += torch.bincount(labels.cpu(), minlength=num_classes).double()

    counts = counts.clamp_min(1.0)
    weights = counts.rsqrt()
    weights = weights / weights.mean()
    return weights.float()


def build_frame_targets(
    *,
    points: torch.Tensor,
    points_xyz: torch.Tensor,
    proj_mask: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    kept_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    owner_all = assign_points_to_boxes(points_xyz=points_xyz, gt_boxes=gt_boxes)
    gate_target = owner_all.ge(0).to(dtype=points.dtype)

    kept_indices = kept_indices.to(device=points.device, dtype=torch.long)
    owner_kept = owner_all[kept_indices] if kept_indices.numel() > 0 else owner_all.new_empty((0,))
    obj_target = owner_kept.ge(0).to(dtype=points.dtype)
    pos_mask = owner_kept.ge(0)

    targets: dict[str, torch.Tensor] = {
        "gate_target": gate_target,
        "obj_target": obj_target,
        "pos_mask": pos_mask,
    }

    if not pos_mask.any():
        targets["cls_target"] = gt_labels.new_empty((0,))
        targets["xy_target"] = points.new_empty((0, 2))
        targets["log_size_target"] = points.new_empty((0, 2))
        targets["yaw_target"] = points.new_empty((0, 2))
        return targets

    owner_pos = owner_kept[pos_mask]
    gt_pos = gt_boxes[owner_pos]
    gt_labels_pos = gt_labels[owner_pos]
    kept_points_pos = points[kept_indices[pos_mask]]

    xy_target = gt_pos[:, 0:2] - kept_points_pos[:, 0:2]
    log_size_target = gt_pos[:, 3:5].clamp_min(1e-4).log()
    yaw = gt_pos[:, 6]
    yaw_target = torch.stack([torch.sin(yaw), torch.cos(yaw)], dim=1)

    targets["cls_target"] = gt_labels_pos
    targets["xy_target"] = xy_target
    targets["log_size_target"] = log_size_target
    targets["yaw_target"] = yaw_target
    return targets


def build_batch_targets(
    batch: dict[str, Sequence[torch.Tensor]],
    outputs: dict[str, Sequence[torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    frame_targets: list[dict[str, torch.Tensor]] = []

    for frame_idx in range(len(batch["points"])):
        points = batch["points"][frame_idx]
        points_xyz = batch["points_xyz"][frame_idx]
        proj_mask = batch["proj_mask"][frame_idx]
        gt_boxes = batch["gt_boxes"][frame_idx]
        gt_labels = batch["gt_labels"][frame_idx]
        kept_indices = outputs["kept_indices"][frame_idx]

        device = outputs["obj_pre_logits"][frame_idx].device
        frame_targets.append(
            build_frame_targets(
                points=points.to(device=device, dtype=torch.float32),
                points_xyz=points_xyz.to(device=device, dtype=torch.float32),
                proj_mask=proj_mask.to(device=device, dtype=torch.bool),
                gt_boxes=gt_boxes.to(device=device, dtype=torch.float32),
                gt_labels=gt_labels.to(device=device, dtype=torch.long),
                kept_indices=kept_indices.to(device=device, dtype=torch.long),
            )
        )

    return frame_targets


def compute_point_bev_losses(
    outputs: dict[str, Sequence[torch.Tensor]],
    batch: dict[str, Sequence[torch.Tensor]],
    *,
    cfg: PointLossConfig = PointLossConfig(),
) -> tuple[torch.Tensor, dict[str, float]]:
    frame_targets = build_batch_targets(batch=batch, outputs=outputs)

    device = outputs["obj_pre_logits"][0].device if outputs["obj_pre_logits"] else torch.device("cpu")
    gate_terms: list[torch.Tensor] = []
    obj_terms: list[torch.Tensor] = []
    cls_terms: list[torch.Tensor] = []
    xy_terms: list[torch.Tensor] = []
    size_terms: list[torch.Tensor] = []
    yaw_terms: list[torch.Tensor] = []

    class_weights = None
    if cfg.class_weights is not None:
        class_weights = torch.tensor(cfg.class_weights, device=device, dtype=torch.float32)

    zero = torch.zeros((), device=device, dtype=torch.float32)

    for frame_idx, targets in enumerate(frame_targets):
        gate_terms.append(
            sigmoid_focal_loss(
                outputs["obj_pre_logits"][frame_idx],
                targets["gate_target"],
                alpha=cfg.gate_alpha,
                gamma=cfg.gate_gamma,
            )
        )

        obj_terms.append(
            weighted_bce_loss(
                outputs["obj_logits"][frame_idx],
                targets["obj_target"],
                pos_weight=cfg.obj_pos_weight,
            )
        )

        pos_mask = targets["pos_mask"]
        if not pos_mask.any():
            continue

        cls_logits = outputs["cls_logits"][frame_idx][pos_mask]
        box_pred = outputs["box_pred"][frame_idx][pos_mask].float()
        size_prior_log = outputs["size_prior_log"][frame_idx][pos_mask].float()

        pred_xy = box_pred[:, 0:2]
        pred_log_size = size_prior_log + box_pred[:, 2:4]
        pred_yaw = F.normalize(box_pred[:, 4:6], dim=1, eps=1e-6)

        cls_terms.append(
            F.cross_entropy(
                cls_logits.float(),
                targets["cls_target"],
                weight=class_weights,
            )
        )
        xy_terms.append(F.smooth_l1_loss(pred_xy, targets["xy_target"]))
        size_terms.append(F.smooth_l1_loss(pred_log_size, targets["log_size_target"]))
        yaw_terms.append(F.smooth_l1_loss(pred_yaw, targets["yaw_target"]))

    gate_loss = torch.stack(gate_terms).mean() if gate_terms else zero
    obj_loss = torch.stack(obj_terms).mean() if obj_terms else zero
    cls_loss = torch.stack(cls_terms).mean() if cls_terms else zero
    xy_loss = torch.stack(xy_terms).mean() if xy_terms else zero
    size_loss = torch.stack(size_terms).mean() if size_terms else zero
    yaw_loss = torch.stack(yaw_terms).mean() if yaw_terms else zero

    total = (
        cfg.lambda_gate * gate_loss
        + cfg.lambda_obj * obj_loss
        + cfg.lambda_cls * cls_loss
        + cfg.lambda_xy * xy_loss
        + cfg.lambda_size * size_loss
        + cfg.lambda_yaw * yaw_loss
    )

    metrics = {
        "gate": float(gate_loss.detach().item()),
        "obj": float(obj_loss.detach().item()),
        "cls": float(cls_loss.detach().item()),
        "xy": float(xy_loss.detach().item()),
        "size": float(size_loss.detach().item()),
        "yaw": float(yaw_loss.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, metrics
