"""
train.py - training loop for the PointBEV v0 detector.

Flat Colab-friendly script. No if __name__ guard.

Usage in Colab:
    %run train.py
Or:
    exec(open("train.py").read())
"""
from __future__ import annotations

import gc
import json
import math
import os
import time

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from point_bev_v0.dataset import (
    DEFAULT_CLASS_NAMES,
    PointBevDataset,
    build_loader,
)
from point_bev_v0.model import PointBEVConfig, PointBEVModel
from point_bev_v0.losses import (
    PointLossConfig,
    compute_point_bev_losses,
    estimate_class_mean_sizes_from_samples,
    estimate_class_weights_from_samples,
)


# ============================================================
# CONFIG - edit this block
# ============================================================
DATAROOT = "/content/data/nuscenes"
VERSION = "v1.0-trainval"
SAVE_DIR = "/content/drive/MyDrive/point_bev"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# T4-friendly default:
# - unfrozen backbone
# - batch 2 usually gives the best throughput / stability trade-off on 16 GB
# - if you hit OOM, drop to batch 1 and double ACCUM_STEPS
BATCH_SIZE = 2
ACCUM_STEPS = 4
EPOCHS = 30
MAX_LR = 3e-4
BACKBONE_LR_MULT = 0.1
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 10.0
NUM_WORKERS = 4
AMP_ENABLED = True
LOG_EVERY = 20

USE_IMAGENET = True
VAL_GATE_THRESHOLD = 0.0

# loss weights
LAMBDA_GATE = 1.0
LAMBDA_OBJ = 1.0
LAMBDA_CLS = 1.0
LAMBDA_XY = 1.0
LAMBDA_SIZE = 1.0
LAMBDA_YAW = 0.5
OBJ_POS_WEIGHT = 20.0

# scene limits (None = use all)
TRAIN_SCENE_LIMIT = None
VAL_SCENE_LIMIT = None

# resume from checkpoint (None = train from scratch)
RESUME_PATH = None


# ============================================================
# HELPERS
# ============================================================
def select_split_tokens(nusc, split_name: str, limit: int | None = None) -> list[str]:
    from nuscenes.utils.splits import create_splits_scenes

    split_names = set(create_splits_scenes()[split_name])
    scenes = [s for s in nusc.scene if s["name"] in split_names]
    if limit is not None:
        scenes = scenes[:limit]

    tokens = []
    for scene in scenes:
        tok = scene["first_sample_token"]
        while tok:
            tokens.append(tok)
            tok = nusc.get("sample", tok)["next"]
    return tokens


def collect_gt_stats(dataset: PointBevDataset) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gt_boxes_list, gt_labels_list = [], []
    for tok in tqdm(dataset.sample_tokens, desc="GT stats", leave=False):
        sample = dataset.nusc.get("sample", tok)
        boxes_np, labels_np = dataset._load_targets(sample)
        gt_boxes_list.append(torch.from_numpy(boxes_np))
        gt_labels_list.append(torch.from_numpy(labels_np))
    return gt_boxes_list, gt_labels_list


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def build_optimizer(model: PointBEVModel) -> AdamW:
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("image_backbone."):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": MAX_LR * BACKBONE_LR_MULT,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": MAX_LR,
                "weight_decay": WEIGHT_DECAY,
            }
        )
    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")
    return AdamW(param_groups)


def scheduler_max_lrs(optimizer: AdamW) -> float | list[float]:
    max_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    return max_lrs[0] if len(max_lrs) == 1 else max_lrs


# ============================================================
# INIT NUSCENES
# ============================================================
print("Loading nuScenes...")
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)

train_split = "mini_train" if VERSION == "v1.0-mini" else "train"
val_split = "mini_val" if VERSION == "v1.0-mini" else "val"
train_tokens = select_split_tokens(nusc, train_split, TRAIN_SCENE_LIMIT)
val_tokens = select_split_tokens(nusc, val_split, VAL_SCENE_LIMIT)
print(f"  train: {len(train_tokens)} frames, val: {len(val_tokens)} frames")


# ============================================================
# DATASETS & LOADERS
# ============================================================
train_ds = PointBevDataset(nusc, sample_tokens=train_tokens)
val_ds = PointBevDataset(nusc, sample_tokens=val_tokens)

train_loader = build_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = build_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


# ============================================================
# CLASS STATS (one-time scan of GT metadata)
# ============================================================
print("Collecting GT stats for class weights / mean sizes...")
gt_boxes_list, gt_labels_list = collect_gt_stats(train_ds)

num_classes = len(DEFAULT_CLASS_NAMES)
class_weights = estimate_class_weights_from_samples(gt_labels_list, num_classes=num_classes)
class_mean_sizes = estimate_class_mean_sizes_from_samples(
    gt_boxes_list,
    gt_labels_list,
    num_classes=num_classes,
)

print(f"  class_weights: {class_weights.tolist()}")
print("  class_mean_sizes (w, l):")
for i, name in enumerate(DEFAULT_CLASS_NAMES):
    print(f"    {name:>22s}: {class_mean_sizes[i, 0]:.2f} x {class_mean_sizes[i, 1]:.2f}")


# ============================================================
# MODEL
# ============================================================
model_cfg = PointBEVConfig(
    num_classes=num_classes,
    use_imagenet_backbone=USE_IMAGENET,
    class_mean_sizes=tuple(tuple(row) for row in class_mean_sizes.tolist()),
)

model = PointBEVModel(model_cfg).to(DEVICE)
trainable, total = count_parameters(model)
print(f"Model: {total:,} params total, {trainable:,} trainable")


# ============================================================
# LOSS CONFIG
# ============================================================
loss_cfg = PointLossConfig(
    obj_pos_weight=OBJ_POS_WEIGHT,
    class_weights=tuple(class_weights.tolist()),
    lambda_gate=LAMBDA_GATE,
    lambda_obj=LAMBDA_OBJ,
    lambda_cls=LAMBDA_CLS,
    lambda_xy=LAMBDA_XY,
    lambda_size=LAMBDA_SIZE,
    lambda_yaw=LAMBDA_YAW,
)


# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================
optimizer = build_optimizer(model)

steps_per_epoch = math.ceil(len(train_loader) / ACCUM_STEPS)
total_opt_steps = steps_per_epoch * EPOCHS

scheduler = OneCycleLR(
    optimizer,
    max_lr=scheduler_max_lrs(optimizer),
    total_steps=total_opt_steps,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1000,
)

scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED and DEVICE.startswith("cuda"))


# ============================================================
# TRAINING STATE
# ============================================================
best_val_loss = float("inf")
start_epoch = 1
os.makedirs(SAVE_DIR, exist_ok=True)
history: list[dict] = []

if DEVICE.startswith("cuda"):
    torch.backends.cudnn.benchmark = True

if RESUME_PATH and os.path.isfile(RESUME_PATH):
    print(f"Resuming from {RESUME_PATH} ...")
    ckpt = torch.load(RESUME_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("best_val_loss", float("inf"))
    print(
        f"  resumed at epoch {start_epoch}, "
        f"best_val_loss={best_val_loss:.5f}"
    )
    del ckpt
    gc.collect()
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================
def train_epoch(epoch: int) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running = {k: 0.0 for k in ("gate", "obj", "cls", "xy", "size", "yaw", "total")}
    running["grad_norm"] = 0.0
    overflow_steps = 0
    attempted_steps = 0
    n_opt_steps = 0
    micro_step = 0
    accum_metrics = {k: 0.0 for k in ("gate", "obj", "cls", "xy", "size", "yaw", "total")}
    accum_count = 0

    def finish_optimizer_step() -> tuple[bool, float]:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRAD_CLIP,
            error_if_nonfinite=False,
        )
        grad_norm_value = float(grad_norm.item())

        old_scale = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        new_scale = float(scaler.get_scale())
        step_skipped = new_scale < old_scale

        optimizer.zero_grad(set_to_none=True)
        if not step_skipped:
            scheduler.step()

        return step_skipped, grad_norm_value

    pbar = tqdm(train_loader, desc=f"Train E{epoch}", leave=False)
    for batch in pbar:
        images = batch["images"].to(DEVICE, non_blocking=True)
        bev_stats = batch["bev_stats"].to(DEVICE, non_blocking=True)

        with autocast(device_type="cuda", enabled=AMP_ENABLED and DEVICE.startswith("cuda")):
            outputs = model(
                images=images,
                bev_stats=bev_stats,
                points=batch["points"],
                proj_uv=batch["proj_uv"],
                proj_mask=batch["proj_mask"],
                image_hw=batch["image_hw"],
                apply_gate=False,
            )

        loss, metrics = compute_point_bev_losses(outputs, batch, cfg=loss_cfg)
        loss = loss / ACCUM_STEPS

        scaler.scale(loss).backward()
        micro_step += 1
        for k in metrics:
            if k in accum_metrics:
                accum_metrics[k] += metrics[k]
        accum_count += 1

        if micro_step % ACCUM_STEPS == 0:
            step_skipped, grad_norm_value = finish_optimizer_step()
            attempted_steps += 1
            if step_skipped:
                overflow_steps += 1
            else:
                n_opt_steps += 1
                for k in accum_metrics:
                    running[k] += accum_metrics[k] / accum_count
                if math.isfinite(grad_norm_value):
                    running["grad_norm"] += grad_norm_value
            accum_metrics = {k: 0.0 for k in accum_metrics}
            accum_count = 0

            if step_skipped or (n_opt_steps > 0 and n_opt_steps % LOG_EVERY == 0):
                avg = {k: running[k] / max(n_opt_steps, 1) for k in running}
                overflow_rate = overflow_steps / max(attempted_steps, 1)
                pbar.set_postfix(
                    loss=f"{avg['total']:.4f}",
                    gate=f"{avg['gate']:.4f}",
                    obj=f"{avg['obj']:.4f}",
                    gn=f"{avg['grad_norm']:.1f}",
                    ovf=f"{overflow_steps}",
                    ovr=f"{overflow_rate:.3f}",
                    lr=f"{optimizer.param_groups[-1]['lr']:.1e}",
                )

    if micro_step % ACCUM_STEPS != 0:
        step_skipped, grad_norm_value = finish_optimizer_step()
        attempted_steps += 1
        if step_skipped:
            overflow_steps += 1
        else:
            n_opt_steps += 1
            if accum_count > 0:
                for k in accum_metrics:
                    running[k] += accum_metrics[k] / accum_count
            if math.isfinite(grad_norm_value):
                running["grad_norm"] += grad_norm_value

    avg = {k: running[k] / max(n_opt_steps, 1) for k in running}
    avg["overflow_steps"] = float(overflow_steps)
    avg["attempted_steps"] = float(attempted_steps)
    avg["overflow_rate"] = float(overflow_steps / max(attempted_steps, 1))
    return avg


# ============================================================
# VALIDATE
# ============================================================
@torch.no_grad()
def validate(epoch: int, gate_threshold: float = 0.0) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    loss_keys = ("gate", "obj", "cls", "xy", "size", "yaw", "total")

    running_loss = {k: 0.0 for k in loss_keys}
    running_deploy = {k: 0.0 for k in loss_keys}
    running_deploy["kept_frac"] = 0.0
    n_batches = 0

    for batch in tqdm(val_loader, desc=f"Val E{epoch}", leave=False):
        images = batch["images"].to(DEVICE, non_blocking=True)
        bev_stats = batch["bev_stats"].to(DEVICE, non_blocking=True)
        fwd_kwargs = dict(
            images=images,
            bev_stats=bev_stats,
            points=batch["points"],
            proj_uv=batch["proj_uv"],
            proj_mask=batch["proj_mask"],
            image_hw=batch["image_hw"],
        )

        with autocast(device_type="cuda", enabled=AMP_ENABLED and DEVICE.startswith("cuda")):
            outputs_full = model(**fwd_kwargs, apply_gate=False)
        _, metrics_full = compute_point_bev_losses(outputs_full, batch, cfg=loss_cfg)

        with autocast(device_type="cuda", enabled=AMP_ENABLED and DEVICE.startswith("cuda")):
            outputs_gated = model(**fwd_kwargs, apply_gate=True, gate_threshold=gate_threshold)
        _, metrics_gated = compute_point_bev_losses(outputs_gated, batch, cfg=loss_cfg)

        total_pts = sum(int(pm.any(dim=1).sum().item()) for pm in batch["proj_mask"])
        kept_pts = sum(k.shape[0] for k in outputs_gated["kept_indices"])
        frac = kept_pts / max(total_pts, 1)

        for k in loss_keys:
            running_loss[k] += metrics_full[k]
            running_deploy[k] += metrics_gated[k]
        running_deploy["kept_frac"] += frac
        n_batches += 1

    avg_loss = {k: running_loss[k] / max(n_batches, 1) for k in loss_keys}
    deploy_keys = list(loss_keys) + ["kept_frac"]
    avg_deploy = {k: running_deploy[k] / max(n_batches, 1) for k in deploy_keys}

    print("  [VAL-loss]   " + " | ".join(f"{k}={avg_loss[k]:.4f}" for k in loss_keys))
    print("  [VAL-deploy] " + " | ".join(f"{k}={avg_deploy[k]:.4f}" for k in deploy_keys))
    return avg_loss, avg_deploy


# ============================================================
# CHECKPOINT
# ============================================================
def save_checkpoint(
    epoch: int,
    *,
    val_loss_metrics: dict | None = None,
    val_deploy_metrics: dict | None = None,
    tag: str = "best",
) -> str:
    path = os.path.join(SAVE_DIR, f"point_bev_{tag}.pt")
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "model_cfg": {
            "num_classes": model_cfg.num_classes,
            "use_imagenet_backbone": model_cfg.use_imagenet_backbone,
            "class_mean_sizes": model_cfg.class_mean_sizes,
        },
        "loss_cfg": {
            "class_weights": loss_cfg.class_weights,
            "obj_pos_weight": loss_cfg.obj_pos_weight,
        },
        "val_loss_metrics": val_loss_metrics,
        "val_deploy_metrics": val_deploy_metrics,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)
    return path


# ============================================================
# MAIN LOOP
# ============================================================
print(f"\nStarting training: {EPOCHS} epochs, batch={BATCH_SIZE}, accum={ACCUM_STEPS}")
print(f"  optimizer steps/epoch: {steps_per_epoch}, total: {total_opt_steps}")
print(f"  max_lr={MAX_LR} | backbone_lr={MAX_LR * BACKBONE_LR_MULT}")
print("  backbone frozen: False")
print(f"  checkpoint metric: val_loss | val_deploy gate_threshold={VAL_GATE_THRESHOLD}")
print()

t0_total = time.time()

for epoch in range(start_epoch, EPOCHS + 1):
    t0 = time.time()
    train_metrics = train_epoch(epoch)
    val_loss_metrics, val_deploy_metrics = validate(epoch, gate_threshold=VAL_GATE_THRESHOLD)
    dt = time.time() - t0

    record = {
        "epoch": epoch,
        "time_sec": round(dt, 1),
        "train": {k: round(v, 5) for k, v in train_metrics.items()},
        "val_loss": {k: round(v, 5) for k, v in val_loss_metrics.items()},
        "val_deploy": {k: round(v, 5) for k, v in val_deploy_metrics.items()},
        "lr": optimizer.param_groups[-1]["lr"],
        "backbone_lr": optimizer.param_groups[0]["lr"],
    }
    history.append(record)

    improved_loss = val_loss_metrics["total"] < best_val_loss
    if improved_loss:
        best_val_loss = val_loss_metrics["total"]
        ckpt_path = save_checkpoint(
            epoch,
            val_loss_metrics=val_loss_metrics,
            val_deploy_metrics=val_deploy_metrics,
            tag="best",
        )
        print(f"  >> NEW BEST val_loss={best_val_loss:.5f} saved to {ckpt_path}")

    if epoch % 5 == 0:
        save_checkpoint(
            epoch,
            val_loss_metrics=val_loss_metrics,
            val_deploy_metrics=val_deploy_metrics,
            tag=f"epoch_{epoch}",
        )

    print(f"  epoch {epoch} done in {dt:.0f}s\n")

    hist_path = os.path.join(SAVE_DIR, "history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

total_time = time.time() - t0_total
print(
    f"Training complete in {total_time / 3600:.1f}h. "
    f"Best val_loss={best_val_loss:.5f}"
)
