from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


@dataclass(frozen=True)
class PointBEVConfig:
    num_classes: int
    num_cams: int = 6
    raw_lidar_dim: int = 7
    bev_in_channels: int = 4
    lfm_channels: int = 16
    gate_img_dim: int = 16
    c2_dim: int = 64
    p2_dim: int = 64
    p3_dim: int = 32
    p4_dim: int = 16
    view_dim: int = 96
    h_sem_dim: int = 128
    h_obj_dim: int = 64
    lidar_feat_dim: int = 32
    gate_threshold: float = 0.0
    x_range: tuple[float, float] = (-70.0, 70.0)
    y_range: tuple[float, float] = (-70.0, 70.0)
    use_imagenet_backbone: bool = False
    class_mean_sizes: tuple[tuple[float, float], ...] | None = None


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        final_activation: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        ]
        if final_activation:
            layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyLidarFeatureMap(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels),
        )

    def forward(self, bev_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fine = self.net(bev_stats)
        ctx = F.max_pool2d(fine, kernel_size=2, stride=2)
        return fine, ctx


class ResNet18FPN(nn.Module):
    def __init__(
        self,
        *,
        c2_dim: int = 64,
        p2_dim: int = 64,
        p3_dim: int = 32,
        p4_dim: int = 16,
        gate_dim: int = 16,
        use_imagenet_backbone: bool = False,
    ):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_imagenet_backbone else None
        resnet = models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
        )
        self.layer1 = resnet.layer1  # H/4, 64
        self.layer2 = resnet.layer2  # H/8, 128
        self.layer3 = resnet.layer3  # H/16, 256
        self.layer4 = resnet.layer4  # H/32, 512

        fpn_dim = 64
        self.lat2 = nn.Conv2d(64, fpn_dim, kernel_size=1)
        self.lat3 = nn.Conv2d(128, fpn_dim, kernel_size=1)
        self.lat4 = nn.Conv2d(256, fpn_dim, kernel_size=1)
        self.lat5 = nn.Conv2d(512, fpn_dim, kernel_size=1)

        self.c2_proj = nn.Conv2d(64, c2_dim, kernel_size=1)
        self.p2_proj = nn.Conv2d(fpn_dim, p2_dim, kernel_size=3, padding=1)
        self.p3_proj = nn.Conv2d(fpn_dim, p3_dim, kernel_size=3, padding=1)
        self.p4_proj = nn.Conv2d(fpn_dim, p4_dim, kernel_size=3, padding=1)
        self.p2_gate_proj = nn.Conv2d(p2_dim, gate_dim, kernel_size=1)

        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        x = (images - self.pixel_mean) / self.pixel_std
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p4_td = self.lat4(c4) + F.interpolate(self.lat5(c5), size=c4.shape[-2:], mode="nearest")
        p3_td = self.lat3(c3) + F.interpolate(p4_td, size=c3.shape[-2:], mode="nearest")
        p2_td = self.lat2(c2) + F.interpolate(p3_td, size=c2.shape[-2:], mode="nearest")

        c2_out = self.c2_proj(c2)
        p2_out = self.p2_proj(p2_td)
        p3_out = self.p3_proj(p3_td)
        p4_out = self.p4_proj(p4_td)
        p2_gate = self.p2_gate_proj(p2_out)

        return {
            "c2": c2_out,
            "p2": p2_out,
            "p3": p3_out,
            "p4": p4_out,
            "p2_gate": p2_gate,
        }


class PointBEVModel(nn.Module):
    def __init__(self, cfg: PointBEVConfig):
        super().__init__()
        self.cfg = cfg
        self._patch_kernels = (3, 5, 7)

        self.image_backbone = ResNet18FPN(
            c2_dim=cfg.c2_dim,
            p2_dim=cfg.p2_dim,
            p3_dim=cfg.p3_dim,
            p4_dim=cfg.p4_dim,
            gate_dim=cfg.gate_img_dim,
            use_imagenet_backbone=cfg.use_imagenet_backbone,
        )
        self.lfm = TinyLidarFeatureMap(cfg.bev_in_channels, cfg.lfm_channels)

        self.gate_lidar_encoder = MLP(cfg.lfm_channels * 3 * 3, 64, 16, final_activation=True)
        self.gate_head = MLP(cfg.gate_img_dim + 16, 32, 1)

        self.e2_encoder = MLP((cfg.c2_dim + cfg.p2_dim) * 3 * 3, 256, 64, final_activation=True)
        self.e3_encoder = MLP(cfg.p3_dim * 5 * 5, 128, 32, final_activation=True)
        self.e4_encoder = MLP(cfg.p4_dim, 32, 16, final_activation=True)
        # Per-view semantic input:
        # - e2/e3/e4 image descriptors
        # - border prior
        # - one-hot camera identity (avoid feeding an ordinal camera index)
        self.view_encoder = MLP(64 + 32 + 16 + 1 + cfg.num_cams, 128, cfg.view_dim, final_activation=True)

        self.lidar_local_encoder = MLP(cfg.lfm_channels * 3 * 3, 64, 16, final_activation=True)
        self.lidar_ctx_encoder = MLP(cfg.lfm_channels * 7 * 7, 128, 16, final_activation=True)
        self.lidar_feat_proj = MLP(32, 64, cfg.lidar_feat_dim, final_activation=True)

        self.h_sem_proj = MLP(cfg.view_dim + cfg.raw_lidar_dim, 192, cfg.h_sem_dim, final_activation=True)
        self.obj_trunk = MLP(cfg.h_sem_dim + cfg.lidar_feat_dim, 128, cfg.h_obj_dim, final_activation=True)

        self.obj_head = nn.Linear(cfg.h_obj_dim, 1)
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.h_sem_dim + cfg.h_obj_dim, 128),
            nn.GELU(),
            nn.Linear(128, cfg.num_classes),
        )
        self.box_head = nn.Sequential(
            nn.Linear(cfg.h_sem_dim + cfg.lidar_feat_dim + 2, 128),
            nn.GELU(),
            nn.Linear(128, 6),
        )

        self.register_buffer(
            "class_mean_log_sizes",
            self._build_class_mean_log_sizes(cfg),
            persistent=False,
        )

        # Cache patch offsets once to avoid rebuilding meshgrids on every forward.
        for kernel_size in self._patch_kernels:
            self.register_buffer(
                f"_patch_offsets_{kernel_size}",
                self._build_patch_offsets(kernel_size),
                persistent=False,
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _build_class_mean_log_sizes(cfg: PointBEVConfig) -> torch.Tensor:
        if cfg.class_mean_sizes is None:
            return torch.zeros((cfg.num_classes, 2), dtype=torch.float32)

        if len(cfg.class_mean_sizes) != cfg.num_classes:
            raise ValueError(
                "class_mean_sizes length must match num_classes: "
                f"{len(cfg.class_mean_sizes)} vs {cfg.num_classes}"
            )

        sizes = torch.tensor(cfg.class_mean_sizes, dtype=torch.float32)
        if sizes.ndim != 2 or sizes.shape[1] != 2:
            raise ValueError("class_mean_sizes must have shape [num_classes, 2]")
        return sizes.clamp_min(1e-4).log()

    def _compute_size_prior_log(self, cls_logits: torch.Tensor) -> torch.Tensor:
        probs = cls_logits.detach().softmax(dim=1)
        return probs @ self.class_mean_log_sizes

    def _normalize_image_uv(
        self,
        uv: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        x = ((uv[:, 0] + 0.5) / max(img_w, 1)) * 2.0 - 1.0
        y = ((uv[:, 1] + 0.5) / max(img_h, 1)) * 2.0 - 1.0
        return torch.stack([x, y], dim=-1)

    def _normalize_bev_xy(
        self,
        xy: torch.Tensor,
    ) -> torch.Tensor:
        x_min, x_max = self.cfg.x_range
        y_min, y_max = self.cfg.y_range
        nx = (xy[:, 0] - x_min) / max(x_max - x_min, 1e-6) * 2.0 - 1.0
        ny = (xy[:, 1] - y_min) / max(y_max - y_min, 1e-6) * 2.0 - 1.0
        return torch.stack([nx, ny], dim=-1)

    def _sample_center(
        self,
        feat_map: torch.Tensor,
        centers_norm: torch.Tensor,
    ) -> torch.Tensor:
        if centers_norm.numel() == 0:
            return feat_map.new_zeros((0, feat_map.shape[0]))
        grid = centers_norm.view(1, -1, 1, 2)
        sampled = F.grid_sample(
            feat_map.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.squeeze(0).squeeze(-1).transpose(0, 1).contiguous().to(dtype=feat_map.dtype)

    def _sample_patch(
        self,
        feat_map: torch.Tensor,
        centers_norm: torch.Tensor,
        kernel_size: int,
    ) -> torch.Tensor:
        if centers_norm.numel() == 0:
            return feat_map.new_zeros((0, feat_map.shape[0] * kernel_size * kernel_size))

        _, h, w = feat_map.shape
        base_offsets = getattr(self, f"_patch_offsets_{kernel_size}").to(
            device=feat_map.device,
            dtype=centers_norm.dtype,
        )
        offsets = base_offsets.clone()
        offsets[:, 0] *= 2.0 / max(w, 1)
        offsets[:, 1] *= 2.0 / max(h, 1)

        grid = centers_norm[:, None, :] + offsets[None, :, :]
        grid = grid.view(1, centers_norm.shape[0], kernel_size * kernel_size, 2)
        sampled = F.grid_sample(
            feat_map.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled = sampled.squeeze(0).permute(1, 0, 2).contiguous()
        return sampled.flatten(1).to(dtype=feat_map.dtype)

    @staticmethod
    def _build_patch_offsets(kernel_size: int) -> torch.Tensor:
        radius = kernel_size // 2
        ys = torch.arange(-radius, radius + 1, dtype=torch.float32)
        xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)

    def _encode_lidar_gate(
        self,
        lfm_fine_frame: torch.Tensor,
        point_xy: torch.Tensor,
    ) -> torch.Tensor:
        bev_norm = self._normalize_bev_xy(point_xy)
        patch = self._sample_patch(lfm_fine_frame, bev_norm, kernel_size=3)
        return self.gate_lidar_encoder(patch)

    def _encode_lidar_feat(
        self,
        lfm_fine_frame: torch.Tensor,
        lfm_ctx_frame: torch.Tensor,
        point_xy: torch.Tensor,
    ) -> torch.Tensor:
        bev_norm = self._normalize_bev_xy(point_xy)
        local_patch = self._sample_patch(lfm_fine_frame, bev_norm, kernel_size=3)
        ctx_patch = self._sample_patch(lfm_ctx_frame, bev_norm, kernel_size=7)
        local_feat = self.lidar_local_encoder(local_patch)
        ctx_feat = self.lidar_ctx_encoder(ctx_patch)
        return self.lidar_feat_proj(torch.cat([local_feat, ctx_feat], dim=1))

    def _encode_point_views(
        self,
        c2_views: torch.Tensor,
        p2_views: torch.Tensor,
        p3_views: torch.Tensor,
        p4_views: torch.Tensor,
        proj_uv: torch.Tensor,
        proj_mask: torch.Tensor,
        padded_img_h: int,
        padded_img_w: int,
        valid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points, num_views = proj_mask.shape
        if num_views != self.cfg.num_cams:
            raise ValueError(
                f"proj_mask has {num_views} camera views, but cfg.num_cams={self.cfg.num_cams}"
            )

        view_feats = []
        any_valid = proj_mask.any(dim=1)

        for view_idx in range(num_views):
            valid = proj_mask[:, view_idx]
            view_feat = c2_views.new_zeros((num_points, self.cfg.view_dim))
            if valid.any():
                valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                uv_valid = proj_uv[valid, view_idx]
                uv_norm = self._normalize_image_uv(uv_valid, img_h=padded_img_h, img_w=padded_img_w)

                c2_patch = self._sample_patch(c2_views[view_idx], uv_norm, kernel_size=3)
                p2_patch = self._sample_patch(p2_views[view_idx], uv_norm, kernel_size=3)
                p3_patch = self._sample_patch(p3_views[view_idx], uv_norm, kernel_size=5)
                p4_center = self._sample_center(p4_views[view_idx], uv_norm)

                e2 = self.e2_encoder(torch.cat([c2_patch, p2_patch], dim=1))
                e3 = self.e3_encoder(p3_patch)
                e4 = self.e4_encoder(p4_center)

                if valid_hw is None:
                    valid_w = float(padded_img_w)
                    valid_h = float(padded_img_h)
                elif valid_hw.ndim == 1:
                    valid_w = float(valid_hw[0].item())
                    valid_h = float(valid_hw[1].item())
                else:
                    valid_w = float(valid_hw[view_idx, 0].item())
                    valid_h = float(valid_hw[view_idx, 1].item())

                border_x = torch.minimum(uv_valid[:, 0], (valid_w - 1.0) - uv_valid[:, 0])
                border_y = torch.minimum(uv_valid[:, 1], (valid_h - 1.0) - uv_valid[:, 1])
                border = torch.minimum(border_x, border_y).unsqueeze(1) / max(min(valid_w, valid_h) - 1.0, 1.0)
                cam_id = torch.full(
                    (uv_valid.shape[0],),
                    view_idx,
                    device=uv_valid.device,
                    dtype=torch.long,
                )
                cam_ohe = F.one_hot(cam_id, num_classes=self.cfg.num_cams).to(dtype=e2.dtype)
                view_raw = torch.cat([e2, e3, e4, border, cam_ohe], dim=1)
                view_feat_valid = self.view_encoder(view_raw)
                view_feat[valid_idx] = view_feat_valid
            view_feats.append(view_feat)

        view_stack = torch.stack(view_feats, dim=1)
        masked = view_stack.masked_fill(~proj_mask.unsqueeze(-1), float("-inf"))
        img_feat = masked.max(dim=1).values
        img_feat[~any_valid] = 0.0
        return img_feat, any_valid

    def _build_gate_scores(
        self,
        p2_gate_views: torch.Tensor,
        lfm_fine_frame: torch.Tensor,
        points_frame: torch.Tensor,
        proj_uv_frame: torch.Tensor,
        proj_mask_frame: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_points, num_views = proj_mask_frame.shape
        gate_img = p2_gate_views.new_full((num_points, self.cfg.gate_img_dim), float("-inf"))
        any_valid = proj_mask_frame.any(dim=1)

        for view_idx in range(num_views):
            valid = proj_mask_frame[:, view_idx]
            if not valid.any():
                continue
            valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
            uv_norm = self._normalize_image_uv(proj_uv_frame[valid, view_idx], img_h=img_h, img_w=img_w)
            sampled = self._sample_center(p2_gate_views[view_idx], uv_norm)
            gate_img[valid_idx] = torch.maximum(gate_img[valid_idx], sampled)

        gate_img[~any_valid] = 0.0
        gate_lidar = self._encode_lidar_gate(lfm_fine_frame, points_frame[:, :2])
        gate_logits = self.gate_head(torch.cat([gate_img, gate_lidar], dim=1)).squeeze(1)
        return gate_logits, any_valid

    def forward(
        self,
        *,
        images: torch.Tensor,
        bev_stats: torch.Tensor,
        points: Sequence[torch.Tensor],
        proj_uv: Sequence[torch.Tensor],
        proj_mask: Sequence[torch.Tensor],
        image_hw: Sequence[torch.Tensor] | None = None,
        apply_gate: bool | None = None,
        gate_threshold: float | None = None,
    ) -> dict[str, list[torch.Tensor]]:
        if apply_gate is None:
            apply_gate = not self.training
        gate_threshold = self.cfg.gate_threshold if gate_threshold is None else float(gate_threshold)

        batch_size, num_cams, _, img_h, img_w = images.shape
        flat_images = images.view(batch_size * num_cams, images.shape[2], img_h, img_w)
        image_feats = self.image_backbone(flat_images)
        c2_all = image_feats["c2"].view(batch_size, num_cams, self.cfg.c2_dim, image_feats["c2"].shape[-2], image_feats["c2"].shape[-1])
        p2_all = image_feats["p2"].view(batch_size, num_cams, self.cfg.p2_dim, image_feats["p2"].shape[-2], image_feats["p2"].shape[-1])
        p3_all = image_feats["p3"].view(batch_size, num_cams, self.cfg.p3_dim, image_feats["p3"].shape[-2], image_feats["p3"].shape[-1])
        p4_all = image_feats["p4"].view(batch_size, num_cams, self.cfg.p4_dim, image_feats["p4"].shape[-2], image_feats["p4"].shape[-1])
        p2_gate_all = image_feats["p2_gate"].view(batch_size, num_cams, self.cfg.gate_img_dim, image_feats["p2_gate"].shape[-2], image_feats["p2_gate"].shape[-1])

        lfm_fine, lfm_ctx = self.lfm(bev_stats)

        obj_pre_logits_out: list[torch.Tensor] = []
        kept_indices_out: list[torch.Tensor] = []
        obj_logits_out: list[torch.Tensor] = []
        cls_logits_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        size_prior_log_out: list[torch.Tensor] = []
        point_xyz_out: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            points_frame = points[batch_idx].to(device=images.device, dtype=torch.float32)
            proj_uv_frame = proj_uv[batch_idx].to(device=images.device, dtype=torch.float32)
            proj_mask_frame = proj_mask[batch_idx].to(device=images.device, dtype=torch.bool)
            if image_hw is None:
                valid_hw_frame = None
            else:
                valid_hw_frame = image_hw[batch_idx].to(device=images.device, dtype=torch.float32)

            if points_frame.numel() == 0:
                empty_long = torch.empty((0,), device=images.device, dtype=torch.long)
                empty_float = torch.empty((0,), device=images.device, dtype=torch.float32)
                obj_pre_logits_out.append(empty_float)
                kept_indices_out.append(empty_long)
                obj_logits_out.append(empty_float)
                cls_logits_out.append(torch.empty((0, self.cfg.num_classes), device=images.device))
                box_out.append(torch.empty((0, 6), device=images.device))
                size_prior_log_out.append(torch.empty((0, 2), device=images.device))
                point_xyz_out.append(torch.empty((0, 2), device=images.device))
                continue

            gate_logits, any_valid = self._build_gate_scores(
                p2_gate_views=p2_gate_all[batch_idx],
                lfm_fine_frame=lfm_fine[batch_idx],
                points_frame=points_frame,
                proj_uv_frame=proj_uv_frame,
                proj_mask_frame=proj_mask_frame,
                img_h=img_h,
                img_w=img_w,
            )
            obj_pre_logits_out.append(gate_logits)

            if apply_gate:
                keep_mask = (gate_logits > gate_threshold) & any_valid
            else:
                keep_mask = any_valid

            keep_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)
            kept_indices_out.append(keep_idx)

            if keep_idx.numel() == 0:
                empty_float = torch.empty((0,), device=images.device, dtype=torch.float32)
                obj_logits_out.append(empty_float)
                cls_logits_out.append(torch.empty((0, self.cfg.num_classes), device=images.device))
                box_out.append(torch.empty((0, 6), device=images.device))
                size_prior_log_out.append(torch.empty((0, 2), device=images.device))
                point_xyz_out.append(torch.empty((0, 2), device=images.device))
                continue

            kept_points = points_frame[keep_idx]
            kept_uv = proj_uv_frame[keep_idx]
            kept_mask = proj_mask_frame[keep_idx]

            img_feat, _ = self._encode_point_views(
                c2_views=c2_all[batch_idx],
                p2_views=p2_all[batch_idx],
                p3_views=p3_all[batch_idx],
                p4_views=p4_all[batch_idx],
                proj_uv=kept_uv,
                proj_mask=kept_mask,
                padded_img_h=img_h,
                padded_img_w=img_w,
                valid_hw=valid_hw_frame,
            )

            lidar_feat = self._encode_lidar_feat(
                lfm_fine_frame=lfm_fine[batch_idx],
                lfm_ctx_frame=lfm_ctx[batch_idx],
                point_xy=kept_points[:, :2],
            )

            raw_attrs = kept_points[:, : self.cfg.raw_lidar_dim]
            h_sem = self.h_sem_proj(torch.cat([img_feat, raw_attrs], dim=1))
            h_obj = self.obj_trunk(torch.cat([h_sem, lidar_feat], dim=1))

            obj_logits = self.obj_head(h_obj).squeeze(1)
            cls_logits = self.cls_head(torch.cat([h_sem, h_obj.detach()], dim=1))
            size_prior_log = self._compute_size_prior_log(cls_logits)
            box_pred = self.box_head(torch.cat([h_sem, lidar_feat, size_prior_log], dim=1))

            obj_logits_out.append(obj_logits)
            cls_logits_out.append(cls_logits)
            box_out.append(box_pred)
            size_prior_log_out.append(size_prior_log)
            point_xyz_out.append(kept_points[:, :2])

        return {
            "obj_pre_logits": obj_pre_logits_out,
            "kept_indices": kept_indices_out,
            "obj_logits": obj_logits_out,
            "cls_logits": cls_logits_out,
            "box_pred": box_out,
            "size_prior_log": size_prior_log_out,
            "point_xy": point_xyz_out,
        }
