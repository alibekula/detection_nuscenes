from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from torch.utils.data import DataLoader, Dataset


CAM_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

DEFAULT_CLASS_NAMES = (
    "car",
    "truck",
    "bus",
    "trailer",
    "construction_vehicle",
    "pedestrian",
    "motorcycle",
    "bicycle",
    "barrier",
)

_PEDESTRIAN_RAW_NAMES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
}


def normalize_detection_class(name: str | None) -> str | None:
    """Map raw nuScenes category names to the reduced detection set.

    This mapping is intentionally strict:
    - traffic cone is dropped
    - stroller / wheelchair / personal mobility are dropped
    - only official raw nuScenes category names are accepted
    """
    if not name:
        return None

    value = str(name).lower()

    if value == "vehicle.car":
        return "car"
    if value == "vehicle.truck":
        return "truck"
    if value in {"vehicle.bus.bendy", "vehicle.bus.rigid"}:
        return "bus"
    if value == "vehicle.trailer":
        return "trailer"
    if value == "vehicle.construction":
        return "construction_vehicle"
    if value in _PEDESTRIAN_RAW_NAMES:
        return "pedestrian"
    if value == "vehicle.motorcycle":
        return "motorcycle"
    if value == "vehicle.bicycle":
        return "bicycle"
    if value == "movable_object.barrier":
        return "barrier"
    return None


class PointBevDataset(Dataset):
    """One item = one frame."""

    def __init__(
        self,
        nusc,
        nsweeps: int = 1,
        bev_x_range: tuple[float, float] = (-70.0, 70.0),
        bev_y_range: tuple[float, float] = (-70.0, 70.0),
        bev_resolution: float = 0.2,
        class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
        target_hw: tuple[int, int] = (928, 1600),
        sample_tokens: list[str] | None = None,
    ):
        self.nusc = nusc
        self.nsweeps = int(nsweeps)
        self.bev_x_range = tuple(float(v) for v in bev_x_range)
        self.bev_y_range = tuple(float(v) for v in bev_y_range)
        self.bev_resolution = float(bev_resolution)
        self.class_names = tuple(class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.target_h = int(target_hw[0])
        self.target_w = int(target_hw[1])

        self.bev_w = int(np.ceil((self.bev_x_range[1] - self.bev_x_range[0]) / self.bev_resolution))
        self.bev_h = int(np.ceil((self.bev_y_range[1] - self.bev_y_range[0]) / self.bev_resolution))

        if sample_tokens is not None:
            self.sample_tokens = list(sample_tokens)
        else:
            self.sample_tokens: list[str] = []
            for scene in self.nusc.scene:
                token = scene["first_sample_token"]
                while token:
                    self.sample_tokens.append(token)
                    token = self.nusc.get("sample", token)["next"]

    def __len__(self):
        return len(self.sample_tokens)

    def __getitem__(self, idx: int):
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get("sample", sample_token)

        images_np, image_hw_np = self._load_images(sample)
        points_np, lidar_xyz_np = self._load_points(sample)
        proj_uv_np, proj_mask_np = self._build_projection(sample, lidar_xyz_np)
        bev_stats_np = self._build_bev_stats(points_np)
        gt_boxes_np, gt_labels_np = self._load_targets(sample)

        return {
            "images": torch.from_numpy(images_np).float(),          # [6, 3, H, W]
            "image_hw": torch.from_numpy(image_hw_np).float(),      # [6, 2] -> W, H
            "points": torch.from_numpy(points_np).float(),          # [N, 7]
            "points_xyz": torch.from_numpy(lidar_xyz_np).float(),   # [N, 3] raw lidar xyz
            "proj_uv": torch.from_numpy(proj_uv_np).float(),        # [N, 6, 2]
            "proj_mask": torch.from_numpy(proj_mask_np),            # [N, 6]
            "bev_stats": torch.from_numpy(bev_stats_np).float(),    # [4, Hb, Wb]
            "gt_boxes": torch.from_numpy(gt_boxes_np).float(),      # [M, 7]
            "gt_labels": torch.from_numpy(gt_labels_np).long(),     # [M]
            "token": sample_token,
        }

    def _load_images(self, sample):
        images = []
        image_hw = []

        # We keep raw [0, 1] inputs and pad with raw ImageNet mean so that
        # model-side normalization turns padded regions into exact zeros.
        pad_value = np.array([0.485, 0.456, 0.406], dtype=np.float32)

        for cam_name in CAM_NAMES:
            sd = self.nusc.get("sample_data", sample["data"][cam_name])
            img_path = Path(self.nusc.dataroot) / sd["filename"]

            with Image.open(img_path) as img:
                img = img.convert("RGB")
                arr = np.asarray(img, dtype=np.float32) / 255.0

            h, w = arr.shape[:2]
            image_hw.append([w, h])

            padded = np.empty((self.target_h, self.target_w, 3), dtype=np.float32)
            padded[...] = pad_value
            padded[:h, :w, :] = arr

            images.append(np.transpose(padded, (2, 0, 1)))

        return (
            np.stack(images, axis=0).astype(np.float32, copy=False),
            np.asarray(image_hw, dtype=np.float32),
        )

    def _load_points(self, sample):
        pc, _ = LidarPointCloud.from_file_multisweep(
            self.nusc,
            sample,
            chan="LIDAR_TOP",
            ref_chan="LIDAR_TOP",
            nsweeps=self.nsweeps,
            min_distance=0.5,
        )
        pts = pc.points[:4].T.astype(np.float32, copy=False)  # x, y, z, intensity

        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        calib = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_height = float(calib["translation"][2])

        z_rel_ground = pts[:, 2] + lidar_height

        mask = (
            (z_rel_ground > 0.15) &
            (z_rel_ground < 7.0)
        )

        pts = pts[mask]
        z_rel_ground = z_rel_ground[mask]
        lidar_xyz = np.ascontiguousarray(pts[:, :3].astype(np.float32, copy=False))

        x = pts[:, 0]
        y = pts[:, 1]
        intensity = pts[:, 3]

        r = np.hypot(x, y)
        theta = np.arctan2(y, x)

        point_feats = np.stack(
            [
                x,
                y,
                z_rel_ground,
                intensity,
                r,
                np.sin(theta),
                np.cos(theta),
            ],
            axis=1,
        ).astype(np.float32, copy=False)

        return point_feats, lidar_xyz


    def _lidar_to_camera_transform(self, sample, cam_name: str):
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cam_sd = self.nusc.get("sample_data", sample["data"][cam_name])

        lidar_calib = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        lidar_ego = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        cam_calib = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        cam_ego = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        r_lidar_ego = Quaternion(lidar_calib["rotation"]).rotation_matrix
        t_lidar_ego = np.asarray(lidar_calib["translation"], dtype=np.float64)

        r_ego_global = Quaternion(lidar_ego["rotation"]).rotation_matrix
        t_ego_global = np.asarray(lidar_ego["translation"], dtype=np.float64)

        r_global_ego_cam = Quaternion(cam_ego["rotation"]).rotation_matrix.T
        t_global_ego_cam = -r_global_ego_cam @ np.asarray(cam_ego["translation"], dtype=np.float64)

        r_ego_cam_sensor = Quaternion(cam_calib["rotation"]).rotation_matrix.T
        t_ego_cam_sensor = -r_ego_cam_sensor @ np.asarray(cam_calib["translation"], dtype=np.float64)

        r_lidar_global = r_ego_global @ r_lidar_ego
        t_lidar_global = r_ego_global @ t_lidar_ego + t_ego_global

        r_lidar_ego_cam = r_global_ego_cam @ r_lidar_global
        t_lidar_ego_cam = r_global_ego_cam @ t_lidar_global + t_global_ego_cam

        r_total = r_ego_cam_sensor @ r_lidar_ego_cam
        t_total = r_ego_cam_sensor @ t_lidar_ego_cam + t_ego_cam_sensor

        intrinsic = np.asarray(cam_calib["camera_intrinsic"], dtype=np.float64)
        width = int(cam_sd["width"])
        height = int(cam_sd["height"])
        return r_total, t_total, intrinsic, width, height

    def _build_projection(self, sample, lidar_xyz):
        n = int(lidar_xyz.shape[0])
        proj_uv = np.full((n, len(CAM_NAMES), 2), -1.0, dtype=np.float32)
        proj_mask = np.zeros((n, len(CAM_NAMES)), dtype=np.bool_)

        if n == 0:
            return proj_uv, proj_mask

        pts_xyz = np.ascontiguousarray(lidar_xyz.astype(np.float64, copy=False))

        for cam_idx, cam_name in enumerate(CAM_NAMES):
            r_total, t_total, intrinsic, width, height = self._lidar_to_camera_transform(sample, cam_name)

            p_cam = (r_total @ pts_xyz.T).T + t_total
            z = p_cam[:, 2]
            in_front = z > 0.1
            if not np.any(in_front):
                continue

            valid_idx = np.flatnonzero(in_front)
            pixels = intrinsic @ p_cam[in_front].T
            u = pixels[0] / pixels[2]
            v = pixels[1] / pixels[2]

            inside = (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
            if not np.any(inside):
                continue

            good_idx = valid_idx[inside]
            proj_uv[good_idx, cam_idx, 0] = u[inside].astype(np.float32, copy=False)
            proj_uv[good_idx, cam_idx, 1] = v[inside].astype(np.float32, copy=False)
            proj_mask[good_idx, cam_idx] = True

        return proj_uv, proj_mask

    def _build_bev_stats(self, points):
        x_min, x_max = self.bev_x_range
        y_min, y_max = self.bev_y_range
        res = self.bev_resolution

        if points.shape[0] == 0:
            return np.zeros((4, self.bev_h, self.bev_w), dtype=np.float32)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        ix = np.floor((x - x_min) / res).astype(np.int32)
        iy = np.floor((y - y_min) / res).astype(np.int32)

        valid = (
            (ix >= 0) & (ix < self.bev_w) &
            (iy >= 0) & (iy < self.bev_h)
        )

        if not np.any(valid):
            return np.zeros((4, self.bev_h, self.bev_w), dtype=np.float32)

        ix = ix[valid]
        iy = iy[valid]
        z = z[valid]
        flat = iy * self.bev_w + ix
        size = self.bev_h * self.bev_w

        count = np.bincount(flat, minlength=size).astype(np.float32)

        z_min_arr = np.full(size, np.inf, dtype=np.float32)
        z_max_arr = np.full(size, -np.inf, dtype=np.float32)
        np.minimum.at(z_min_arr, flat, z)
        np.maximum.at(z_max_arr, flat, z)

        empty = count == 0
        z_min_arr[empty] = 0.0
        z_max_arr[empty] = 0.0

        occupied = (count > 0).astype(np.float32)
        height_range = z_max_arr - z_min_arr

        return np.stack(
            [
                count.reshape(self.bev_h, self.bev_w),
                occupied.reshape(self.bev_h, self.bev_w),
                z_max_arr.reshape(self.bev_h, self.bev_w),
                height_range.reshape(self.bev_h, self.bev_w),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _load_targets(self, sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        _, boxes, _ = self.nusc.get_sample_data(
            lidar_token,
            selected_anntokens=sample["anns"],
        )

        gt_boxes = []
        gt_labels = []

        for box in boxes:
            cls_name = normalize_detection_class(getattr(box, "name", None))
            if cls_name is None or cls_name not in self.class_to_idx:
                continue

            cx, cy, cz = map(float, box.center.tolist())
            w, l, h = map(float, box.wlh.tolist())
            yaw = float(box.orientation.yaw_pitch_roll[0])

            gt_boxes.append([cx, cy, cz, w, l, h, yaw])
            gt_labels.append(self.class_to_idx[cls_name])

        if not gt_boxes:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return (
            np.asarray(gt_boxes, dtype=np.float32),
            np.asarray(gt_labels, dtype=np.int64),
        )


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([b["images"] for b in batch], dim=0),        # [B, 6, 3, H, W]
        "image_hw": torch.stack([b["image_hw"] for b in batch], dim=0),    # [B, 6, 2] -> W, H
        "bev_stats": torch.stack([b["bev_stats"] for b in batch], dim=0),  # [B, 4, Hb, Wb]
        "points": [b["points"] for b in batch],                            # list[[Ni, 7]]
        "points_xyz": [b["points_xyz"] for b in batch],                    # list[[Ni, 3]]
        "proj_uv": [b["proj_uv"] for b in batch],                          # list[[Ni, 6, 2]]
        "proj_mask": [b["proj_mask"] for b in batch],                      # list[[Ni, 6]]
        "gt_boxes": [b["gt_boxes"] for b in batch],                        # list[[Mi, 7]]
        "gt_labels": [b["gt_labels"] for b in batch],                      # list[[Mi]]
        "tokens": [b["token"] for b in batch],
    }


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
