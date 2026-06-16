import math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

METRIC_KEYS = ("foreground_iou", "foreground_dice", "hd", "hd95")
RESULT_KEYS = ("loss", "foreground_iou", "foreground_dice", "hd", "hd95", "num_samples")

# 定义得到边界像素的方法
def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask

    structure = np.ones((3, 3), dtype=bool)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded

# 定义对角线距离的方法
def _empty_distance(shape: Tuple[int, int]) -> float:
    shape_arr = np.asarray(shape, dtype=np.float64)
    return float(np.linalg.norm(shape_arr - 1.0))

# 定义计算豪斯多夫距离和95%距离的方法
def _hausdorff_and_hd95(pred_mask: np.ndarray, target_mask: np.ndarray) -> Tuple[float, float]:
    pred_mask = pred_mask.astype(bool)
    target_mask = target_mask.astype(bool)

    pred_empty = not pred_mask.any()
    target_empty = not target_mask.any()

    if pred_empty and target_empty:
        return float("nan"), float("nan")

    if pred_empty != target_empty:
        penalty = _empty_distance(pred_mask.shape)
        return penalty, penalty

    pred_surface = _surface(pred_mask)
    target_surface = _surface(target_mask)

    # 如果两个都没有边界像素，则返回nan
    if not pred_surface.any() and not target_surface.any():
        return float("nan"), float("nan")

    # 计算到目标的距离和到预测的距离
    dt_to_target = distance_transform_edt(~target_surface, sampling=(1.0, 1.0))
    dt_to_pred = distance_transform_edt(~pred_surface, sampling=(1.0, 1.0))

    pred_to_target = dt_to_target[pred_surface]
    target_to_pred = dt_to_pred[target_surface]
    
    distances = np.concatenate([pred_to_target, target_to_pred], axis=0)

    if distances.size == 0:
        return float("nan"), float("nan")

    return float(np.max(distances)), float(np.percentile(distances, 95))


def _nanmean(values) -> float:
    valid = [float(value) for value in values if not math.isnan(float(value))]
    return float(sum(valid) / len(valid)) if valid else 0.0

# 定义单类计算IoU的方法
def _class_iou(pred, target, class_id, eps=1e-6) -> float:
    pred_cls = pred.reshape(-1) == class_id
    target_cls = target.reshape(-1) == class_id

    # 如果两个都没有像素，则返回1.0
    if pred_cls.sum().item() == 0 and target_cls.sum().item() == 0:
        return 1.0
    intersection = (pred_cls & target_cls).sum().float()
    union = (pred_cls | target_cls).sum().float()
    return ((intersection + eps) / (union + eps)).item()

# 定义单类计算Dice的方法
def _class_dice(pred, target, class_id, eps=1e-6) -> float:
    pred_cls = pred.reshape(-1) == class_id
    target_cls = target.reshape(-1) == class_id
    pred_sum = pred_cls.sum().float()
    target_sum = target_cls.sum().float()
    if pred_sum.item() == 0 and target_sum.item() == 0:
        return 1.0
    intersection = (pred_cls & target_cls).sum().float()
    return ((2.0 * intersection + eps) / (pred_sum + target_sum + eps)).item()


def compute_foreground_metrics(pred, target, num_classes, eps=1e-6) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    fg_classes = range(1, num_classes)

    class_ious = [_class_iou(pred, target, class_id, eps) for class_id in fg_classes]
    class_dices = [_class_dice(pred, target, class_id, eps) for class_id in fg_classes]

    hd_values = []
    hd95_values = []
    for class_id in fg_classes:
        hd, hd95 = _hausdorff_and_hd95(pred_np == class_id, target_np == class_id)
        if not math.isnan(hd):
            hd_values.append(hd)
        if not math.isnan(hd95):
            hd95_values.append(hd95)

    n = max(len(class_ious), 1)
    return {
        "foreground_iou": float(sum(class_ious) / n) if class_ious else 1.0,
        "foreground_dice": float(sum(class_dices) / n) if class_dices else 1.0,
        "hd": _nanmean(hd_values),
        "hd95": _nanmean(hd95_values),
    }

# 定义按样本累加、再按有效样本数平均的方法
def accumulate_sample_metrics(totals, counts, sample_metrics: Dict[str, float]) -> None:
    for key in METRIC_KEYS:
        value = sample_metrics[key]
        if math.isnan(float(value)):
            continue
        totals[key] += float(value)
        counts[key] += 1


def average_accumulated_metrics(totals, counts) -> Dict[str, float]:
    return {
        key: (totals[key] / counts[key] if counts[key] > 0 else 0.0)
        for key in METRIC_KEYS
    }


def init_metric_accumulators() -> Tuple[defaultdict, defaultdict]:
    return defaultdict(float), defaultdict(int)

# 返回结果
def build_result_metrics(metrics: Dict[str, float], loss: float, num_samples: int) -> Dict[str, float]:
    return {
        "loss": loss,
        "foreground_iou": metrics["foreground_iou"],
        "foreground_dice": metrics["foreground_dice"],
        "hd": metrics["hd"],
        "hd95": metrics["hd95"],
        "num_samples": num_samples,
    }
