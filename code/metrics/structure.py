"""Model-free structure metrics used by the Phase 3 audit.

These values are geometry proxies. They do not establish cultural correctness;
binding labels and VLM judgements remain a separate axis in experiment reports.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _mask(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError("masks must be two-dimensional")
    return array.astype(bool, copy=False)


def normalized_mask_iou(predicted: Any, reference: Any) -> float:
    """Return intersection-over-union in the closed interval ``[0, 1]``."""
    pred, ref = _mask(predicted), _mask(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {ref.shape}")
    union = np.logical_or(pred, ref).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, ref).sum() / union)


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
    x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
    sy0, sy1 = max(0, -dy), max(0, -dy) + max(0, y1 - y0)
    sx0, sx1 = max(0, -dx), max(0, -dx) + max(0, x1 - x0)
    if y1 > y0 and x1 > x0:
        out[y0:y1, x0:x1] = mask[sy0:sy1, sx0:sx1]
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    for distance in range(1, max(0, int(radius)) + 1):
        for dy, dx in ((-distance, 0), (distance, 0), (0, -distance), (0, distance)):
            result |= _shift(mask, dy, dx)
    return result


def boundary_mask(mask: Any) -> np.ndarray:
    """Extract the one-pixel outer boundary of a binary mask."""
    value = _mask(mask)
    interior = value.copy()
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        interior &= _shift(value, dy, dx)
    return value & ~interior


def boundary_f_score(predicted: Any, reference: Any, tolerance: int = 1) -> dict[str, float]:
    """Compute tolerant precision, recall, and F-score for two boundaries."""
    pred, ref = _mask(predicted), _mask(reference)
    if pred.shape != ref.shape:
        raise ValueError(f"mask shapes differ: {pred.shape} vs {ref.shape}")
    pred_edge, ref_edge = boundary_mask(pred), boundary_mask(ref)
    pred_count, ref_count = int(pred_edge.sum()), int(ref_edge.sum())
    if pred_count == 0 and ref_count == 0:
        return {"precision": 1.0, "recall": 1.0, "fscore": 1.0,
                "pred_boundary": 0.0, "ref_boundary": 0.0}
    if pred_count == 0 or ref_count == 0:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0,
                "pred_boundary": float(pred_count), "ref_boundary": float(ref_count)}
    ref_near, pred_near = _dilate(ref_edge, tolerance), _dilate(pred_edge, tolerance)
    precision = float((pred_edge & ref_near).sum() / pred_count)
    recall = float((ref_edge & pred_near).sum() / ref_count)
    fscore = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "fscore": fscore,
            "pred_boundary": float(pred_count), "ref_boundary": float(ref_count)}


def structure_metrics(predicted: Any, reference: Any, tolerance: int = 1) -> dict[str, Any]:
    result = boundary_f_score(predicted, reference, tolerance=tolerance)
    result["normalized_mask_iou"] = normalized_mask_iou(predicted, reference)
    return result


def _gray(image: Any) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 3:
        if array.shape[-1] < 3:
            raise ValueError("RGB image must have at least three channels")
        array = 0.299 * array[..., 0] + 0.587 * array[..., 1] + 0.114 * array[..., 2]
    if array.ndim != 2:
        raise ValueError("image must be a grayscale or RGB array")
    return array


def image_structure_metrics(image: Any, boundary: Optional[Any] = None) -> dict[str, float]:
    """Compute edge and Laplacian proxies for a generated image."""
    gray = _gray(image)
    gy = np.diff(gray, axis=0, prepend=gray[:1])
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    grad = np.hypot(gx, gy)
    if boundary is None:
        band = np.zeros(gray.shape, dtype=bool)
    else:
        band = _mask(boundary)
        if band.shape != gray.shape:
            raise ValueError("boundary mask shape must match image")
    boundary_grad = float(grad[band].mean()) if band.any() else 0.0
    outside = grad[~band]
    outside_mean = float(outside.mean()) if outside.size else 0.0
    lap = np.concatenate((np.diff(gray, n=2, axis=0).ravel(), np.diff(gray, n=2, axis=1).ravel()))
    return {
        "edge_mean": float(grad.mean()),
        "boundary_grad": boundary_grad,
        "boundary_excess": boundary_grad / outside_mean if outside_mean > 1e-8 else 0.0,
        "lap_var": float(np.var(lap)) if lap.size else 0.0,
    }


__all__ = [
    "boundary_f_score", "boundary_mask", "image_structure_metrics",
    "normalized_mask_iou", "structure_metrics",
]
