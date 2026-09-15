"""Evaluation metrics package.

Heavy image/VLM metrics remain in their existing modules; the model-free
structure proxies are exposed from :mod:`metrics.structure`.
"""

from .structure import boundary_f_score, boundary_mask, image_structure_metrics, normalized_mask_iou, structure_metrics

__all__ = ["boundary_f_score", "boundary_mask", "image_structure_metrics", "normalized_mask_iou", "structure_metrics"]
