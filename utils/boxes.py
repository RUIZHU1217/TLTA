from __future__ import annotations

import torch
from torchvision.ops import generalized_box_iou


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), -1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), -1)


def canonicalize_boxes(boxes: torch.Tensor, minimum: float = 1e-4) -> torch.Tensor:
    """Clamp normalized xyxy boxes and make every width/height positive."""
    boxes = boxes.clamp(0.0, 1.0)
    lo = torch.minimum(boxes[..., :2], boxes[..., 2:])
    hi = torch.maximum(boxes[..., :2], boxes[..., 2:])
    hi = torch.maximum(hi, lo + minimum).clamp(max=1.0)
    lo = torch.minimum(lo, hi - minimum).clamp(min=0.0)
    return torch.cat((lo, hi), dim=-1)


def normalize_xyxy(boxes: torch.Tensor, size_hw: torch.Tensor | tuple[int, int]) -> torch.Tensor:
    if isinstance(size_hw, torch.Tensor):
        h, w = size_hw.to(device=boxes.device, dtype=boxes.dtype).unbind(-1)
    else:
        h, w = size_hw
        h = boxes.new_tensor(float(h))
        w = boxes.new_tensor(float(w))
    scale = torch.stack((w, h, w, h))
    return boxes / scale


def denormalize_xyxy(boxes: torch.Tensor, size_hw: torch.Tensor | tuple[int, int]) -> torch.Tensor:
    if isinstance(size_hw, torch.Tensor):
        h, w = size_hw.to(device=boxes.device, dtype=boxes.dtype).unbind(-1)
    else:
        h, w = size_hw
        h = boxes.new_tensor(float(h))
        w = boxes.new_tensor(float(w))
    scale = torch.stack((w, h, w, h))
    return boxes * scale


def pairwise_giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum()
    return (1.0 - torch.diag(generalized_box_iou(pred, target))).mean()


def rescale_boxes(boxes: torch.Tensor, from_hw: torch.Tensor, to_hw: torch.Tensor) -> torch.Tensor:
    from_h, from_w = from_hw.to(boxes).unbind(-1)
    to_h, to_w = to_hw.to(boxes).unbind(-1)
    scale = torch.stack((to_w / from_w, to_h / from_h, to_w / from_w, to_h / from_h))
    return boxes * scale

