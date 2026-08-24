from __future__ import annotations

import torch
from torchvision.transforms import functional as F


class DetectionResize:
    """Resize an image and its xyxy boxes to the configured paper input size.

    Stretch-resizing is an implementation detail because the manuscript gives
    only the final input resolution and does not describe padding/aspect policy.
    """

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = tuple(size)

    def __call__(self, image, target: dict):
        old_w, old_h = image.size
        new_h, new_w = self.size
        image = F.resize(image, [new_h, new_w], antialias=True)
        boxes = target["boxes"].clone()
        if boxes.numel():
            boxes[:, [0, 2]] *= new_w / old_w
            boxes[:, [1, 3]] *= new_h / old_h
        target["boxes"] = boxes
        target["area"] = target["area"] * (new_w / old_w) * (new_h / old_h)
        target["size"] = torch.tensor([new_h, new_w], dtype=torch.int64)
        return F.to_tensor(image), target
