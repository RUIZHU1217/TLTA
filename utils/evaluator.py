from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from pycocotools.cocoeval import COCOeval

from .boxes import rescale_boxes


class CocoEvaluator:
    """COCO bbox evaluator reporting the metrics used in Section 4.3."""

    METRIC_NAMES = ("AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L")

    def __init__(self, coco_gt, category_ids: list[int]) -> None:
        self.coco_gt = coco_gt
        self.category_ids = category_ids
        self.results: list[dict[str, float | int | list[float]]] = []

    def update(self, detections: list[dict[str, torch.Tensor]], targets: list[dict]) -> None:
        for detection, target in zip(detections, targets):
            boxes = detection["boxes"].detach().cpu()
            boxes = rescale_boxes(boxes, target["size"].cpu(), target["orig_size"].cpu())
            scores = detection["scores"].detach().cpu()
            labels = detection["labels"].detach().cpu()
            image_id = int(target["image_id"].item())
            for box, score, label in zip(boxes, scores, labels):
                x0, y0, x1, y1 = box.tolist()
                category_index = max(0, min(int(label.item()) - 1, len(self.category_ids) - 1))
                self.results.append(
                    {
                        "image_id": image_id,
                        "category_id": int(self.category_ids[category_index]),
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        "score": float(score.item()),
                    }
                )

    def summarize(self) -> dict[str, float]:
        if not self.results:
            return {name: 0.0 for name in self.METRIC_NAMES}
        coco_dt = self.coco_gt.loadRes(self.results)
        evaluator = COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        return {name: float(value) for name, value in zip(self.METRIC_NAMES, evaluator.stats[:6])}


@torch.no_grad()
def evaluate_model(model, data_loader: Iterable, device: torch.device) -> dict[str, float]:
    model.eval()
    evaluator = CocoEvaluator(data_loader.dataset.coco, data_loader.dataset.category_ids)
    for images, targets in data_loader:
        images = torch.stack([image.to(device) for image in images])
        # Ground truth is intentionally not passed to the model during
        # evaluation; it is used only by COCOeval after prediction.
        detections = model(images)
        evaluator.update(detections, targets)
    return evaluator.summarize()
