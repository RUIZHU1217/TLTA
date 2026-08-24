from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class SARShipCocoDataset(Dataset):
    """COCO-format SAR ship dataset used for SSDD and HRSID configurations."""

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        transforms=None,
        category_ids: list[int] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(annotation_file))
        self.ids = sorted(self.coco.getImgIds())
        self.category_ids = category_ids or sorted(self.coco.getCatIds())
        self.category_to_label = {category_id: i + 1 for i, category_id in enumerate(self.category_ids)}
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image = Image.open(self.image_dir / image_info["file_name"]).convert("RGB")
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id], iscrowd=None))

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd: list[int] = []
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id not in self.category_to_label:
                continue
            x, y, w, h = annotation["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.category_to_label[category_id])
            areas.append(float(annotation.get("area", w * h)))
            iscrowd.append(int(annotation.get("iscrowd", 0)))

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([image.height, image.width], dtype=torch.int64),
            "size": torch.tensor([image.height, image.width], dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

