from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .coco import SARShipCocoDataset
from .transforms import DetectionResize


def build_dataset(config: dict, split: str) -> SARShipCocoDataset:
    dataset_config = config["dataset"]
    root = Path(dataset_config["root"])
    image_key = "train_images" if split == "train" else "val_images"
    annotation_key = "train_annotations" if split == "train" else "val_annotations"
    return SARShipCocoDataset(
        root / dataset_config[image_key],
        root / dataset_config[annotation_key],
        DetectionResize(tuple(dataset_config["image_size"])),
        list(dataset_config["category_ids"]),
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataloader(config: dict, split: str) -> DataLoader:
    dataset = build_dataset(config, split)
    section = config["train"] if split == "train" else config["test"]
    return DataLoader(
        dataset,
        batch_size=int(section["batch_size"]),
        shuffle=split == "train",
        num_workers=int(section["num_workers"]),
        pin_memory=True,
        collate_fn=collate_fn,
    )

