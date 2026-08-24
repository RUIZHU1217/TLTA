from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_tlta  # noqa: E402
from utils import load_config, seed_everything  # noqa: E402


def main() -> None:
    config = load_config(ROOT / "configs" / "smoke.yaml")
    seed_everything(7)
    model = build_tlta(config)
    images = torch.rand(1, 3, 64, 64)
    targets = [
        {
            "boxes": torch.tensor([[16.0, 18.0, 36.0, 34.0]]),
            "labels": torch.tensor([1], dtype=torch.long),
            "size": torch.tensor([64, 64]),
            "orig_size": torch.tensor([64, 64]),
            "image_id": torch.tensor(1),
        }
    ]
    model.train()
    losses = model(images, targets)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise AssertionError(losses)
    total.backward()
    model.eval()
    with torch.no_grad():
        detections = model(images)
    assert len(detections) == 1
    assert detections[0]["boxes"].shape[-1] == 4
    print("TLTA smoke test passed")
    print({name: float(value.detach()) for name, value in losses.items()})


if __name__ == "__main__":
    main()

