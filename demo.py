from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as F

from models import build_tlta
from utils import load_config, seed_everything
from utils.boxes import rescale_boxes
from utils.checkpoint import load_checkpoint
from utils.visualization import draw_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image TLTA inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="demo_output.png")
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else config["train"]["seed"]))
    device = torch.device(args.device)
    model = build_tlta(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    original_size = torch.tensor([image.height, image.width])
    input_h, input_w = config["dataset"]["image_size"]
    tensor = F.to_tensor(F.resize(image, [input_h, input_w], antialias=True)).unsqueeze(0).to(device)
    detection = model(tensor)[0]
    boxes = rescale_boxes(
        detection["boxes"].cpu(),
        torch.tensor([input_h, input_w]),
        original_size,
    )
    threshold = (
        float(args.score_threshold)
        if args.score_threshold is not None
        else float(config["test"]["score_threshold"])
    )
    visualization = draw_detections(
        image,
        boxes,
        detection["scores"].cpu(),
        detection["labels"].cpu(),
        threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

