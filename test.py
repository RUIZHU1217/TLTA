from __future__ import annotations

import argparse
import json

import torch

from datasets import build_dataloader
from models import build_tlta
from utils import load_config, seed_everything
from utils.checkpoint import load_checkpoint
from utils.evaluator import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TLTA with COCO metrics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(args.seed if args.seed is not None else config["train"]["seed"]))
    device = torch.device(args.device)
    model = build_tlta(config).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    data_loader = build_dataloader(config, "val")
    metrics = evaluate_model(model, data_loader, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

