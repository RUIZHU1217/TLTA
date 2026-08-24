from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from datasets import build_dataloader
from models import build_tlta
from utils import load_config, seed_everything
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.evaluator import evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TLTA")
    parser.add_argument("--config", required=True, help="Path to SSDD/HRSID YAML config")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--output", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override reproducible seed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move_targets(targets, device: torch.device) -> list[dict]:
    return [
        {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(model, data_loader, optimizer, device: torch.device, epoch: int, log_interval: int) -> dict:
    model.train()
    running: dict[str, float] = {}
    progress = tqdm(data_loader, desc=f"train {epoch:03d}")
    for step, (images, targets) in enumerate(progress):
        images = torch.stack([image.to(device) for image in images])
        targets = move_targets(targets, device)
        losses = model(images, targets)
        total_loss = sum(losses.values())
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {losses}")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        for name, value in losses.items():
            running[name] = running.get(name, 0.0) + float(value.detach())
        if step % log_interval == 0:
            progress.set_postfix(loss=f"{float(total_loss.detach()):.4f}")
    denominator = max(len(data_loader), 1)
    return {name: value / denominator for name, value in running.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["train"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device)
    output_dir = Path(args.output or config["train"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_tlta(config).to(device)
    train_loader = build_dataloader(config, "train")
    val_loader = build_dataloader(config, "val")
    if config["train"]["optimizer"] != "SGD":
        raise ValueError("The paper experimental setting requires SGD")
    optimizer = SGD(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        momentum=float(config["train"]["momentum"]),
        weight_decay=float(config["train"]["weight_decay"]),
        nesterov=bool(config["train"]["nesterov"]),
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=list(config["train"]["milestones"]),
        gamma=float(config["train"]["gamma"]),
    )
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)

    history_path = output_dir / "metrics.jsonl"
    epochs = int(config["train"]["epochs"])
    for epoch in range(start_epoch, epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            int(config["train"]["log_interval"]),
        )
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], **train_metrics}
        if (epoch + 1) % int(config["train"]["eval_interval"]) == 0 or epoch + 1 == epochs:
            eval_metrics = evaluate_model(model, val_loader, device)
            record.update({f"val_{key}": value for key, value in eval_metrics.items()})
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record, indent=2))
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler, epoch, config)


if __name__ == "__main__":
    main()

