from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and resolve no machine-specific paths."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return config


def require_keys(mapping: dict[str, Any], keys: list[str], prefix: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise KeyError(f"Missing configuration keys in {prefix}: {missing}")

