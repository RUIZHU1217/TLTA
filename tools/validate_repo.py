from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_tlta  # noqa: E402
from utils import load_config  # noqa: E402


def main() -> None:
    python_files = sorted(ROOT.rglob("*.py"))
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for name in ("smoke", "ssdd", "hrsid"):
        config = load_config(ROOT / "configs" / f"{name}.yaml")
        model = build_tlta(config)
        assert model.i_lta.cacn.blocks.__len__() == 5
        assert len(model.f_lta.stages) == 4
    print(f"Validated syntax/imports for {len(python_files)} Python files and all configurations")


if __name__ == "__main__":
    main()

