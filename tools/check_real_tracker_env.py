#!/usr/bin/env python3
"""Check whether the real ByteTrack-DMA-LTC pipeline can run locally."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "bytetrack_dma_caviar.json"
REQUIRED_IMPORTS = [
    "cv2",
    "torch",
    "torchvision",
    "loguru",
    "numpy",
    "scipy",
    "pycocotools",
    "cython_bbox",
    "lap",
    "motmetrics",
    "filterpy",
    "yaml",
    "yacs",
]
DMA_IMPORTS = ["lightgbm"]
FAST_REID_IMPORTS = ["fvcore", "iopath", "tabulate", "faiss", "tensorboard"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight real tracker dependencies.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "data" / "caviar" / "videos" / "Browse_WhileWaiting1.mpg",
    )
    return parser


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def has_import(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    args = make_parser().parse_args()
    args.config = resolve_project_path(args.config)
    args.video = resolve_project_path(args.video)

    with args.config.open(encoding="utf-8") as handle:
        config = json.load(handle)

    model_repo = resolve_project_path(Path(config["model_repo"]))
    checkpoint = model_repo / config["checkpoint"]

    checks = [
        ("config", args.config.exists(), str(args.config)),
        ("model_repo", model_repo.exists(), str(model_repo)),
        ("video", args.video.exists(), str(args.video)),
        ("checkpoint", checkpoint.exists(), str(checkpoint)),
    ]

    ltc = config.get("ltc", {})
    if ltc.get("motion_ckpt"):
        ltc_ckpt = model_repo / ltc["motion_ckpt"]
        checks.append(("ltc_motion_ckpt", ltc_ckpt.exists(), str(ltc_ckpt)))

    reid = config.get("reid", {})
    if reid.get("enabled", False):
        if reid.get("fast_reid_config"):
            reid_config = model_repo / reid["fast_reid_config"]
            checks.append(("fast_reid_config", reid_config.exists(), str(reid_config)))
        if reid.get("fast_reid_weights"):
            reid_weights = model_repo / reid["fast_reid_weights"]
            checks.append(("fast_reid_weights", reid_weights.exists(), str(reid_weights)))

    dma = config.get("dma", {})
    if dma.get("ml_weights"):
        dma_weights = model_repo / dma["ml_weights"]
        checks.append(("dma_ml_weights", dma_weights.exists(), str(dma_weights)))
    if dma.get("weights"):
        dma_weights = model_repo / dma["weights"]
        checks.append(("dma_weights", dma_weights.exists(), str(dma_weights)))

    missing = []
    for name, ok, detail in checks:
        status = "ok" if ok else "missing"
        print(f"{status:8} {name}: {detail}")
        if not ok:
            missing.append(name)

    imports = list(REQUIRED_IMPORTS)
    if config.get("dma", {}).get("ml") == "gbm":
        imports.extend(DMA_IMPORTS)
    if config.get("reid", {}).get("enabled", False) and config.get("reid", {}).get("backend") == "fast":
        imports.extend(FAST_REID_IMPORTS)

    for module_name in imports:
        ok = has_import(module_name)
        status = "ok" if ok else "missing"
        print(f"{status:8} import {module_name}")
        if not ok:
            missing.append(module_name)

    if missing:
        print()
        print("Real tracker is not ready yet.")
        print("Install the missing Python packages and download the configured weights.")
        raise SystemExit(1)

    print()
    print("Real tracker environment looks ready.")


if __name__ == "__main__":
    main()
