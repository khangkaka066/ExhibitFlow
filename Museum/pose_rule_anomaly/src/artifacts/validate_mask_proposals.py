#!/usr/bin/env python3
"""Validate the structure and safety status of artifact-mask results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    manifest = root / "mask_proposals.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    if len(rows) != args.expected_count:
        errors.append(f"expected {args.expected_count} proposals, found {len(rows)}")
    seen: set[tuple[int, str]] = set()
    for row in rows:
        key = (int(row["camera_id"]), row["artifact_id"])
        if key in seen:
            errors.append(f"duplicate proposal: {key}")
        seen.add(key)
        if row.get("approval_status") != "needs_review":
            errors.append(f"{key}: unsafe approval_status={row.get('approval_status')!r}")
        if row.get("source") != "real_museum":
            errors.append(f"{key}: source is not real_museum")
        mask_path = root / row["mask_path"]
        overlay_path = root / row["overlay_path"]
        if not mask_path.is_file() or not overlay_path.is_file():
            errors.append(f"{key}: missing mask or overlay")
            continue
        mask = np.asarray(Image.open(mask_path).convert("L")) > 0
        if mask.shape != (800, 1280):
            errors.append(f"{key}: mask shape is {mask.shape}, expected (800, 1280)")
        if not mask.any():
            errors.append(f"{key}: empty mask")
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"VALIDATION PASSED: {len(rows)} review-only proposals; no masks were auto-approved.")


if __name__ == "__main__":
    main()
