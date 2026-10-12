#!/usr/bin/env python3
"""Create reusable museum backgrounds from the MuseumVisitors frame dataset.

The source data is never modified. People are excluded with the provided person
boxes and the background is estimated with a temporal median inside each
contiguous camera segment.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=61)
    parser.add_argument("--min-segment-frames", type=int, default=100)
    parser.add_argument("--bbox-margin", type=float, default=0.10)
    return parser.parse_args()


def load_annotations(dataset: Path):
    boxes = defaultdict(list)
    with (dataset / "annotations.dat").open(newline="") as handle:
        for row in csv.reader(handle):
            rel = row[13]
            # Published camera-3 annotations incorrectly point at directory 4.
            if row[1] == "3" and rel.startswith("4/"):
                rel = "3/" + rel[2:]
            boxes[rel].append(tuple(map(int, row[3:7])))
    return boxes


def contiguous_segments(paths: list[Path]):
    ordered = sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    segments = []
    current = [ordered[0]]
    previous = int(ordered[0].stem.rsplit("_", 1)[1])
    for path in ordered[1:]:
        frame_id = int(path.stem.rsplit("_", 1)[1])
        if frame_id != previous + 1:
            segments.append(current)
            current = []
        current.append(path)
        previous = frame_id
    segments.append(current)
    return segments


def evenly_sample(paths: list[Path], maximum: int) -> list[Path]:
    count = min(maximum, len(paths))
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[index] for index in sorted(set(indices.tolist()))]


def expanded_box(box, width, height, margin):
    x, y, w, h = box
    mx, my = round(w * margin), round(h * margin)
    return (
        max(0, x - mx),
        max(0, y - my),
        min(width, x + w + mx),
        min(height, y + h + my),
    )


def masked_temporal_median(images, relpaths, boxes, margin, strip_height=80):
    height, width = images[0].shape[:2]
    result = np.zeros_like(images[0])
    coverage = np.zeros((height, width), dtype=np.uint8)

    for top in range(0, height, strip_height):
        bottom = min(height, top + strip_height)
        strip = np.stack([im[top:bottom] for im in images]).astype(np.float32)
        valid = np.ones(strip.shape[:3], dtype=bool)

        for index, relpath in enumerate(relpaths):
            for box in boxes.get(relpath, []):
                x1, y1, x2, y2 = expanded_box(box, width, height, margin)
                iy1, iy2 = max(y1, top) - top, min(y2, bottom) - top
                if iy1 < iy2:
                    valid[index, iy1:iy2, x1:x2] = False

        work = np.where(valid[..., None], strip, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            estimate = np.nanmedian(work, axis=0)
        fallback = np.median(strip, axis=0)
        missing = np.isnan(estimate[..., 0])
        estimate[missing] = fallback[missing]
        result[top:bottom] = np.clip(estimate, 0, 255).astype(np.uint8)
        coverage[top:bottom] = np.clip(valid.sum(axis=0) * 255 / len(images), 0, 255)

    # Pixels hidden by a person in every sample are unreliable. Inpaint only
    # small holes; the coverage image remains available for quality filtering.
    no_observation = (coverage == 0).astype(np.uint8) * 255
    if no_observation.any():
        result = cv2.inpaint(result, no_observation, 5, cv2.INPAINT_TELEA)
    return result, coverage


def main():
    args = parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    boxes = load_annotations(dataset)
    manifest_rows = []

    for camera_id in ("1", "2", "3"):
        paths = list((dataset / "frames" / camera_id).rglob("*.jpg"))
        segments = contiguous_segments(paths)
        kept_index = 0
        for original_index, segment in enumerate(segments, 1):
            if len(segment) < args.min_segment_frames:
                continue
            kept_index += 1
            selected = evenly_sample(segment, args.max_samples)
            images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in selected]
            if any(image is None for image in images):
                raise RuntimeError(f"Failed to read an image in camera {camera_id}")
            relpaths = [str(path.relative_to(dataset / "frames")) for path in selected]
            background, coverage = masked_temporal_median(
                images, relpaths, boxes, args.bbox_margin
            )

            segment_name = f"cam{camera_id}_segment{original_index:02d}"
            segment_dir = output / segment_name
            segment_dir.mkdir(parents=True, exist_ok=True)
            background_path = segment_dir / "background_masked_median.jpg"
            coverage_path = segment_dir / "background_coverage.png"
            cv2.imwrite(str(background_path), background, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(coverage_path), coverage)

            cleanest = min(
                selected,
                key=lambda path: len(
                    boxes[str(path.relative_to(dataset / "frames"))]
                ),
            )
            cleanest_path = segment_dir / "least_people_source.jpg"
            shutil.copy2(cleanest, cleanest_path)

            manifest_rows.append(
                {
                    "segment_id": segment_name,
                    "camera_id": camera_id,
                    "source_frames": len(segment),
                    "sampled_frames": len(selected),
                    "first_frame": segment[0].name,
                    "last_frame": segment[-1].name,
                    "background": str(background_path.relative_to(output)),
                    "coverage": str(coverage_path.relative_to(output)),
                    "least_people_source": str(cleanest_path.relative_to(output)),
                    "least_people_count": len(
                        boxes[str(cleanest.relative_to(dataset / "frames"))]
                    ),
                }
            )

    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Created {len(manifest_rows)} background sets in {output}")


if __name__ == "__main__":
    main()
