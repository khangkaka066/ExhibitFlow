#!/usr/bin/env python3
"""Create a short local clip from a longer video for CPU smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a short video clip.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.frames <= 0:
        raise SystemExit("--frames must be positive")
    if args.start_frame < 0:
        raise SystemExit("--start-frame must be non-negative")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output file already exists: {args.output}")

    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("OpenCV is required: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise SystemExit(f"could not open input video: {args.input}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    if width <= 0 or height <= 0:
        raise SystemExit("input video has invalid dimensions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise SystemExit(f"could not open output video: {args.output}")

    written = 0
    while written < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1

    cap.release()
    writer.release()

    if written == 0:
        raise SystemExit("no frames were written")

    print(f"wrote {written} frame(s) to {args.output}")


if __name__ == "__main__":
    main()
