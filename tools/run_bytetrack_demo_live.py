#!/usr/bin/env python3
"""Run ByteTrack's demo script with a live annotated OpenCV window."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch ByteTrack demo_track.py with live display enabled."
    )
    parser.add_argument("--model-repo", required=True, type=Path)
    parser.add_argument("--window-name", default="ExhibitFlow Tracking")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("demo_args", nargs=argparse.REMAINDER)
    return parser


class LiveVideoWriter:
    def __init__(self, original_writer_factory, window_name: str, display_scale: float, *args, **kwargs):
        self._writer = original_writer_factory(*args, **kwargs)
        self._window_name = window_name
        self._display_scale = display_scale

    def write(self, frame):
        import cv2

        shown = frame
        if self._display_scale != 1.0:
            shown = cv2.resize(
                frame,
                None,
                fx=self._display_scale,
                fy=self._display_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        cv2.imshow(self._window_name, shown)
        return self._writer.write(frame)

    def __getattr__(self, name: str):
        return getattr(self._writer, name)


def install_live_display(window_name: str, display_scale: float) -> None:
    if display_scale <= 0:
        raise SystemExit("--display-scale must be positive")

    import cv2

    original_video_writer = cv2.VideoWriter
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def video_writer_factory(*args, **kwargs):
        return LiveVideoWriter(original_video_writer, window_name, display_scale, *args, **kwargs)

    cv2.VideoWriter = video_writer_factory


def main() -> None:
    args = make_parser().parse_args()
    if args.demo_args and args.demo_args[0] == "--":
        args.demo_args = args.demo_args[1:]
    if not args.demo_args:
        raise SystemExit("missing ByteTrack demo args after --")

    model_repo = args.model_repo.resolve()
    demo_script = model_repo / "tools" / "demo_track.py"
    if not demo_script.exists():
        raise SystemExit(f"missing ByteTrack demo script: {demo_script}")

    sys.path.insert(0, str(model_repo))
    os.chdir(model_repo)
    install_live_display(args.window_name, args.display_scale)
    sys.argv = [str(demo_script), *args.demo_args]

    try:
        runpy.run_path(str(demo_script), run_name="__main__")
    finally:
        try:
            import cv2

            cv2.destroyWindow(args.window_name)
        except Exception:
            pass


if __name__ == "__main__":
    main()
