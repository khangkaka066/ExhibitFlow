#!/usr/bin/env python3
"""Convert ByteTrack MOT-style results into ExhibitFlow track JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class VideoMeta:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class TrackRow:
    frame_id: int
    track_id: int
    bbox: list[float]
    score: float


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert ByteTrack MOT txt output to ExhibitFlow JSONL."
    )
    parser.add_argument("--mot-txt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sequence-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--video", type=Path, help="Optional video path for metadata.")
    parser.add_argument("--fps", type=float, help="Frames per second if --video is omitted.")
    parser.add_argument("--width", type=int, help="Image width if --video is omitted.")
    parser.add_argument("--height", type=int, help="Image height if --video is omitted.")
    parser.add_argument("--frame-count", type=int, help="Frame count if --video is omitted.")
    parser.add_argument(
        "--input-frame-base",
        choices=("auto", "0", "1"),
        default="auto",
        help="Frame index base used by the MOT txt file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser


def read_video_meta(video_path: Path) -> VideoMeta:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required to read video metadata. Install the real tracker "
            "environment or pass --fps --width --height --frame-count."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise SystemExit(f"invalid video metadata for: {video_path}")

    return VideoMeta(width=width, height=height, fps=fps, frame_count=frame_count)


def read_manual_meta(args: argparse.Namespace) -> VideoMeta:
    missing = [
        name
        for name in ("fps", "width", "height", "frame_count")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(
            "missing video metadata fields: "
            + ", ".join("--" + name.replace("_", "-") for name in missing)
        )

    meta = VideoMeta(
        width=args.width,
        height=args.height,
        fps=args.fps,
        frame_count=args.frame_count,
    )
    if meta.width <= 0 or meta.height <= 0 or meta.fps <= 0 or meta.frame_count <= 0:
        raise SystemExit("video metadata values must be positive")
    return meta


def read_mot_rows(path: Path) -> list[TrackRow]:
    rows: list[TrackRow] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for line_number, raw in enumerate(reader, start=1):
            if not raw:
                continue
            if len(raw) < 7:
                raise SystemExit(f"{path}:{line_number}: expected at least 7 columns")

            try:
                frame_id = int(float(raw[0]))
                track_id = int(float(raw[1]))
                x = float(raw[2])
                y = float(raw[3])
                width = float(raw[4])
                height = float(raw[5])
                score = float(raw[6])
            except ValueError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid numeric value") from exc

            if frame_id < 0 or track_id < 0:
                raise SystemExit(f"{path}:{line_number}: frame_id and track_id must be non-negative")
            if not all(math.isfinite(value) for value in (x, y, width, height, score)):
                raise SystemExit(f"{path}:{line_number}: bbox and score must be finite")
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise SystemExit(f"{path}:{line_number}: bbox must have positive size inside image space")

            rows.append(
                TrackRow(
                    frame_id=frame_id,
                    track_id=track_id,
                    bbox=[x, y, width, height],
                    score=score,
                )
            )
    return rows


def normalize_frame_base(rows: Iterable[TrackRow], requested_base: str) -> tuple[list[TrackRow], int]:
    materialized = list(rows)
    if not materialized:
        return materialized, 0

    min_frame = min(row.frame_id for row in materialized)
    if requested_base == "auto":
        base = 0 if min_frame == 0 else 1
    else:
        base = int(requested_base)

    normalized = [
        TrackRow(
            frame_id=row.frame_id - base,
            track_id=row.track_id,
            bbox=row.bbox,
            score=row.score,
        )
        for row in materialized
    ]
    if any(row.frame_id < 0 for row in normalized):
        raise SystemExit("input frame base makes at least one frame negative")
    return normalized, base


def compact_number(value: float) -> int | float:
    if float(value).is_integer():
        return int(value)
    return round(value, 6)


def build_frame_record(
    *,
    frame_id: int,
    timestamp_ms: int,
    meta: VideoMeta,
    sequence_id: str,
    camera_id: str,
    tracks: list[TrackRow],
) -> dict:
    track_records = []
    for row in sorted(tracks, key=lambda item: item.track_id):
        x, y, width, height = row.bbox
        if x + width > meta.width or y + height > meta.height:
            raise SystemExit(
                f"track {row.track_id} on frame {frame_id} is outside image bounds "
                f"{meta.width}x{meta.height}"
            )

        track_records.append(
            {
                "track_id": row.track_id,
                "bbox": [compact_number(value) for value in row.bbox],
                "point_image": [
                    compact_number(x + width / 2.0),
                    compact_number(y + height),
                ],
                "score": compact_number(row.score),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "sequence_id": sequence_id,
        "camera_id": camera_id,
        "frame_id": frame_id,
        "timestamp_ms": timestamp_ms,
        "image_size": {"width": meta.width, "height": meta.height},
        "tracks": track_records,
    }


def convert(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output file already exists: {args.output}")

    meta = read_video_meta(args.video) if args.video else read_manual_meta(args)
    rows, _ = normalize_frame_base(read_mot_rows(args.mot_txt), args.input_frame_base)

    rows_by_frame: dict[int, list[TrackRow]] = defaultdict(list)
    for row in rows:
        rows_by_frame[row.frame_id].append(row)

    max_row_frame = max(rows_by_frame, default=-1)
    total_frames = max(meta.frame_count, max_row_frame + 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_output.open("w", encoding="utf-8") as handle:
        for frame_id in range(total_frames):
            timestamp_ms = int(round(frame_id * 1000.0 / meta.fps))
            record = build_frame_record(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                meta=meta,
                sequence_id=args.sequence_id,
                camera_id=args.camera_id,
                tracks=rows_by_frame.get(frame_id, []),
            )
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    tmp_output.replace(args.output)


def main() -> None:
    args = make_parser().parse_args()
    convert(args)


if __name__ == "__main__":
    main()
