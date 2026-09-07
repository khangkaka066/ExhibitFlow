#!/usr/bin/env python3
"""Run the ByteTrack-DMA-LTC repo and export ExhibitFlow JSONL."""

from __future__ import annotations

import argparse
import glob
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CONFIG = Path("configs/bytetrack_dma_caviar.json")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real ByteTrack-DMA-LTC tracking on a video and convert results."
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sequence-id", help="Defaults to the video stem.")
    parser.add_argument("--camera-id", default="cam_01")
    parser.add_argument("--model-repo", type=Path)
    parser.add_argument("--exp-file")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=("cpu", "gpu"), help="Model inference device.")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download the configured ByteTrack checkpoint if it is missing.",
    )
    parser.add_argument("--weights-name", default="mot17_x")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != "0.1":
        raise SystemExit(f"unsupported config schema_version in {path}")
    if config.get("backend") != "bytetrack_video_demo":
        raise SystemExit(f"unsupported backend in {path}: {config.get('backend')}")
    return config


def resolve_child_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def snapshot_result_files(model_repo: Path) -> set[Path]:
    pattern = str(model_repo / "YOLOX_outputs" / "*" / "track_vis" / "*.txt")
    return {Path(path) for path in glob.glob(pattern)}


def newest_result_file(model_repo: Path, before: set[Path]) -> Path:
    after = snapshot_result_files(model_repo)
    created = sorted(after - before, key=lambda path: path.stat().st_mtime, reverse=True)
    if created:
        return created[0]

    all_results = sorted(after, key=lambda path: path.stat().st_mtime, reverse=True)
    if not all_results:
        raise SystemExit("ByteTrack finished but no result .txt file was found.")
    return all_results[0]


def build_demo_command(args: argparse.Namespace, config: dict, model_repo: Path, checkpoint: Path) -> list[str]:
    exp_file = args.exp_file or config["exp_file"]
    exp_path = resolve_child_path(model_repo, exp_file)
    device = args.device or config.get("device", "cpu")

    command = [
        sys.executable,
        "tools/demo_track.py",
        "video",
        "-f",
        str(exp_path),
        "-c",
        str(checkpoint),
        "--path",
        str(args.video.resolve()),
        "--device",
        device,
    ]

    numeric_flags = {
        "--fps": config.get("fps"),
        "--tsize": config.get("tsize"),
        "--conf": config.get("conf"),
        "--nms": config.get("nms"),
        "--track_thresh": config.get("track_thresh"),
        "--track_buffer": config.get("track_buffer"),
        "--match_thresh": config.get("match_thresh"),
        "--min_box_area": config.get("min_box_area"),
    }
    for flag, value in numeric_flags.items():
        if value is not None:
            command.extend([flag, str(value)])

    if config.get("save_result", True):
        command.append("--save_result")

    for extra in config.get("extra_args", []):
        command.append(str(extra))

    return command


def build_convert_command(args: argparse.Namespace, config: dict, result_txt: Path) -> list[str]:
    sequence_id = args.sequence_id or args.video.stem
    return [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "convert_bytetrack_results.py"),
        "--mot-txt",
        str(result_txt),
        "--video",
        str(args.video),
        "--output",
        str(args.output),
        "--sequence-id",
        sequence_id,
        "--camera-id",
        args.camera_id,
        "--fps",
        str(config.get("fps", 25)),
        "--input-frame-base",
        "0",
        *(["--overwrite"] if args.overwrite else []),
    ]


def main() -> None:
    args = make_parser().parse_args()
    args.config = resolve_project_path(args.config)
    args.video = resolve_project_path(args.video)
    args.output = resolve_project_path(args.output)
    config = load_config(args.config)

    model_repo = args.model_repo or Path(config["model_repo"])
    model_repo = resolve_project_path(model_repo).resolve()
    if not model_repo.exists():
        raise SystemExit(
            f"model repo is missing: {model_repo}\n"
            "Clone it with: git clone https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker.git "
            "external/ByteTrack-DMA-LTC-Motion-Tracker"
        )
    if not args.video.exists():
        raise SystemExit(f"video is missing: {args.video}")

    checkpoint_value = args.checkpoint or config["checkpoint"]
    checkpoint = resolve_child_path(model_repo, checkpoint_value)
    checkpoint_missing = not checkpoint.exists()
    if checkpoint_missing and args.download_weights and not args.dry_run:
        download_script = model_repo / "tools" / "download_bytetrack_weights.py"
        subprocess.run(
            [
                sys.executable,
                str(download_script),
                "--name",
                args.weights_name,
                "--output-dir",
                str(checkpoint.parent),
            ],
            cwd=model_repo,
            check=True,
        )
        checkpoint_missing = not checkpoint.exists()

    if checkpoint_missing and not args.dry_run:
        raise SystemExit(
            f"checkpoint is missing: {checkpoint}\n"
            "Run again with --download-weights, or from the model repo run: "
            "python tools/download_bytetrack_weights.py --name mot17_x"
        )

    demo_command = build_demo_command(args, config, model_repo, checkpoint)
    if args.dry_run:
        if checkpoint_missing:
            print(
                f"warning: checkpoint is missing: {checkpoint}\n"
                "run again with --download-weights, or from the model repo run: "
                "python tools/download_bytetrack_weights.py --name mot17_x",
                file=sys.stderr,
            )
        print(" ".join(shlex.quote(part) for part in demo_command))
        return

    before = snapshot_result_files(model_repo)
    started_at = time.time()
    subprocess.run(demo_command, cwd=model_repo, check=True)
    result_txt = newest_result_file(model_repo, before)
    if result_txt.stat().st_mtime + 1 < started_at:
        raise SystemExit(f"latest ByteTrack result looks stale: {result_txt}")

    convert_command = build_convert_command(args, config, result_txt)
    subprocess.run(convert_command, cwd=PROJECT_ROOT, check=True)
    print(f"ByteTrack result: {result_txt}")
    print(f"ExhibitFlow JSONL: {args.output}")


if __name__ == "__main__":
    main()
