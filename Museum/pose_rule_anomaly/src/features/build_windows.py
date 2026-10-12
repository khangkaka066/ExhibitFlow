"""Build fixed-time pose/trajectory windows without assuming a fixed FPS.

Run from Museum/pose_rule_anomaly/:
    python -m src.features.build_windows

The output stores NaN for unavailable numeric values and a separate keypoint
mask. Missing keypoints are never represented as fake (0, 0) coordinates.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
POSES = REPO_ROOT / "outputs" / "poses.jsonl"
OUT_WINDOWS = REPO_ROOT / "outputs" / "windows.npz"
OUT_EXCLUDED = REPO_ROOT / "outputs" / "windows_excluded.jsonl"
OUT_REPORT = REPO_ROOT / "reports" / "feature_windows.md"

WINDOW_SECONDS = 2.5
STRIDE_SECONDS = 1.0
N_STEPS = 16
MAX_INTERPOLATION_GAP_SECONDS = 0.75
MIN_VALID_STEPS = 0.80
MIN_KEYPOINT_VISIBILITY = 0.60
KEYPOINT_VISIBLE_SCORE = 0.5
IMAGE_WIDTH = 1280.0
IMAGE_HEIGHT = 800.0

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_frame_times() -> dict[tuple[int, str, int], float]:
    """Estimate sub-second time using observed frames within each source second."""
    by_segment: dict[tuple[int, str], list[dict]] = defaultdict(list)
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            by_segment[(int(row["camera_id"]), row["segment_id"])].append(row)

    result: dict[tuple[int, str, int], float] = {}
    for (camera_id, segment_id), rows in by_segment.items():
        rows.sort(key=lambda r: int(r["frame_id"]))
        by_second: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_second[int(row["timestamp_source"])].append(row)
        for second, second_rows in by_second.items():
            n = len(second_rows)
            for rank, row in enumerate(second_rows):
                # Timestamp filenames only have second precision. Distribute the
                # actually observed frames inside that second; do not assume FPS.
                time_value = second + (rank + 0.5) / n
                result[(camera_id, segment_id, int(row["frame_id"]))] = time_value
    return result


def load_tracks(frame_times: dict) -> dict[tuple[int, str, int], list[dict]]:
    tracks: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    with POSES.open() as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            frame_key = (int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))
            if frame_key not in frame_times:
                raise KeyError(f"pose line {line_number} has no matching manifest frame: {frame_key}")
            row["time"] = frame_times[frame_key]
            key = (frame_key[0], frame_key[1], int(row["track_id"]))
            tracks[key].append(row)
    for rows in tracks.values():
        rows.sort(key=lambda r: (r["time"], int(r["frame_id"])))
    return tracks


def row_features(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if row["quality_status"] not in ("ok", "low_confidence") or not row["keypoints"] or not row["box"]:
        return None

    keypoints = {point["name"]: point for point in row["keypoints"]}
    if any(name not in keypoints for name in KEYPOINT_NAMES):
        return None
    box = np.asarray(row["box"], dtype=np.float32)
    x1, y1, x2, y2 = box
    width = max(float(x2 - x1), 1.0)
    height = max(float(y2 - y1), 1.0)

    coords = np.asarray([[keypoints[name]["x"], keypoints[name]["y"]] for name in KEYPOINT_NAMES], dtype=np.float32)
    confidence = np.asarray([keypoints[name]["score"] for name in KEYPOINT_NAMES], dtype=np.float32)
    visible = confidence >= KEYPOINT_VISIBLE_SCORE

    hips = [KEYPOINT_NAMES.index("left_hip"), KEYPOINT_NAMES.index("right_hip")]
    if visible[hips].all():
        origin = coords[hips].mean(axis=0)
    else:
        origin = np.asarray([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale = max(height, math.sqrt(width * height), 1.0)
    normalized = (coords - origin) / scale
    normalized[~visible] = np.nan

    shoulders = [KEYPOINT_NAMES.index("left_shoulder"), KEYPOINT_NAMES.index("right_shoulder")]
    torso_angle = np.nan
    hip_height = np.nan
    if visible[hips].all():
        hip_center = coords[hips].mean(axis=0)
        hip_height = (hip_center[1] - y1) / height
        if visible[shoulders].all():
            shoulder_center = coords[shoulders].mean(axis=0)
            vector = shoulder_center - hip_center
            torso_angle = math.atan2(float(vector[0]), float(-vector[1]))

    bbox_features = np.asarray([
        ((x1 + x2) / 2.0) / IMAGE_WIDTH,
        ((y1 + y2) / 2.0) / IMAGE_HEIGHT,
        width / IMAGE_WIDTH,
        height / IMAGE_HEIGHT,
        width / height,
        hip_height,
        torso_angle,
    ], dtype=np.float32)
    return normalized.reshape(-1), confidence, bbox_features


def interpolate_pair(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    valid = np.isfinite(a) & np.isfinite(b)
    result = np.full_like(a, np.nan)
    result[valid] = a[valid] + (b[valid] - a[valid]) * fraction
    return result


def resample_track(rows: list[dict], grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    usable = []
    for row in rows:
        values = row_features(row)
        if values is not None:
            usable.append((row["time"], *values))

    pose = np.full((len(grid), 34), np.nan, dtype=np.float32)
    confidence = np.full((len(grid), 17), np.nan, dtype=np.float32)
    bbox = np.full((len(grid), 7), np.nan, dtype=np.float32)
    step_valid = np.zeros(len(grid), dtype=bool)
    if len(usable) < 2:
        return pose, confidence, bbox, step_valid

    times = np.asarray([item[0] for item in usable])
    for index, target in enumerate(grid):
        right = int(np.searchsorted(times, target, side="left"))
        if right == 0 or right == len(times):
            continue
        left = right - 1
        gap = times[right] - times[left]
        if gap <= 0 or gap > MAX_INTERPOLATION_GAP_SECONDS:
            continue
        fraction = float((target - times[left]) / gap)
        pose[index] = interpolate_pair(usable[left][1], usable[right][1], fraction)
        confidence[index] = interpolate_pair(usable[left][2], usable[right][2], fraction)
        bbox[index] = interpolate_pair(usable[left][3], usable[right][3], fraction)
        step_valid[index] = np.isfinite(bbox[index, :5]).all()
    return pose, confidence, bbox, step_valid


def feature_names() -> list[str]:
    names = [f"pose_{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y")]
    names += [f"keypoint_confidence_{name}" for name in KEYPOINT_NAMES]
    names += [f"pose_velocity_{name}_{axis}" for name in KEYPOINT_NAMES for axis in ("x", "y")]
    names += ["bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height", "bbox_aspect", "hip_height", "torso_angle"]
    names += ["bbox_velocity_x", "bbox_velocity_y"]
    return names


def build_window(rows: list[dict], start: float) -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, str]:
    grid = np.linspace(start, start + WINDOW_SECONDS, N_STEPS, dtype=np.float64)
    pose, confidence, bbox, step_valid = resample_track(rows, grid)
    valid_ratio = float(step_valid.mean())
    if valid_ratio < MIN_VALID_STEPS:
        return None, None, "insufficient_temporal_coverage"

    keypoint_mask = np.isfinite(pose.reshape(N_STEPS, 17, 2)).all(axis=2)
    visibility = float(keypoint_mask[step_valid].mean()) if step_valid.any() else 0.0
    if visibility < MIN_KEYPOINT_VISIBILITY:
        return None, None, "insufficient_keypoint_visibility"

    dt = WINDOW_SECONDS / (N_STEPS - 1)
    pose_velocity = np.full_like(pose, np.nan)
    pose_velocity[1:] = (pose[1:] - pose[:-1]) / dt
    bbox_velocity = np.full((N_STEPS, 2), np.nan, dtype=np.float32)
    bbox_velocity[1:] = (bbox[1:, :2] - bbox[:-1, :2]) / dt
    features = np.concatenate([pose, confidence, pose_velocity, bbox, bbox_velocity], axis=1).astype(np.float32)
    quality = {"valid_step_ratio": valid_ratio, "keypoint_visibility": visibility}
    return features, keypoint_mask, quality


def main() -> None:
    frame_times = load_frame_times()
    tracks = load_tracks(frame_times)
    names = feature_names()
    windows, masks, metadata, exclusions = [], [], [], []
    reason_counts: Counter = Counter()

    print(f"loaded {len(frame_times)} frames and {len(tracks)} tracks")
    for track_index, ((camera_id, segment_id, track_id), rows) in enumerate(sorted(tracks.items()), 1):
        start_time = rows[0]["time"]
        end_time = rows[-1]["time"]
        first_start = math.ceil(start_time / STRIDE_SECONDS) * STRIDE_SECONDS
        candidate = first_start
        while candidate + WINDOW_SECONDS <= end_time:
            result, mask, detail = build_window(rows, candidate)
            if result is None:
                reason_counts[detail] += 1
                exclusions.append({
                    "camera_id": camera_id, "segment_id": segment_id, "track_id": track_id,
                    "start_time": candidate, "end_time": candidate + WINDOW_SECONDS,
                    "reason": detail, "source": "real_museum",
                })
            else:
                in_window = [row for row in rows if candidate <= row["time"] <= candidate + WINDOW_SECONDS]
                windows.append(result)
                masks.append(mask)
                metadata.append({
                    "camera_id": camera_id, "segment_id": segment_id, "track_id": track_id,
                    "start_time": candidate, "end_time": candidate + WINDOW_SECONDS,
                    "start_frame": min(int(row["frame_id"]) for row in in_window),
                    "end_frame": max(int(row["frame_id"]) for row in in_window),
                    "valid_step_ratio": detail["valid_step_ratio"],
                    "keypoint_visibility": detail["keypoint_visibility"],
                    "source": "real_museum",
                })
            candidate += STRIDE_SECONDS
        if track_index % 25 == 0 or track_index == len(tracks):
            print(f"[{track_index}/{len(tracks)}] tracks; accepted={len(windows)}, excluded={len(exclusions)}")

    if not windows:
        raise RuntimeError("no valid windows were produced")
    x = np.stack(windows)
    mask_array = np.stack(masks)
    meta_json = np.asarray([json.dumps(row, separators=(",", ":")) for row in metadata])
    np.savez_compressed(
        OUT_WINDOWS,
        features=x,
        keypoint_mask=mask_array,
        metadata_json=meta_json,
        feature_names=np.asarray(names),
        sample_times=np.linspace(0.0, WINDOW_SECONDS, N_STEPS, dtype=np.float32),
    )
    with OUT_EXCLUDED.open("w") as f:
        for row in exclusions:
            f.write(json.dumps(row) + "\n")

    by_camera = Counter(row["camera_id"] for row in metadata)
    lines = [
        "# T05 temporal feature windows\n\n",
        f"- Accepted windows: {len(metadata)}\n",
        f"- Excluded windows: {len(exclusions)}\n",
        f"- Window: {WINDOW_SECONDS:.1f}s, stride: {STRIDE_SECONDS:.1f}s, resampled steps: {N_STEPS}\n",
        f"- Feature tensor: `{tuple(x.shape)}` (`float32`; missing values are NaN)\n",
        f"- Keypoint mask tensor: `{tuple(mask_array.shape)}`\n",
        "- Time base: source filename seconds with observed frames distributed within each second; no fixed FPS assumption.\n",
        "- Normalization: keypoints relative to hip center (bbox center fallback), scaled by person size.\n",
        "- Quality gates: interpolation gap <= 0.75s, >=80% valid time steps, >=60% visible keypoints.\n",
        "- Windows never cross a segment or track boundary.\n\n",
        "## Accepted by camera\n",
    ]
    for camera_id in (1, 2, 3):
        lines.append(f"- Camera {camera_id}: {by_camera[camera_id]}\n")
    lines.append("\n## Exclusion reasons\n")
    for reason, count in sorted(reason_counts.items()):
        lines.append(f"- {reason}: {count}\n")
    lines.append("\n## Feature groups\n\n")
    lines.append("Relative pose (34), keypoint confidence (17), pose velocity (34), bbox/shape/body geometry (7), and bbox-center velocity (2). Absolute image position is retained only as normalized bbox trajectory for motion modeling; ROI membership remains separate for rules.\n")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {OUT_WINDOWS} ({x.shape})")
    print(f"wrote {OUT_EXCLUDED}")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
