"""T01: normalize raw MuseumVisitors annotations into a manifest + jsonl.

Run from Museum/pose_rule_anomaly/:
    python -m src.ingest.parse_annotations
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # Museum/pose_rule_anomaly
DATASET_ROOT = REPO_ROOT.parent / "dataset" / "MuseumVisitors"
ANNOTATIONS_PATH = DATASET_ROOT / "annotations.dat"
FRAMES_ROOT = DATASET_ROOT / "frames"

OUT_MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
OUT_JSONL = REPO_ROOT / "outputs" / "annotations.jsonl"
OUT_REPORT = REPO_ROOT / "reports" / "t01_validation.md"

IMG_W, IMG_H = 1280, 800

COLUMNS = [
    "visitor_id", "camera_id", "frame_id",
    "bb_x", "bb_y", "bb_w", "bb_h",
    "bbV_x", "bbV_y", "bbV_w", "bbV_h",
    "gaze_x", "gaze_y",
    "filename_raw", "opera_id", "group_id",
]

# empirically: cam 3 rows point at "4/..." instead of "3/..."; fix read-time only.
CAMERA3_BUG_PREFIX = "4/"
CAMERA3_FIX_PREFIX = "3/"

# frame_id gap <= this is considered the same recording segment.
SEGMENT_GAP_THRESHOLD = 1
SHORT_SEGMENT_MAX_FRAMES = 5

FILENAME_TS_RE = re.compile(r"(\d{2})h(\d{2})m(\d{2})s")


def fix_camera3_path(camera_id: int, filename_raw: str) -> str:
    if camera_id == 3 and filename_raw.startswith(CAMERA3_BUG_PREFIX):
        return CAMERA3_FIX_PREFIX + filename_raw[len(CAMERA3_BUG_PREFIX):]
    return filename_raw


def parse_timestamp_source(filename_raw: str) -> int | None:
    m = FILENAME_TS_RE.search(filename_raw)
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def assign_segments(frame_ids: pd.Series) -> dict[int, str]:
    """Given all frame_ids for one camera, split into segments on frame_id gaps."""
    unique_sorted = sorted(frame_ids.unique())
    segments: list[list[int]] = []
    current = [unique_sorted[0]]
    for prev, cur in zip(unique_sorted, unique_sorted[1:]):
        if cur - prev <= SEGMENT_GAP_THRESHOLD:
            current.append(cur)
        else:
            segments.append(current)
            current = [cur]
    segments.append(current)

    frame_to_seg: dict[int, str] = {}
    for idx, seg in enumerate(segments, start=1):
        seg_id = f"seg{idx:02d}"
        for f in seg:
            frame_to_seg[f] = seg_id
    return frame_to_seg, segments


def main() -> None:
    assert ANNOTATIONS_PATH.exists(), f"missing {ANNOTATIONS_PATH}"

    df = pd.read_csv(
        ANNOTATIONS_PATH,
        header=None,
        names=COLUMNS,
        dtype={
            "visitor_id": int, "camera_id": int, "frame_id": int,
            "bb_x": int, "bb_y": int, "bb_w": int, "bb_h": int,
            "bbV_x": int, "bbV_y": int, "bbV_w": int, "bbV_h": int,
            "gaze_x": float, "gaze_y": float,
            "filename_raw": str, "opera_id": int, "group_id": int,
        },
        skipinitialspace=True,
    )

    df["filename_fixed"] = [
        fix_camera3_path(cam, fn) for cam, fn in zip(df["camera_id"], df["filename_raw"])
    ]
    df["image_path"] = df["filename_fixed"].apply(lambda fn: str(FRAMES_ROOT / fn))
    df["timestamp_source"] = df["filename_raw"].apply(parse_timestamp_source)
    df["source"] = "real_museum"

    # per-camera segment assignment
    df["segment_id"] = ""
    segment_tables: dict[int, list[list[int]]] = {}
    for cam, sub in df.groupby("camera_id"):
        frame_to_seg, segments = assign_segments(sub["frame_id"])
        segment_tables[cam] = segments
        df.loc[sub.index, "segment_id"] = [
            f"cam{cam}_{frame_to_seg[f]}" for f in sub["frame_id"]
        ]

    # ---- validation ----
    report_lines: list[str] = ["# T01 validation report\n"]

    report_lines.append(f"Total annotation rows: {len(df)}\n")

    # 1. image existence
    unique_images = df[["camera_id", "image_path"]].drop_duplicates()
    missing = [p for p in unique_images["image_path"] if not Path(p).exists()]
    report_lines.append(f"\n## Image existence\nUnique referenced images: {len(unique_images)}\n")
    report_lines.append(f"Missing images: {len(missing)}\n")
    if missing:
        report_lines.append("First 20 missing:\n")
        for p in missing[:20]:
            report_lines.append(f"- {p}\n")

    # 2. bbox sanity
    def bad_bbox(row) -> bool:
        if row["bb_w"] <= 0 or row["bb_h"] <= 0 or row["bbV_w"] <= 0 or row["bbV_h"] <= 0:
            return True
        for x, y in ((row["bb_x"], row["bb_y"]), (row["bbV_x"], row["bbV_y"])):
            if not (0 <= x <= IMG_W and 0 <= y <= IMG_H):
                return True
        return False

    bad_mask = df.apply(bad_bbox, axis=1)
    bad_rows = df[bad_mask]
    report_lines.append(f"\n## Bbox sanity (bounds 0..{IMG_W} x 0..{IMG_H})\n")
    report_lines.append(f"Rows with invalid bbox: {len(bad_rows)}\n")
    if len(bad_rows):
        report_lines.append("Rows (visitor_id, camera_id, frame_id, bb_x, bb_y, bbV_x, bbV_y):\n")
        for _, r in bad_rows.iterrows():
            report_lines.append(
                f"- v{r.visitor_id} cam{r.camera_id} f{r.frame_id} "
                f"bb=({r.bb_x},{r.bb_y}) bbV=({r.bbV_x},{r.bbV_y})\n"
            )

    # 3. duplicate (visitor_id, frame_id) within same camera+segment
    dup_mask = df.duplicated(subset=["camera_id", "segment_id", "visitor_id", "frame_id"], keep=False)
    dup_rows = df[dup_mask]
    report_lines.append(f"\n## Duplicate (visitor_id, frame_id) within camera+segment\n")
    report_lines.append(f"Duplicate rows: {len(dup_rows)}\n")

    # 4. segment table
    report_lines.append("\n## Segments\n")
    total_segments = 0
    for cam in sorted(segment_tables):
        report_lines.append(f"\n### Camera {cam}\n")
        for idx, seg in enumerate(segment_tables[cam], start=1):
            total_segments += 1
            n = len(seg)
            flag = " SHORT (exclude from sequence/window learning)" if n <= SHORT_SEGMENT_MAX_FRAMES else ""
            report_lines.append(
                f"- seg{idx:02d}: frame {seg[0]}..{seg[-1]} ({n} frames){flag}\n"
            )
    report_lines.append(f"\nTotal segments across all cameras: {total_segments} "
                         f"(PLAN.md expects 11)\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("".join(report_lines))

    # ---- outputs ----
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        df[["camera_id", "segment_id", "frame_id", "timestamp_source", "image_path", "source"]]
        .drop_duplicates(subset=["camera_id", "segment_id", "frame_id"])
        .sort_values(["camera_id", "frame_id"])
    )
    manifest.to_csv(OUT_MANIFEST, index=False)

    with OUT_JSONL.open("w") as f:
        for _, row in df.iterrows():
            record = {
                "visitor_id": int(row.visitor_id),
                "camera_id": int(row.camera_id),
                "frame_id": int(row.frame_id),
                "bb_x": int(row.bb_x), "bb_y": int(row.bb_y),
                "bb_w": int(row.bb_w), "bb_h": int(row.bb_h),
                "bbV_x": int(row.bbV_x), "bbV_y": int(row.bbV_y),
                "bbV_w": int(row.bbV_w), "bbV_h": int(row.bbV_h),
                "gaze_x": float(row.gaze_x), "gaze_y": float(row.gaze_y),
                "filename_raw": row.filename_raw,
                "opera_id": int(row.opera_id),
                "group_id": int(row.group_id),
                "segment_id": row.segment_id,
                "timestamp_source": row.timestamp_source,
                "image_path": row.image_path,
                "source": row.source,
            }
            f.write(json.dumps(record) + "\n")

    print(f"wrote {len(manifest)} manifest rows -> {OUT_MANIFEST}")
    print(f"wrote {len(df)} annotation rows -> {OUT_JSONL}")
    print(f"wrote validation report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
