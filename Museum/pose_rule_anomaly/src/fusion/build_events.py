"""Fuse T04 rules with T06 per-camera anomaly percentiles.

Run from Museum/pose_rule_anomaly/:
    python -m src.fusion.build_events
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
RULES = REPO_ROOT / "outputs" / "rule_candidates.jsonl"
SCORES = REPO_ROOT / "outputs" / "anomaly" / "anomaly_scores.jsonl"
POSES = REPO_ROOT / "outputs" / "poses.jsonl"
MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
THRESHOLDS = REPO_ROOT / "configs" / "rule_thresholds.json"
OUT_EVENTS = REPO_ROOT / "outputs" / "events.jsonl"
OUT_PREVIEWS = REPO_ROOT / "outputs" / "event_previews"
OUT_REPORT = REPO_ROOT / "reports" / "fusion_events.md"


def overlaps(a: dict, b: dict) -> bool:
    return a["start_time"] <= b["end_time"] and b["start_time"] <= a["end_time"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def attach_source_evidence(events: list[dict]) -> None:
    frame_paths = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            path = Path(row["image_path"])
            parts = path.parts
            frames_index = len(parts) - 1 - parts[::-1].index("frames")
            portable = str(Path(*parts[frames_index:]))
            frame_paths[(int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))] = portable
    for event in events:
        rule_evidence = event.get("evidence", {}).get("rule", {})
        frame = int(rule_evidence.get("evidence_frame_id", (event["start_frame"] + event["end_frame"]) // 2))
        key = (event["camera_id"], event["segment_id"], frame)
        if key not in frame_paths:
            candidates = [item for item in frame_paths if item[:2] == key[:2]]
            key = min(candidates, key=lambda item: abs(item[2] - frame))
            frame = key[2]
        event["evidence"]["review_frame_id"] = frame
        event["evidence"]["review_image_path"] = frame_paths[key]


def merge_windows(rows: list[dict], event_type: str, merge_gap: float) -> list[dict]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: (row["camera_id"], row["segment_id"], row["track_id"], row["start_time"]))
    merged = []
    current = None
    for row in rows:
        key = (row["camera_id"], row["segment_id"], row["track_id"])
        if current is None or current["key"] != key or row["start_time"] > current["end_time"] + merge_gap:
            if current:
                merged.append(current)
            current = {
                "key": key, "camera_id": row["camera_id"], "segment_id": row["segment_id"], "track_id": row["track_id"],
                "start_time": row["start_time"], "end_time": row["end_time"],
                "start_frame": row["start_frame"], "end_frame": row["end_frame"],
                "event_type": event_type, "members": [row],
            }
        else:
            current["end_time"] = max(current["end_time"], row["end_time"])
            current["end_frame"] = max(current["end_frame"], row["end_frame"])
            current["members"].append(row)
    if current:
        merged.append(current)
    return merged


def build_events() -> list[dict]:
    rules = [row for row in load_jsonl(RULES) if row["event_type"] == "touch_candidate"]
    scores = [
        row for row in load_jsonl(SCORES)
        if row.get("evidence_status") == "scored" and row.get("camera_percentile") is not None
    ]
    thresholds = json.loads(THRESHOLDS.read_text())
    anomaly_threshold = float(thresholds["fusion"]["unusual_motion_percentile"])
    merge_gap = float(thresholds["fusion"]["merge_gap_seconds"])

    events = []
    for rule in rules:
        matching_scores = [
            row for row in scores
            if int(row["camera_id"]) == int(rule["camera_id"])
            and row["segment_id"] == rule["segment_id"]
            and int(row["track_id"]) == int(rule["track_id"])
            and overlaps(row, rule)
        ]
        peak = max(matching_scores, key=lambda row: float(row["camera_percentile"]), default=None)
        events.append({
            "camera_id": rule["camera_id"], "segment_id": rule["segment_id"], "track_id": rule["track_id"],
            "start_time": rule["start_time"], "end_time": rule["end_time"],
            "start_frame": rule["start_frame"], "end_frame": rule["end_frame"],
            "event_type": rule["event_type"], "rule_score": rule["rule_score"],
            "anomaly_score": float(peak["anomaly_score"]) if peak else None,
            "camera_percentile": float(peak["camera_percentile"]) if peak else None,
            "pose_quality": rule["pose_quality"],
            "evidence": {
                "rule": rule["evidence"],
                "decision_basis": "touch_rule_with_anomaly_context" if peak else "touch_rule_only",
                "anomaly_model_used": peak is not None,
                "anomaly_context_windows": len(matching_scores),
            },
            "review_status": "pending", "roi_review_status": rule["roi_review_status"], "source": "real_museum",
        })

    high_windows = [
        row for row in scores
        if float(row["camera_percentile"]) >= anomaly_threshold
    ]
    anomaly_groups = merge_windows(high_windows, "unusual_motion_candidate", merge_gap)
    touch_events = [event for event in events if event["event_type"] == "touch_candidate"]
    for group in anomaly_groups:
        # A high score overlapping a touch event enriches that event; do not
        # create a duplicate unusual-motion alert for the same track/time.
        duplicate_of_touch = any(
            event["camera_id"] == group["camera_id"]
            and event["segment_id"] == group["segment_id"]
            and event["track_id"] == group["track_id"]
            and overlaps(event, group)
            for event in touch_events
        )
        if duplicate_of_touch:
            continue
        members = group["members"]
        peak = max(members, key=lambda row: float(row["camera_percentile"]))
        events.append({
            "camera_id": group["camera_id"], "segment_id": group["segment_id"], "track_id": group["track_id"],
            "start_time": group["start_time"], "end_time": group["end_time"],
            "start_frame": group["start_frame"], "end_frame": group["end_frame"],
            "event_type": "unusual_motion_candidate", "rule_score": None,
            "anomaly_score": float(peak["anomaly_score"]),
            "camera_percentile": float(peak["camera_percentile"]),
            "pose_quality": {
                "valid_step_ratio_min": min(float(row["valid_step_ratio"]) for row in members),
                "keypoint_visibility_min": min(float(row["keypoint_visibility"]) for row in members),
            },
            "evidence": {
                "decision_basis": "normal_only_isolation_forest_outlier",
                "anomaly_model_used": True,
                "threshold_camera_percentile": anomaly_threshold,
                "scored_windows": len(members),
                "interpretation": "unusual motion only; behavior/violation type is unknown",
            },
            "review_status": "pending", "roi_review_status": "not_applicable", "source": "real_museum",
        })

    events.sort(key=lambda row: (row["camera_id"], row["segment_id"], row["start_time"], row["track_id"], row["event_type"]))
    for index, event in enumerate(events, 1):
        event["event_id"] = f"evt_{index:06d}"
    return events


def render_previews(events: list[dict]) -> int:
    paths = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            paths[(int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))] = row["image_path"]
    wanted = []
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for event in events:
        if event["event_type"] != "insufficient_evidence":
            groups[(event["camera_id"], event["event_type"])].append(event)
    for _, rows in sorted(groups.items()):
        rows.sort(key=lambda row: ((row["camera_percentile"] or 0.0), (row["rule_score"] or 0.0)), reverse=True)
        wanted.extend(rows[:10])
    frame_keys = {(row["camera_id"], row["segment_id"], row["track_id"], (row["start_frame"] + row["end_frame"]) // 2) for row in wanted}
    boxes = {}
    with POSES.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["camera_id"], row["segment_id"], row["track_id"], row["frame_id"])
            if key in frame_keys:
                boxes[key] = row["box"]
    for index, event in enumerate(wanted):
        frame = (event["start_frame"] + event["end_frame"]) // 2
        path = paths.get((event["camera_id"], event["segment_id"], frame))
        if not path or not Path(path).is_file():
            continue
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        box = boxes.get((event["camera_id"], event["segment_id"], event["track_id"], frame))
        if box:
            draw.rectangle(box, outline=(0, 255, 0), width=5)
        label = f"{event['event_type']} rule={event['rule_score']} pct={event['camera_percentile']}"
        draw.text((8, 8), label, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        out_dir = OUT_PREVIEWS / f"cam{event['camera_id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        image.save(out_dir / f"{index:04d}_{event['event_id']}_{event['event_type']}.jpg", quality=90)
    return len(wanted)


def main() -> None:
    events = build_events()
    attach_source_evidence(events)
    with OUT_EVENTS.open("w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")
    preview_count = render_previews(events)
    counts = Counter((row["camera_id"], row["event_type"]) for row in events)
    lines = ["# T07 fused events\n\n", f"Total event records: {len(events)}\n\n"]
    for camera in (1, 2, 3):
        lines.append(f"## Camera {camera}\n")
        for event_type in ("touch_candidate", "unusual_motion_candidate"):
            lines.append(f"- {event_type}: {counts[(camera, event_type)]}\n")
        lines.append("\n")
    lines.append("T06 contributes `unusual_motion_candidate` events at the configured per-camera percentile threshold. This normal-only model flags outlier motion; it does not identify or confirm a violation. Touch remains a separate geometry/rule candidate. All events require review.\n")
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {len(events)} events -> {OUT_EVENTS}")
    print(f"selected {preview_count} event previews -> {OUT_PREVIEWS}")


if __name__ == "__main__":
    main()
