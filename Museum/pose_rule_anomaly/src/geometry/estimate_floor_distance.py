#!/usr/bin/env python3
"""Prototype relative floor distance using an approximate fixed-camera homography."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "floor_geometry.json"
POSES = ROOT / "outputs" / "poses.jsonl"
EVENTS = ROOT / "outputs" / "events.jsonl"
MASK_ROOT = ROOT / "outputs" / "artifact_masks_v1"
OUT_DIR = ROOT / "outputs" / "floor_geometry_trial"
OUT_ROWS = OUT_DIR / "floor_distances.jsonl"
OUT_REPORT = ROOT / "reports" / "floor_geometry_trial.md"
DATASET = ROOT.parent / "dataset" / "MuseumVisitors"


def transform(point: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    result = cv2.perspectiveTransform(np.asarray([[point]], dtype=np.float32), matrix)[0, 0]
    return float(result[0]), float(result[1])


def foot_anchor(row: dict) -> tuple[tuple[float, float], str]:
    keypoints = {item["name"]: item for item in row.get("keypoints", [])}
    ankles = [
        keypoints[name] for name in ("left_ankle", "right_ankle")
        if name in keypoints and keypoints[name]["score"] >= 0.5
    ]
    if ankles:
        return ((sum(item["x"] for item in ankles) / len(ankles),
                 sum(item["y"] for item in ankles) / len(ankles)), "ankles")
    box = row["box"]
    return (((box[0] + box[2]) / 2, box[3]), "bbox_bottom")


def load_artifact_anchors() -> dict[tuple[int, str], tuple[float, float]]:
    grouped: dict[tuple[int, str], list[np.ndarray]] = {}
    for line in (MASK_ROOT / "mask_manifest.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = (int(row["camera_id"]), row["rule_group"])
        grouped.setdefault(key, []).append(
            np.asarray(Image.open(MASK_ROOT / row["mask_path"]).convert("L")) > 0
        )
    anchors = {}
    for key, masks in grouped.items():
        union = np.logical_or.reduce(masks)
        ys, xs = np.where(union)
        bottom = int(ys.max())
        bottom_x = xs[ys >= bottom - 4]
        anchors[key] = (float(np.median(bottom_x)), float(bottom))
    return anchors


def locate_image(row: dict) -> Path:
    path = Path(row.get("image_path", ""))
    if path.is_file():
        return path
    matches = list((DATASET / "frames" / str(row["camera_id"])).rglob(f"*_{row['frame_id']}.jpg"))
    if not matches:
        raise FileNotFoundError(f"frame {row['camera_id']}/{row['frame_id']}")
    return matches[0]


def main() -> None:
    config = json.loads(CONFIG.read_text())
    matrices = {
        int(camera): cv2.getPerspectiveTransform(
            np.asarray(item["image_points"], dtype=np.float32),
            np.asarray(item["floor_points_assumed_m"], dtype=np.float32),
        )
        for camera, item in config["cameras"].items()
    }
    anchors = load_artifact_anchors()
    cases = list(config["diagnostic_cases"])
    for event in (json.loads(line) for line in EVENTS.read_text().splitlines() if line.strip()):
        rule = event.get("evidence", {}).get("rule", {})
        cases.append({
            "case_id": event["event_id"], "camera_id": event["camera_id"],
            "segment_id": event["segment_id"], "track_id": event["track_id"],
            "frame_id": event["evidence"]["review_frame_id"],
            "artwork_id": rule["artwork_id"], "known_review": "pending",
        })
    wanted = {(int(c["camera_id"]), int(c["track_id"]), int(c["frame_id"])) for c in cases}
    pose_rows = {}
    for line in POSES.read_text().splitlines():
        row = json.loads(line)
        key = (int(row["camera_id"]), int(row["track_id"]), int(row["frame_id"]))
        if key in wanted:
            pose_rows[key] = row
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for case in cases:
        camera = int(case["camera_id"])
        key = (camera, int(case["track_id"]), int(case["frame_id"]))
        row = pose_rows[key]
        person_px, source = foot_anchor(row)
        artwork_px = anchors[(camera, case["artwork_id"])]
        person_floor = transform(person_px, matrices[camera])
        artwork_floor = transform(artwork_px, matrices[camera])
        distance = math.dist(person_floor, artwork_floor)
        reach_limit = float(config["assumed_maximum_reachable_distance_m"])
        result = {
            **case,
            "person_floor_anchor_px": [round(v, 3) for v in person_px],
            "person_floor_anchor_source": source,
            "artwork_floor_anchor_px": [round(v, 3) for v in artwork_px],
            "person_floor_assumed_m_xy": [round(v, 6) for v in person_floor],
            "artwork_floor_assumed_m_xy": [round(v, 6) for v in artwork_floor],
            "assumed_floor_distance_m": round(distance, 6),
            "assumed_maximum_reachable_distance_m": reach_limit,
            "touch_reachable_under_assumption": distance <= reach_limit,
            "calibration_status": config["calibration_status"],
            "operational_gate_allowed": False,
        }
        results.append(result)
        image = Image.open(locate_image(row)).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line((person_px, artwork_px), fill=(255, 220, 0), width=5)
        for point, color in ((person_px, (255, 40, 40)), (artwork_px, (30, 255, 120))):
            x, y = point
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline=(0, 0, 0), width=2)
        draw.rectangle((0, 0, 720, 58), fill=(8, 15, 30))
        draw.text((10, 8), f"{case['case_id']} | assumed floor distance={distance:.2f} m", fill="white")
        draw.text((10, 31), "ASSUMED AB=10m, AC=12m - NOT FIELD-CALIBRATED / NOT AN ALERT GATE", fill=(255, 200, 80))
        image.save(OUT_DIR / f"{case['case_id']}.jpg", quality=90)
    OUT_ROWS.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results))
    lines = [
        "# Floor geometry trial\n\n",
        "This trial assumes the observed floor quadrilateral is 10 m wide (AB) and 12 m deep (AC). Values are approximate meters, not field measurements.\n\n",
        "Point convention: A = near-left, B = near-right, C = far-left, D = far-right; homography input order is A-B-D-C.\n\n",
        "| Case | Review | Camera/track/frame | Assumed distance | Reachable at 0.65 m? |\n",
        "|---|---|---|---:|---|\n",
    ]
    for item in results:
        lines.append(
            f"| `{item['case_id']}` | {item['known_review']} | "
            f"{item['camera_id']}/{item['track_id']}/{item['frame_id']} | {item['assumed_floor_distance_m']:.2f} m | "
            f"{'yes' if item['touch_reachable_under_assumption'] else 'no'} |\n"
        )
    lines += [
        "\nThe known false positive and the remaining candidate are compared only to test whether floor geometry separates them. "
        "The 0.65 m reach threshold is diagnostic only. No operational threshold is activated until real dimensions are provided.\n"
    ]
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {len(results)} cases -> {OUT_ROWS}")
    for item in results:
        print(item["case_id"], item["assumed_floor_distance_m"], "m", item["known_review"])


if __name__ == "__main__":
    main()
