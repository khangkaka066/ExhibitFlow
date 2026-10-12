"""Generate reviewable T04 spatial-rule candidates from pose tracks.

Run from Museum/pose_rule_anomaly/:
    python -m src.rules.generate_candidates
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from src.features.build_windows import load_frame_times

REPO_ROOT = Path(__file__).resolve().parents[2]
POSES = REPO_ROOT / "outputs" / "poses.jsonl"
CONFIG_DIR = REPO_ROOT / "configs"
OUT_CANDIDATES = REPO_ROOT / "outputs" / "rule_candidates.jsonl"
OUT_PREVIEWS = REPO_ROOT / "outputs" / "rule_previews_v2"
OUT_REPORT = REPO_ROOT / "reports" / "rule_candidates.md"
DATASET_ROOT = REPO_ROOT.parent / "dataset" / "MuseumVisitors"
ARTIFACT_MASK_ROOT = REPO_ROOT / "outputs" / "artifact_masks_v1"
VISIBLE_SCORE = 0.5


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


def segment_distance(point: tuple[float, float], a: list[float], b: list[float]) -> float:
    px, py = point
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polygon_distance(point: tuple[float, float], polygon: list[list[float]]) -> float:
    if point_in_polygon(point, polygon):
        return 0.0
    return min(segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)]) for i in range(len(polygon)))


def load_artwork_masks() -> dict[tuple[int, str], dict]:
    """Load approved masks and union artifacts belonging to the same rule group."""
    manifest = ARTIFACT_MASK_ROOT / "mask_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Approved artifact masks are required: {manifest}. "
            "Run src.artifacts.finalize_reviewed_masks first."
        )
    grouped: dict[tuple[int, str], list[tuple[dict, np.ndarray]]] = defaultdict(list)
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("approval_status") != "approved":
            continue
        mask = np.asarray(Image.open(ARTIFACT_MASK_ROOT / row["mask_path"]).convert("L")) > 0
        grouped[(int(row["camera_id"]), row["rule_group"])].append((row, mask))
    registry = {}
    for key, members in grouped.items():
        union = np.logical_or.reduce([mask for _, mask in members])
        registry[key] = {
            "mask": union,
            "distance": cv2.distanceTransform(
                (~union).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ),
            "inside_distance": cv2.distanceTransform(
                union.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
            ),
            "artifact_ids": [row["artifact_id"] for row, _ in members],
            "mask_version": "artifact_masks_v1",
        }
    return registry


def load_inputs() -> tuple[dict, dict, dict, dict]:
    configs = {camera: json.loads((CONFIG_DIR / f"cam{camera}_roi.json").read_text()) for camera in (1, 2, 3)}
    thresholds = json.loads((CONFIG_DIR / "rule_thresholds.json").read_text())
    frame_times = load_frame_times()
    tracks: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    with POSES.open() as f:
        for line in f:
            row = json.loads(line)
            key = (int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))
            row["time"] = frame_times[key]
            tracks[(key[0], key[1], int(row["track_id"]))].append(row)
    for rows in tracks.values():
        rows.sort(key=lambda row: row["time"])
    return configs, thresholds, tracks, load_artwork_masks()


def pose_geometry(row: dict) -> dict | None:
    if row["quality_status"] not in ("ok", "low_confidence") or not row["box"] or not row["keypoints"]:
        return None
    keypoints = {point["name"]: point for point in row["keypoints"]}
    box = row["box"]
    height = max(float(box[3] - box[1]), 1.0)
    ankles = [keypoints[name] for name in ("left_ankle", "right_ankle") if keypoints[name]["score"] >= VISIBLE_SCORE]
    if ankles:
        anchor = (sum(p["x"] for p in ankles) / len(ankles), sum(p["y"] for p in ankles) / len(ankles))
        anchor_source = "ankles"
    else:
        anchor = ((box[0] + box[2]) / 2.0, box[3])
        anchor_source = "bbox_bottom"
    wrists = {
        name: (keypoints[name]["x"], keypoints[name]["y"])
        for name in ("left_wrist", "right_wrist")
        if keypoints[name]["score"] >= VISIBLE_SCORE
    }
    arms = {}
    for side in ("left", "right"):
        names = [f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist"]
        if all(name in keypoints and keypoints[name]["score"] >= VISIBLE_SCORE for name in names):
            arms[f"{side}_wrist"] = {
                joint: (keypoints[f"{side}_{joint}"]["x"], keypoints[f"{side}_{joint}"]["y"])
                for joint in ("shoulder", "elbow", "wrist")
            }
    return {
        "anchor": anchor, "anchor_source": anchor_source, "wrists": wrists,
        "arms": arms, "height": height,
    }


def sample_distance(field: np.ndarray, point: tuple[float, float]) -> float:
    height, width = field.shape
    x = max(0, min(width - 1, int(round(point[0]))))
    y = max(0, min(height - 1, int(round(point[1]))))
    return float(field[y, x])


def touch_geometry_metrics(row: dict, geometry: dict, wrist_name: str,
                           mask_geometry: dict, thresholds: dict) -> dict | None:
    """Reject 2D overlap unless the arm approaches the artifact boundary cleanly."""
    arm = geometry["arms"].get(wrist_name)
    if arm is None:
        return None
    height = geometry["height"]
    mask = mask_geometry["mask"]
    outside = mask_geometry["distance"]
    inside = mask_geometry["inside_distance"]
    wrist = arm["wrist"]
    wx = max(0, min(mask.shape[1] - 1, int(round(wrist[0]))))
    wy = max(0, min(mask.shape[0] - 1, int(round(wrist[1]))))
    wrist_inside = bool(mask[wy, wx])
    wrist_boundary = sample_distance(inside if wrist_inside else outside, wrist) / height
    wrist_inside_depth = sample_distance(inside, wrist) / height if wrist_inside else 0.0
    elbow_distance = sample_distance(outside, arm["elbow"]) / height
    shoulder_distance = sample_distance(outside, arm["shoulder"]) / height
    ex = max(0, min(mask.shape[1] - 1, int(round(arm["elbow"][0]))))
    ey = max(0, min(mask.shape[0] - 1, int(round(arm["elbow"][1]))))
    sx = max(0, min(mask.shape[1] - 1, int(round(arm["shoulder"][0]))))
    sy = max(0, min(mask.shape[0] - 1, int(round(arm["shoulder"][1]))))
    x1, y1, x2, y2 = [int(round(value)) for value in row["box"]]
    x1, x2 = max(0, x1), min(mask.shape[1], x2)
    y1, y2 = max(0, y1), min(mask.shape[0], y2)
    bbox_overlap = float(mask[y1:y2, x1:x2].mean()) if x2 > x1 and y2 > y1 else 1.0
    keypoints = {point["name"]: point for point in row["keypoints"]}
    side = wrist_name.removesuffix("_wrist")
    other_side = "right" if side == "left" else "left"
    shoulder_x = arm["shoulder"][0]
    other_shoulder_x = keypoints[f"{other_side}_shoulder"]["x"]
    shoulder_min, shoulder_max = sorted((shoulder_x, other_shoulder_x))
    lateral_reach = max(shoulder_min - wrist[0], wrist[0] - shoulder_max, 0.0) / height
    valid = (
        not bool(mask[ey, ex])
        and not bool(mask[sy, sx])
        and wrist_inside_depth <= thresholds["maximum_wrist_inside_depth_person_heights"]
        and elbow_distance >= thresholds["minimum_elbow_clearance_person_heights"]
        and shoulder_distance >= thresholds["minimum_shoulder_clearance_person_heights"]
        and elbow_distance - wrist_boundary >= thresholds["minimum_wrist_advantage_over_elbow_person_heights"]
        and lateral_reach >= thresholds["minimum_wrist_lateral_reach_person_heights"]
        and bbox_overlap <= thresholds["maximum_person_bbox_mask_overlap"]
    )
    return {
        "valid": valid,
        "wrist_boundary_distance": wrist_boundary,
        "wrist_inside_depth": wrist_inside_depth,
        "elbow_clearance": elbow_distance,
        "shoulder_clearance": shoulder_distance,
        "person_bbox_mask_overlap": bbox_overlap,
        "wrist_lateral_reach": lateral_reach,
    }


def barrier_candidates(key: tuple, rows: list[dict], config: dict, thresholds: dict) -> list[dict]:
    camera, segment, track = key
    results = []
    max_gap = thresholds["max_observation_gap_seconds"]
    required_allowed = thresholds["barrier"]["minimum_allowed_observations"]
    dwell = thresholds["barrier"]["restricted_dwell_seconds"]
    for barrier in config["barriers"]:
        allowed_count = 0
        entry = None
        last = None
        for row in rows:
            geometry = pose_geometry(row)
            if geometry is None:
                continue
            if last is not None and row["time"] - last["time"] > max_gap:
                allowed_count, entry = 0, None
            inside = point_in_polygon(geometry["anchor"], barrier["restricted_zone"])
            if not inside:
                allowed_count += 1
                entry = None
            elif allowed_count >= required_allowed:
                if entry is None:
                    entry = {"row": row, "geometry": geometry, "previous": last}
                if row["time"] - entry["row"]["time"] >= dwell:
                    start_row = entry["previous"] or entry["row"]
                    results.append({
                        "camera_id": camera, "segment_id": segment, "track_id": track,
                        "start_time": start_row["time"], "end_time": row["time"],
                        "start_frame": int(start_row["frame_id"]), "end_frame": int(row["frame_id"]),
                        "event_type": "cross_barrier_candidate", "rule_score": 1.0,
                        "evidence": {
                            "barrier_id": barrier["id"], "anchor_source": entry["geometry"]["anchor_source"],
                            "transition": "allowed->restricted", "restricted_dwell_seconds": row["time"] - entry["row"]["time"],
                        },
                        "pose_quality": entry["row"]["quality_status"], "source": "real_museum",
                        "review_status": "pending", "roi_review_status": config["assignment"]["review_status"],
                    })
                    allowed_count, entry = 0, None
            last = row
    return results


def artwork_candidates(key: tuple, rows: list[dict], config: dict, thresholds: dict,
                       artwork_masks: dict[tuple[int, str], dict]) -> list[dict]:
    camera, segment, track = key
    art = thresholds["artwork"]
    max_gap = thresholds["max_observation_gap_seconds"]
    cooldown = art["episode_cooldown_seconds"]
    results = []
    for artwork in config["artwork_rois"]:
        if not artwork.get("rule_enabled", True):
            continue
        mask_geometry = artwork_masks.get((camera, artwork["id"]))
        if mask_geometry is None:
            raise KeyError(f"No approved artifact mask for camera {camera}, rule group {artwork['id']}")
        episode = []
        previous_time = None

        def finish() -> None:
            nonlocal episode
            if not episode:
                return
            duration = episode[-1]["row"]["time"] - episode[0]["row"]["time"]
            if duration < art["minimum_near_dwell_seconds"]:
                episode = []
                return
            minimum = min(episode, key=lambda item: item["distance"])
            initial = max(item["distance"] for item in episode[: max(1, len(episode) // 3)])
            approach_delta = initial - minimum["distance"]
            contact_time = minimum["row"]["time"]
            motion_rows = [
                item for item in rows
                if item.get("box") and abs(item["time"] - contact_time) <= art["body_speed_context_seconds"]
            ]
            motion_rows = motion_rows or [item["row"] for item in episode]
            first_box, last_box = motion_rows[0]["box"], motion_rows[-1]["box"]
            first_center = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
            last_center = ((last_box[0] + last_box[2]) / 2, (last_box[1] + last_box[3]) / 2)
            elapsed = max(motion_rows[-1]["time"] - motion_rows[0]["time"], 1e-6)
            mean_height = sum(float(item["box"][3] - item["box"][1]) for item in motion_rows) / len(motion_rows)
            body_speed = math.dist(first_center, last_center) / mean_height / elapsed
            is_touch = (
                len(episode) >= art["minimum_touch_observations"]
                and minimum["distance"] <= art["touch_distance_person_heights"]
                and approach_delta >= art["minimum_approach_delta_person_heights"]
                and body_speed <= art["maximum_body_speed_person_heights_per_second"]
            )
            # Near-artwork is an internal state only. The operational output is
            # deliberately touch-only to avoid alerting on ordinary viewing.
            if not is_touch:
                episode = []
                return
            event_type = "touch_candidate"
            score_limit = art["touch_distance_person_heights"]
            proximity_score = max(0.0, min(1.0, 1.0 - minimum["distance"] / score_limit))
            approach_score = max(0.0, min(1.0, approach_delta / max(art["minimum_approach_delta_person_heights"], 1e-6)))
            score = 0.7 * proximity_score + 0.3 * approach_score
            results.append({
                "camera_id": camera, "segment_id": segment, "track_id": track,
                "start_time": episode[0]["row"]["time"], "end_time": episode[-1]["row"]["time"],
                "start_frame": int(episode[0]["row"]["frame_id"]), "end_frame": int(episode[-1]["row"]["frame_id"]),
                "event_type": event_type, "rule_score": score,
                "evidence": {
                    "artwork_id": artwork["id"], "wrist": minimum["wrist"],
                    "evidence_frame_id": int(minimum["row"]["frame_id"]),
                    "evidence_wrist_xy": [
                        round(float(minimum["geometry"]["wrists"][minimum["wrist"]][0]), 2),
                        round(float(minimum["geometry"]["wrists"][minimum["wrist"]][1]), 2),
                    ],
                    "minimum_distance_person_heights": minimum["distance"],
                    "approach_delta_person_heights": approach_delta,
                    "body_speed_person_heights_per_second": body_speed,
                    "contact_geometry": minimum["contact_geometry"],
                    "interaction_zone_required": True,
                    "geometry_source": mask_geometry["mask_version"],
                    "mask_artifact_ids": mask_geometry["artifact_ids"],
                    "occlusion": artwork.get("occlusion", "unknown"),
                    "claim_limit": "candidate_only_not_confirmed_contact",
                },
                "pose_quality": minimum["row"]["quality_status"], "source": "real_museum",
                "review_status": "pending", "roi_review_status": config["assignment"]["review_status"],
            })
            episode = []

        for row in rows:
            geometry = pose_geometry(row)
            if geometry is None or not geometry["wrists"]:
                if previous_time is not None and row["time"] - previous_time > max_gap:
                    finish()
                continue
            if not point_in_polygon(geometry["anchor"], artwork["interaction_zone"]):
                if episode:
                    finish()
                previous_time = row["time"]
                continue
            distances = []
            metrics_by_wrist = {}
            for name, point in geometry["wrists"].items():
                metrics = touch_geometry_metrics(row, geometry, name, mask_geometry, art)
                if metrics is not None and metrics["valid"]:
                    distances.append((metrics["wrist_boundary_distance"], name))
                    metrics_by_wrist[name] = metrics
            if not distances:
                if episode:
                    finish()
                previous_time = row["time"]
                continue
            distance, wrist = min(distances)
            near = distance <= art["near_distance_person_heights"]
            if episode and row["time"] - episode[-1]["row"]["time"] > cooldown:
                finish()
            if near:
                episode.append({
                    "row": row, "geometry": geometry, "distance": distance,
                    "wrist": wrist, "contact_geometry": metrics_by_wrist[wrist],
                })
            elif episode:
                finish()
            previous_time = row["time"]
        finish()
    return results


def render_preview(candidate: dict, config: dict, rows_by_frame: dict[int, dict], index: int,
                   artwork_masks: dict[tuple[int, str], dict]) -> None:
    frame_id = (candidate["start_frame"] + candidate["end_frame"]) // 2
    row = min(rows_by_frame.values(), key=lambda item: abs(int(item["frame_id"]) - frame_id))
    image_path = Path(row.get("image_path", ""))
    if not image_path.is_file():
        # Recover the portable path from the manifest-derived frame location.
        camera = candidate["camera_id"]
        matches = list((DATASET_ROOT / "frames" / str(camera)).rglob(f"*_{row['frame_id']}.jpg"))
        if not matches:
            return
        image_path = matches[0]
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if row.get("box"):
        draw.rectangle(row["box"], outline=(0, 255, 0), width=4)
    if candidate["event_type"] == "cross_barrier_candidate":
        item = next(x for x in config["barriers"] if x["id"] == candidate["evidence"]["barrier_id"])
        draw.polygon(item["restricted_zone"], outline=(255, 0, 255), width=4)
        draw.line(item["line"], fill=(255, 0, 0), width=5)
    else:
        item = next(x for x in config["artwork_rois"] if x["id"] == candidate["evidence"]["artwork_id"])
        mask_geometry = artwork_masks[(candidate["camera_id"], item["id"])]
        pixels = np.asarray(image).copy()
        mask = mask_geometry["mask"]
        pixels[mask] = (0.55 * pixels[mask] + 0.45 * np.array([0, 255, 120])).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        draw.polygon(item["interaction_zone"], outline=(50, 255, 120), width=4)
    draw.text((8, 8), f"{candidate['event_type']} score={candidate['rule_score']:.2f}", fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    out_dir = OUT_PREVIEWS / f"cam{candidate['camera_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / f"{index:04d}_{candidate['event_type']}_f{row['frame_id']}.jpg", quality=90)


def main() -> None:
    configs, thresholds, tracks, artwork_masks = load_inputs()
    candidates = []
    for index, (key, rows) in enumerate(sorted(tracks.items()), 1):
        config = configs[key[0]]
        if config["rules_enabled"].get("cross_barrier", False):
            candidates.extend(barrier_candidates(key, rows, config, thresholds))
        if config["rules_enabled"].get("near_artwork", False):
            candidates.extend(artwork_candidates(key, rows, config, thresholds, artwork_masks))
        if index % 25 == 0 or index == len(tracks):
            print(f"[{index}/{len(tracks)}] tracks; candidates={len(candidates)}")
    candidates.sort(key=lambda item: (item["camera_id"], item["segment_id"], item["start_time"], item["track_id"], item["event_type"]))
    with OUT_CANDIDATES.open("w") as f:
        for item in candidates:
            f.write(json.dumps(item) + "\n")

    # Render a bounded, deterministic review set: top 10 per camera/event type.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in candidates:
        groups[(item["camera_id"], item["event_type"])].append(item)
    preview_index = 0
    for (camera, _), items in sorted(groups.items()):
        for item in sorted(items, key=lambda value: value["rule_score"], reverse=True)[:10]:
            rows = tracks[(item["camera_id"], item["segment_id"], item["track_id"])]
            render_preview(item, configs[camera], {int(row["frame_id"]): row for row in rows}, preview_index, artwork_masks)
            preview_index += 1

    counts = Counter((item["camera_id"], item["event_type"]) for item in candidates)
    lines = ["# T04 spatial rule candidates\n\n", f"Total candidates: {len(candidates)}\n\n"]
    for camera in (1, 2, 3):
        lines.append(f"## Camera {camera}\n")
        for event_type in ("touch_candidate",):
            lines.append(f"- {event_type}: {counts[(camera, event_type)]}\n")
        lines.append("\n")
    lines.append("Operational output is touch-only. Near-artwork remains an internal state and is not emitted. Artwork distance uses the reviewed `artifact_masks_v1` mask set, constrained by each camera's interaction zone. `touch_candidate` still requires review and does not confirm physical contact.\n")
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {len(candidates)} candidates -> {OUT_CANDIDATES}")
    print(f"wrote {preview_index} previews -> {OUT_PREVIEWS}")


if __name__ == "__main__":
    main()
