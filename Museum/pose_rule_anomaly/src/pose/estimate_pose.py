"""T02: run a pretrained pose model once per unique frame, IoU-match each
original annotation row to the best detection, and emit poses + quality gate.

Run from Museum/pose_rule_anomaly/:
    python -m src.pose.estimate_pose
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor

REPO_ROOT = Path(__file__).resolve().parents[2]  # Museum/pose_rule_anomaly
DATASET_ROOT = REPO_ROOT.parent / "dataset" / "MuseumVisitors"

IN_MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
IN_ANNOTATIONS = REPO_ROOT / "outputs" / "annotations.jsonl"
OUT_POSES = REPO_ROOT / "outputs" / "poses.jsonl"
OUT_PREVIEW_DIR = REPO_ROOT / "outputs" / "pose_previews"
OUT_REPORT = REPO_ROOT / "reports" / "pose_quality.md"

DETECTION_SCORE_OK = 0.5
IOU_MATCH_THRESHOLD = 0.3
KEYPOINT_VISIBLE_SCORE = 0.5
PREVIEW_SAMPLES_PER_CAMERA = 9

SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
]


def resolve_image_path(raw_path: str) -> str:
    """Resolve a manifest path against this machine's dataset root.

    T01 outputs may have been generated on another machine and therefore can
    contain an absolute path such as ``/Users/.../frames/1/...jpg``.  Keep a
    valid path unchanged; otherwise rebuild it from the portion below
    ``frames/`` so the same payload works on Colab and locally.
    """
    path = Path(raw_path)
    if path.is_file():
        return str(path)

    parts = path.parts
    try:
        frames_index = len(parts) - 1 - parts[::-1].index("frames")
    except ValueError as exc:
        raise ValueError(f"image_path does not contain a 'frames' component: {raw_path}") from exc

    return str(DATASET_ROOT / "frames" / Path(*parts[frames_index + 1 :]))


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_annotations_by_image() -> dict[str, list[dict]]:
    by_image: dict[str, list[dict]] = defaultdict(list)
    with IN_ANNOTATIONS.open() as f:
        for line in f:
            row = json.loads(line)
            row["image_path"] = resolve_image_path(row["image_path"])
            by_image[row["image_path"]].append(row)
    return by_image


def load_frame_list() -> list[dict]:
    frames = []
    with IN_MANIFEST.open() as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.rstrip("\n").split(",")
            frame = dict(zip(header, parts))
            frame["image_path"] = resolve_image_path(frame["image_path"])
            frames.append(frame)
    return frames


def validate_frame_paths(frames: list[dict]) -> None:
    missing = [frame["image_path"] for frame in frames if not Path(frame["image_path"]).is_file()]
    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:5])
        raise FileNotFoundError(
            f"Could not find {len(missing)}/{len(frames)} frame images under {DATASET_ROOT}.\n"
            f"First missing paths:\n{examples}"
        )


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def classify_row(
    row_box: tuple[float, float, float, float],
    detections: list[dict],
) -> tuple[str, dict | None, float | None]:
    if not detections:
        return "no_person_detected", None, None

    best_det, best_iou = None, 0.0
    for det in detections:
        score = iou(row_box, det["box"])
        if score > best_iou:
            best_iou, best_det = score, det

    if best_det is None or best_iou < IOU_MATCH_THRESHOLD:
        return "no_match", None, None

    status = "ok" if best_det["score"] >= DETECTION_SCORE_OK else "low_confidence"
    return status, best_det, best_iou


def draw_overlay(image: Image.Image, keypoints: list[dict] | None, box: tuple | None, status: str) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    color = {"ok": (0, 220, 0), "low_confidence": (230, 170, 0),
             "no_match": (220, 0, 0), "no_person_detected": (120, 120, 120)}[status]
    if box:
        draw.rectangle(box, outline=color, width=2)
    if keypoints:
        pts = [(kp["x"], kp["y"]) if kp["score"] >= KEYPOINT_VISIBLE_SCORE else None for kp in keypoints]
        for a, b in SKELETON_EDGES:
            if pts[a] and pts[b]:
                draw.line([pts[a], pts[b]], fill=color, width=2)
        for p in pts:
            if p:
                draw.ellipse([p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3], fill=color)
    draw.text((5, 5), status, fill=color)
    return img


def main() -> None:
    device = pick_device()
    print(f"device = {device}")

    annotations_by_image = load_annotations_by_image()
    frames = load_frame_list()
    validate_frame_paths(frames)
    print(f"unique frames: {len(frames)}, annotation rows: {sum(len(v) for v in annotations_by_image.values())}")

    weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    keypoint_names = list(weights.meta["keypoint_names"])
    model = keypointrcnn_resnet50_fpn(weights=weights)
    model.eval().to(device)

    OUT_POSES.parent.mkdir(parents=True, exist_ok=True)
    OUT_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # metrics accumulators
    status_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    wrist_visible: dict[int, int] = defaultdict(int)
    ankle_visible: dict[int, int] = defaultdict(int)
    pose_eligible: dict[int, int] = defaultdict(int)  # ok + low_confidence rows
    track_frame_sets: dict[tuple, set[int]] = defaultdict(set)
    preview_candidates: dict[int, list[tuple]] = defaultdict(list)  # cam -> [(n_people, area, frame_row)]

    poses_written = 0
    with OUT_POSES.open("w") as out_f:
        for i, frame in enumerate(frames):
            camera_id = int(frame["camera_id"])
            image_path = frame["image_path"]
            rows = annotations_by_image.get(image_path, [])

            try:
                image = Image.open(image_path)
            except FileNotFoundError:
                image = None

            detections: list[dict] = []
            if image is not None:
                tensor = to_tensor(image).to(device)
                with torch.no_grad():
                    output = model([tensor])[0]
                boxes = output["boxes"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                kpts = output["keypoints"].cpu().numpy()
                kpt_scores = output["keypoints_scores"].cpu().numpy()
                for b, lbl, sc, kp, kps in zip(boxes, labels, scores, kpts, kpt_scores):
                    if lbl != 1:
                        continue
                    detections.append({
                        "box": tuple(float(x) for x in b),
                        "score": float(sc),
                        "keypoints": [
                            {"name": keypoint_names[j], "x": float(kp[j][0]), "y": float(kp[j][1]),
                             "score": float(kps[j])}
                            for j in range(len(keypoint_names))
                        ],
                    })

            if image is not None:
                preview_candidates[camera_id].append(
                    (len(detections), max((d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]) for d in detections) if detections else 0.0, frame)
                )

            for row in rows:
                row_box = (
                    float(row["bb_x"]), float(row["bb_y"]),
                    float(row["bb_x"]) + float(row["bb_w"]),
                    float(row["bb_y"]) + float(row["bb_h"]),
                )
                status, det, matched_iou = classify_row(row_box, detections) if image is not None else ("no_person_detected", None, None)

                record = {
                    "visitor_id": row["visitor_id"],
                    "camera_id": row["camera_id"],
                    "frame_id": row["frame_id"],
                    "segment_id": row["segment_id"],
                    "track_id": row["visitor_id"],
                    "quality_status": status,
                    "detection_score": det["score"] if det else None,
                    "iou": matched_iou,
                    "box": list(det["box"]) if det else None,
                    "keypoints": det["keypoints"] if det else None,
                    "source": row["source"],
                }
                out_f.write(json.dumps(record) + "\n")
                poses_written += 1

                status_counts[camera_id][status] += 1
                track_frame_sets[(row["camera_id"], row["segment_id"], row["visitor_id"])].add(int(row["frame_id"]))

                if status in ("ok", "low_confidence"):
                    pose_eligible[camera_id] += 1
                    kp_by_name = {kp["name"]: kp for kp in det["keypoints"]}
                    if any(kp_by_name[n]["score"] >= KEYPOINT_VISIBLE_SCORE for n in ("left_wrist", "right_wrist")):
                        wrist_visible[camera_id] += 1
                    if any(kp_by_name[n]["score"] >= KEYPOINT_VISIBLE_SCORE for n in ("left_ankle", "right_ankle")):
                        ankle_visible[camera_id] += 1

            if i % 500 == 0:
                print(f"[{i}/{len(frames)}] frames processed, {poses_written} pose rows written")

    print(f"wrote {poses_written} pose rows -> {OUT_POSES}")

    # ---- build overlay previews: sample across person-count / bbox-area spread,
    # re-running inference just for the handful of sampled frames so we can draw
    # the actual matched skeleton + bbox + quality_status per row ----
    random.seed(0)
    for camera_id, candidates in preview_candidates.items():
        candidates_sorted = sorted(candidates, key=lambda c: (c[0], c[1]))
        n = len(candidates_sorted)
        if n == 0:
            continue
        idxs = sorted(set(int(p * (n - 1)) for p in
                           [k / (PREVIEW_SAMPLES_PER_CAMERA - 1) for k in range(PREVIEW_SAMPLES_PER_CAMERA)]))
        cam_dir = OUT_PREVIEW_DIR / f"cam{camera_id}"
        cam_dir.mkdir(parents=True, exist_ok=True)
        for idx in idxs:
            _, _, frame = candidates_sorted[idx]
            image_path = frame["image_path"]
            rows = annotations_by_image.get(image_path, [])
            try:
                image = Image.open(image_path)
            except FileNotFoundError:
                continue

            tensor = to_tensor(image).to(device)
            with torch.no_grad():
                output = model([tensor])[0]
            boxes = output["boxes"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            kpts = output["keypoints"].cpu().numpy()
            kpt_scores = output["keypoints_scores"].cpu().numpy()
            detections = []
            for b, lbl, sc, kp, kps in zip(boxes, labels, scores, kpts, kpt_scores):
                if lbl != 1:
                    continue
                detections.append({
                    "box": tuple(float(x) for x in b),
                    "score": float(sc),
                    "keypoints": [
                        {"name": keypoint_names[j], "x": float(kp[j][0]), "y": float(kp[j][1]),
                         "score": float(kps[j])}
                        for j in range(len(keypoint_names))
                    ],
                })

            img = image.convert("RGB")
            for row in rows:
                row_box = (
                    float(row["bb_x"]), float(row["bb_y"]),
                    float(row["bb_x"]) + float(row["bb_w"]),
                    float(row["bb_y"]) + float(row["bb_h"]),
                )
                status, det, _ = classify_row(row_box, detections)
                img = draw_overlay(
                    img,
                    det["keypoints"] if det else None,
                    tuple(det["box"]) if det else row_box,
                    status,
                )
            out_path = cam_dir / f"{frame['frame_id']}.jpg"
            img.save(out_path)

    # ---- report ----
    lines = ["# T02 pose quality report\n"]
    lines.append(
        "\nModel: torchvision `keypointrcnn_resnet50_fpn` (COCO, 17 keypoints), "
        "one forward pass per unique frame; each annotation row matched to the "
        "best-IoU detection (threshold 0.3). `track_id` reuses the dataset's "
        "`visitor_id` — no new tracker was introduced, so there are no ID-switch "
        "events to count. The 'track break' metric below is instead a "
        "frame-coverage continuity measure per (camera, segment, visitor_id).\n"
    )

    for camera_id in sorted(status_counts):
        counts = status_counts[camera_id]
        total = sum(counts.values())
        lines.append(f"\n## Camera {camera_id}\n")
        lines.append(f"Total annotation rows: {total}\n")
        for status in ("ok", "low_confidence", "no_match", "no_person_detected"):
            c = counts.get(status, 0)
            lines.append(f"- {status}: {c} ({100.0 * c / total:.1f}%)\n")

        eligible = pose_eligible[camera_id]
        if eligible:
            wv = wrist_visible[camera_id]
            av = ankle_visible[camera_id]
            lines.append(f"\nWrist visibility rate (of ok/low_confidence rows): {wv}/{eligible} ({100.0*wv/eligible:.1f}%)\n")
            lines.append(f"Ankle visibility rate (of ok/low_confidence rows): {av}/{eligible} ({100.0*av/eligible:.1f}%)\n")
        else:
            lines.append("\nNo rows with a matched pose on this camera.\n")

    lines.append("\n## Track frame-coverage continuity\n")
    for camera_id in sorted(status_counts):
        coverages = []
        for (cam, seg, vid), frame_ids in track_frame_sets.items():
            if int(cam) != camera_id:
                continue
            span = max(frame_ids) - min(frame_ids) + 1
            coverages.append(len(frame_ids) / span)
        if coverages:
            mean_cov = sum(coverages) / len(coverages)
            lines.append(f"- Camera {camera_id}: {len(coverages)} tracks, mean coverage {mean_cov:.2f}, min {min(coverages):.2f}\n")

    lines.append("\n## Reliability verdict per camera\n")
    for camera_id in sorted(status_counts):
        eligible = pose_eligible[camera_id]
        total = sum(status_counts[camera_id].values())
        frac_ok = eligible / total if total else 0.0
        wv_rate = (wrist_visible[camera_id] / eligible) if eligible else 0.0
        verdict = "reliable" if frac_ok > 0.7 and wv_rate > 0.5 else "unreliable / use with caution"
        lines.append(f"- Camera {camera_id}: {verdict} (pose-match rate {frac_ok:.2f}, wrist visibility {wv_rate:.2f})\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
