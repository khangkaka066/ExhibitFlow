#!/usr/bin/env python3
"""Render one self-contained evidence image per pending fused event."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EVENTS = ROOT / "outputs" / "events.jsonl"
POSES = ROOT / "outputs" / "poses.jsonl"
MANIFEST = ROOT / "outputs" / "manifest.csv"
MASK_ROOT = ROOT / "outputs" / "artifact_masks_v1"
OUT_DIR = ROOT / "outputs" / "event_evidence"


def load_masks() -> dict[int, list[dict]]:
    masks: dict[int, list[dict]] = defaultdict(list)
    for line in (MASK_ROOT / "mask_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("approval_status") != "approved":
            continue
        mask = np.asarray(Image.open(MASK_ROOT / row["mask_path"]).convert("L")) > 0
        ys, xs = np.where(mask)
        masks[int(row["camera_id"])].append({
            "artifact_id": row["artifact_id"], "rule_group": row["rule_group"], "mask": mask,
            "label_xy": (int(np.median(xs)), int(ys.min())) if len(xs) else (0, 0),
        })
    return masks


def main() -> None:
    all_events = [json.loads(line) for line in EVENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Render every fused record, including sampled insufficient-evidence rows,
    # so the audit UI never falls back to a different visualization format.
    events = all_events
    wanted = {
        (int(event["camera_id"]), event["segment_id"], int(event["evidence"]["review_frame_id"]))
        for event in events
    }
    frame_paths = {}
    with MANIFEST.open() as handle:
        for row in csv.DictReader(handle):
            key = (int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))
            if key in wanted:
                frame_paths[key] = Path(row["image_path"])
    people: dict[tuple[int, str, int], list[dict]] = defaultdict(list)
    with POSES.open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))
            if key in wanted and row.get("box"):
                people[key].append(row)
    masks = load_masks()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rendered = 0
    for event in events:
        key = (int(event["camera_id"]), event["segment_id"], int(event["evidence"]["review_frame_id"]))
        image_path = frame_paths.get(key)
        if image_path is None or not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        pixels = np.asarray(image).copy()
        rule = event.get("evidence", {}).get("rule", {})
        active_group = rule.get("artwork_id")
        visible_masks = [item for item in masks[key[0]] if active_group is None or item["rule_group"] == active_group]
        for item in visible_masks:
            mask = item["mask"]
            pixels[mask] = (0.5 * pixels[mask] + 0.5 * np.array([0, 255, 110])).astype(np.uint8)
        image = Image.fromarray(pixels)
        draw = ImageDraw.Draw(image)
        for item in visible_masks:
            x, y = item["label_xy"]
            label = f"ART {item['artifact_id']}"
            draw.text((x, max(64, y - 14)), label, font=font, fill=(20, 255, 130),
                      stroke_width=2, stroke_fill=(0, 0, 0))
        suspect_id = int(event["track_id"])
        for person in people.get(key, []):
            track_id = int(person["track_id"])
            box = [int(round(value)) for value in person["box"]]
            suspect = track_id == suspect_id
            color = (255, 35, 35) if suspect else (50, 190, 255)
            width = 6 if suspect else 3
            draw.rectangle(box, outline=color, width=width)
            label = f"CANDIDATE ID {track_id}" if suspect else f"ID {track_id}"
            ty = max(64, box[1] - 17)
            draw.text((box[0], ty), label, font=font, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))
        wrist = rule.get("evidence_wrist_xy")
        if wrist:
            x, y = map(int, wrist)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
        rule_score = "—" if event["rule_score"] is None else f"{event['rule_score']:.3f}"
        percentile = "—" if event["camera_percentile"] is None else f"{event['camera_percentile']:.3f}"
        draw.rectangle((0, 0, image.width, 58), fill=(8, 15, 30))
        title = f"{event['event_id']} | {event['event_type']} | candidate track ID {suspect_id}"
        detail = f"Cam {event['camera_id']} | frame {key[2]} | rule {rule_score} | anomaly percentile {percentile} | CANDIDATE, NOT CONFIRMED"
        draw.text((12, 9), title, font=font, fill=(255, 255, 255))
        draw.text((12, 31), detail, font=font, fill=(190, 205, 225))
        image.save(OUT_DIR / f"{event['event_id']}.jpg", quality=90)
        event["evidence"]["evidence_image_path"] = f"event_evidence/{event['event_id']}.jpg"
        rendered += 1
    EVENTS.write_text(
        "".join(json.dumps(event, ensure_ascii=False) + "\n" for event in all_events),
        encoding="utf-8",
    )
    print(f"rendered {rendered}/{len(events)} event evidence images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
