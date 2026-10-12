"""Validate per-camera ROI configs and render review overlays.

Run from Museum/pose_rule_anomaly/:
    python -m src.roi.render_rois
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT.parent / "dataset" / "MuseumVisitors"
CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_DIR = REPO_ROOT / "outputs" / "roi_previews"

COLORS = {
    "floor": (30, 144, 255),
    "artwork": (255, 215, 0),
    "pedestal": (255, 140, 0),
    "barrier": (255, 40, 40),
    "restricted": (220, 20, 120),
}


def validate_polygon(points: list, width: int, height: int, label: str) -> None:
    if len(points) < 3:
        raise ValueError(f"{label} needs at least three points")
    for point in points:
        if len(point) != 2 or not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f"{label} has out-of-bounds point: {point}")


def validate_config(config: dict) -> None:
    width, height = config["image_size"]
    validate_polygon(config["floor_observable"], width, height, "floor_observable")
    for group in ("artwork_rois", "pedestal_rois"):
        ids = set()
        for item in config[group]:
            if item["id"] in ids:
                raise ValueError(f"duplicate {group} id: {item['id']}")
            ids.add(item["id"])
            validate_polygon(item["polygon"], width, height, f"{group}.{item['id']}")
            if item.get("rule_enabled") and group == "artwork_rois":
                if "interaction_zone" not in item:
                    raise ValueError(f"enabled artwork {item['id']} needs an interaction_zone")
                validate_polygon(item["interaction_zone"], width, height, f"{group}.{item['id']}.interaction_zone")
    for barrier in config["barriers"]:
        if len(barrier["line"]) < 2:
            raise ValueError(f"barrier {barrier['id']} needs at least two line points")
        for point in barrier["line"]:
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                raise ValueError(f"barrier {barrier['id']} has out-of-bounds point: {point}")
        validate_polygon(barrier["restricted_zone"], width, height, f"barrier.{barrier['id']}.restricted_zone")


def draw_polygon(draw: ImageDraw.ImageDraw, points: list, color: tuple, label: str, width: int = 4) -> None:
    draw.line([tuple(p) for p in points] + [tuple(points[0])], fill=color, width=width)
    draw.text((points[0][0] + 4, points[0][1] + 4), label, fill=color, stroke_width=2, stroke_fill=(0, 0, 0))


def render(config: dict) -> Path:
    image_path = DATASET_ROOT / config["source_frame"]["image_path"]
    image = Image.open(image_path).convert("RGB")
    if list(image.size) != config["image_size"]:
        raise ValueError(f"image size mismatch for camera {config['camera_id']}: {image.size}")
    draw = ImageDraw.Draw(image, "RGBA")
    draw_polygon(draw, config["floor_observable"], COLORS["floor"] + (220,), "floor")
    for item in config["artwork_rois"]:
        draw_polygon(draw, item["polygon"], COLORS["artwork"] + (255,), f"art:{item['id']}")
        if item.get("rule_enabled") and item.get("interaction_zone"):
            draw_polygon(draw, item["interaction_zone"], (50, 255, 120, 255), f"interaction:{item['id']}", 3)
    for item in config["pedestal_rois"]:
        draw_polygon(draw, item["polygon"], COLORS["pedestal"] + (255,), f"ped:{item['id']}", 3)
    for item in config["barriers"]:
        draw.polygon([tuple(p) for p in item["restricted_zone"]], fill=COLORS["restricted"] + (45,))
        draw_polygon(draw, item["restricted_zone"], COLORS["restricted"] + (255,), f"restricted:{item['id']}")
        draw.line([tuple(p) for p in item["line"]], fill=COLORS["barrier"] + (255,), width=6)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"cam{config['camera_id']}_roi.jpg"
    image.save(out_path, quality=92)
    return out_path


def main() -> None:
    for camera_id in (1, 2, 3):
        path = CONFIG_DIR / f"cam{camera_id}_roi.json"
        config = json.loads(path.read_text())
        validate_config(config)
        out = render(config)
        print(f"camera {camera_id}: valid -> {out}")


if __name__ == "__main__":
    main()
