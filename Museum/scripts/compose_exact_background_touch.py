#!/usr/bin/env python3
"""Place one RGBA touch actor on three museum backgrounds without regenerating them."""

from pathlib import Path
import json

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "dataset" / "MuseumVisitorsProcessed"
ASSET = PROCESSED / "synthetic_exact_background" / "assets" / "touch_person_rgba.png"
OUTPUT = PROCESSED / "synthetic_exact_background"


# The right edge of the cutout is aligned with a local exhibit surface.
# Coordinates refer to the original 1280x800 MuseumVisitors background.
CONFIG = {
    "cam1": {
        "background": PROCESSED / "backgrounds" / "cam1_segment01" / "background_masked_median.jpg",
        "target_height": 230,
        "right_x": 390,
        "top_y": 94,
        "artwork_id": "cam1_near_white_pedestal",
    },
    "cam2": {
        "background": PROCESSED / "backgrounds" / "cam2_segment01" / "background_masked_median.jpg",
        "target_height": 400,
        "right_x": 795,
        "top_y": 260,
        "artwork_id": "cam2_foreground_stone",
    },
    "cam3": {
        "background": PROCESSED / "backgrounds" / "cam3_segment02" / "background_masked_median.jpg",
        "target_height": 390,
        "right_x": 1190,
        "top_y": 285,
        "artwork_id": "cam3_right_foreground_stone",
    },
}


def prepare_actor(path: Path) -> Image.Image:
    actor = Image.open(path).convert("RGBA")
    array = np.array(actor)
    alpha = array[:, :, 3].astype(np.float32)
    ys, xs = np.where(alpha >= 32)
    actor = actor.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    array = np.array(actor)
    alpha = array[:, :, 3]
    # Remove the very low-alpha generation halo while retaining antialiased edges.
    alpha = np.where(alpha < 32, 0, np.clip((alpha - 32.0) * 255.0 / 222.0, 0, 255))
    array[:, :, 3] = alpha.astype(np.uint8)
    return Image.fromarray(array, "RGBA")


def main() -> None:
    image_dir = OUTPUT / "images"
    mask_dir = OUTPUT / "masks"
    annotation_dir = OUTPUT / "annotations"
    for directory in (image_dir, mask_dir, annotation_dir):
        directory.mkdir(parents=True, exist_ok=True)

    source_actor = prepare_actor(ASSET)
    records = []

    for camera_id, cfg in CONFIG.items():
        background = Image.open(cfg["background"]).convert("RGB")
        assert background.size == (1280, 800), background.size

        height = cfg["target_height"]
        width = round(source_actor.width * height / source_actor.height)
        actor = source_actor.resize((width, height), Image.Resampling.LANCZOS)
        left = cfg["right_x"] - width
        top = cfg["top_y"]
        assert left >= 0 and top >= 0 and left + width <= 1280 and top + height <= 800

        foreground = Image.new("RGBA", background.size, (0, 0, 0, 0))
        foreground.alpha_composite(actor, (left, top))
        result = Image.alpha_composite(background.convert("RGBA"), foreground).convert("RGB")

        image_path = image_dir / f"touch_001_{camera_id}.png"
        mask_path = mask_dir / f"touch_001_{camera_id}_change_mask.png"
        result.save(image_path, compress_level=3)
        foreground.getchannel("A").save(mask_path, compress_level=3)

        bg_array = np.array(background)
        out_array = np.array(result)
        mask_array = np.array(foreground.getchannel("A"))
        unchanged_outside_mask = bool(np.array_equal(
            bg_array[mask_array == 0], out_array[mask_array == 0]
        ))

        records.append({
            "image_id": f"touch_001_{camera_id}",
            "camera_id": int(camera_id[-1]),
            "label": "near_touch",
            "file_name": str(image_path.relative_to(OUTPUT)),
            "change_mask": str(mask_path.relative_to(OUTPUT)),
            "background": str(cfg["background"].relative_to(PROCESSED)),
            "artwork_id": cfg["artwork_id"],
            "person_bbox_xywh": [left, top, width, height],
            "synthetic": True,
            "same_actor_asset": True,
            "same_physical_event": False,
            "background_pixel_exact_outside_mask": unchanged_outside_mask,
            "quality_status": "rejected_unnatural_contact",
            "use_for_training": False,
            "review_required": True,
        })

    with (annotation_dir / "views.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
