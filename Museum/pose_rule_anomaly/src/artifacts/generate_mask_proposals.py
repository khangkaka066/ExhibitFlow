#!/usr/bin/env python3
"""Generate review-only museum artifact masks with Grounding DINO + SAM 2.

The detector is constrained by a camera-specific spatial prior. Every output is
marked ``needs_review``; this script never enables a rule or overwrites an ROI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def bbox_from_polygon(polygon: Iterable[Iterable[float]]) -> list[float]:
    points = np.asarray(list(polygon), dtype=float)
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


def expand_box(box: list[float], scale: float, width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = max(x2 - x1, 96), max(y2 - y1, 96)
    return [
        max(0, int(math.floor(cx - bw * scale / 2))),
        max(0, int(math.floor(cy - bh * scale / 2))),
        min(width, int(math.ceil(cx + bw * scale / 2))),
        min(height, int(math.ceil(cy + bh * scale / 2))),
    ]


def box_iou(a: list[float], b: list[float]) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def post_process_dino(processor: Any, outputs: Any, inputs: Any, size: tuple[int, int],
                      box_threshold: float, text_threshold: float) -> dict[str, Any]:
    """Support both recent and older Transformers Grounding DINO APIs."""
    kwargs = {"target_sizes": [size]}
    try:
        result = processor.post_process_grounded_object_detection(
            outputs, threshold=box_threshold, text_threshold=text_threshold, **kwargs
        )[0]
    except TypeError:
        result = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold,
            text_threshold=text_threshold, **kwargs
        )[0]
    return result


def choose_box(result: dict[str, Any], offset: tuple[int, int], expected: list[float],
               min_iou: float) -> tuple[list[float] | None, float | None, str | None, float]:
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])
    labels = result.get("text_labels", result.get("labels", [""] * len(boxes)))
    best: tuple[float, list[float], float, str, float] | None = None
    for index, raw_box in enumerate(boxes):
        local = raw_box.detach().cpu().tolist() if hasattr(raw_box, "detach") else list(raw_box)
        full = [local[0] + offset[0], local[1] + offset[1],
                local[2] + offset[0], local[3] + offset[1]]
        score = float(scores[index].detach().cpu()) if hasattr(scores[index], "detach") else float(scores[index])
        iou = box_iou(full, expected)
        combined = 0.70 * score + 0.30 * iou
        label = str(labels[index]) if index < len(labels) else ""
        if iou >= min_iou and (best is None or combined > best[0]):
            best = (combined, full, score, label, iou)
    if best is None:
        return None, None, None, 0.0
    return best[1], best[2], best[3], best[4]


def overlay_image(image: Image.Image, mask: np.ndarray, polygon: list[list[int]],
                  box: list[float], title: str, method: str) -> Image.Image:
    base = image.convert("RGBA")
    tint = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    tint[mask] = [0, 255, 120, 105]
    base = Image.alpha_composite(base, Image.fromarray(tint, mode="RGBA"))
    draw = ImageDraw.Draw(base)
    draw.line([tuple(p) for p in polygon] + [tuple(polygon[0])], fill=(0, 220, 255, 255), width=3)
    draw.rectangle(tuple(box), outline=(255, 70, 70, 255), width=3)
    draw.rectangle((8, 8, min(image.width - 8, 620), 58), fill=(0, 0, 0, 190))
    draw.text((16, 15), f"{title} | {method} | NEEDS REVIEW", fill="white", font=ImageFont.load_default())
    return base.convert("RGB")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent.parent
    output_dir = args.output_dir.resolve()
    masks_dir, overlays_dir = output_dir / "masks", output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. In Colab select a GPU runtime.")

    detector_id = config["models"]["grounding_dino"]
    sam_id = config["models"]["sam2"]
    print(f"Loading Grounding DINO: {detector_id}", flush=True)
    processor = AutoProcessor.from_pretrained(detector_id)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(detector_id).to(args.device).eval()
    print(f"Loading SAM 2: {sam_id}", flush=True)
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor.from_pretrained(sam_id, device=args.device)

    proposals: list[dict[str, Any]] = []
    for camera in config["cameras"]:
        camera_id = int(camera["camera_id"])
        image_path = (base_dir / camera["image_path"]).resolve()
        image = Image.open(image_path).convert("RGB")
        expected_size = tuple(camera["image_size"])
        if image.size != expected_size:
            raise ValueError(f"{image_path}: expected {expected_size}, got {image.size}")
        predictor.set_image(np.asarray(image))

        for artifact in camera["artifacts"]:
            artifact_id = artifact["artifact_id"]
            expected_box = bbox_from_polygon(artifact["expected_polygon"])
            crop_box = expand_box(expected_box, float(artifact.get("crop_scale", 2.0)), *image.size)
            crop = image.crop(tuple(crop_box))
            prompt = ". ".join(artifact["prompts"]) + "."
            inputs = processor(images=crop, text=prompt, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                outputs = detector(**inputs)
            result = post_process_dino(
                processor, outputs, inputs, (crop.height, crop.width),
                float(config["inference"]["box_threshold"]),
                float(config["inference"]["text_threshold"]),
            )
            box, dino_score, matched_label, prior_iou = choose_box(
                result, (crop_box[0], crop_box[1]), expected_box,
                float(config["inference"]["minimum_prior_iou"]),
            )
            method = "grounding_dino_box_to_sam2"
            flags: list[str] = []
            if box is None:
                box = expected_box
                method = "manual_prior_box_to_sam2_fallback"
                flags.append("grounding_dino_no_spatially_valid_box")

            with torch.inference_mode():
                sam_masks, sam_scores, _ = predictor.predict(
                    point_coords=None, point_labels=None,
                    box=np.asarray(box, dtype=np.float32)[None, :],
                    multimask_output=False,
                )
            mask = np.asarray(sam_masks)
            while mask.ndim > 2:
                mask = mask[0]
            mask = mask.astype(bool)
            if mask.shape != (image.height, image.width):
                raise ValueError(f"SAM 2 mask shape {mask.shape} does not match image {image.size}")

            mask_area = int(mask.sum())
            expected_area = max(1.0, (expected_box[2] - expected_box[0]) * (expected_box[3] - expected_box[1]))
            area_ratio = mask_area / expected_area
            if mask_area < int(config["quality"]["minimum_mask_pixels"]):
                flags.append("mask_too_small")
            if not (float(config["quality"]["minimum_area_ratio"]) <= area_ratio <=
                    float(config["quality"]["maximum_area_ratio"])):
                flags.append("mask_area_implausible_for_spatial_prior")

            stem = f"cam{camera_id}_{artifact_id}"
            mask_path = masks_dir / f"{stem}.png"
            overlay_path = overlays_dir / f"{stem}.jpg"
            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
            overlay_image(image, mask, artifact["expected_polygon"], box, stem, method).save(
                overlay_path, quality=92
            )
            sam_score = float(np.asarray(sam_scores).reshape(-1)[0]) if np.asarray(sam_scores).size else None
            row = {
                "schema_version": 1,
                "source": "real_museum",
                "camera_id": camera_id,
                "artifact_id": artifact_id,
                "rule_group": artifact.get("rule_group", artifact_id),
                "source_image": camera["image_path"],
                "prompts": artifact["prompts"],
                "expected_polygon": artifact["expected_polygon"],
                "expected_bbox_xyxy": [round(v, 2) for v in expected_box],
                "detected_bbox_xyxy": [round(float(v), 2) for v in box],
                "proposal_method": method,
                "grounding_dino_score": None if dino_score is None else round(dino_score, 6),
                "grounding_dino_label": matched_label,
                "spatial_prior_iou": round(prior_iou, 6),
                "sam2_score": None if sam_score is None else round(sam_score, 6),
                "mask_pixels": mask_area,
                "mask_to_prior_area_ratio": round(area_ratio, 6),
                "quality_flags": flags,
                "approval_status": "needs_review",
                "mask_path": str(mask_path.relative_to(output_dir)),
                "overlay_path": str(overlay_path.relative_to(output_dir)),
            }
            proposals.append(row)
            print(f"[{len(proposals):02d}] {stem}: {method}, flags={flags or ['none']}", flush=True)

    manifest = output_dir / "mask_proposals.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in proposals), encoding="utf-8")
    report = {
        "schema_version": 1,
        "proposal_count": len(proposals),
        "approval_status": "needs_review",
        "models": config["models"],
        "device": args.device,
        "fallback_count": sum(p["proposal_method"].startswith("manual") for p in proposals),
        "flagged_count": sum(bool(p["quality_flags"]) for p in proposals),
        "operational_use_allowed": False,
    }
    (output_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
