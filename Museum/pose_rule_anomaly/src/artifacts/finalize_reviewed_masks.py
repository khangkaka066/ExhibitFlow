#!/usr/bin/env python3
"""Materialize the explicitly reviewed mask set used by spatial rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
DECISIONS = ROOT / "configs" / "artifact_mask_review_v1.json"
DESTINATION = OUTPUTS / "artifact_masks_v1"


def load_rows(source_dir: Path) -> dict[tuple[int, str], dict]:
    path = source_dir / "mask_proposals.jsonl"
    return {
        (int(row["camera_id"]), row["artifact_id"]): row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def main() -> None:
    decisions = json.loads(DECISIONS.read_text(encoding="utf-8"))
    sources = {
        name: load_rows(OUTPUTS / name)
        for name in sorted({item["source_set"] for item in decisions["items"]})
    }
    masks_dir = DESTINATION / "masks"
    overlays_dir = DESTINATION / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    final_rows = []
    rejected = []
    for decision in decisions["items"]:
        key = (int(decision["camera_id"]), decision["artifact_id"])
        row = dict(sources[decision["source_set"]][key])
        if decision["decision"] != "approved":
            rejected.append({**decision, "original_proposal_method": row["proposal_method"]})
            continue
        source_root = OUTPUTS / decision["source_set"]
        mask_source = source_root / row["mask_path"]
        overlay_source = source_root / row["overlay_path"]
        mask_name = f"cam{key[0]}_{key[1]}.png"
        overlay_name = f"cam{key[0]}_{key[1]}.jpg"
        if Image.open(mask_source).size != (1280, 800):
            raise ValueError(f"Unexpected mask size: {mask_source}")
        shutil.copy2(mask_source, masks_dir / mask_name)
        shutil.copy2(overlay_source, overlays_dir / overlay_name)
        row.update({
            "approval_status": "approved",
            "reviewed_by": decisions["reviewed_by"],
            "reviewed_at": decisions["reviewed_at"],
            "operational_scope": decisions["operational_scope"],
            "review_note": decision.get("note", "Visual mask boundary matches the intended artifact."),
            "mask_path": f"masks/{mask_name}",
            "overlay_path": f"overlays/{overlay_name}",
        })
        final_rows.append(row)
    final_rows.sort(key=lambda row: (int(row["camera_id"]), row["artifact_id"]))
    (DESTINATION / "mask_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "approved_count": len(final_rows),
        "rejected_count": len(rejected),
        "operational_scope": decisions["operational_scope"],
        "rejected": rejected,
    }
    (DESTINATION / "review_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
