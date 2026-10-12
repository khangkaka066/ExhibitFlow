"""Create a deterministic T08 human-review queue and pilot coverage report."""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVENTS = REPO_ROOT / "outputs" / "events.jsonl"
MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
OUT_QUEUE = REPO_ROOT / "outputs" / "audit_queue.jsonl"
OUT_REPORT = REPO_ROOT / "reports" / "candidate_audit.md"


def observed_minutes() -> dict[int, float]:
    seconds: dict[int, set[int]] = defaultdict(set)
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            seconds[int(row["camera_id"])].add(int(row["timestamp_source"]))
    return {camera: len(values) / 60.0 for camera, values in seconds.items()}


def main() -> None:
    with EVENTS.open() as f:
        events = [json.loads(line) for line in f]
    selected = [event for event in events if event["event_type"] == "touch_candidate"]
    reasons = {event["event_id"]: "touch_only_full_review" for event in selected}

    selected.sort(key=lambda row: (row["camera_id"], row["event_type"], row["start_time"], row["track_id"]))
    with OUT_QUEUE.open("w") as f:
        for event in selected:
            audit = {
                "event_id": event["event_id"], "selection_reason": reasons[event["event_id"]],
                "camera_id": event["camera_id"], "segment_id": event["segment_id"], "track_id": event["track_id"],
                "event_type": event["event_type"], "start_frame": event["start_frame"], "end_frame": event["end_frame"],
                "review_frame_id": event["evidence"]["review_frame_id"],
                "review_image_path": event["evidence"]["review_image_path"],
                "rule_score": event["rule_score"], "camera_percentile": event["camera_percentile"],
                "evidence": event["evidence"],
                "review_label": None, "review_notes": None,
                "allowed_labels": ["valid_candidate", "false_alarm", "uncertain", "pose_failure"],
                "source": "real_museum",
            }
            f.write(json.dumps(audit) + "\n")

    minutes = observed_minutes()
    alert_events = [event for event in events if event["review_status"] == "pending"]
    counts = Counter((event["camera_id"], event["event_type"]) for event in alert_events)
    lines = [
        "# T08 pilot candidate audit\n\n",
        f"Audit queue: {len(selected)} touch candidates. Labels are intentionally blank for human review.\n\n",
        "## Observed time and raw candidate rate\n\n",
    ]
    for camera in (1, 2, 3):
        total = counts[(camera, "touch_candidate")]
        rate = total / minutes[camera] if minutes[camera] else 0.0
        lines.append(f"- Camera {camera}: {minutes[camera]:.2f} observed minutes, {total} pending candidates, {rate:.2f}/minute\n")
    lines += [
        "\n## Interpretation limits\n\n",
        "Only `touch_candidate` is present. No precision, recall, or false-alarm rate is reported yet because `review_label` is blank. Candidate rates are raw review workload, not confirmed contact prevalence.\n",
    ]
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {len(selected)} audit rows -> {OUT_QUEUE}")
    print(f"wrote pilot report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
