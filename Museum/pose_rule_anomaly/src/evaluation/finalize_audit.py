"""Validate a browser-exported audit JSONL and report reviewed-candidate metrics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE = REPO_ROOT / "outputs" / "audit_queue.jsonl"
OUT_REVIEWED = REPO_ROOT / "outputs" / "audit_reviewed.jsonl"
OUT_REPORT = REPO_ROOT / "reports" / "candidate_audit_reviewed.md"
ALLOWED = {"valid_candidate", "false_alarm", "uncertain", "pose_failure"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reviewed_jsonl", type=Path)
    args = parser.parse_args()
    source = {row["event_id"]: row for row in map(json.loads, QUEUE.read_text().splitlines())}
    reviewed = [json.loads(line) for line in args.reviewed_jsonl.read_text().splitlines() if line.strip()]
    ids = [row["event_id"] for row in reviewed]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate event_id in reviewed file")
    for row in reviewed:
        if row["event_id"] not in source:
            raise ValueError(f"unknown event_id: {row['event_id']}")
        if row.get("review_label") not in ALLOWED:
            raise ValueError(f"invalid review_label for {row['event_id']}: {row.get('review_label')}")
    with OUT_REVIEWED.open("w") as f:
        for row in reviewed:
            f.write(json.dumps(row) + "\n")
    counts = Counter(row["review_label"] for row in reviewed)
    false_alarm_denominator = counts["valid_candidate"] + counts["false_alarm"]
    false_alarm_rate = counts["false_alarm"] / false_alarm_denominator if false_alarm_denominator else None
    lines = [
        "# T08 reviewed-candidate audit\n\n",
        f"- Reviewed: {len(reviewed)}/{len(source)}\n",
        f"- valid_candidate: {counts['valid_candidate']}\n",
        f"- false_alarm: {counts['false_alarm']}\n",
        f"- uncertain: {counts['uncertain']}\n",
        f"- pose_failure: {counts['pose_failure']}\n",
        f"- False-alarm rate among decisively reviewed candidates: {false_alarm_rate:.1%}\n" if false_alarm_rate is not None else "- False-alarm rate: not available (no decisive labels)\n",
        "\nThis rate applies only to reviewed candidates; it is not dataset-wide precision or recall.\n",
    ]
    OUT_REPORT.write_text("".join(lines))
    print(f"wrote {len(reviewed)} reviewed rows -> {OUT_REVIEWED}")
    print(f"wrote review metrics -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
