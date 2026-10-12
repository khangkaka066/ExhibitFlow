"""Build a standalone browser UI for reviewing the T08 audit queue.

Run from Museum/pose_rule_anomaly/:
    python -m src.evaluation.build_review_ui
"""
from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUE = REPO_ROOT / "outputs" / "audit_queue.jsonl"
POSES = REPO_ROOT / "outputs" / "poses.jsonl"
MANIFEST = REPO_ROOT / "outputs" / "manifest.csv"
OUT_DIR = REPO_ROOT / "outputs" / "audit_review_v2"
IMAGE_DIR = OUT_DIR / "images"
OUT_HTML = OUT_DIR / "index.html"


def load_data() -> tuple[list[dict], dict, dict]:
    with QUEUE.open() as f:
        queue = [json.loads(line) for line in f]
    wanted_tracks = {(row["camera_id"], row["segment_id"], row["track_id"]) for row in queue}
    poses: dict[tuple, list[dict]] = defaultdict(list)
    with POSES.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["camera_id"], row["segment_id"], row["track_id"])
            if key in wanted_tracks:
                poses[key].append(row)
    paths = {}
    with MANIFEST.open() as f:
        for row in csv.DictReader(f):
            paths[(int(row["camera_id"]), row["segment_id"], int(row["frame_id"]))] = Path(row["image_path"])
    return queue, poses, paths


def make_triptych(event: dict, track_rows: list[dict], paths: dict) -> Path:
    target_frames = [
        event["start_frame"],
        (event["start_frame"] + event["end_frame"]) // 2,
        event["end_frame"],
    ]
    panels = []
    for label, target in zip(("START", "MIDDLE", "END"), target_frames):
        row = min(track_rows, key=lambda item: abs(int(item["frame_id"]) - target))
        path = paths[(event["camera_id"], event["segment_id"], int(row["frame_id"]))]
        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        if row.get("box"):
            draw.rectangle(row["box"], outline=(255, 20, 20), width=7)
        draw.rectangle((0, 0, 190, 34), fill=(0, 0, 0))
        draw.text((8, 8), f"{label} frame {row['frame_id']}", fill=(255, 255, 255))
        image.thumbnail((426, 266))
        panel = Image.new("RGB", (426, 266), (20, 20, 20))
        panel.paste(image, ((426 - image.width) // 2, (266 - image.height) // 2))
        panels.append(panel)
    triptych = Image.new("RGB", (1278, 266), (20, 20, 20))
    for index, panel in enumerate(panels):
        triptych.paste(panel, (index * 426, 0))
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output = IMAGE_DIR / f"{event['event_id']}.jpg"
    triptych.save(output, quality=88)
    return output


def build_html(queue: list[dict]) -> str:
    cards = []
    for row in queue:
        score = "—" if row["camera_percentile"] is None else f"{row['camera_percentile']:.3f}"
        rule = "—" if row["rule_score"] is None else f"{row['rule_score']:.3f}"
        cards.append(f"""
<article class="card" data-id="{html.escape(row['event_id'])}" data-camera="{row['camera_id']}" data-type="{html.escape(row['event_type'])}">
  <header><strong>{html.escape(row['event_id'])}</strong> · Cam {row['camera_id']} · {html.escape(row['event_type'])}</header>
  <div class="meta">{html.escape(row['segment_id'])} · track {row['track_id']} · frames {row['start_frame']}–{row['end_frame']} · rule {rule} · percentile {score} · {html.escape(row['selection_reason'])}</div>
  <img src="../event_evidence/{html.escape(row['event_id'])}.jpg" loading="lazy" alt="Single evidence frame for {html.escape(row['event_id'])}">
  <div class="controls">
    <label>Label
      <select class="label"><option value="">— chưa review —</option><option>valid_candidate</option><option>false_alarm</option><option>uncertain</option><option>pose_failure</option></select>
    </label>
    <label>Notes<input class="notes" type="text" placeholder="Lý do / quan sát"></label>
  </div>
</article>""")
    queue_json = json.dumps(queue, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ExhibitFlow T08 Audit</title>
<style>
body{{margin:0;background:#111827;color:#e5e7eb;font:14px system-ui,sans-serif}}.top{{position:sticky;top:0;z-index:2;background:#0f172a;padding:14px 20px;border-bottom:1px solid #334155}}h1{{font-size:20px;margin:0 0 10px}}button,select,input{{font:inherit}}button{{background:#2563eb;color:white;border:0;border-radius:6px;padding:9px 13px;cursor:pointer}}.filters{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}}main{{max-width:1320px;margin:auto;padding:18px}}.card{{background:#1f2937;border:1px solid #374151;border-radius:10px;margin:0 0 18px;overflow:hidden}}.card header{{font-size:16px;padding:12px 14px 5px}}.meta{{padding:0 14px 10px;color:#9ca3af}}.card img{{width:100%;display:block;background:#000}}.controls{{display:grid;grid-template-columns:220px 1fr;gap:12px;padding:12px 14px}}label{{display:flex;gap:8px;align-items:center}}select,input{{width:100%;background:#111827;color:#e5e7eb;border:1px solid #4b5563;border-radius:5px;padding:8px}}.done{{border-color:#22c55e}}.hidden{{display:none}}@media(max-width:700px){{.controls{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="top"><h1>ExhibitFlow · Touch candidate audit</h1><div class="filters"><span id="progress">0 / {len(queue)} reviewed</span><select id="camera"><option value="">All cameras</option><option value="1">Camera 1</option><option value="2">Camera 2</option><option value="3">Camera 3</option></select><button id="export">Export reviewed JSONL</button></div></div>
<main>{''.join(cards)}</main>
<script>
const source={queue_json}; const cards=[...document.querySelectorAll('.card')];
function save(){{const state={{}};cards.forEach(c=>state[c.dataset.id]={{label:c.querySelector('.label').value,notes:c.querySelector('.notes').value}});localStorage.setItem('exhibitflow_t08',JSON.stringify(state));update();}}
function update(){{let n=0;cards.forEach(c=>{{const done=!!c.querySelector('.label').value;c.classList.toggle('done',done);if(done)n++;}});document.getElementById('progress').textContent=`${{n}} / ${{cards.length}} reviewed`;filter();}}
function filter(){{const cam=document.getElementById('camera').value;cards.forEach(c=>c.classList.toggle('hidden',cam&&c.dataset.camera!==cam));}}
const old=JSON.parse(localStorage.getItem('exhibitflow_t08')||'{{}}');cards.forEach(c=>{{if(old[c.dataset.id]){{c.querySelector('.label').value=old[c.dataset.id].label||'';c.querySelector('.notes').value=old[c.dataset.id].notes||'';}}c.querySelectorAll('select,input').forEach(x=>x.addEventListener('change',save));}});document.getElementById('camera').addEventListener('change',filter);
document.getElementById('export').onclick=()=>{{const map=Object.fromEntries(source.map(x=>[x.event_id,x]));const lines=[];cards.forEach(c=>{{const label=c.querySelector('.label').value;if(!label)return;const row={{...map[c.dataset.id],review_label:label,review_notes:c.querySelector('.notes').value}};lines.push(JSON.stringify(row));}});const blob=new Blob([lines.join('\n')+'\n'],{{type:'application/x-ndjson'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='audit_reviewed.jsonl';a.click();URL.revokeObjectURL(a.href);}};update();
</script></body></html>"""


def main() -> None:
    with QUEUE.open() as f:
        queue = [json.loads(line) for line in f]
    missing = [row["event_id"] for row in queue if not (REPO_ROOT / "outputs" / "event_evidence" / f"{row['event_id']}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing event evidence images: {missing[:5]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(queue))
    print(f"wrote single-frame review UI ({len(queue)} events) -> {OUT_HTML}")


if __name__ == "__main__":
    main()
