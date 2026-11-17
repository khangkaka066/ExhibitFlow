#!/usr/bin/env python3
"""Đánh giá kết quả ExhibitFlow (results/*/result.json) theo ground truth (gt_videos.csv).

Metrics trên mỗi video:
- hit: có alert nào giao với khoảng frame GT hay không
- delay: độ trễ báo (frame) = alarm_start_frame của alert hit đầu tiên - start_frame của khoảng GT khớp
- fp: số alert nằm hoàn toàn ngoài mọi khoảng GT
- ids: số track_id khác nhau xuất hiện trong alerts
- red_gt_pct: % số frame GT mà có người bị gắn cờ còn đỏ (giao của các khoảng đỏ với khoảng GT)
- label_ok: nhãn alert khớp nhãn GT (stealing GT chấp nhận holding/stealing)

Chạy:
    python3 scripts/eval_gt.py --results results --gt "../benchmark anomaly/gt_videos.csv"
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GtSpan:
    start_frame: int
    end_frame: int
    label: str = ''


@dataclass
class VideoEval:
    video: str
    gt_label: str
    gt_spans: list[tuple[int, int]]
    has_result: bool
    hit: bool | None
    hit_spans: int
    total_gt_spans: int
    delay_frames: int | None
    n_alerts: int
    tp_alerts: int
    fp_alerts: int
    n_unique_ids: int
    red_gt_pct: float | None
    n_people_tracks: int | None = None
    pred_labels: list[str] = field(default_factory=list)
    label_ok: bool | None = None


def _sort_key(video_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(video_id))
    except ValueError:
        return (1, video_id)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Gộp các đoạn [lo, hi] (inclusive) giao nhau hoặc liền kề thành danh sách tăng dần rời rạc."""
    cleaned = [(min(lo, hi), max(lo, hi)) for lo, hi in intervals]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi + 1:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def interval_length(intervals: list[tuple[int, int]]) -> int:
    """Tổng số frame (inclusive) của danh sách đoạn đã gộp."""
    return sum(hi - lo + 1 for lo, hi in merge_intervals(intervals))


def interval_intersection_length(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Số frame giao nhau giữa hai tập đoạn [lo, hi] (inclusive)."""
    ma, mb = merge_intervals(a), merge_intervals(b)
    i = j = total = 0
    while i < len(ma) and j < len(mb):
        lo = max(ma[i][0], mb[j][0])
        hi = min(ma[i][1], mb[j][1])
        if lo <= hi:
            total += hi - lo + 1
        if ma[i][1] < mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def load_gt_csv(gt_path: Path) -> dict[str, list[GtSpan]]:
    """Đọc gt_videos.csv bằng utf-8-sig (hỗ trợ BOM), gom các khoảng GT theo video ID."""
    grouped: dict[str, list[GtSpan]] = {}
    with gt_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {(k or '').strip(): (v or '').strip() for k, v in raw.items()}
            vid = row.get('video', '')
            if not vid:
                continue
            grouped.setdefault(vid, [])
            s_raw, e_raw = row.get('start_frame', ''), row.get('end_frame', '')
            label = row.get('label', '').lower()
            if not s_raw or not e_raw or label in {'normal', 'none'}:
                continue
            lo, hi = int(float(s_raw)), int(float(e_raw))
            if hi < lo:
                lo, hi = hi, lo
            grouped[vid].append(GtSpan(start_frame=lo, end_frame=hi, label=label))
    return grouped


def discover_results(results_dir: Path) -> dict[str, Path]:
    """Tìm các thư mục con results/<video>/result.json."""
    found: dict[str, Path] = {}
    if not results_dir.is_dir():
        return found
    for child in results_dir.iterdir():
        if child.is_dir():
            rfile = child / 'result.json'
            if rfile.is_file():
                found[child.name] = rfile
    return found


def _infer_max_frame(result: dict[str, Any]) -> int:
    n_frames = (result.get('video') or {}).get('n_frames')
    if isinstance(n_frames, (int, float)) and n_frames > 0:
        return int(n_frames) - 1
    cand = [0]
    for a in result.get('alerts') or []:
        if a.get('alarm_end_frame') is not None:
            cand.append(int(a['alarm_end_frame']))
    for e in result.get('events') or []:
        if e.get('end_frame') is not None:
            cand.append(int(e['end_frame']))
    return max(cand)


def _is_alarm_persist(result: dict[str, Any]) -> bool:
    cfg = result.get('config') or {}
    if cfg.get('alarm_persist') is True:
        return True
    rt = (cfg.get('thresholds') or {}).get('RT') or {}
    return rt.get('alarm_persist') is True


def extract_red_intervals(result: dict[str, Any]) -> list[tuple[int, int]]:
    """Suy ra các đoạn frame [lo, hi] có ít nhất một bbox đỏ từ result.json.

    - Nếu có `flagged_tracks` (TASK-001/002): người bị gắn cờ giữ đỏ từ first_frame tới
      last_frame/end_frame (hoặc hết video nếu không ghi end_frame).
    - Nếu `config` bật `alarm_persist=True`: mỗi alert giữ đỏ tới hết video.
    - Ngược lại (hành vi cũ): dùng trực tiếp khoảng [alarm_start_frame, alarm_end_frame] của alerts.
    """
    max_frame = _infer_max_frame(result)
    persist = _is_alarm_persist(result)
    spans: list[tuple[int, int]] = []

    for a in result.get('alerts') or []:
        lo = a.get('alarm_start_frame')
        hi = a.get('alarm_end_frame')
        if lo is None or hi is None:
            continue
        lo_i, hi_i = int(lo), int(hi)
        if persist:
            hi_i = max(hi_i, max_frame)
        spans.append((min(lo_i, hi_i), max(lo_i, hi_i)))

    flagged = result.get('flagged_tracks')
    entries: list[dict[str, Any]] = []
    if isinstance(flagged, dict):
        for info in flagged.values():
            if isinstance(info, dict):
                entries.append(info)
            elif isinstance(info, (int, float)):
                entries.append({'first_frame': int(info)})
    elif isinstance(flagged, list):
        entries = [x for x in flagged if isinstance(x, dict)]

    for item in entries:
        item_spans = item.get('spans')
        if isinstance(item_spans, list) and item_spans:
            for sp in item_spans:
                if isinstance(sp, (list, tuple)) and len(sp) >= 2:
                    spans.append((int(sp[0]), int(sp[1])))
                elif isinstance(sp, dict) and 'start_frame' in sp and 'end_frame' in sp:
                    spans.append((int(sp['start_frame']), int(sp['end_frame'])))
            continue
        start = item.get('first_frame', item.get('start_frame', item.get('shown_from')))
        if start is None:
            continue
        end = item.get('last_frame', item.get('end_frame'))
        end_i = int(end) if end is not None else max_frame
        start_i = int(start)
        spans.append((min(start_i, end_i), max(start_i, end_i)))

    return merge_intervals(spans)


def _alert_interval(alert: dict[str, Any]) -> tuple[int, int] | None:
    lo = alert.get('alarm_start_frame')
    hi = alert.get('alarm_end_frame')
    if lo is None or hi is None:
        return None
    lo_i, hi_i = int(lo), int(hi)
    return (min(lo_i, hi_i), max(lo_i, hi_i))


def _alert_label(alert: dict[str, Any]) -> str:
    raw = alert.get('behavior') if alert.get('behavior') is not None else alert.get('label')
    return str(raw or '').strip().lower()


def label_matches(gt_label: str, pred_label: str) -> bool:
    """So khớp nhãn dự đoán với GT; GT 'stealing' chấp nhận cả 'holding' và 'stealing'."""
    gt = (gt_label or '').strip().lower()
    pred = (pred_label or '').strip().lower()
    if not gt or not pred:
        return False
    if gt == 'stealing':
        return pred in {'stealing', 'holding'}
    return pred == gt


def _unique_ordered(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def evaluate_video(video_id: str, gt_spans: list[GtSpan], result: dict[str, Any] | None) -> VideoEval:
    gt_pairs = [(g.start_frame, g.end_frame) for g in gt_spans]
    labels = sorted({g.label for g in gt_spans if g.label})
    gt_label = ','.join(labels) if labels else ('normal' if not gt_spans else '')

    if result is None:
        return VideoEval(
            video=video_id,
            gt_label=gt_label,
            gt_spans=gt_pairs,
            has_result=False,
            hit=False if gt_spans else None,
            hit_spans=0,
            total_gt_spans=len(gt_spans),
            delay_frames=None,
            n_alerts=0,
            tp_alerts=0,
            fp_alerts=0,
            n_unique_ids=0,
            red_gt_pct=0.0 if gt_spans else None,
            n_people_tracks=None,
            pred_labels=[],
            label_ok=None,
        )

    alerts = result.get('alerts') or []
    unique_ids = {a['track_id'] for a in alerts if a.get('track_id') is not None}
    n_people = (result.get('summary') or {}).get('n_people_tracks')

    tp_pairs: list[tuple[int, int, GtSpan]] = []
    tp_label_matched = False
    tp_labels: list[str] = []
    all_alert_labels: list[str] = []
    fp_count = 0
    for a in alerts:
        p_lbl = _alert_label(a)
        if p_lbl:
            all_alert_labels.append(p_lbl)
        iv = _alert_interval(a)
        if iv is None:
            continue
        matched = [g for g in gt_spans if intervals_overlap(iv, (g.start_frame, g.end_frame))]
        if matched:
            first_gt = min(matched, key=lambda g: (g.start_frame, g.end_frame))
            tp_pairs.append((iv[0], iv[1], first_gt))
            if p_lbl:
                tp_labels.append(p_lbl)
            if any(label_matches(g.label, p_lbl) for g in matched):
                tp_label_matched = True
        else:
            fp_count += 1

    hit_span_count = sum(
        1
        for g in gt_spans
        if any(
            (iv := _alert_interval(a)) is not None and intervals_overlap(iv, (g.start_frame, g.end_frame))
            for a in alerts
        )
    )

    if gt_spans:
        hit = len(tp_pairs) > 0
        if tp_pairs:
            earliest_start, _, matched_gt = min(tp_pairs, key=lambda x: (x[0], x[1]))
            delay_frames: int | None = earliest_start - matched_gt.start_frame
            label_ok: bool | None = tp_label_matched
        else:
            delay_frames = None
            label_ok = None
        gt_total_frames = interval_length(gt_pairs)
        red_intervals = extract_red_intervals(result)
        red_in_gt = interval_intersection_length(gt_pairs, red_intervals)
        red_gt_pct: float | None = round(100.0 * red_in_gt / gt_total_frames, 1) if gt_total_frames > 0 else None
    else:
        hit = None
        delay_frames = None
        red_gt_pct = None
        label_ok = None

    pred_labels = _unique_ordered(tp_labels if tp_labels else all_alert_labels)

    return VideoEval(
        video=video_id,
        gt_label=gt_label,
        gt_spans=gt_pairs,
        has_result=True,
        hit=hit,
        hit_spans=hit_span_count,
        total_gt_spans=len(gt_spans),
        delay_frames=delay_frames,
        n_alerts=len(alerts),
        tp_alerts=len(tp_pairs),
        fp_alerts=fp_count,
        n_unique_ids=len(unique_ids),
        red_gt_pct=red_gt_pct,
        n_people_tracks=int(n_people) if isinstance(n_people, (int, float)) else None,
        pred_labels=pred_labels,
        label_ok=label_ok,
    )


def evaluate_all(results_dir: Path, gt_path: Path) -> dict[str, Any]:
    gt_by_vid = load_gt_csv(gt_path)
    result_files = discover_results(results_dir)
    all_vids = sorted(set(gt_by_vid) | set(result_files), key=_sort_key)

    rows: list[VideoEval] = []
    for vid in all_vids:
        gt_spans = gt_by_vid.get(vid, [])
        res_data = None
        if vid in result_files:
            res_data = json.loads(result_files[vid].read_text(encoding='utf-8'))
        rows.append(evaluate_video(vid, gt_spans, res_data))

    pos_rows = [r for r in rows if r.total_gt_spans > 0]
    hits = sum(1 for r in pos_rows if r.hit)
    total_pos = len(pos_rows)
    total_gt_spans = sum(r.total_gt_spans for r in pos_rows)
    hit_gt_spans = sum(r.hit_spans for r in pos_rows)
    total_alerts = sum(r.n_alerts for r in rows)
    total_tp = sum(r.tp_alerts for r in rows)
    total_fp = sum(r.fp_alerts for r in rows)
    delays = [r.delay_frames for r in pos_rows if r.delay_frames is not None]
    red_all = [r.red_gt_pct for r in pos_rows if r.red_gt_pct is not None]
    red_hit = [r.red_gt_pct for r in pos_rows if r.hit and r.red_gt_pct is not None]
    label_eval_rows = [r for r in pos_rows if r.hit and r.label_ok is not None]
    label_ok_count = sum(1 for r in label_eval_rows if r.label_ok)
    label_eval_count = len(label_eval_rows)

    summary = {
        'n_videos': len(rows),
        'n_gt_videos': total_pos,
        'hits': hits,
        'video_recall': round(hits / total_pos, 4) if total_pos else None,
        'hit_gt_spans': hit_gt_spans,
        'total_gt_spans': total_gt_spans,
        'span_recall': round(hit_gt_spans / total_gt_spans, 4) if total_gt_spans else None,
        'total_alerts': total_alerts,
        'tp_alerts': total_tp,
        'fp_alerts': total_fp,
        'alert_precision': round(total_tp / total_alerts, 4) if total_alerts else None,
        'mean_delay_frames': round(sum(delays) / len(delays), 2) if delays else None,
        'total_unique_ids': sum(r.n_unique_ids for r in rows),
        'mean_red_gt_pct': round(sum(red_all) / len(red_all), 2) if red_all else None,
        'mean_red_gt_pct_hit': round(sum(red_hit) / len(red_hit), 2) if red_hit else None,
        'label_ok_count': label_ok_count,
        'label_eval_count': label_eval_count,
        'label_acc': round(label_ok_count / label_eval_count, 4) if label_eval_count else None,
    }
    return {'summary': summary, 'videos': [asdict(r) for r in rows]}


def format_report(report: dict[str, Any]) -> str:
    rows = report['videos']
    s = report['summary']
    headers = ['video', 'label', 'gt_frames', 'hit', 'delay(f)', 'alerts', 'fp', 'ids', 'red_gt%', 'tracks', 'label_ok']
    table_rows: list[list[str]] = []
    for r in rows:
        gt_str = ','.join(f'{a}-{b}' for a, b in r['gt_spans']) if r['gt_spans'] else '-'
        if not r['has_result']:
            hit_str = 'MISSING'
        elif r['hit'] is None:
            hit_str = '-'
        else:
            hit_str = 'YES' if r['hit'] else 'NO'
        delay_str = f"{r['delay_frames']:+d}" if r['delay_frames'] is not None else '-'
        red_str = f"{r['red_gt_pct']:.1f}%" if r['red_gt_pct'] is not None else '-'
        tracks_str = str(r['n_people_tracks']) if r['n_people_tracks'] is not None else '-'
        if r.get('label_ok') is True:
            label_ok_str = 'YES'
        elif r.get('label_ok') is False:
            label_ok_str = 'NO'
        else:
            label_ok_str = '-'
        table_rows.append([
            str(r['video']),
            r['gt_label'] or '-',
            gt_str,
            hit_str,
            delay_str,
            str(r['n_alerts']),
            str(r['fp_alerts']),
            str(r['n_unique_ids']),
            red_str,
            tracks_str,
            label_ok_str,
        ])

    widths = [max(len(h), *(len(row[i]) for row in table_rows)) for i, h in enumerate(headers)] if table_rows else [len(h) for h in headers]
    right_cols = {3, 4, 5, 6, 7, 8, 9, 10}

    def fmt_row(cols: list[str]) -> str:
        parts = []
        for i, c in enumerate(cols):
            parts.append(c.rjust(widths[i]) if i in right_cols else c.ljust(widths[i]))
        return ' | '.join(parts)

    lines = [fmt_row(headers), '-+-'.join('-' * w for w in widths)]
    lines.extend(fmt_row(row) for row in table_rows)

    rec = f"{s['video_recall'] * 100:.1f}% ({s['hits']}/{s['n_gt_videos']})" if s['video_recall'] is not None else 'N/A'
    srec = f"{s['span_recall'] * 100:.1f}% ({s['hit_gt_spans']}/{s['total_gt_spans']})" if s['span_recall'] is not None else 'N/A'
    prec = f"{s['alert_precision'] * 100:.1f}% ({s['tp_alerts']}/{s['total_alerts']})" if s['alert_precision'] is not None else 'N/A'
    mdelay = f"{s['mean_delay_frames']:+.1f}f" if s['mean_delay_frames'] is not None else 'N/A'
    mred = f"{s['mean_red_gt_pct']:.1f}%" if s['mean_red_gt_pct'] is not None else 'N/A'
    mred_hit = f"{s['mean_red_gt_pct_hit']:.1f}%" if s['mean_red_gt_pct_hit'] is not None else 'N/A'
    lacc = (
        f"{s['label_acc'] * 100:.1f}% ({s['label_ok_count']}/{s['label_eval_count']})"
        if s.get('label_acc') is not None
        else 'N/A'
    )

    lines.append('')
    lines.append(
        f"Summary: video_recall={rec} | span_recall={srec} | alert_precision={prec} | "
        f"fp={s['fp_alerts']} | mean_delay={mdelay} | total_ids={s['total_unique_ids']} | "
        f"mean_red_gt={mred} (hit_only={mred_hit}) | label_acc={lacc}"
    )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Evaluate ExhibitFlow results against GT CSV.')
    ap.add_argument('--results', type=Path, default=Path('results'), help='Directory containing <video>/result.json')
    ap.add_argument('--gt', type=Path, default=Path('../benchmark anomaly/gt_videos.csv'), help='Path to gt_videos.csv')
    ap.add_argument('--json', action='store_true', help='Print JSON report to stdout')
    ap.add_argument('--out', type=Path, default=None, help='Optional path to write JSON report')
    args = ap.parse_args(argv)

    report = evaluate_all(args.results, args.gt)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
