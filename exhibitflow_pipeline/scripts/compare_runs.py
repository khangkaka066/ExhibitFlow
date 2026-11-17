#!/usr/bin/env python3
"""So sánh 2 thư mục kết quả ExhibitFlow (a -> b): số mask/cảnh, alerts, hit, delay, red_gt%, nhãn và đánh dấu video bị lùi.

Chạy:
    python3 -B scripts/compare_runs.py --a results --b "results (1)"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import eval_gt  # noqa: E402

_DEFAULT_GT = Path('../benchmark anomaly/gt_videos.csv')
_CORE_REGRESSIONS = {'missing', 'hit', 'label', 'red_gt', 'fp'}
_CORE_IMPROVEMENTS = {'hit', 'label', 'red_gt'}


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def _extract_artifacts_info(result: dict[str, Any] | None) -> tuple[int | None, list[int] | None]:
    if result is None:
        return None, None
    scenes = [sc for sc in (result.get('scenes') or []) if isinstance(sc, dict)]
    scene_counts = [int(sc.get('n_artifacts', 0)) for sc in scenes]
    artifacts = result.get('artifacts')
    if isinstance(artifacts, list):
        n_artifacts = len(artifacts)
    else:
        n_artifacts = sum(scene_counts)
    return n_artifacts, scene_counts


def _run_metrics(vid: str, gt_spans: list[eval_gt.GtSpan], result: dict[str, Any] | None) -> dict[str, Any]:
    ev = eval_gt.evaluate_video(vid, gt_spans, result)
    n_artifacts, scene_artifacts = _extract_artifacts_info(result)
    return {
        'has_result': ev.has_result,
        'n_artifacts': n_artifacts,
        'scene_artifacts': scene_artifacts,
        'n_alerts': ev.n_alerts,
        'tp_alerts': ev.tp_alerts,
        'fp_alerts': ev.fp_alerts,
        'hit': ev.hit,
        'hit_spans': ev.hit_spans,
        'total_gt_spans': ev.total_gt_spans,
        'delay_frames': ev.delay_frames,
        'red_gt_pct': ev.red_gt_pct,
        'n_unique_ids': ev.n_unique_ids,
        'n_people_tracks': ev.n_people_tracks,
        'pred_labels': ev.pred_labels,
        'label_ok': ev.label_ok,
    }


def _classify_changes(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    mask_drop_ratio: float,
    red_drop_pct: float,
    delay_inc_frames: int,
) -> tuple[list[str], list[str], bool, str]:
    regressions: list[str] = []
    improvements: list[str] = []

    if a['has_result'] and not b['has_result']:
        regressions.append('missing')

    if a['hit'] is True and b['hit'] is False:
        regressions.append('hit')
    elif a['hit'] is False and b['hit'] is True:
        improvements.append('hit')

    if a['label_ok'] is True and b['label_ok'] is False:
        regressions.append('label')
    elif a['label_ok'] is False and b['label_ok'] is True:
        improvements.append('label')

    a_red, b_red = a['red_gt_pct'], b['red_gt_pct']
    if a_red is not None and b_red is not None:
        if a_red - b_red >= red_drop_pct:
            regressions.append('red_gt')
        elif b_red - a_red >= red_drop_pct:
            improvements.append('red_gt')

    if a['has_result'] and b['has_result']:
        if b['fp_alerts'] > a['fp_alerts']:
            regressions.append('fp')
        elif b['fp_alerts'] < a['fp_alerts']:
            improvements.append('fp')

    a_del, b_del = a['delay_frames'], b['delay_frames']
    if a_del is not None and b_del is not None:
        if b_del - a_del >= delay_inc_frames:
            regressions.append('delay')
        elif a_del - b_del >= delay_inc_frames:
            improvements.append('delay')

    a_art, b_art = a['n_artifacts'], b['n_artifacts']
    if a_art is not None and b_art is not None and a_art > 0:
        drop_frac = (a_art - b_art) / a_art
        if drop_frac >= mask_drop_ratio:
            regressions.append('masks')
        elif (b_art - a_art) / a_art >= mask_drop_ratio:
            improvements.append('masks')

    has_core_reg = any(r in _CORE_REGRESSIONS for r in regressions)
    has_core_imp = any(i in _CORE_IMPROVEMENTS for i in improvements)
    regressed = has_core_reg or (bool(regressions) and not has_core_imp)

    if regressed:
        status = f"REGRESS({','.join(regressions)})"
    elif regressions and has_core_imp:
        status = f"MIXED(+{','.join(improvements)}/-{','.join(regressions)})"
    elif improvements:
        status = f"IMPROVED({','.join(improvements)})"
    else:
        status = '='
    return regressions, improvements, regressed, status


def _summarize_run(dir_path: Path, gt_path: Path | None, rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if gt_path is not None and gt_path.is_file():
        summary = dict(eval_gt.evaluate_all(dir_path, gt_path)['summary'])
    else:
        summary = {
            'n_videos': len(rows),
            'n_gt_videos': 0,
            'hits': 0,
            'video_recall': None,
            'hit_gt_spans': 0,
            'total_gt_spans': 0,
            'span_recall': None,
            'total_alerts': sum(r[key]['n_alerts'] for r in rows),
            'tp_alerts': 0,
            'fp_alerts': 0,
            'alert_precision': None,
            'mean_delay_frames': None,
            'total_unique_ids': sum(r[key]['n_unique_ids'] for r in rows),
            'mean_red_gt_pct': None,
            'mean_red_gt_pct_hit': None,
            'label_ok_count': 0,
            'label_eval_count': 0,
            'label_acc': None,
        }
    summary['total_artifacts'] = sum(r[key]['n_artifacts'] or 0 for r in rows)
    return summary


def compare_runs(
    dir_a: Path,
    dir_b: Path,
    gt_path: Path | None = None,
    *,
    mask_drop_ratio: float = 0.3,
    red_drop_pct: float = 5.0,
    delay_inc_frames: int = 15,
) -> dict[str, Any]:
    """So sánh hai thư mục kết quả dir_a và dir_b theo từng video và tổng hợp."""
    gt_by_vid = eval_gt.load_gt_csv(gt_path) if (gt_path is not None and gt_path.is_file()) else {}
    files_a = eval_gt.discover_results(dir_a)
    files_b = eval_gt.discover_results(dir_b)
    all_vids = sorted(set(gt_by_vid) | set(files_a) | set(files_b), key=eval_gt._sort_key)

    videos: list[dict[str, Any]] = []
    for vid in all_vids:
        gt_spans = gt_by_vid.get(vid, [])
        labels = sorted({g.label for g in gt_spans if g.label})
        gt_label = ','.join(labels) if labels else ('normal' if gt_by_vid and not gt_spans else '-')

        res_a = _load_json(files_a.get(vid))
        res_b = _load_json(files_b.get(vid))
        info_a = _run_metrics(vid, gt_spans, res_a)
        info_b = _run_metrics(vid, gt_spans, res_b)

        regressions, improvements, regressed, status = _classify_changes(
            info_a,
            info_b,
            mask_drop_ratio=mask_drop_ratio,
            red_drop_pct=red_drop_pct,
            delay_inc_frames=delay_inc_frames,
        )
        videos.append({
            'video': vid,
            'gt_label': gt_label,
            'a': info_a,
            'b': info_b,
            'regressions': regressions,
            'improvements': improvements,
            'regressed': regressed,
            'status': status,
        })

    sum_a = _summarize_run(dir_a, gt_path, videos, 'a')
    sum_b = _summarize_run(dir_b, gt_path, videos, 'b')
    summary = {
        'dir_a': str(dir_a),
        'dir_b': str(dir_b),
        'gt_path': str(gt_path) if gt_path is not None else None,
        'run_a': sum_a,
        'run_b': sum_b,
        'regressed_videos': [v['video'] for v in videos if v['regressed']],
        'mask_drop_videos': [v['video'] for v in videos if 'masks' in v['regressions']],
        'improved_videos': [v['video'] for v in videos if v['improvements'] and not v['regressed']],
    }
    return {'summary': summary, 'videos': videos}


def _fmt_opt_int(val: int | None, signed: bool = False) -> str:
    if val is None:
        return '-'
    return f'{val:+d}' if signed else str(val)


def _fmt_hit(info: dict[str, Any]) -> str:
    if not info['has_result']:
        return 'MISS'
    if info['hit'] is None:
        return '-'
    return 'YES' if info['hit'] else 'NO'


def _fmt_bool(val: bool | None) -> str:
    if val is True:
        return 'YES'
    if val is False:
        return 'NO'
    return '-'


def _fmt_pct(val: float | None) -> str:
    return f'{val:.1f}%' if val is not None else '-'


def _fmt_scenes(counts: list[int] | None) -> str:
    if counts is None:
        return '-'
    if not counts:
        return '0'
    return '/'.join(str(c) for c in counts)


def _fmt_labels(labels: list[str]) -> str:
    return ','.join(labels) if labels else '-'


def _pair(a_str: str, b_str: str) -> str:
    return f'{a_str}→{b_str}'


def format_comparison(report: dict[str, Any]) -> str:
    """Định dạng bảng so sánh dễ đọc cho terminal."""
    videos = report['videos']
    s = report['summary']
    sa, sb = s['run_a'], s['run_b']

    headers = [
        'video',
        'gt',
        'masks(a→b)',
        'scenes(a→b)',
        'alerts(a→b)',
        'hit(a→b)',
        'delay(a→b)',
        'red_gt%(a→b)',
        'label_ok(a→b)',
        'labels(a→b)',
        'status',
    ]
    table_rows: list[list[str]] = []
    for v in videos:
        a, b = v['a'], v['b']
        table_rows.append([
            str(v['video']),
            v['gt_label'] or '-',
            _pair(_fmt_opt_int(a['n_artifacts']), _fmt_opt_int(b['n_artifacts'])),
            _pair(_fmt_scenes(a['scene_artifacts']), _fmt_scenes(b['scene_artifacts'])),
            _pair(str(a['n_alerts']), str(b['n_alerts'])),
            _pair(_fmt_hit(a), _fmt_hit(b)),
            _pair(_fmt_opt_int(a['delay_frames'], signed=True), _fmt_opt_int(b['delay_frames'], signed=True)),
            _pair(_fmt_pct(a['red_gt_pct']), _fmt_pct(b['red_gt_pct'])),
            _pair(_fmt_bool(a['label_ok']), _fmt_bool(b['label_ok'])),
            _pair(_fmt_labels(a['pred_labels']), _fmt_labels(b['pred_labels'])),
            v['status'],
        ])

    widths = [max(len(h), *(len(r[i]) for r in table_rows)) for i, h in enumerate(headers)] if table_rows else [len(h) for h in headers]
    right_cols = {2, 4, 5, 6, 7, 8}

    def fmt_row(cols: list[str]) -> str:
        return ' | '.join(c.rjust(widths[i]) if i in right_cols else c.ljust(widths[i]) for i, c in enumerate(cols))

    def fmt_ratio(key: str, ok_key: str | None = None, tot_key: str | None = None) -> tuple[str, str]:
        def one(run_s: dict[str, Any]) -> str:
            val = run_s.get(key)
            if val is None:
                return 'N/A'
            if ok_key and tot_key:
                return f'{val * 100:.1f}% ({run_s[ok_key]}/{run_s[tot_key]})'
            return f'{val * 100:.1f}%'
        return one(sa), one(sb)

    lines = [
        f"Compare: {s['dir_a']} -> {s['dir_b']}",
        fmt_row(headers),
        '-+-'.join('-' * w for w in widths),
    ]
    lines.extend(fmt_row(row) for row in table_rows)
    lines.append('')

    rec_a, rec_b = fmt_ratio('video_recall', 'hits', 'n_gt_videos')
    srec_a, srec_b = fmt_ratio('span_recall', 'hit_gt_spans', 'total_gt_spans')
    prec_a, prec_b = fmt_ratio('alert_precision', 'tp_alerts', 'total_alerts')
    lacc_a, lacc_b = fmt_ratio('label_acc', 'label_ok_count', 'label_eval_count')
    del_a = f"{sa['mean_delay_frames']:+.1f}f" if sa.get('mean_delay_frames') is not None else 'N/A'
    del_b = f"{sb['mean_delay_frames']:+.1f}f" if sb.get('mean_delay_frames') is not None else 'N/A'
    red_a = f"{sa['mean_red_gt_pct']:.1f}%" if sa.get('mean_red_gt_pct') is not None else 'N/A'
    red_b = f"{sb['mean_red_gt_pct']:.1f}%" if sb.get('mean_red_gt_pct') is not None else 'N/A'

    lines.append(
        f"Summary A->B: masks={sa['total_artifacts']}->{sb['total_artifacts']} | "
        f"video_recall={rec_a}->{rec_b} | span_recall={srec_a}->{srec_b} | "
        f"alert_precision={prec_a}->{prec_b} (fp={sa['fp_alerts']}->{sb['fp_alerts']}) | "
        f"mean_delay={del_a}->{del_b} | mean_red_gt={red_a}->{red_b} | label_acc={lacc_a}->{lacc_b}"
    )
    reg_list = ', '.join(s['regressed_videos']) if s['regressed_videos'] else 'none'
    mdrop_list = ', '.join(s['mask_drop_videos']) if s['mask_drop_videos'] else 'none'
    imp_list = ', '.join(s['improved_videos']) if s['improved_videos'] else 'none'
    lines.append(f"Regressed videos: {reg_list} | Mask-drop videos: {mdrop_list} | Improved videos: {imp_list}")
    return '\n'.join(lines)


def _resolve_gt_path(gt_arg: Path | None) -> Path | None:
    if gt_arg is None:
        return None
    if gt_arg.is_file():
        return gt_arg
    if gt_arg == _DEFAULT_GT:
        fallback = Path(__file__).resolve().parents[2] / 'benchmark anomaly' / 'gt_videos.csv'
        if fallback.is_file():
            return fallback
        return None
    raise FileNotFoundError(f'GT CSV not found: {gt_arg}')


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Compare two ExhibitFlow result directories.')
    ap.add_argument('--a', type=Path, required=True, help='Baseline results directory (run A)')
    ap.add_argument('--b', type=Path, required=True, help='New results directory (run B)')
    ap.add_argument('--gt', type=Path, default=_DEFAULT_GT, help='Optional path to gt_videos.csv')
    ap.add_argument('--mask-drop-ratio', type=float, default=0.3, help='Relative mask drop ratio to flag regression (default 0.3)')
    ap.add_argument('--red-drop-pct', type=float, default=5.0, help='Percentage-point drop in red_gt%% to flag regression (default 5.0)')
    ap.add_argument('--delay-inc-frames', type=int, default=15, help='Frame delay increase to flag regression (default 15)')
    ap.add_argument('--json', action='store_true', help='Print JSON report to stdout')
    ap.add_argument('--out', type=Path, default=None, help='Optional path to write JSON report')
    args = ap.parse_args(argv)

    gt_path = _resolve_gt_path(args.gt)
    report = compare_runs(
        args.a,
        args.b,
        gt_path,
        mask_drop_ratio=args.mask_drop_ratio,
        red_drop_pct=args.red_drop_pct,
        delay_inc_frames=args.delay_inc_frames,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_comparison(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
