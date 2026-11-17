"""Unit tests cho scripts/eval_gt.py (chỉ dùng JSON/CSV giả lập, không chạy model)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import eval_gt  # noqa: E402


def _write_result(results_dir: Path, vid: str, n_frames: int, alerts: list[dict], **extra):
    vdir = results_dir / vid
    vdir.mkdir(parents=True, exist_ok=True)
    payload = {
        'video': {'n_frames': n_frames, 'fps': 25.0},
        'summary': {'n_people_tracks': extra.pop('n_people_tracks', 1), 'n_alerts': len(alerts)},
        'alerts': alerts,
        'events': [],
        **extra,
    }
    (vdir / 'result.json').write_text(json.dumps(payload), encoding='utf-8')


def test_load_gt_csv_utf8_bom_and_multi_spans(tmp_path: Path):
    gt_file = tmp_path / 'gt.csv'
    gt_text = (
        'video,start_frame,end_frame,label,note\n'
        '1,95,120,touching,ok\n'
        '3,33,120,touching,span1\n'
        '3,115,120,touching,span2\n'
        '4,,,normal,no anomaly\n'
    )
    gt_file.write_bytes(b'\xef\xbb\xbf' + gt_text.encode('utf-8'))

    gt = eval_gt.load_gt_csv(gt_file)
    assert set(gt.keys()) == {'1', '3', '4'}
    assert len(gt['1']) == 1 and gt['1'][0].start_frame == 95
    assert len(gt['3']) == 2
    assert gt['4'] == []


def test_evaluate_all_metrics_hit_fp_delay_ids_red_pct(tmp_path: Path):
    gt_file = tmp_path / 'gt.csv'
    gt_file.write_text(
        'video,start_frame,end_frame,label,note\n'
        '1,50,100,touching,\n'
        '2,100,150,stealing,\n'
        '3,20,60,touching,\n'
        '4,,,normal,\n'
        '10,200,300,stealing,\n',
        encoding='utf-8-sig',
    )
    res_dir = tmp_path / 'results'

    # Video 1: hit trễ +10f, 1 FP ngoài GT, 2 track IDs khác nhau, red 60..80 (21/51 = 41.2%)
    _write_result(
        res_dir,
        '1',
        n_frames=120,
        alerts=[
            {'track_id': 2, 'alarm_start_frame': 60, 'alarm_end_frame': 80},
            {'track_id': 5, 'alarm_start_frame': 5, 'alarm_end_frame': 15},
        ],
    )
    # Video 2: miss (0 alerts)
    _write_result(res_dir, '2', n_frames=160, alerts=[])
    # Video 3: hit sớm -5f, 2 IDs cùng hit, phủ trọn GT -> 100%
    _write_result(
        res_dir,
        '3',
        n_frames=100,
        alerts=[
            {'track_id': 11, 'alarm_start_frame': 15, 'alarm_end_frame': 60},
            {'track_id': 20, 'alarm_start_frame': 40, 'alarm_end_frame': 60},
        ],
    )
    # Video 4: negative video nhưng có 1 FP alert
    _write_result(
        res_dir,
        '4',
        n_frames=80,
        alerts=[{'track_id': 1, 'alarm_start_frame': 10, 'alarm_end_frame': 20}],
    )
    # Video 10: thiếu result.json

    report = eval_gt.evaluate_all(res_dir, gt_file)
    by_vid = {r['video']: r for r in report['videos']}

    # Thứ tự sort số: 1, 2, 3, 4, 10
    assert [r['video'] for r in report['videos']] == ['1', '2', '3', '4', '10']

    v1 = by_vid['1']
    assert v1['hit'] is True
    assert v1['delay_frames'] == 10
    assert v1['fp_alerts'] == 1
    assert v1['tp_alerts'] == 1
    assert v1['n_unique_ids'] == 2
    assert v1['red_gt_pct'] == 41.2

    v2 = by_vid['2']
    assert v2['hit'] is False
    assert v2['delay_frames'] is None
    assert v2['red_gt_pct'] == 0.0

    v3 = by_vid['3']
    assert v3['hit'] is True
    assert v3['delay_frames'] == -5
    assert v3['fp_alerts'] == 0
    assert v3['n_unique_ids'] == 2
    assert v3['red_gt_pct'] == 100.0

    v4 = by_vid['4']
    assert v4['hit'] is None
    assert v4['fp_alerts'] == 1

    v10 = by_vid['10']
    assert v10['has_result'] is False
    assert v10['hit'] is False

    s = report['summary']
    assert s['n_gt_videos'] == 4
    assert s['hits'] == 2
    assert s['video_recall'] == 0.5
    assert s['total_alerts'] == 5
    assert s['tp_alerts'] == 3
    assert s['fp_alerts'] == 2
    assert s['alert_precision'] == 0.6


def test_flagged_tracks_and_alarm_persist_extend_red_coverage(tmp_path: Path):
    gt_file = tmp_path / 'gt.csv'
    gt_file.write_text(
        'video,start_frame,end_frame,label,note\n'
        '10,260,795,stealing,\n'
        '11,260,795,stealing,\n',
        encoding='utf-8',
    )
    res_dir = tmp_path / 'results'

    # Video 10: có flagged_tracks (TASK-001) từ frame 328 -> đỏ tới hết video (frame 818), phủ 328..795 = 468/536 = 87.3%
    _write_result(
        res_dir,
        '10',
        n_frames=819,
        alerts=[{'track_id': 3, 'alarm_start_frame': 328, 'alarm_end_frame': 358}],
        flagged_tracks={'3': {'behavior': 'touching', 'first_frame': 328}},
    )
    # Video 11: config.alarm_persist=True cũng giữ đỏ tới hết GT
    _write_result(
        res_dir,
        '11',
        n_frames=819,
        alerts=[{'track_id': 3, 'alarm_start_frame': 328, 'alarm_end_frame': 358}],
        config={'alarm_persist': True},
    )

    report = eval_gt.evaluate_all(res_dir, gt_file)
    by_vid = {r['video']: r for r in report['videos']}
    assert by_vid['10']['red_gt_pct'] == 87.3
    assert by_vid['11']['red_gt_pct'] == 87.3


def test_cli_outputs_table_and_json_file(tmp_path: Path, capsys):
    gt_file = tmp_path / 'gt.csv'
    gt_file.write_text('video,start_frame,end_frame,label,note\n1,10,19,touching,\n', encoding='utf-8')
    res_dir = tmp_path / 'results'
    _write_result(res_dir, '1', n_frames=30, alerts=[{'track_id': 1, 'alarm_start_frame': 12, 'alarm_end_frame': 19}])
    out_json = tmp_path / 'report.json'

    rc = eval_gt.main(['--results', str(res_dir), '--gt', str(gt_file), '--out', str(out_json)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert 'Summary:' in captured
    assert 'YES' in captured
    assert out_json.is_file()
    saved = json.loads(out_json.read_text(encoding='utf-8'))
    assert saved['summary']['hits'] == 1
    assert saved['videos'][0]['red_gt_pct'] == 80.0
