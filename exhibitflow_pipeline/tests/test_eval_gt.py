"""Unit tests cho scripts/eval_gt.py (chỉ dùng JSON/CSV giả lập, không chạy model)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import eval_gt  # noqa: E402
import compare_runs  # noqa: E402


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


def test_label_matching_and_label_acc_in_eval_gt(tmp_path: Path):
    assert eval_gt.label_matches('touching', 'touching') is True
    assert eval_gt.label_matches('touching', 'stealing') is False
    assert eval_gt.label_matches('stealing', 'stealing') is True
    assert eval_gt.label_matches('stealing', 'holding') is True
    assert eval_gt.label_matches('stealing', 'touching') is False
    assert eval_gt.label_matches('stealing', '') is False

    gt_file = tmp_path / 'gt.csv'
    gt_file.write_text(
        'video,start_frame,end_frame,label,note\n'
        '1,50,100,touching,\n'
        '2,50,100,stealing,\n'
        '3,50,100,stealing,\n'
        '4,50,100,stealing,\n'
        '5,50,100,stealing,\n',
        encoding='utf-8',
    )
    res_dir = tmp_path / 'results'
    # 1: touching == touching -> label_ok True
    _write_result(res_dir, '1', n_frames=120, alerts=[{'track_id': 1, 'behavior': 'touching', 'alarm_start_frame': 55, 'alarm_end_frame': 80}])
    # 2: stealing GT, holding pred -> label_ok True
    _write_result(res_dir, '2', n_frames=120, alerts=[{'track_id': 1, 'behavior': 'holding', 'alarm_start_frame': 55, 'alarm_end_frame': 80}])
    # 3: stealing GT, stealing pred -> label_ok True
    _write_result(res_dir, '3', n_frames=120, alerts=[{'track_id': 1, 'behavior': 'stealing', 'alarm_start_frame': 55, 'alarm_end_frame': 80}])
    # 4: stealing GT, touching pred -> hit True nhưng label_ok False ("đúng alert nhưng sai nhãn")
    _write_result(res_dir, '4', n_frames=120, alerts=[{'track_id': 1, 'behavior': 'touching', 'alarm_start_frame': 55, 'alarm_end_frame': 80}])
    # 5: miss -> label_ok None (không tính vào label_eval_count)
    _write_result(res_dir, '5', n_frames=120, alerts=[])

    report = eval_gt.evaluate_all(res_dir, gt_file)
    by_vid = {r['video']: r for r in report['videos']}
    assert by_vid['1']['label_ok'] is True
    assert by_vid['2']['label_ok'] is True
    assert by_vid['3']['label_ok'] is True
    assert by_vid['4']['hit'] is True
    assert by_vid['4']['label_ok'] is False
    assert by_vid['4']['pred_labels'] == ['touching']
    assert by_vid['5']['hit'] is False
    assert by_vid['5']['label_ok'] is None

    s = report['summary']
    assert s['label_ok_count'] == 3
    assert s['label_eval_count'] == 4
    assert s['label_acc'] == 0.75

    txt = eval_gt.format_report(report)
    header_line = txt.splitlines()[0]
    for col in ['video', 'label', 'gt_frames', 'hit', 'delay(f)', 'alerts', 'fp', 'ids', 'red_gt%', 'tracks', 'label_ok']:
        assert col in header_line
    assert 'label_acc=75.0% (3/4)' in txt


def test_compare_runs_detects_regressions_improvements_and_masks(tmp_path: Path, capsys):
    gt_file = tmp_path / 'gt.csv'
    gt_file.write_text(
        'video,start_frame,end_frame,label,note\n'
        '1,50,100,touching,\n'
        '2,50,100,stealing,\n'
        '3,50,100,stealing,\n'
        '4,50,100,touching,\n'
        '5,,,normal,\n',
        encoding='utf-8',
    )
    dir_a = tmp_path / 'run_a'
    dir_b = tmp_path / 'run_b'

    # Video 1: tụt hit + red_gt + masks (19 -> 6) => REGRESS
    _write_result(
        dir_a, '1', n_frames=120,
        alerts=[{'track_id': 1, 'behavior': 'touching', 'alarm_start_frame': 50, 'alarm_end_frame': 100}],
        artifacts=[{}] * 19, scenes=[{'start_frame': 0, 'n_artifacts': 19}],
    )
    _write_result(
        dir_b, '1', n_frames=120, alerts=[],
        artifacts=[{}] * 6, scenes=[{'start_frame': 0, 'n_artifacts': 6}],
    )

    # Video 2: tụt label_ok (holding -> touching trên stealing GT) => REGRESS
    _write_result(
        dir_a, '2', n_frames=120,
        alerts=[{'track_id': 1, 'behavior': 'holding', 'alarm_start_frame': 50, 'alarm_end_frame': 100}],
        artifacts=[{}] * 8, scenes=[{'start_frame': 0, 'n_artifacts': 8}],
    )
    _write_result(
        dir_b, '2', n_frames=120,
        alerts=[{'track_id': 1, 'behavior': 'touching', 'alarm_start_frame': 50, 'alarm_end_frame': 100}],
        artifacts=[{}] * 8, scenes=[{'start_frame': 0, 'n_artifacts': 8}],
    )

    # Video 3: cải thiện hit (miss -> holding) => IMPROVED
    _write_result(
        dir_a, '3', n_frames=120, alerts=[],
        artifacts=[{}] * 5, scenes=[{'start_frame': 0, 'n_artifacts': 5}],
    )
    _write_result(
        dir_b, '3', n_frames=120,
        alerts=[{'track_id': 2, 'behavior': 'holding', 'alarm_start_frame': 55, 'alarm_end_frame': 100}],
        artifacts=[{}] * 6, scenes=[{'start_frame': 0, 'n_artifacts': 6}],
    )

    # Video 4: cải thiện hit + red_gt nhưng masks tụt mạnh 10 -> 3 => MIXED
    _write_result(
        dir_a, '4', n_frames=120, alerts=[],
        artifacts=[{}] * 10, scenes=[{'start_frame': 0, 'n_artifacts': 10}],
    )
    _write_result(
        dir_b, '4', n_frames=120,
        alerts=[{'track_id': 3, 'behavior': 'touching', 'alarm_start_frame': 50, 'alarm_end_frame': 100}],
        artifacts=[{}] * 3, scenes=[{'start_frame': 0, 'n_artifacts': 3}],
    )

    # Video 5: normal video phát sinh FP mới => REGRESS(fp)
    _write_result(dir_a, '5', n_frames=100, alerts=[], artifacts=[{}] * 4, scenes=[{'start_frame': 0, 'n_artifacts': 4}])
    _write_result(
        dir_b, '5', n_frames=100,
        alerts=[{'track_id': 1, 'behavior': 'touching', 'alarm_start_frame': 10, 'alarm_end_frame': 20}],
        artifacts=[{}] * 4, scenes=[{'start_frame': 0, 'n_artifacts': 4}],
    )

    out_json = tmp_path / 'cmp.json'
    rc = compare_runs.main(['--a', str(dir_a), '--b', str(dir_b), '--gt', str(gt_file), '--out', str(out_json)])
    assert rc == 0
    stdout = capsys.readouterr().out
    assert 'Summary A->B:' in stdout
    assert 'Regressed videos:' in stdout
    assert out_json.is_file()

    rep = json.loads(out_json.read_text(encoding='utf-8'))
    by_vid = {v['video']: v for v in rep['videos']}
    assert by_vid['1']['regressed'] is True
    assert {'hit', 'red_gt', 'masks'} <= set(by_vid['1']['regressions'])
    assert by_vid['2']['regressed'] is True
    assert 'label' in by_vid['2']['regressions']
    assert by_vid['3']['regressed'] is False
    assert 'hit' in by_vid['3']['improvements']
    assert 'MIXED' in by_vid['4']['status']
    assert by_vid['5']['regressed'] is True
    assert 'fp' in by_vid['5']['regressions']
    assert rep['summary']['regressed_videos'] == ['1', '2', '5']
    assert rep['summary']['mask_drop_videos'] == ['1', '4']

    # Chạy không có GT vẫn so sánh được số mask & alerts
    no_gt = compare_runs.compare_runs(dir_a, dir_b, gt_path=None)
    no_gt_v1 = next(v for v in no_gt['videos'] if v['video'] == '1')
    assert no_gt_v1['a']['n_artifacts'] == 19 and no_gt_v1['b']['n_artifacts'] == 6
    assert no_gt_v1['regressed'] is True

