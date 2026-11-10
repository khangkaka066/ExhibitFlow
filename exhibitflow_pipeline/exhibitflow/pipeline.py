"""Điều phối cả pipeline: nạp model -> segment hiện vật -> nạp VLM -> chạy từng video -> ghi JSON."""
import datetime as _dt
import platform

from .common import *  # noqa: F401,F403
from . import perception  # noqa: F401  (đăng ký detector / tracker / pose / hand)
from .engine import run_video
from .interaction import decide, vlm_contact_indices  # noqa: F401
from .engine import finalize_video
from .segmenter import SegmenterSession, prepare_artifacts
from .vlm import VLMWorker

PERCEPTION = ('detector', 'tracker', 'pose', 'hand')
EXP = 'run'                      # tiền tố task_id; thư mục kết quả = <out>/<video_key>/
SCHEMA_VERSION = '1.0'


# ---------------- Các bước ----------------
def build_models(models):
    """Nạp model perception. Segmenter và VLM được nạp riêng (segment_all_videos / start_vlm)."""
    M = SimpleNamespace(specs=copy.deepcopy(models))
    for kind in PERCEPTION:
        setattr(M, kind, build(kind, models[kind]))
    M.load_times = {k: dict(load_s=getattr(M, k).load_s, warmup_s=getattr(M, k).warmup_s) for k in PERCEPTION}
    return M


def segment_all_videos(models, M, video_paths):
    """Segment hiện vật cho mọi video, rồi giải phóng segmenter trước khi nạp VLM (trả VRAM)."""
    seg = SegmenterSession(models['segmenter'])
    arts = {}
    try:
        for key, vpath in video_paths.items():
            d = dict(path=vpath, **video_meta(vpath))
            arts[key] = a = prepare_artifacts(key, d, M, seg)
            sc = ', '.join(f"f{s['start']}(seg f{s['ref_idx']}): {len(s['artifacts'])}" for s in a.scenes)
            print(f"  {key}: {len(a.artifacts)} hiện vật, {len(a.scenes)} cảnh [{sc}] "
                  f"({a.timing['source'] or 'lỗi'}, {a.timing['seconds']}s)")
    finally:
        seg.close()
    M.load_times['segmenter'] = seg.load_times
    return arts


def start_vlm(models, M):
    vlm = VLMWorker(models['vlm'])
    vlm.start()                                   # nạp trên thread riêng
    if RUN['wait_vlm_loaded']:
        vlm.loaded.wait()
        if vlm.load_error:
            raise RuntimeError('VLM nạp lỗi -> dừng (xem log ở trên).')
    M.load_times['vlm'] = dict(load_s=vlm.load_s, warmup_s=None)
    return vlm


# ---------------- JSON ----------------
def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _pct(x, q):
    return round(float(np.percentile(x, q)), 3) if len(x) else None


def _t(frame, fps):
    return None if frame is None else round(float(frame) / fps, 3)


def _span(lo, hi, fps):
    return dict(start_frame=int(lo), end_frame=int(hi), start_time_s=_t(lo, fps), end_time_s=_t(hi, fps))


def _performance(ctx, vlm_records):
    df = ctx.rec.df()
    skip = 10
    stages = {}
    for st, g in df.groupby('stage'):
        g = g[g.frame >= skip] if len(g) > skip else g
        stages[st] = dict(mean_ms=round(float(g.compute_ms.mean()), 2), p95_ms=_pct(g.compute_ms, 95), frames=int(len(g)))
    fl = ctx.frame_log
    e2e = [r['e2e_ms'] for r in fl[skip:]] or [r['e2e_ms'] for r in fl]
    span = fl[-1]['t_out'] - fl[0]['t_out'] - ctx.pause_overlap(fl[0]['t_out'], fl[-1]['t_out']) if len(fl) > 1 else 0.0
    fps = (len(fl) - 1) / span if span > 0 else 0.0
    pause_s = sum(e - s for s, e in ctx.pauses)
    inf = [r['infer_s'] for r in vlm_records if r.get('status') == 'ok']
    return dict(
        pipeline_fps=round(fps, 2), vlm_sync_pauses=len(ctx.pauses), vlm_sync_pause_s=round(pause_s, 2), wall_s=round(ctx.wall_s, 2), wall_with_vlm_s=round(ctx.wall_with_vlm_s, 2),
        frames_processed=len(fl), frames_dropped=len(ctx.dropped), e2e_ms=dict(mean=round(float(np.mean(e2e)), 1) if e2e else None,
                                                                              p95=_pct(e2e, 95)),
        stages=stages,
        vlm=dict(n_tasks=len(vlm_records), n_ok=len(inf), infer_mean_s=round(float(np.mean(inf)), 3) if inf else None,
                 infer_p95_s=_pct(inf, 95),
                 queue_wait_mean_s=round(float(np.mean([r['queue_wait_s'] for r in vlm_records])), 3) if vlm_records else None),
        gpu_peak_gb=ctx.gpu_peak_gb)


def video_result(ctx, vpath, d, models, vlm_records, load_times):
    """Gom kết quả của 1 video thành dict JSON-safe."""
    fps = ctx.fps
    events = []
    for e in ctx.inter.events:
        if 'eid' not in e:
            continue
        events.append(dict(
            event_id=e['eid'], track_id=int(e['track_id']), artifact_id=int(e['artifact_id']), hand=e.get('hand'),
            start_frame=int(e['start_frame']), end_frame=int(e['end_frame']),
            start_time_s=_t(e['start_frame'], fps), end_time_s=_t(e['end_frame'], fps),
            decision=e.get('decision'), behavior=e.get('behavior'), vlm_confidence=e.get('vlm_conf'),
            vlm_evidence=e.get('vlm_evidence'), task=e.get('task'),
            alarm_spans=[_span(lo, hi, fps) for lo, hi in (e.get('alarm_spans') or [])],
            alarm_onset_source=e.get('alarm_onset_source')))
    alerts = []
    for e in events:
        if e['decision'] == 'ANOMALY':
            first = e['alarm_spans'][0] if e['alarm_spans'] else None
            alerts.append(dict(event_id=e['event_id'], track_id=e['track_id'], artifact_id=e['artifact_id'],
                               behavior=e['behavior'], confidence=e['vlm_confidence'], evidence=e['vlm_evidence'],
                               alarm_start_frame=first['start_frame'] if first else e['start_frame'],
                               alarm_start_time_s=first['start_time_s'] if first else e['start_time_s'],
                               alarm_end_frame=first['end_frame'] if first else e['end_frame'],
                               alarm_end_time_s=first['end_time_s'] if first else e['end_time_s'],
                               onset_source=e['alarm_onset_source'], task=e['task']))
    by_task = {r['task_id']: r for r in vlm_records}
    tasks = []
    for g in ctx.inter.tasks:
        r = by_task.get(g.get('task_id'), {})
        tasks.append(dict(
            task=g.get('name'), task_id=g.get('task_id'), track_id=int(g['tid']), artifact_ids=[int(a) for a in g['artifact_ids']],
            window_frames=[int(x) for x in g['window']], status=g.get('vlm_status'),
            decision=g.get('decision'), label=r.get('label'), confidence=g.get('vlm_conf'), evidence=g.get('vlm_evidence'),
            raw=r.get('raw'), n_images=r.get('n_images'), n_input_tokens=r.get('n_input_tokens'),
            latency_s=r.get('latency_s'), infer_s=r.get('infer_s'), prefill_s=r.get('prefill_s'),
            queue_wait_s=r.get('queue_wait_s'), error=r.get('error')))
    labels = Counter(a['behavior'] for a in alerts)
    return dict(
        schema_version=SCHEMA_VERSION,
        generated_at=_dt.datetime.now().astimezone().isoformat(timespec='seconds'),
        video=dict(path=str(vpath), name=Path(vpath).name, key=ctx.key, fps=round(fps, 3), n_frames=int(ctx.n),
                   width=int(d['NW']), height=int(d['NH'])),
        summary=dict(n_people_tracks=len({r['id'] for fr in ctx.tracks.values() for r in fr}), n_events=len(events),
                     n_alerts=len(alerts), alerts_by_behavior=dict(labels), n_vlm_tasks=len(tasks),
                     has_anomaly=bool(alerts)),
        alerts=alerts, events=events, vlm_tasks=tasks,
        artifacts=[dict(id=int(a['id']), label=a.get('label'), box=[round(float(v), 1) for v in a['box']],
                        pedestal=bool(a.get('pedestal'))) for a in ctx.art.artifacts],
        scenes=[dict(start_frame=int(s['start']), segment_frame=int(s['ref_idx']), n_artifacts=len(s['artifacts']))
                for s in ctx.art.scenes],
        performance=_performance(ctx, vlm_records),
        warnings=([f'segmentation lỗi: {ctx.art.error[-600:]}'] if ctx.art.error else []),
        config=dict(version=settings.VERSION, models=describe_models(models), vlm=models['vlm'], load_times=load_times,
                    thresholds=dict(CFG=CFG, EXPORT=EXPORT), alarm_onset=settings.ALARM_ONSET,
                    environment=dict(python=platform.python_version(), gpus=settings.GPU_NAMES)))


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable), encoding='utf-8')


# ---------------- Chạy ----------------
def collect_videos(videos):
    """Danh sách đường dẫn / thư mục -> {key: Path}. key = tên file bỏ đuôi (thêm hậu tố nếu trùng)."""
    paths = []
    for v in videos:
        p = Path(v)
        if p.is_dir():
            paths += sorted(x for x in p.iterdir() if x.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        elif p.is_file():
            paths.append(p)
        else:
            raise FileNotFoundError(f'Không thấy video: {v}')
    if not paths:
        raise FileNotFoundError('Không có video nào để chạy.')
    out = {}
    for p in paths:
        key, k = p.stem, 2
        while key in out:
            key, k = f'{p.stem}_{k}', k + 1
        out[key] = p
    return out


def run_pipeline(videos, out_dir=None, work_dir=None, config_files=(), overrides=None, max_frames=None,
                 save_video=False, dry_run_vlm=False, clean_vlm_inputs=False):
    """Chạy ExhibitFlow trên các video. Trả dict tổng hợp; kết quả từng video ở <out_dir>/<key>/result.json."""
    for f in config_files:
        settings.load_config_file(f)
    if overrides:
        settings.apply_overrides(overrides)
    settings.set_paths(work=work_dir, out=out_dir)
    if max_frames:
        RUN['max_frames'] = int(max_frames)
    RT['render_live'] = bool(save_video)
    models = copy.deepcopy(settings.MODELS)
    if dry_run_vlm:
        models['vlm'] = dict(backend='dry_run', delay_s=0.0)
    video_paths = collect_videos(videos)
    out_root = settings.OUT
    print(f'ExhibitFlow {settings.VERSION} | {len(video_paths)} video | GPU: {settings.GPU_NAMES or "không có (CPU)"}')
    print(describe_models(models))
    summary = dict(schema_version=SCHEMA_VERSION, version=settings.VERSION, videos=[], errors=[])
    t_all = time.perf_counter()
    M = vlm = None
    try:
        print('\n① Nạp model perception')
        M = build_models(models)
        print('\n② Segmentation hiện vật (trước pipeline)')
        arts = segment_all_videos(models, M, video_paths)
        print('\n③ Nạp VLM')
        vlm = start_vlm(models, M)
        print('\n④ Chạy pipeline')
        for key, vpath in video_paths.items():
            d = dict(path=vpath, **video_meta(vpath, RUN['max_frames']))
            print(f"\n▶ {key} | {d['n']} frames @ {d['fps']:.1f}fps | {d['NW']}x{d['NH']}")
            try:
                ctx = run_video(EXP, key, d, M, vlm, arts[key], RT['mode'], RT)
                vlm.drain()
                finalize_video(ctx)
                recs = [r for r in vlm.records if r['task_id'].startswith(f'{EXP}/{key}/')]
                res = video_result(ctx, vpath, d, models, recs, M.load_times)
                vdir = ctx.dir
                _write_json(vdir / 'result.json', res)
                if clean_vlm_inputs:
                    shutil.rmtree(vdir / 'vlm_inputs', ignore_errors=True)
                s = res['summary']
                print(f"  {res['performance']['pipeline_fps']} FPS | event {s['n_events']} | đoạn VLM {s['n_vlm_tasks']} | "
                      f"ALERT {s['n_alerts']} {s['alerts_by_behavior']} -> {vdir / 'result.json'}")
                summary['videos'].append(dict(key=key, video=str(vpath), result=str((vdir / 'result.json').relative_to(out_root)),
                                              **s))
                ctx.M = ctx.vlm = None
            except Exception:
                tb = traceback.format_exc()
                print(f'!!! {key} lỗi:\n{tb[-2000:]}')
                summary['errors'].append(dict(video=key, error=tb[-2000:]))
    finally:
        if vlm is not None:
            vlm.stop()
        del M, vlm
        free_gpu()
    summary['total_s'] = round(time.perf_counter() - t_all, 1)
    summary['n_videos'] = len(video_paths)
    summary['n_alerts'] = sum(v['n_alerts'] for v in summary['videos'])
    _write_json(out_root / 'summary.json', summary)
    print(f"\nXong {len(summary['videos'])}/{len(video_paths)} video sau {summary['total_s']}s | "
          f"{summary['n_alerts']} cảnh báo | {out_root / 'summary.json'}")
    return summary
