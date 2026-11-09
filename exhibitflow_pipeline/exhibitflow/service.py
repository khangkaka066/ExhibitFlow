"""Engine "nạp một lần, chạy nhiều video" cho dashboard.

`run_pipeline` (CLI) nạp model rồi giải phóng sau mỗi lần chạy. Ở dashboard ta muốn model nằm sẵn trong bộ nhớ
ngay khi server lên, nên tách riêng: `Engine.load()` nạp model, `Engine.run(video)` chỉ chạy video.

Ba chế độ (tăng dần):
    tracking      detector + tracker                                    (không hiện vật, không pose/hand, không VLM)
    tracking_seg  + segmentation hiện vật + pose/hand + phát hiện chạm  (event "CANDIDATE", chưa có cảnh báo)
    full          + VLM xác nhận chạm / cầm / lấy -> cảnh báo
Chỉ nạp phần cần cho chế độ đã chọn. Chọn chế độ nhẹ hơn khi model đã nạp thì không phải nạp lại.
"""
from types import SimpleNamespace

from .common import *  # noqa: F401,F403
from .engine import finalize_video, run_video
from .pipeline import _write_json, collect_videos, video_result
from .segmenter import ArtifactResult, SegmenterSession, prepare_artifacts
from .vlm import VLMWorker

MODES = ('tracking', 'tracking_seg', 'full')
MODE_COMPONENTS = {'tracking': frozenset(), 'tracking_seg': frozenset({'seg', 'pose'}),
                   'full': frozenset({'seg', 'pose', 'vlm'})}


def components_for(mode):
    return MODE_COMPONENTS.get(mode, MODE_COMPONENTS['full'])


class NullVLM:
    """VLM tắt: nhận task rồi bỏ qua -> event ở trạng thái CANDIDATE, không có cảnh báo."""
    records, results = [], {}

    def submit(self, task, callback):
        pass

    def pending(self):
        return 0

    def drain(self):
        pass

    def cancel_pending(self):
        return 0

    def stop(self):
        pass

    def __init__(self):
        import threading
        self.gate = threading.Event()
        self.gate.set()


class Engine:
    def __init__(self):
        self.models = self.M = self.vlm = self.seg = None
        self.key = None            # khoá cấu hình model đã nạp (để biết khi nào phải nạp lại)
        self.components = frozenset()   # {'seg', 'pose', 'vlm'} đã nạp (detector + tracker luôn có)
        self.ready = False
        self.n_runs = 0
        self.stage = ''             # đang nạp gì (dashboard hiển thị tiến độ)
        self.on_stage = None        # callback(msg) khi đổi bước (dashboard hiển thị)

    def covers(self, components):
        return self.ready and frozenset(components) <= self.components

    def load(self, overrides=None, work_dir=None, out_dir=None, dry_run_vlm=False, key=None, components=None):
        components = frozenset(MODE_COMPONENTS['full'] if components is None else components)
        self.close()
        try:
            if overrides:
                settings.apply_overrides(overrides)
            settings.set_paths(work=work_dir, out=out_dir)
            self.models = copy.deepcopy(settings.MODELS)
            if dry_run_vlm:
                self.models['vlm'] = dict(backend='dry_run', delay_s=0.0)
            if 'vlm' in components:
                # VLM (tải llama.cpp + GGUF ~6 GB, khởi động llama-server) chạy ở thread riêng ngay từ đầu, song song với việc
                # nạp perception + segmenter -> tổng thời gian ~ max(các phần) thay vì cộng dồn.
                self.stage = 'Tải + nạp VLM (nền) và nạp model perception'
                print('\n[nạp] VLM bắt đầu ở nền (lần đầu tải llama.cpp + GGUF)', flush=True)
                self.vlm = VLMWorker(self.models['vlm'])
                self.vlm.start()
            else:
                self.stage = 'Nạp detector + tracker'
            print('\n[nạp] model perception', flush=True)
            self.M = SimpleNamespace(specs=copy.deepcopy(self.models))
            self.M.detector = build('detector', self.models['detector'])
            self.M.tracker = build('tracker', self.models['tracker'])
            if 'pose' in components:
                self.M.pose = build('pose', self.models['pose'])
                self.M.hand = build('hand', self.models['hand'])
            else:                                          # không cần pose/hand: đối tượng rỗng (không nạp gì)
                self.M.pose = self.M.hand = build('hand', dict(backend='none'))
            self.M.load_times = {k: dict(load_s=getattr(getattr(self.M, k), 'load_s', None),
                                         warmup_s=getattr(getattr(self.M, k), 'warmup_s', None))
                                 for k in ('detector', 'tracker', 'pose', 'hand')}
            if 'seg' in components:
                self.set_stage('Nạp segmenter hiện vật (Grounding-DINO + SAM)')
                print('\n[nạp] segmenter hiện vật (Grounding-DINO + SAM)', flush=True)
                self.seg = SegmenterSession(self.models['segmenter'])
                self.seg.get()
                self.M.load_times['segmenter'] = self.seg.load_times
            if self.vlm is not None:
                self.set_stage('Đợi VLM (tải trọng số / khởi động llama-server)')
                print('\n[nạp] đợi VLM nạp xong', flush=True)
                self.vlm.loaded.wait()
                if self.vlm.load_error:
                    raise RuntimeError('VLM nạp lỗi (xem log ở trên).')
                self.M.load_times['vlm'] = dict(load_s=self.vlm.load_s, warmup_s=None)
            self.stage = ''
            self.key, self.components, self.ready = key, components, True
            print(f'\n[nạp] XONG — sẵn sàng ({"+".join(sorted(components)) or "chỉ tracking"})', flush=True)
        except BaseException:
            self.close()
            raise

    def set_stage(self, msg):
        self.stage = msg
        if self.on_stage:
            try:
                self.on_stage(msg)
            except Exception:
                pass

    def run(self, video, max_frames=None, mode='full'):
        """Chạy 1 video với model đã nạp theo `mode`. Trả dict result.json."""
        need = components_for(mode)
        if not self.covers(need):
            raise RuntimeError(f'Chế độ "{mode}" cần {sorted(need)} nhưng model đã nạp chỉ có {sorted(self.components)}')
        RUN['max_frames'] = int(max_frames) if max_frames else None
        RT['render_live'] = False
        key, vpath = next(iter(collect_videos([video]).items()))
        self.n_runs += 1
        exp = f'run{self.n_runs}'                      # tiền tố task_id riêng mỗi lần chạy: không lẫn kết quả VLM cũ
        M = self.M
        if 'pose' not in need:                         # chế độ tracking: bỏ pose/hand dù đang nạp sẵn
            M = SimpleNamespace(**vars(self.M))
            M.pose = M.hand = build('hand', dict(backend='none'))
        vlm = self.vlm if 'vlm' in need else NullVLM()
        if 'seg' in need:
            self.set_stage('Tách cảnh + tìm hiện vật (Grounding-DINO + SAM)')
            t0 = time.perf_counter()
            art = prepare_artifacts(key, dict(path=vpath, **video_meta(vpath)), M, self.seg)
            print(f'  chuẩn bị hiện vật: {time.perf_counter() - t0:.1f}s ({art.timing.get("source") or "lỗi"}, '
                  f'{len(art.scenes)} cảnh)', flush=True)
        else:
            art = ArtifactResult(key)                  # không hiện vật -> không có event chạm, chỉ tracking
        self.set_stage('Chạy pipeline')
        d = dict(path=vpath, **video_meta(vpath, RUN['max_frames']))
        print(f"\n> {key} [{mode}] | {d['n']} frames @ {d['fps']:.1f}fps | {d['NW']}x{d['NH']}", flush=True)
        ctx = run_video(exp, key, d, M, vlm, art, RT['mode'], RT)
        vlm.drain()
        finalize_video(ctx)
        recs = [r for r in vlm.records if r['task_id'].startswith(f'{exp}/{key}/')]
        res = video_result(ctx, vpath, d, self.models, recs, M.load_times)
        res['mode'] = mode
        _write_json(ctx.dir / 'result.json', res)
        ctx.M = ctx.vlm = None
        return res

    def close(self):
        self.ready = False
        self.components = frozenset()
        if self.vlm is not None:
            try:
                self.vlm.stop()
            except Exception:
                pass
        if self.seg is not None:
            try:
                self.seg.close()
            except Exception:
                pass
        self.models = self.M = self.vlm = self.seg = None
        free_gpu()
