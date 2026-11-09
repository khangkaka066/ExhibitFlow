"""Engine "nạp một lần, chạy nhiều video" cho dashboard.

`run_pipeline` (CLI) nạp model rồi giải phóng sau mỗi lần chạy. Ở dashboard ta muốn model nằm sẵn trong bộ nhớ
ngay khi server lên, nên tách riêng: `Engine.load()` nạp perception + segmenter + VLM, `Engine.run(video)` chỉ chạy video.
"""
from .common import *  # noqa: F401,F403
from .engine import finalize_video, run_video
from .pipeline import _write_json, build_models, collect_videos, video_result
from .segmenter import SegmenterSession, prepare_artifacts
from .vlm import VLMWorker


class Engine:
    def __init__(self):
        self.models = self.M = self.vlm = self.seg = None
        self.key = None            # khoá cấu hình model đã nạp (để biết khi nào phải nạp lại)
        self.ready = False
        self.n_runs = 0
        self.stage = ''             # đang nạp gì (dashboard hiển thị tiến độ)
        self.on_stage = None        # callback(msg) khi đổi bước (dashboard hiển thị)

    def load(self, overrides=None, work_dir=None, out_dir=None, dry_run_vlm=False, key=None):
        self.close()
        try:
            if overrides:
                settings.apply_overrides(overrides)
            settings.set_paths(work=work_dir, out=out_dir)
            self.models = copy.deepcopy(settings.MODELS)
            if dry_run_vlm:
                self.models['vlm'] = dict(backend='dry_run', delay_s=0.0)
            # VLM (tải llama.cpp + GGUF ~6 GB, khởi động llama-server) chạy ở thread riêng ngay từ đầu, song song với việc
            # nạp perception + segmenter -> tổng thời gian ~ max(các phần) thay vì cộng dồn.
            self.stage = 'Tải + nạp VLM (nền) và nạp model perception'
            print('\n[nạp] VLM bắt đầu ở nền (lần đầu tải llama.cpp + GGUF)', flush=True)
            self.vlm = VLMWorker(self.models['vlm'])
            self.vlm.start()
            print('\n[nạp] model perception (detector / tracker / pose / hand)', flush=True)
            self.M = build_models(self.models)
            self.stage = 'Nạp segmenter hiện vật (Grounding-DINO + SAM)'
            print('\n[nạp] segmenter hiện vật (Grounding-DINO + SAM)', flush=True)
            self.seg = SegmenterSession(self.models['segmenter'])
            self.seg.get()
            self.M.load_times['segmenter'] = self.seg.load_times
            self.stage = 'Đợi VLM (tải trọng số / khởi động llama-server)'
            print('\n[nạp] đợi VLM nạp xong', flush=True)
            self.vlm.loaded.wait()
            if self.vlm.load_error:
                raise RuntimeError('VLM nạp lỗi (xem log ở trên).')
            self.M.load_times['vlm'] = dict(load_s=self.vlm.load_s, warmup_s=None)
            self.stage = ''
            self.key, self.ready = key, True
            print('\n[nạp] XONG — model sẵn sàng', flush=True)
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

    def run(self, video, max_frames=None):
        """Chạy 1 video với model đã nạp. Trả dict result.json."""
        if not self.ready:
            raise RuntimeError('Model chưa nạp xong')
        RUN['max_frames'] = int(max_frames) if max_frames else None
        RT['render_live'] = False
        key, vpath = next(iter(collect_videos([video]).items()))
        self.n_runs += 1
        exp = f'run{self.n_runs}'                      # tiền tố task_id riêng mỗi lần chạy: không lẫn kết quả VLM cũ
        self.set_stage('Tách cảnh + tìm hiện vật (Grounding-DINO + SAM)')
        t0 = time.perf_counter()
        art = prepare_artifacts(key, dict(path=vpath, **video_meta(vpath)), self.M, self.seg)
        print(f'  chuẩn bị hiện vật: {time.perf_counter() - t0:.1f}s ({art.timing.get("source") or "lỗi"}, {len(art.scenes)} cảnh)', flush=True)
        self.set_stage('Chạy pipeline')
        d = dict(path=vpath, **video_meta(vpath, RUN['max_frames']))
        print(f"\n> {key} | {d['n']} frames @ {d['fps']:.1f}fps | {d['NW']}x{d['NH']}", flush=True)
        ctx = run_video(exp, key, d, self.M, self.vlm, art, RT['mode'], RT)
        self.vlm.drain()
        finalize_video(ctx)
        recs = [r for r in self.vlm.records if r['task_id'].startswith(f'{exp}/{key}/')]
        res = video_result(ctx, vpath, d, self.models, recs, self.M.load_times)
        _write_json(ctx.dir / 'result.json', res)
        ctx.M = ctx.vlm = None
        return res

    def close(self):
        self.ready = False
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
