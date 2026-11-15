"""Nhiều camera trong một lần chạy (multi-camera).

Mỗi camera là một "job con" (một lần chạy pipeline như bình thường). Có hai cách chạy:
- Tuần tự (parallel=1): xong camera này mới tới camera sau. Engine trong MỘT tiến trình chỉ chạy một video tại một thời điểm.
- Song song (parallel>1): tối đa `parallel` job con chạy cùng lúc; camera chờ được cấp chỗ khi một camera xong. Server dùng
  tiến trình riêng cho từng job con (xem exhibitflow/cam_worker.py) nên các job không đụng biến toàn cục của engine.
Chung cho cả hai:
- Camera IP (RTSP) chạy tới khi hết thời lượng `camera_max_min` của mỗi lượt (hoặc bấm Dừng). File video chạy hết file.
- Cảnh báo của mọi camera được gộp vào một danh sách, mỗi cảnh báo mang nhãn `camera` (tên camera) và `cam` (thứ tự).
- Không đồng bộ thời gian giữa các camera: t_s là giây trong video của camera đó.
- Dừng (`finish`): các job con đang chạy kết thúc và xử lý nốt; camera chưa tới lượt bị bỏ qua ('skipped').
- `release(job_id)` (nếu có) được gọi khi job con đã xong hoặc lỗi, để server giải phóng tiến trình / VRAM của nó.

Lớp MultiRun không biết gì về HTTP / PipelineRunner: server truyền vào các hàm gọi lại, nhờ vậy kiểm thử được trên CPU
(xem tests/test_pipeline_cpu.py).
"""
import threading
import time
import uuid

MIN_CAMS, MAX_CAMS = 2, 8


def from_result_alerts(res):
    """Cảnh báo trong result.json của một camera -> dạng gộp (track, behavior, confidence, evidence, art, t_s)."""
    return [dict(track=a.get('track_id'), behavior=a.get('behavior'), confidence=a.get('confidence'), evidence=a.get('evidence') or '',
                 art=a.get('artifact_id'), t_s=a.get('alarm_start_time_s') or 0)
            for a in (res.get('alerts') or [])]


def from_live_alerts(alerts):
    """Cảnh báo đang có trong State (lúc đang chạy) -> dạng gộp."""
    return [dict(track=a.get('track_id'), behavior=a.get('behavior'), confidence=a.get('confidence'), evidence=a.get('evidence') or '',
                 art=(a.get('artifact_ids') or [None])[0], t_s=a.get('video_time') or 0)
            for a in alerts]


class MultiRun:
    """Chạy các camera trong một nhóm luồng nền (số luồng = `parallel`; 1 = tuần tự).

    cams: [dict(name=..., ...)] — `launch(spec)` nhận đúng dict này.
    launch(spec) -> job_id          (ném lỗi nếu không khởi chạy được; camera đó ghi nhận 'error' rồi chạy camera sau)
    poll(job_id) -> dict(status in {running, done, error, cancelled}, key, error)
    stop(job_id)                    (yêu cầu job con kết thúc và xử lý nốt)
    alerts_of(job_id, final) -> list | None
        final=False: cảnh báo đang có (khi đang chạy); final=True: cảnh báo đã chốt trong kết quả, None nếu chưa có kết quả.
    release(job_id)                 (tuỳ chọn) gọi khi job con đã xong / lỗi
    """

    def __init__(self, cams, launch, poll, stop, alerts_of, interval=0.5, parallel=1, release=None):
        self.id = uuid.uuid4().hex[:12]
        self._specs = list(cams)
        self.cams = [dict(idx=i, name=c['name'], status='pending', job_id=None, key=None, error=None, t0=None, t1=None)
                     for i, c in enumerate(self._specs)]
        self._launch, self._poll, self._stop, self._alerts_of = launch, poll, stop, alerts_of
        self._release = release
        self._parallel = max(1, int(parallel))
        self._interval = interval
        self._lock = threading.Lock()
        self._stop_requested = False
        self._next = 0                        # chỉ số camera kế tiếp chờ tới lượt
        self._running = set()                 # chỉ số các camera đang chạy
        self._done = False
        self._alerts = []                     # cảnh báo đã chốt của các camera đã xong (đã gắn nhãn camera)
        self._live = {}                       # idx -> cảnh báo đang có của camera đó (đã gắn nhãn)
        self.thread = threading.Thread(target=self._run, daemon=True, name=f'multi-{self.id}')

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        """Dừng: các job con đang chạy kết thúc và xử lý nốt; các camera còn lại bị bỏ qua."""
        with self._lock:
            self._stop_requested = True

    @staticmethod
    def _tag(alerts, cam):
        return [dict(a, camera=cam['name'], cam=cam['idx']) for a in (alerts or [])]

    def _take(self):
        """Lấy camera kế tiếp: (idx, True) nếu chạy, (idx, False) nếu bỏ qua vì đã dừng, None khi hết."""
        with self._lock:
            if self._next >= len(self.cams):
                return None
            idx = self._next
            self._next += 1
            if self._stop_requested:
                self.cams[idx]['status'] = 'skipped'
                return idx, False
            self.cams[idx]['status'] = 'running'
            self._running.add(idx)
            return idx, True

    def _worker(self):
        while (t := self._take()) is not None:
            idx, go = t
            if go:
                self._run_one(idx)

    def _run(self):
        try:
            workers = [threading.Thread(target=self._worker, daemon=True) for _ in range(min(self._parallel, len(self.cams)))]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
        finally:
            with self._lock:
                self._done = True

    def _run_one(self, idx):
        cam, spec = self.cams[idx], self._specs[idx]
        with self._lock:
            cam['t0'] = time.time()
        try:
            jid = self._launch(spec)
        except Exception as e:  # noqa: BLE001 - một camera lỗi không làm hỏng các camera khác
            with self._lock:
                cam['status'], cam['error'], cam['t1'] = 'error', str(e)[:300], time.time()
                self._running.discard(idx)
            return
        with self._lock:
            cam['job_id'] = jid
        sent, live, p, final = False, [], {'status': 'running'}, []
        try:
            while True:
                p = self._poll(jid)
                if p['status'] in ('done', 'error', 'cancelled'):
                    break
                with self._lock:
                    stop_now = self._stop_requested
                if stop_now and not sent:
                    self._stop(jid)
                    sent = True
                live = self._alerts_of(jid, False) or []
                with self._lock:
                    self._live[idx] = self._tag(live, cam)
                time.sleep(self._interval)
            final = self._alerts_of(jid, True)
            final = live if final is None else final
            with self._lock:
                cam['status'], cam['key'], cam['error'] = p['status'], p.get('key'), p.get('error')
        except Exception as e:  # noqa: BLE001
            final = live
            with self._lock:
                cam['status'], cam['error'] = 'error', str(e)[:300]
        finally:
            if self._release is not None:
                try:
                    self._release(jid)
                except Exception:  # noqa: BLE001 - dọn dẹp lỗi không được làm hỏng kết quả đã có
                    pass
            with self._lock:
                self._alerts += self._tag(final, cam)
                self._live.pop(idx, None)
                self._running.discard(idx)
                cam['t1'] = time.time()

    def _status(self):
        if not self._done:
            return 'running'
        if any(c['status'] in ('done', 'cancelled') for c in self.cams):
            return 'done'
        return 'error' if any(c['status'] == 'error' for c in self.cams) else 'done'

    def snapshot(self):
        with self._lock:
            alerts = self._alerts + [a for v in self._live.values() for a in v]
            cams = [dict(c, n_alerts=sum(1 for a in alerts if a['cam'] == c['idx'])) for c in self.cams]
            running = sorted(self._running)
            return dict(id=self.id, status=self._status(), cur=running[0] if running else None, running=running,
                        stop=self._stop_requested, cams=cams, alerts=list(alerts))
