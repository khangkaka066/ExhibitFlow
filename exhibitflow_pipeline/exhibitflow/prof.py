"""Đo thời gian chi tiết bên trong từng stage (mục `performance.breakdown` của result.json, in ra log khi video xong).

`span('pose.predict')` cộng dồn số lần gọi + tổng thời gian theo tên. Tiến trình con (tracker / hand / pose) có bộ đếm riêng: cuối video
tiến trình chính gọi `collect_prof()` của từng model để lấy về. Chi phí đo: 1 lần perf_counter + 1 khoá mỗi span (không đáng kể)."""
import threading
import time
from contextlib import contextmanager

_LOCK = threading.Lock()
_D = {}


def add(name, dt, n=1):
    with _LOCK:
        e = _D.setdefault(name, [0, 0.0])
        e[0] += n
        e[1] += dt


@contextmanager
def span(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        add(name, time.perf_counter() - t0)


def reset():
    with _LOCK:
        _D.clear()


def take():
    """Lấy toàn bộ bộ đếm rồi xoá (dùng ở tiến trình con để gửi về tiến trình chính)."""
    with _LOCK:
        out = {k: list(v) for k, v in _D.items()}
        _D.clear()
    return out


def merge(snap):
    for k, (n, s) in (snap or {}).items():
        add(k, s, n)


def summary(frames):
    """{tên: dict(calls, total_ms, ms_per_frame, ms_per_call)}; ms_per_frame = tổng / số frame đã xử lý."""
    with _LOCK:
        items = sorted(_D.items())
    fr = max(int(frames or 0), 1)
    return {k: dict(calls=int(n), total_ms=round(s * 1000, 1), ms_per_frame=round(s * 1000 / fr, 2),
                    ms_per_call=round(s * 1000 / max(n, 1), 2)) for k, (n, s) in items}


def format_table(summ):
    rows = ['  chi tiết thời gian (ms/frame | ms/lần gọi | số lần gọi):']
    for k, v in summ.items():
        rows.append(f"    {k:28s} {v['ms_per_frame']:8.2f} | {v['ms_per_call']:8.2f} | {v['calls']}")
    return '\n'.join(rows)
