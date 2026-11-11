"""Tiến trình con chạy 1 model nặng mã Python (tracker ByteTrack-DMA-LTC, RTMW hand hoặc YOLOv8-pose) riêng, để stage tương ứng thoát GIL
của tiến trình chính.

Chạy bởi _RemoteProc (perception.py):  python -m exhibitflow.tracker_worker <địa chỉ socket>  (khoá xác thực qua EF_TRK_KEY).
Giao thức (request -> reply, đồng bộ từng lần): init / shm / reset / upd (tracker) / hand / pose / quit. Ảnh đi qua shared memory, không qua pickle.
"""
import os
import sys
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import Client

import numpy as np


def _attach(name):
    shm = shared_memory.SharedMemory(name=name)
    try:                                   # tiến trình chính sở hữu / unlink: không để resource_tracker của con cảnh báo "leaked"
        from multiprocessing import resource_tracker
        resource_tracker.unregister(shm._name, 'shared_memory')
    except Exception:
        pass
    return shm


def main():
    conn = Client(sys.argv[1], family='AF_UNIX', authkey=bytes.fromhex(os.environ['EF_TRK_KEY']))
    model, shm = None, None
    try:
        while True:
            msg = conn.recv()
            op = msg[0]
            try:
                if op == 'init':
                    cfg = msg[1]
                    import torch
                    n_thr = int(cfg.get('threads') or 1)
                    torch.set_num_threads(n_thr)
                    if n_thr > 1:                  # =1 giữ nguyên mặc định OpenCV như trước; >1 thì OpenCV cũng dùng bấy nhiêu luồng
                        import cv2
                        cv2.setNumThreads(n_thr)
                    from pathlib import Path
                    from exhibitflow import settings
                    settings.WEIGHT_ROOTS[:] = [Path(p) for p in cfg['weight_roots']]
                    settings.set_paths(work=cfg['work'])
                    for k, v in (cfg.get('cfg') or {}).items():
                        settings.CFG[k] = v
                    import exhibitflow.perception  # noqa: F401  đăng ký backend
                    from exhibitflow.common import build
                    model = build(cfg['kind'], cfg['spec'])
                    conn.send(('ok', None))
                elif op == 'shm':
                    if shm is not None:
                        shm.close()
                    shm = _attach(msg[1])
                    conn.send(('ok', None))
                elif op == 'reset':
                    model.reset(msg[1])
                    conn.send(('ok', None))
                elif op == 'upd':
                    _, shape, dets, s, use_native = msg
                    img = np.ndarray(shape, np.uint8, buffer=shm.buf)
                    out = model.update(dets, img, native=img if use_native else None, S=s)
                    conn.send(('ok', out))
                elif op == 'hand':
                    _, shape, recs, s = msg
                    img = np.ndarray(shape, np.uint8, buffer=shm.buf)
                    model(img, recs, s)
                    keys = ('hand_kp', 'hand_kc', 'hand_quality')
                    conn.send(('ok', [{k: r[k] for k in keys} if 'hand_kp' in r else None for r in recs]))
                elif op == 'pose':
                    _, shape, boxes, s = msg
                    img = np.ndarray(shape, np.uint8, buffer=shm.buf)
                    recs = [dict(box=b) for b in boxes]
                    model(img, recs, s)
                    keys = ('kp', 'kc', 'pose_crop', 'pose_box')
                    conn.send(('ok', [{k: r[k] for k in keys if k in r} if 'kp' in r else None for r in recs]))
                elif op == 'quit':
                    conn.send(('ok', None))
                    break
            except Exception:
                conn.send(('err', traceback.format_exc()))
    except (EOFError, ConnectionError):
        pass                                # tiến trình chính đã thoát
    finally:
        if shm is not None:
            shm.close()


if __name__ == '__main__':
    main()
