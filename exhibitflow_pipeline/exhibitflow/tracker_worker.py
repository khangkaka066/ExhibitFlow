"""Tiến trình con chạy tracker (ByteTrack-DMA-LTC + ReID) riêng, để stage `track` thoát GIL của tiến trình chính.

Chạy bởi RemoteTracker (perception.py):  python -m exhibitflow.tracker_worker <địa chỉ socket>  (khoá xác thực qua biến EF_TRK_KEY).
Giao thức (request -> reply, đồng bộ từng lần): init / shm / reset / upd / quit. Ảnh đi qua shared memory, không qua pickle.
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
    trk, shm = None, None
    try:
        while True:
            msg = conn.recv()
            op = msg[0]
            try:
                if op == 'init':
                    cfg = msg[1]
                    import torch
                    torch.set_num_threads(int(cfg.get('threads') or 1))
                    from exhibitflow import settings
                    settings.WEIGHT_ROOTS[:] = [__import__('pathlib').Path(p) for p in cfg['weight_roots']]
                    settings.set_paths(work=cfg['work'])
                    import exhibitflow.perception  # noqa: F401  đăng ký backend
                    from exhibitflow.common import build
                    trk = build('tracker', cfg['spec'])
                    conn.send(('ok', None))
                elif op == 'shm':
                    if shm is not None:
                        shm.close()
                    shm = _attach(msg[1])
                    conn.send(('ok', None))
                elif op == 'reset':
                    trk.reset(msg[1])
                    conn.send(('ok', None))
                elif op == 'upd':
                    _, shape, dets, s, use_native = msg
                    img = np.ndarray(shape, np.uint8, buffer=shm.buf)
                    out = trk.update(dets, img, native=img if use_native else None, S=s)
                    conn.send(('ok', out))
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
