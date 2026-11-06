#!/usr/bin/env python
"""ExhibitFlow — điểm vào dòng lệnh.

Ví dụ:
    python main.py --videos video1.mp4 video2.mp4 --out results
    python main.py --video-dir /data/videos --out results --weights-dir /data/weights
    python main.py --videos v.mp4 --config configs/default.yaml --set models.vlm.max_new_tokens=96
    python main.py --check                      # kiểm tra môi trường + trọng số, không chạy video

Kết quả: <out>/summary.json và <out>/<tên video>/result.json (+ events.csv, vlm_inputs/, live.mp4 nếu --save-video).
Mã thoát: 0 = mọi video chạy xong, 1 = có video lỗi, 2 = tham số / môi trường sai.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_set(items):
    """['models.vlm.max_new_tokens=96', 'ALARM_ONSET=event'] -> {'models': {'vlm': {'max_new_tokens': 96}}, 'ALARM_ONSET': 'event'}"""
    out = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'--set cần dạng khoá=giá_trị, nhận: {it!r}')
        path, raw = it.split('=', 1)
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw                                   # chuỗi trần, vd event
        node = out
        parts = path.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def check_environment(weights_dir=None):
    """In trạng thái GPU, driver, gói Python và trọng số. Trả True nếu đủ để chạy."""
    from exhibitflow import settings
    from exhibitflow.common import locate_weight, resolve_weight
    ok = True
    print(f'ExhibitFlow {settings.VERSION} | Python {sys.version.split()[0]}')
    try:
        import torch
        print(f'torch {torch.__version__} | GPU dùng được: {settings.GPU_NAMES or "không có (CPU)"}')
    except ImportError:
        print('!!! chưa cài torch'); ok = False
    for mod in ('cv2', 'numpy', 'pandas', 'ultralytics', 'transformers', 'rtmlib', 'onnxruntime', 'huggingface_hub', 'yaml'):
        try:
            m = __import__(mod)
            print(f'  {mod:16s} {getattr(m, "__version__", "ok")}')
        except ImportError:
            print(f'  {mod:16s} !!! THIẾU'); ok = False
    if weights_dir:
        settings.WEIGHT_ROOTS.insert(0, Path(weights_dir))
    m = settings.MODELS
    need = [m['detector']['ckpt'], m['tracker']['reid_ckpt'], m['tracker']['ltc_ckpt'], m['tracker']['dma_ckpt']]
    print('Trọng số cần có (tự tìm trong WEIGHT_ROOTS):')
    for name in need:
        hit = locate_weight(Path(str(name)).name) if not Path(str(name)).exists() else Path(str(name))
        print(f'  {"OK " if hit else "!!!"} {name} -> {hit or "KHÔNG THẤY"}')
        ok &= bool(hit)
    for name in (m['pose']['weights'], m['segmenter']['sam']):
        hit = resolve_weight(name)
        print(f'  ... {name} -> {hit} (nếu chưa có file thì ultralytics tự tải)')
    print('Tự tải lần đầu: Grounding-DINO (HF), GGUF Qwen3-VL-8B Q4_K_M + mmproj (~6 GB), llama.cpp prebuilt (GitHub).')
    print('Thư mục tìm trọng số:', ', '.join(str(p) for p in settings.WEIGHT_ROOTS))
    print('=> SẴN SÀNG' if ok else '=> THIẾU THÀNH PHẦN (xem dòng !!!)')
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description='ExhibitFlow: phát hiện chạm / cầm / lấy hiện vật trong video bảo tàng',
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('--videos', nargs='*', default=[], help='file video (mp4/avi/mov/mkv/webm)')
    ap.add_argument('--video-dir', action='append', default=[], help='thư mục chứa video (có thể lặp lại)')
    ap.add_argument('--out', default=None, help='thư mục kết quả (mặc định <work>/results)')
    ap.add_argument('--work-dir', default=None, help='thư mục tạm / cache (llama.cpp, mask hiện vật). Mặc định ./work')
    ap.add_argument('--weights-dir', action='append', default=[], help='thư mục chứa trọng số (có thể lặp lại)')
    ap.add_argument('--config', action='append', default=[], help='file YAML / JSON ghi đè cấu hình (có thể lặp lại)')
    ap.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                    help='ghi đè 1 giá trị, vd models.vlm.max_side=768 hoặc rt.queue_size=4 (JSON hoặc chuỗi)')
    ap.add_argument('--max-frames', type=int, default=None, help='chỉ chạy N frame đầu mỗi video (thử nhanh)')
    ap.add_argument('--save-video', action='store_true', help='ghi live.mp4 (video có khung + cảnh báo)')
    ap.add_argument('--dry-run-vlm', action='store_true', help='không nạp VLM (trả nhãn giả) — kiểm thử phần còn lại')
    ap.add_argument('--clean-vlm-inputs', action='store_true', help='xoá ảnh gửi VLM (vlm_inputs/) sau khi xong')
    ap.add_argument('--cpu', action='store_true', help='ép chạy CPU (CUDA_VISIBLE_DEVICES rỗng)')
    ap.add_argument('--check', action='store_true', help='chỉ kiểm tra môi trường + trọng số rồi thoát')
    a = ap.parse_args(argv)

    if a.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''               # phải đặt trước khi import torch
    try:
        overrides = parse_set(a.set)
        if a.weights_dir:
            overrides['WEIGHT_ROOTS'] = a.weights_dir
        if a.check:
            if a.config:
                from exhibitflow import settings
                for f in a.config:
                    settings.load_config_file(f)
            if overrides:
                from exhibitflow import settings
                settings.apply_overrides(overrides)
            return 0 if check_environment() else 2
        videos = list(a.videos) + list(a.video_dir)
        if not videos:
            ap.error('cần --videos hoặc --video-dir (hoặc --check)')
        from exhibitflow import run_pipeline
        summary = run_pipeline(videos, out_dir=a.out, work_dir=a.work_dir, config_files=a.config, overrides=overrides,
                               max_frames=a.max_frames, save_video=a.save_video, dry_run_vlm=a.dry_run_vlm,
                               clean_vlm_inputs=a.clean_vlm_inputs)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        print(f'LỖI: {type(e).__name__}: {e}', file=sys.stderr)
        return 2
    return 1 if summary['errors'] else 0


if __name__ == '__main__':
    sys.exit(main())
