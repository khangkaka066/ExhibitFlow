"""ExhibitFlow: phát hiện hành vi bất thường (chạm / cầm / lấy hiện vật) trong video bảo tàng.

    from exhibitflow import run_pipeline
    run_pipeline(['video.mp4'], out_dir='results')

Chạy bằng dòng lệnh: `python main.py --videos video.mp4 --out results` (xem README.md).
"""
from .settings import VERSION as __version__  # noqa: F401


def run_pipeline(*args, **kwargs):
    from .pipeline import run_pipeline as _run          # import muộn: tránh nạp torch / cv2 khi chỉ cần __version__
    return _run(*args, **kwargs)
