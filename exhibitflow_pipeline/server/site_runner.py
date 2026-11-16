"""Chạy repo `multicam-2d-floorplan-tracker` cho nhiều camera ĐÃ HIỆU CHUẨN (hồ sơ địa điểm) và đọc kết quả.

Hồ sơ địa điểm: `server/sites/<site_id>.json` (K, hệ số méo, R, t từng camera trong cùng một hệ toạ độ mét).
Luồng: video của từng camera -> khung đã khử méo ở tần số lấy mẫu -> manifest.json -> `main.py --mode all` của repo
(dựng mặt bằng, YOLO pose, ngoại suy điểm chân, hợp nhất người giữa các camera, ID chung) -> ảnh map + vị trí người theo thời gian.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITES_DIR = HERE / 'sites'


def find_repo():
    """Thư mục repo multicam-2d-floorplan-tracker (FLOORPLAN_DIR, external/, hoặc cạnh dự án)."""
    cands = [os.environ.get('FLOORPLAN_DIR'), HERE.parent / 'external' / 'multicam-2d-floorplan-tracker',
             HERE.parent.parent.parent / 'multicam-2d-floorplan-tracker', HERE.parent.parent / 'multicam-2d-floorplan-tracker',
             Path('/kaggle/working/multicam-2d-floorplan-tracker')]
    for c in cands:
        if c and (Path(c) / 'main.py').is_file() and (Path(c) / 'src' / 'map_builder.py').is_file():
            return Path(c).resolve()
    return None


def list_sites():
    out = {}
    for p in sorted(SITES_DIR.glob('*.json')):
        try:
            s = json.loads(p.read_text(encoding='utf-8'))
            out[s['site_id']] = dict(site_id=s['site_id'], name=s.get('name', s['site_id']), cameras=list(s['cameras']),
                                     labels={k: v.get('label', k) for k, v in s['cameras'].items()})
        except Exception:                                  # noqa: BLE001
            continue
    return out


def load_site(site_id):
    p = SITES_DIR / f'{Path(site_id).name}.json'
    if not p.is_file():
        raise FileNotFoundError(f'Không có hồ sơ địa điểm "{site_id}"')
    return json.loads(p.read_text(encoding='utf-8'))


def prepare_dataset(site, videos, work_dir, seconds=None, log=print):
    """Cắt khung đã khử méo của từng camera ở site['sample_fps'] rồi ghi manifest.json. Trả (thư mục dataset, số mẫu)."""
    import cv2
    import numpy as np
    work = Path(work_dir) / site['room_id']
    work.mkdir(parents=True, exist_ok=True)
    sfps = float(site.get('sample_fps', 4.0))
    caps = {}
    for cam, path in videos.items():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f'Không mở được video của {cam}: {path}')
        caps[cam] = cap
    secs = [c.get(cv2.CAP_PROP_FRAME_COUNT) / max(c.get(cv2.CAP_PROP_FPS), 1) for c in caps.values()]
    total = min(secs) if seconds is None else min(float(seconds), min(secs))
    n = max(1, int(total * sfps))
    cameras = []
    for cam, cap in caps.items():
        spec = site['cameras'][cam]
        K = np.asarray(spec['K'], float)
        dist = np.asarray(spec.get('dist', [0, 0, 0, 0, 0]), float)
        undist = bool(np.any(dist))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (w, h) != (spec['width'], spec['height']):
            raise RuntimeError(f'{cam}: video {w}x{h} khác kích thước hiệu chuẩn {spec["width"]}x{spec["height"]}')
        folder = work / cam / 'frames'
        folder.mkdir(parents=True, exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        want = [int(round(i * fps / sfps)) for i in range(n)]
        idx, k = 0, 0
        while k < n:
            ok, frame = cap.read()
            if not ok:
                n = k
                break
            if idx == want[k]:
                if undist:
                    frame = cv2.undistort(frame, K, dist)
                cv2.imwrite(str(folder / f'{k:06d}.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                k += 1
            idx += 1
        cap.release()
        log(f'[site] {cam}: {k} khung ({sfps:g}/s)')
        cameras.append(dict(id=cam, room_id=site['room_id'], width=w, height=h, fps=sfps, frames=n, frames_dir=f'{cam}/frames',
                            frame_pattern='{index:06d}.jpg', K=spec['K'], R_world_to_camera=spec['R_world_to_camera'],
                            t_world_to_camera=spec['t_world_to_camera']))
    for c in cameras:
        c['frames'] = n
    manifest = dict(room_id=site['room_id'], coordinate_units='meters', map_y_axis='up', allow_open_ends=True,
                    source=f"ExhibitFlow web: {site.get('name', site['site_id'])}", pose_scales=[1920],
                    timing=dict(clock_id=f"site-{site['site_id']}", time_offset_seconds=0.0), cameras=cameras)
    (work / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    return Path(work_dir), n


def run_repo(repo, data_dir, out_dir, frames, site_room, device='cpu', log=print, timeout=3600):
    """Chạy main.py của repo (mode all). Trả thư mục kết quả (chứa global_2d_map.png, projections.json...)."""
    repo = Path(repo)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = repo / 'models' / 'yolo11n-pose.pt'
    # site_driver.py chạy các bước lõi của repo (collect + process) trên camera đã hiệu chuẩn, bỏ qua bước dựng tường (cần phòng kín)
    cmd = [sys.executable, '-u', str(HERE / 'site_driver.py'), '--data-dir', str(Path(data_dir) / site_room), '--output-dir', str(out_dir),
           '--frames', str(frames), '--device', device, '--weights', str(weights)]
    env = dict(os.environ, YOLO_CONFIG_DIR=str(out_dir / 'yolo_config'), PYTHONUNBUFFERED='1')
    p = subprocess.Popen(cmd, cwd=str(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    tail = []
    try:
        for line in p.stdout:
            line = line.rstrip()
            tail.append(line)
            if line and not line.startswith(('WARNING', 'Warning')):
                log('[map2d] ' + line[:200])
        p.wait(timeout=timeout)
    except BaseException:
        p.kill()
        raise
    if p.returncode != 0:
        raise RuntimeError('map2d lỗi: ' + ' | '.join(tail[-5:])[-600:])
    return out_dir
