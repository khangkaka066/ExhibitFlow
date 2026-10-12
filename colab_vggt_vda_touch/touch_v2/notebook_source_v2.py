# %% [markdown]
# # Touch Interaction V2 — Visual-Temporal Hand–Artifact Pipeline
#
# Triển khai các phase MVP của `PLAN_TOUCH_V2.md`: tái dùng tracker
# ByteTrack-DMA-LTC-OSNet và pipeline artifact MM-Grounding-DINO + SAM2.1
# của V1, thêm pose bàn tay/thân trên ảnh **gốc** (không downscale), ghép
# người–hiện vật trong ảnh, state machine theo thời gian, và coi
# geometry (VGGT + Depth Anything V2) cùng Isolation Forest là tín hiệu
# **tùy chọn/shadow** (mặc định tắt) theo đúng quy tắc fusion của plan.
# Notebook tự hoàn tất từng phase và chuyển tiếp khi `Run all`, không hỏi
# xác nhận `TIEP`/`DUNG` — giống hành vi V1 đã triển khai.
#
# **Phạm vi đã triển khai (MVP, Section 10):** Phase 0–8 dưới đây.
# **Chưa triển khai** (ghi rõ trong manifest, không giả lập kết quả):
# T00 (gán nhãn ground-truth thủ công), T08 (retrain Isolation Forest V2
# trên dữ liệu tương tác thật), T12 (nghiên cứu 3D contact verifier mở).
# Các phần này cần dữ liệu có nhãn hoặc nghiên cứu riêng, không thể
# "chạy suy luận" trong một notebook.

# %%
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

print('Python', sys.version)
print('OpenCV', cv2.__version__)

# %% [markdown]
# ## Cài thư viện + repo tracker (giống V1, không đổi tên file checkpoint)

# %%
def _pip(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)

# Không install requirements.txt gốc của YOLOX (repo ByteTrack-DMA-LTC-Motion-Tracker):
# có thể kéo NumPy/PyTorch không phù hợp Kaggle. Thay vào đó cài đúng các gói mà
# `yolox/utils/__init__.py` cần khi import (loguru, pycocotools, thop, tabulate,
# tensorboard, yacs) — giống cách V1 (`notebook_source.py`) đã làm.
_pip('ultralytics==8.3.* '.strip(), 'gdown', 'imageio-ffmpeg', 'lap', 'cython_bbox',
     'scikit-learn', 'joblib', 'scipy', 'rtmlib', 'onnxruntime-gpu',
     'loguru', 'pycocotools', 'thop', 'tabulate', 'tensorboard', 'yacs')

TRACKER_REPO = Path('/kaggle/working/ByteTrack-DMA-LTC-Motion-Tracker')
if not TRACKER_REPO.exists():
    subprocess.run(['git', 'clone', '--depth', '1',
                     'https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker',
                     str(TRACKER_REPO)], check=True)
sys.path.insert(0, str(TRACKER_REPO))

DA_V2_REPO = Path('/kaggle/working/Depth-Anything-V2')

def ensure_depth_anything_v2_repo() -> Path:
    if not DA_V2_REPO.exists():
        subprocess.run(['git', 'clone', '--depth', '1',
                         'https://github.com/DepthAnything/Depth-Anything-V2',
                         str(DA_V2_REPO)], check=True)
    if str(DA_V2_REPO) not in sys.path:
        sys.path.insert(0, str(DA_V2_REPO))
    return DA_V2_REPO

import torch
print('Torch', torch.__version__, 'CUDA', torch.cuda.is_available())
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# %% [markdown]
# ## Input: tìm file dưới `/kaggle/input`, giải nén bundle nếu có (path-traversal guard)

# %%
WORK = Path('/kaggle/working/exhibit_touch_v2')
WORK.mkdir(parents=True, exist_ok=True)

REQUIRED_FILES = [
    '1.mp4', '2.mp4', '3.mp4',
    'best_ckpt', 'dma_gbm_mot17.gbm', 'ltc_motion_mot17.pth',
    'sam2.1_b.pt',
]
OPTIONAL_FILES = [
    'isolation_forest_normal_yolov8_pose.joblib', 'evaluation.json',
]


def _safe_extract(zf: zipfile.ZipFile, dest: Path):
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if not str(target).startswith(str(dest)):
            raise RuntimeError(f'Unsafe path trong bundle zip: {member.filename}')
    zf.extractall(dest)


def repack_extracted_torch_checkpoint(folder: Path) -> Path:
    """Đóng gói lại checkpoint PyTorch mà Kaggle bung thành thư mục (torch.save ZIP)."""
    data_pkls = [p for p in folder.rglob('data.pkl') if p.is_file()]
    if len(data_pkls) != 1:
        raise RuntimeError(f'Checkpoint đã bung phải có đúng một data.pkl; thấy {len(data_pkls)} trong {folder}')
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    if not files:
        raise RuntimeError(f'Thư mục checkpoint rỗng: {folder}')
    destination = WORK / 'repacked_best_ckpt.zip'
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, arcname=path.relative_to(folder).as_posix())
    print(f'Đóng gói lại checkpoint đã bung: {len(files)} files -> {destination}')
    return destination


def resolve_best_ckpt_hit(hit: Path) -> Path:
    """Kaggle có thể mount best_ckpt dưới dạng thư mục đã bung; torch.load cần file."""
    if hit.is_file():
        return hit
    if hit.is_dir():
        if any(p.is_file() for p in hit.rglob('data.pkl')):
            return repack_extracted_torch_checkpoint(hit)
        files = sorted(p for p in hit.rglob('*') if p.is_file())
        if len(files) == 1:
            return files[0]
        raise RuntimeError(f'best_ckpt là thư mục không rõ cấu trúc torch.save: {hit} ({len(files)} file)')
    raise RuntimeError(f'best_ckpt không tồn tại: {hit}')


def locate_input_file(name: str, dataset_dir: str | None = None) -> Path | None:
    roots = [Path(dataset_dir)] if dataset_dir else sorted(Path('/kaggle/input').glob('*'))
    for root in roots:
        if not root.exists():
            continue
        for bundle in root.glob('*.zip'):
            with zipfile.ZipFile(bundle) as zf:
                if any(Path(n).name == name for n in zf.namelist()):
                    extract_to = WORK / 'bundle_extracted'
                    if not (extract_to / name).exists():
                        _safe_extract(zf, extract_to)
                    hit = next(extract_to.rglob(name), None)
                    if hit:
                        return resolve_best_ckpt_hit(hit) if name == 'best_ckpt' else hit
        hit = next(root.rglob(name), None)
        if hit:
            return resolve_best_ckpt_hit(hit) if name == 'best_ckpt' else hit
    return None


DATASET_DIR = None  # đổi thành '/kaggle/input/<dataset-slug>' nếu có nhiều dataset trùng tên file
INPUT_PATHS = {}
missing = []
for name in REQUIRED_FILES:
    p = locate_input_file(name, DATASET_DIR)
    if p is None:
        missing.append(name)
    INPUT_PATHS[name] = p
for name in OPTIONAL_FILES:
    INPUT_PATHS[name] = locate_input_file(name, DATASET_DIR)

if missing:
    raise RuntimeError(f'Thiếu file input bắt buộc: {missing}. Kiểm tra Add Input/bundle.')

print('Input files:')
for k, v in INPUT_PATHS.items():
    print(f'  {k}: {v}')

HAS_IF_MODEL = INPUT_PATHS.get('isolation_forest_normal_yolov8_pose.joblib') is not None

# %% [markdown]
# ## CFG — ngưỡng V2. Geometry và Isolation Forest mặc định tắt (tùy chọn/shadow)

# %%
CFG = dict(
    width=960,
    max_frames=None,
    # Phase 2 — pose trên ảnh gốc (không downscale), padding quanh bbox track
    pose_crop_pad_frac=0.30,
    pose_min_crop_px=16,
    pose_conf_thr=0.30,
    pose_iou_match_thr=0.30,
    pose_margin_thr=0.15,
    # Phase 2b — pose bàn tay (RTMW wholebody 133 điểm, bbox từ track, không tự detect)
    enable_hand_pose=True,
    hand_backend_mode='lightweight',
    hand_conf_thr=0.30,
    hand_min_bbox_diag_px=18,   # bbox tay (trong ảnh gốc) nhỏ hơn mức này -> too_small, không dùng làm chứng cứ dương
    # Phase 3 — ghép người–hiện vật trong ảnh
    pairing_max_norm_dist=3.0,   # khoảng cách wrist->mask tối đa / chiều cao bbox người
    # Phase 4 — state machine theo thời gian (ngưỡng theo "norm_dist" = px / bbox_height)
    state_near_thr=0.9,
    state_contact_thr=0.35,
    state_hold_seconds=0.6,
    state_history_seconds=2.0,
    state_lost_grace_frames=5,
    # Phase 5 — geometry tùy chọn (tắt mặc định; chỉ dùng để REFINE, không bao giờ tự nới ngưỡng 2D)
    enable_geometry=False,
    da_v2_input_size=336,
    reference_frames=3,
    geometry_contact_m=0.12,
    max_alignment_median_rel=0.25,
    max_alignment_p90_rel=0.45,
    max_scale_drift=0.35,
    # Phase 6 — Isolation Forest shadow (tắt mặc định; không bao giờ gate quyết định touch)
    enable_isolation_forest=HAS_IF_MODEL,
    if_window_seconds=2.5,
)

print(json.dumps(CFG, indent=2, ensure_ascii=False))

# %% [markdown]
# ## Phase scaffolding — giống V1: tự hoàn tất, PASS/WARN/FAIL không chặn, không hỏi xác nhận

# %%
PHASE_NAMES = [
    'input', 'tracking', 'pose_original_res', 'artifact_masks',
    'pairing_state_machine', 'geometry_optional', 'isolation_forest_shadow',
    'fusion_events', 'evaluation',
]

RUN = WORK / datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
RUN.mkdir(parents=True, exist_ok=True)
STATE = dict(clip=None, config=CFG, completed=[], ready=True, attempts={})


def begin_phase(idx: int) -> Path:
    name = PHASE_NAMES[idx]
    STATE['attempts'][name] = STATE['attempts'].get(name, 0) + 1
    phase_dir = RUN / f'{idx:02d}_{name}' / f'attempt_{STATE["attempts"][name]:02d}'
    phase_dir.mkdir(parents=True, exist_ok=True)
    STATE['completed'] = [n for n in STATE['completed'] if PHASE_NAMES.index(n) < idx]
    print(f'\n=== Phase {idx} [{name}] — attempt {STATE["attempts"][name]} ===')
    return phase_dir


def phase_diagnostics(phase_dir: Path, name: str, metrics: dict, checks: list, artifacts: list):
    rows = []
    for status, label, value, hint in checks:
        rows.append(dict(status=status, label=label, value=value, hint=hint))
        print(f'[{status}] {label}: {value}  ({hint})')
    payload = dict(phase=name, metrics=metrics, checks=rows, artifacts=artifacts,
                    generated_at=datetime.now().isoformat())
    (phase_dir / 'phase_diagnostic.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    import csv
    with open(phase_dir / 'phase_diagnostic.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['status', 'label', 'value', 'hint'])
        w.writeheader()
        w.writerows(rows)
    fails = [r for r in rows if r['status'] == 'FAIL']
    if fails:
        print(f'  -> {len(fails)} FAIL check(s): xem {phase_dir}/phase_diagnostic.json trước khi tin kết quả.')


def review_phase(idx: int):
    name = PHASE_NAMES[idx]
    if name not in STATE['completed']:
        STATE['completed'].append(name)
    (RUN / 'phase_review.json').write_text(json.dumps(STATE, indent=2, ensure_ascii=False, default=str))
    print(f'Phase {idx} [{name}] done. Completed so far: {STATE["completed"]}')

# %% [markdown]
# ## Phase 0 — Input

# %%
CLIP = '1'  # đổi thành '1' | '2' | '3'

def read_clip(video_path: Path, width: int, max_frames: int | None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Không mở được video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, native_frames = [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        native_frames.append(frame)
        h, w = frame.shape[:2]
        scale = width / w
        frames.append(cv2.resize(frame, (width, int(round(h * scale)))))
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f'Video rỗng, không đọc được frame nào: {video_path}')
    return frames, native_frames, fps


phase_dir = begin_phase(0)
video_path = INPUT_PATHS[f'{CLIP}.mp4']
frames, native_frames, FPS = read_clip(video_path, CFG['width'], CFG['max_frames'])
H, W = frames[0].shape[:2]
NATIVE_H, NATIVE_W = native_frames[0].shape[:2]
SCALE_TO_NATIVE = NATIVE_W / W
STATE['clip'] = CLIP
cv2.imwrite(str(phase_dir / 'frame0.jpg'), frames[0])
manifest = dict(clip=CLIP, fps=FPS, n_frames=len(frames), width=W, height=H,
                native_width=NATIVE_W, native_height=NATIVE_H, config=CFG)
(RUN / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
phase_diagnostics(phase_dir, 'input', dict(n_frames=len(frames), fps=FPS), [
    ('PASS' if len(frames) > 10 else 'WARN', 'so_frame', len(frames), 'clip qua ngan co the thieu bang chung thoi gian'),
    ('PASS', 'native_resolution', f'{NATIVE_W}x{NATIVE_H}', 'dung de crop pose goc'),
], ['frame0.jpg', 'manifest.json'])
review_phase(0)

# %% [markdown]
# ## Phase 1 — Tracking (ByteTrack-DMA-LTC + OSNet ReID, tái dùng từ V1)

# %%
def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _ensure_reid_weights(folder: Path) -> Path:
    weights = folder / 'osnet_x1_0_imagenet.pth'
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        import gdown
        gdown.download(id='1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY', output=str(weights), quiet=False)
    if not weights.exists() or weights.stat().st_size < 1_000_000:
        raise RuntimeError('Tải trọng số OSNet ReID thất bại hoặc file quá nhỏ.')
    return weights


def track_people(frames, fps, folder: Path):
    from yolox.exp import get_exp
    from yolox.tracker.byte_tracker import BYTETracker
    from yolox.tracker.basetrack import BaseTrack

    exp = get_exp(str(TRACKER_REPO / 'exps/example/mot/yolox_nano_mix_det.py'), None)
    model = exp.get_model().to(DEVICE).eval()
    ckpt = torch.load(str(INPUT_PATHS['best_ckpt']), map_location='cpu', weights_only=True)
    state = ckpt.get('model', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    reid_weights = _ensure_reid_weights(folder)
    args = SimpleNamespace(
        track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False,
        dma_gbm_path=str(INPUT_PATHS['dma_gbm_mot17.gbm']),
        ltc_motion_path=str(INPUT_PATHS['ltc_motion_mot17.pth']),
        reid_weights=str(reid_weights), reid_name='osnet_x1_0', device=DEVICE,
    )
    BaseTrack._count = 0
    tracker = BYTETracker(args, frame_rate=int(round(fps)))
    assert getattr(tracker, 'dma', None) is not None, 'DMA GBM khong duoc nap'
    assert getattr(tracker.motion_predictor, 'enabled', False), 'LTC motion khong duoc bat'
    assert getattr(tracker, 'reid_extractor', None) is not None, 'ReID extractor khong duoc nap'

    tracks, detections, total_dets = [], [], 0
    for i, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            raw = model(tensor)
            dets = exp.postprocess(raw, exp.num_classes, 0.3, 0.45)[0]
        frame_dets = []
        if dets is not None:
            for d in dets.cpu().numpy():
                x1, y1, x2, y2, s1, s2, cls = d[:7]
                if int(cls) != 0:
                    continue
                frame_dets.append([float(x1), float(y1), float(x2), float(y2), float(s1 * s2)])
        total_dets += len(frame_dets)
        online = tracker.update(np.array(frame_dets) if frame_dets else np.empty((0, 5)), frame)
        recs = []
        for t in online:
            tb = list(map(float, t.tlbr))
            best_iou, best_det = 0.0, None
            for d in frame_dets:
                iou = box_iou(tb, d[:4])
                if iou > best_iou:
                    best_iou, best_det = iou, d
            if best_iou >= 0.30:
                recs.append(dict(id=int(t.track_id), box=tb, kp=None, kc=None))
        tracks.append(recs)
        detections.append(frame_dets)
        if i == 0:
            vis = frame.copy()
            for d in frame_dets:
                cv2.rectangle(vis, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 2)
            cv2.imwrite(str(folder / 'detector_smoke.jpg'), vis)

    if total_dets == 0:
        raise RuntimeError('Khong co detection nao tren toan bo clip — dung lai, khong gia lap track.')

    (folder / 'tracks.json').write_text(json.dumps(tracks, indent=1))
    (folder / 'detections.json').write_text(json.dumps(detections, indent=1))
    n_with_track = sum(1 for r in tracks if r)
    report = dict(total_detections=total_dets, frames_with_track=n_with_track, n_frames=len(frames))
    (folder / 'tracker_report.json').write_text(json.dumps(report, indent=2))
    return tracks


phase_dir = begin_phase(1)
RAW_TRACKS = track_people(frames, FPS, phase_dir)
n_with_track = sum(1 for r in RAW_TRACKS if r)
vis = frames[0].copy()
for r in RAW_TRACKS[0]:
    x1, y1, x2, y2 = map(int, r['box'])
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(vis, f"ID {r['id']}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
cv2.imwrite(str(phase_dir / 'tracking_preview.jpg'), vis)
phase_diagnostics(phase_dir, 'tracking',
                   dict(frames_with_track=n_with_track, n_frames=len(frames)),
                   [('PASS' if n_with_track / len(frames) > 0.5 else 'WARN', 'ty_le_frame_co_track',
                     round(n_with_track / len(frames), 3), 'ty le thap co the do occlusion hoac detector yeu')],
                   ['tracking_preview.jpg', 'tracks.json'])
review_phase(1)

# %% [markdown]
# ## Phase 2 — Pose trên ảnh GỐC (không downscale) — khác biệt chính so với V1
#
# V1 chạy YOLOv8n-pose trên crop lấy từ frame đã resize (`width=960`), làm
# mất độ chính xác cổ tay ở khoảng cách xa. V2 crop trực tiếp từ
# `native_frames` (độ phân giải gốc của video) theo bbox track đã quy đổi
# tỉ lệ, chạy pose rồi quy đổi keypoint ngược về không gian làm việc.

# %%
def estimate_track_poses_v2(native_frames, tracks, scale_to_native, folder: Path):
    from ultralytics import YOLO
    import copy

    model = YOLO('yolov8n-pose.pt')
    out_tracks = copy.deepcopy(tracks)
    n_pose, n_total = 0, 0
    for i, recs in enumerate(out_tracks):
        native = native_frames[i]
        nh, nw = native.shape[:2]
        for rec in recs:
            n_total += 1
            x1, y1, x2, y2 = [v * scale_to_native for v in rec['box']]
            bw, bh = x2 - x1, y2 - y1
            dx, dy = bw * CFG['pose_crop_pad_frac'], bh * CFG['pose_crop_pad_frac'] * 0.5
            cx1, cy1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
            cx2, cy2 = min(nw, int(x2 + dx)), min(nh, int(y2 + dy))
            if (cx2 - cx1) < CFG['pose_min_crop_px'] or (cy2 - cy1) < CFG['pose_min_crop_px']:
                continue
            crop = native[cy1:cy2, cx1:cx2]
            res = model(crop, verbose=False)[0]
            if res.keypoints is None or len(res.boxes) == 0:
                continue
            track_box_native = [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]
            best_iou, best_j, second_iou = 0.0, None, 0.0
            for j, pb in enumerate(res.boxes.xyxy.cpu().numpy()):
                iou = box_iou(track_box_native, pb.tolist())
                if iou > best_iou:
                    second_iou = best_iou
                    best_iou, best_j = iou, j
                elif iou > second_iou:
                    second_iou = iou
            if best_j is None or best_iou < CFG['pose_conf_thr'] or (best_iou - second_iou) < CFG['pose_margin_thr']:
                continue
            kp = res.keypoints.xy[best_j].cpu().numpy()
            kc = res.keypoints.conf[best_j].cpu().numpy() if res.keypoints.conf is not None else np.ones(len(kp))
            kp_native = kp + np.array([cx1, cy1])
            kp_work = kp_native / scale_to_native
            rec['kp'] = kp_work.tolist()
            rec['kc'] = kc.tolist()
            n_pose += 1
    rate = n_pose / n_total if n_total else 0.0
    (folder / 'tracks_pose.json').write_text(json.dumps(out_tracks, indent=1))
    return out_tracks, rate, n_pose, n_total


phase_dir = begin_phase(2)
TRACKS, pose_rate, n_pose, n_total = estimate_track_poses_v2(native_frames, RAW_TRACKS, SCALE_TO_NATIVE, phase_dir)
vis = frames[0].copy()
for rec in TRACKS[0]:
    if rec.get('kp') is None:
        continue
    for (x, y), c in zip(rec['kp'], rec['kc']):
        if c >= CFG['pose_conf_thr']:
            cv2.circle(vis, (int(x), int(y)), 3, (0, 0, 255), -1)
cv2.imwrite(str(phase_dir / 'pose_preview.jpg'), vis)
phase_diagnostics(phase_dir, 'pose_original_res', dict(pose_rate=pose_rate, n_pose=n_pose, n_total=n_total),
                   [('PASS' if pose_rate > 0.5 else 'WARN', 'ty_le_pose_khop', round(pose_rate, 3),
                     'thap co the do crop qua nho, occlusion hoac nguoi qua xa camera')],
                   ['pose_preview.jpg', 'tracks_pose.json'])
review_phase(2)

# %% [markdown]
# ## Phase 2b — Pose bàn tay (RTMW wholebody 133 điểm, bbox từ track)
#
# COCO17 (Phase 2) chỉ có khớp cổ tay, không có ngón tay. Theo PLAN_TOUCH_V2
# T04, thử RTMW wholebody (133 điểm: 0–16 thân, 91–111 tay trái, 112–132 tay
# phải theo chuẩn COCO-WholeBody) qua `rtmlib`, dùng API pose cấp thấp với
# bbox do track cung cấp — không để model tự detect lại người. Trái/phải lấy
# theo nhãn giải phẫu gốc của model (không map theo vị trí trái/phải trên
# ảnh). Tay bị che/quá nhỏ được đánh dấu rõ, không dùng làm chứng cứ chạm
# dương. Nếu `rtmlib` không khởi tạo được (thiếu mạng, backend lỗi…), phase
# này tự tắt (`HAS_HAND_POSE=False`) và Phase 4 dùng lại cổ tay COCO17 như cũ
# — đây là nhánh tùy chọn, lỗi ở đây không làm dừng toàn bộ notebook.

# %%
HAND_WB_LEFT = slice(91, 112)
HAND_WB_RIGHT = slice(112, 133)


def _hand_quality(kp_hand, kc_hand, conf_thr, min_diag_px):
    visible = kc_hand >= conf_thr
    n_visible = int(visible.sum())
    if n_visible < 5:
        return 'occluded_or_missing', n_visible
    pts = kp_hand[visible]
    diag = float(np.hypot(*(pts.max(axis=0) - pts.min(axis=0)))) if len(pts) >= 2 else 0.0
    if diag < min_diag_px:
        return 'too_small', n_visible
    return 'ok', n_visible


def estimate_hand_poses_v2(native_frames, tracks, scale_to_native, folder: Path):
    try:
        from rtmlib import Wholebody
    except Exception as exc:
        (folder / 'hand_pose_disabled.txt').write_text(f'rtmlib import failed: {exc}\n')
        return tracks, 0.0, 0, 0, False

    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        wholebody = Wholebody(mode=CFG['hand_backend_mode'], backend='onnxruntime', device=device)
        pose_model = wholebody.pose_model  # API cấp thấp: nhận bbox ngoài, không tự detect
    except Exception as exc:
        (folder / 'hand_pose_disabled.txt').write_text(f'rtmlib Wholebody init failed: {exc}\n')
        return tracks, 0.0, 0, 0, False

    n_hand, n_total = 0, 0
    preview_crops = []
    for i, recs in enumerate(tracks):
        native = native_frames[i]
        nh, nw = native.shape[:2]
        for rec in recs:
            if rec.get('kp') is None:
                continue
            n_total += 1
            x1, y1, x2, y2 = [v * scale_to_native for v in rec['box']]
            bw, bh = x2 - x1, y2 - y1
            dx, dy = bw * CFG['pose_crop_pad_frac'], bh * CFG['pose_crop_pad_frac'] * 0.5
            cx1, cy1 = max(0, int(x1 - dx)), max(0, int(y1 - dy))
            cx2, cy2 = min(nw, int(x2 + dx)), min(nh, int(y2 + dy))
            if (cx2 - cx1) < CFG['pose_min_crop_px'] or (cy2 - cy1) < CFG['pose_min_crop_px']:
                continue
            crop = native[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]
            try:
                kp_all, kc_all = pose_model(crop, bboxes=np.array([[0, 0, cw, ch]], dtype=np.float32))
            except Exception:
                continue
            if kp_all is None or len(kp_all) == 0 or kp_all.shape[1] < 133:
                continue
            kp_wb, kc_wb = kp_all[0], kc_all[0]
            hand_kp, hand_kc, hand_quality = {}, {}, {}
            for side, sl in (('left', HAND_WB_LEFT), ('right', HAND_WB_RIGHT)):
                kp_hand_native = kp_wb[sl] + np.array([cx1, cy1])
                kc_hand = kc_wb[sl]
                quality, n_vis = _hand_quality(kp_hand_native, kc_hand, CFG['hand_conf_thr'],
                                                CFG['hand_min_bbox_diag_px'])
                hand_kp[side] = (kp_hand_native / scale_to_native).tolist()
                hand_kc[side] = kc_hand.tolist()
                hand_quality[side] = dict(status=quality, n_visible=n_vis)
                if quality == 'ok' and i == 0 and len(preview_crops) < 6:
                    preview_crops.append((crop.copy(), kp_wb[sl], kc_hand, side, CFG['hand_conf_thr']))
            rec['hand_kp'] = hand_kp
            rec['hand_kc'] = hand_kc
            rec['hand_quality'] = hand_quality
            if any(q['status'] == 'ok' for q in hand_quality.values()):
                n_hand += 1
    rate = n_hand / n_total if n_total else 0.0
    (folder / 'tracks_pose.json').write_text(json.dumps(tracks, indent=1))
    if preview_crops:
        tiles = []
        for crop, pts, conf, side, thr in preview_crops:
            tile = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
            sx, sy = 160 / crop.shape[1], 160 / crop.shape[0]
            for (x, y), c in zip(pts, conf):
                if c >= thr:
                    cv2.circle(tile, (int(x * sx), int(y * sy)), 2, (0, 255, 255), -1)
            cv2.putText(tile, side[:1].upper(), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            tiles.append(tile)
        cv2.imwrite(str(folder / 'hand_roi_preview.jpg'), np.hstack(tiles))
    return tracks, rate, n_hand, n_total, True


phase_dir2b = phase_dir / 'hand_pose'
phase_dir2b.mkdir(parents=True, exist_ok=True)
if CFG['enable_hand_pose']:
    TRACKS, hand_rate, n_hand, n_hand_total, HAS_HAND_POSE = estimate_hand_poses_v2(
        native_frames, TRACKS, SCALE_TO_NATIVE, phase_dir2b)
else:
    hand_rate, n_hand, n_hand_total, HAS_HAND_POSE = 0.0, 0, 0, False

track_hand_stats = {}
for recs in TRACKS:
    for rec in recs:
        for side, hq in (rec.get('hand_quality') or {}).items():
            key = (rec['id'], side)
            st = track_hand_stats.setdefault(key, dict(n_frames=0, n_ok=0, n_too_small=0, n_occluded_or_missing=0))
            st['n_frames'] += 1
            st[f"n_{hq['status']}"] += 1
import csv
with open(phase_dir2b / 'pose_quality_by_track.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['track_id', 'hand', 'n_frames', 'n_ok', 'n_too_small', 'n_occluded_or_missing', 'ok_rate'])
    for (track_id, side), st in sorted(track_hand_stats.items()):
        ok_rate = st['n_ok'] / st['n_frames'] if st['n_frames'] else 0.0
        writer.writerow([track_id, side, st['n_frames'], st['n_ok'], st['n_too_small'],
                          st['n_occluded_or_missing'], round(ok_rate, 3)])

vis_hand = frames[0].copy()
for rec in TRACKS[0]:
    for side, pts in (rec.get('hand_kp') or {}).items():
        conf = rec['hand_kc'][side]
        color = (0, 255, 255) if side == 'left' else (255, 0, 255)
        for (x, y), c in zip(pts, conf):
            if c >= CFG['hand_conf_thr']:
                cv2.circle(vis_hand, (int(x), int(y)), 2, color, -1)
cv2.imwrite(str(phase_dir2b / 'pose_hand_preview.jpg'), vis_hand)
phase_diagnostics(phase_dir2b, 'pose_hand_rtmw',
                   dict(hand_pose_rate=hand_rate, n_hand=n_hand, n_total=n_hand_total,
                        backend_available=HAS_HAND_POSE),
                   [('PASS' if HAS_HAND_POSE and hand_rate > 0.3 else 'WARN',
                     'ty_le_tay_quan_sat_duoc', round(hand_rate, 3),
                     'HAS_HAND_POSE=False nghia la rtmlib khong khoi tao duoc, Phase 4 dung lai cay tay'
                     ' co the-wrist COCO17; thap du backend OK co the do tay bi che hoac qua nho')],
                   ['pose_hand_preview.jpg', 'tracks_pose.json', 'pose_quality_by_track.csv'])

# %% [markdown]
# ## Phase 3 — Artifact segmentation (MM-Grounding-DINO + SAM2.1, tái dùng cấu hình V1)
#
# Để ghép người–hiện vật cần vị trí hiện vật trong ảnh ngay cả khi không
# dùng geometry 3D. Dùng lại checkpoint SAM2.1 B có sẵn trong bundle; tải
# MM-Grounding-DINO (base, nhẹ hơn bản `large` của V1 để giảm VRAM khi
# geometry tắt — vẫn strict, không fallback âm thầm).
#
# PLAN_TOUCH_V2 T03 yêu cầu chọn reference frame ít bị người che khuất, thay vì
# luôn lấy `frames[0]` — một frame có thể vô tình bị người đứng che hiện vật.
# `select_reference_frame` quét một số frame đầu clip, chọn frame có tổng diện
# tích bbox người chồng lên khung hình nhỏ nhất làm proxy "ít che khuất". Đây
# vẫn là một mask/reference duy nhất cho cả scene (giả định camera tĩnh, không
# cắt cảnh) — T03 còn yêu cầu đăng ký lại camera khi đổi góc, tách
# `reference_mask`/`visible_mask_t` theo che khuất từng frame, `artifact_part_id`
# riêng tượng/bệ; các phần này **chưa làm** trong bản vá này, ghi rõ trong
# diagnostics để không báo "xong" quá mức.

# %%
def select_reference_frame(frames_list, tracks, max_scan=30):
    """Chọn frame có diện tích bbox người che khung hình nhỏ nhất trong
    max_scan frame đầu — proxy cho 'ít che khuất tại vùng tương tác'."""
    n = min(max_scan, len(frames_list))
    best_idx, best_cover = 0, None
    h, w = frames_list[0].shape[:2]
    frame_area = float(h * w)
    for i in range(n):
        cover = 0.0
        for rec in tracks[i]:
            x1, y1, x2, y2 = rec['box']
            cover += max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cover_frac = cover / frame_area
        if best_cover is None or cover_frac < best_cover:
            best_idx, best_cover = i, cover_frac
    return best_idx, best_cover if best_cover is not None else 0.0


SEGMENTATION_CFG = dict(
    dino_model='IDEA-Research/grounding-dino-base',
    sam_model=str(INPUT_PATHS['sam2.1_b.pt']),
    box_thr=0.30, text_thr=0.25,
    prompts=['sculpture.', 'painting.', 'pedestal.', 'exhibit.'],
    min_mask_fill=0.15,
)


def artifact_masks(frame, folder: Path, ref_frame_idx: int, ref_person_cover_frac: float):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from ultralytics import SAM

    processor = AutoProcessor.from_pretrained(SEGMENTATION_CFG['dino_model'])
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(SEGMENTATION_CFG['dino_model']).to(DEVICE).eval()
    text = ' '.join(SEGMENTATION_CFG['prompts'])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb, text=text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        outputs = dino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=SEGMENTATION_CFG['box_thr'],
        text_threshold=SEGMENTATION_CFG['text_thr'], target_sizes=[rgb.shape[:2]])[0]
    boxes = results['boxes'].cpu().numpy().tolist()
    if not boxes:
        raise RuntimeError('MM-Grounding-DINO khong tim thay hien vat nao — khong gia lap mask.')

    sam = SAM(SEGMENTATION_CFG['sam_model'])
    artifacts = []
    vis = frame.copy()
    for i, box in enumerate(boxes):
        res = sam.predict(rgb, bboxes=[box], verbose=False)[0]
        if res.masks is None or len(res.masks.data) == 0:
            continue
        mask = res.masks.data[0].cpu().numpy().astype(np.uint8)
        if mask.sum() / mask.size < SEGMENTATION_CFG['min_mask_fill'] / 10:
            continue
        artifacts.append(dict(id=i, box=box, label=str(results['labels'][i])))
        np.save(folder / f'artifact_{i}_mask.npy', mask)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, results['labels'][i], (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if not artifacts:
        raise RuntimeError('Khong co mask hien vat nao qua duoc quality filter.')
    cv2.imwrite(str(folder / 'artifact_review.jpg'), vis)
    registry = dict(
        reference=dict(
            frame_idx=ref_frame_idx,
            person_cover_frac=round(ref_person_cover_frac, 4),
            selection_method='min_person_cover_frac_in_first_30_frames',
        ),
        not_implemented=['camera_registration', 'visible_mask_t_per_frame',
                          'artifact_part_id', 'artifact_quality_csv',
                          'artifact_visibility_preview_video'],
        artifacts=artifacts,
    )
    (folder / 'artifacts.json').write_text(json.dumps(registry, indent=2))
    return artifacts


def artifact_mask_distance_map(artifact: dict, folder: Path, shape):
    mask = np.load(folder / f"artifact_{artifact['id']}_mask.npy")
    inv = (1 - mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    return dist


phase_dir = begin_phase(3)
REF_FRAME_IDX, REF_PERSON_COVER_FRAC = select_reference_frame(frames, TRACKS)
ARTIFACTS = artifact_masks(frames[REF_FRAME_IDX], phase_dir, REF_FRAME_IDX, REF_PERSON_COVER_FRAC)
phase_diagnostics(phase_dir, 'artifact_masks',
                   dict(n_artifacts=len(ARTIFACTS), reference_frame_idx=REF_FRAME_IDX,
                        reference_person_cover_frac=round(REF_PERSON_COVER_FRAC, 4)),
                   [('PASS' if len(ARTIFACTS) > 0 else 'FAIL', 'so_hien_vat', len(ARTIFACTS),
                     'khong co hien vat thi khong the ghep cap voi nguoi'),
                    ('PASS' if REF_PERSON_COVER_FRAC < 0.5 else 'WARN', 'reference_frame_occlusion',
                     round(REF_PERSON_COVER_FRAC, 4),
                     'frame tham chieu van bi nguoi che > 50% khung hinh trong ca 30 frame dau quet duoc — '
                     'mask hien vat co the khong chinh xac')],
                   ['artifact_review.jpg', 'artifacts.json'])
review_phase(3)
ARTIFACTS_DIR = phase_dir  # tai dung cho Phase 4 (pairing) va Phase 7 (anh/video bang chung)

# %% [markdown]
# ## Phase 4 — Ghép người–hiện vật + state machine theo thời gian (2D, luôn chạy)
#
# Trạng thái: `FAR → APPROACHING → NEAR → CONTACT_LIKE → WITHDRAWING`,
# cùng `OCCLUDED` (mất pose tạm thời), `LOST` (mất track quá lâu),
# `AMBIGUOUS` (nhiều hiện vật gần bằng nhau). Không bao giờ nới ngưỡng để
# ép ra `CONTACT_LIKE` khi thiếu bằng chứng — giữ đúng nguyên tắc V1.

# %%
WRIST_IDX = {'left': 9, 'right': 10}  # COCO17


def hand_anchor_point(rec, hand):
    """Điểm mốc để đo khoảng cách tới hiện vật: ưu tiên tâm các điểm ngón tay
    RTMW khi chất lượng 'ok' (Phase 2b), nếu không rơi về cổ tay COCO17."""
    hq = (rec.get('hand_quality') or {}).get(hand)
    if hq and hq['status'] == 'ok':
        pts = np.array(rec['hand_kp'][hand])
        conf = np.array(rec['hand_kc'][hand])
        vis = conf >= CFG['hand_conf_thr']
        if vis.sum() > 0:
            return pts[vis].mean(axis=0), 'hand'
    idx = WRIST_IDX[hand]
    if rec['kc'][idx] >= CFG['pose_conf_thr']:
        return np.array(rec['kp'][idx]), 'wrist'
    return None, None


def norm_distance(wrist_xy, dist_map, box_height):
    x, y = int(round(wrist_xy[0])), int(round(wrist_xy[1]))
    h, w = dist_map.shape
    if not (0 <= x < w and 0 <= y < h):
        return None
    px_dist = float(dist_map[y, x])
    return px_dist / max(box_height, 1e-3)


class PairState:
    """State machine theo T06. `history` lưu (frame_idx, norm_dist, low) để suy ra
    bằng chứng hướng tay A (approach: khoảng cách đang giảm) và W (withdraw: khoảng
    cách tăng sau một cực tiểu) — dùng để làm giàu nhãn state, KHÔNG dùng để chặn
    CONTACT_LIKE của motif Hold (plan T06 nói rõ Hold không bắt buộc approach/withdraw)."""

    def __init__(self):
        self.state = 'FAR'
        self.history = deque()
        self.lost_counter = 0
        self.min_dist_since_low = None

    def update(self, norm_dist, frame_idx, fps):
        if norm_dist is None:
            self.lost_counter += 1
            if self.lost_counter > CFG['state_lost_grace_frames']:
                self.state = 'OCCLUDED'
            return self.state, False, False
        self.lost_counter = 0
        low = norm_dist <= CFG['state_contact_thr']
        self.history.append((frame_idx, norm_dist, low))
        window = CFG['state_history_seconds'] * fps
        while self.history and frame_idx - self.history[0][0] > window:
            self.history.popleft()
        need = math.ceil(CFG['state_hold_seconds'] * fps)
        recent_low = sum(1 for _, _, l in self.history if l)

        dists = [d for _, d, _ in self.history]
        # A: khoang cach dang giam ro rang trong cua so gan nhat (khong chi 2 frame)
        approach_evidence = len(dists) >= 3 and (dists[0] - dists[-1]) > 0.02 and dists[-1] < dists[-2]
        # W: da tung xuong gan cuc tieu (low) truoc do, gio khoang cach dang tang lai
        withdraw_evidence = bool(
            len(dists) >= 2 and self.min_dist_since_low is not None
            and norm_dist > self.min_dist_since_low + 0.02 and dists[-1] > dists[-2])

        if norm_dist > CFG['state_near_thr']:
            self.state = 'FAR'
            self.min_dist_since_low = None
        elif low and recent_low >= need:
            self.state = 'CONTACT_LIKE'
            self.min_dist_since_low = norm_dist if self.min_dist_since_low is None else min(self.min_dist_since_low, norm_dist)
        elif low:
            self.state = 'NEAR'
            self.min_dist_since_low = norm_dist if self.min_dist_since_low is None else min(self.min_dist_since_low, norm_dist)
        elif withdraw_evidence:
            self.state = 'WITHDRAWING'
        elif approach_evidence or self.state in ('FAR', 'APPROACHING'):
            self.state = 'APPROACHING'
        else:
            self.state = 'WITHDRAWING'
        return self.state, approach_evidence, withdraw_evidence


def pairing_and_state_machine(tracks, artifacts, phase_dir_artifacts: Path, fps):
    dist_maps = {a['id']: artifact_mask_distance_map(a, phase_dir_artifacts, (H, W)) for a in artifacts}
    pair_states: dict[tuple, PairState] = {}
    per_frame_states = []
    for i, recs in enumerate(tracks):
        frame_states = []
        for rec in recs:
            if rec.get('kp') is None:
                continue
            box_h = rec['box'][3] - rec['box'][1]
            best = None
            for hand in WRIST_IDX:
                anchor, anchor_source = hand_anchor_point(rec, hand)
                if anchor is None:
                    continue
                candidates = []
                for a in artifacts:
                    d = norm_distance(anchor, dist_maps[a['id']], box_h)
                    if d is not None and d <= CFG['pairing_max_norm_dist']:
                        candidates.append((d, a['id']))
                if not candidates:
                    continue
                candidates.sort()
                d0, aid0 = candidates[0]
                ambiguous = len(candidates) > 1 and (candidates[1][0] - d0) < 0.15
                key = (rec['id'], aid0, hand)
                ps = pair_states.setdefault(key, PairState())
                state, approach_evidence, withdraw_evidence = ps.update(d0, i, fps)
                if ambiguous and state in ('NEAR', 'CONTACT_LIKE'):
                    state = 'AMBIGUOUS'
                frame_states.append(dict(track_id=rec['id'], artifact_id=aid0, hand=hand,
                                          norm_dist=d0, state=state, anchor_source=anchor_source,
                                          anchor_xy=(float(anchor[0]), float(anchor[1])),
                                          approach_evidence=approach_evidence, withdraw_evidence=withdraw_evidence))
        per_frame_states.append(frame_states)
    return per_frame_states, pair_states


phase_dir = begin_phase(4)
PER_FRAME_STATES, PAIR_STATES = pairing_and_state_machine(TRACKS, ARTIFACTS, ARTIFACTS_DIR, FPS)
contact_states = [s for fs in PER_FRAME_STATES for s in fs if s['state'] == 'CONTACT_LIKE']
n_contact_frames = len(contact_states)
n_contact_with_approach = sum(1 for s in contact_states if s['approach_evidence'])
n_contact_with_withdraw = sum(1 for s in contact_states if s['withdraw_evidence'])
import csv
with open(phase_dir / 'pairing_states.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame', 'track_id', 'artifact_id', 'hand', 'norm_dist', 'state',
                'approach_evidence', 'withdraw_evidence'])
    for i, fs in enumerate(PER_FRAME_STATES):
        for s in fs:
            w.writerow([i, s['track_id'], s['artifact_id'], s['hand'], round(s['norm_dist'], 4), s['state'],
                        s['approach_evidence'], s['withdraw_evidence']])
phase_diagnostics(phase_dir, 'pairing_state_machine',
                   dict(n_contact_like_frames=n_contact_frames, n_pairs=len(PAIR_STATES),
                        n_contact_with_approach_evidence=n_contact_with_approach,
                        n_contact_with_withdraw_evidence=n_contact_with_withdraw),
                   [('PASS' if len(PAIR_STATES) > 0 else 'FAIL', 'so_cap_track_artifact', len(PAIR_STATES),
                     'khong co cap track-hien vat nao duoc theo doi thi khong co gi de fuse/evaluate'),
                    ('PASS' if n_contact_frames == 0 or n_contact_with_approach > 0 or n_contact_with_withdraw > 0
                     else 'WARN', 'contact_without_approach_withdraw',
                     dict(n_contact_frames=n_contact_frames, n_with_approach=n_contact_with_approach,
                          n_with_withdraw=n_contact_with_withdraw),
                     'toan bo frame CONTACT_LIKE khong co bang chung approach/withdraw nao trong lich su quan sat '
                     '— co the la Hold bi cat dau/cuoi clip (duoc plan T06 cho phep), nhung cung co the la nhieu '
                     'do nguong state_contact_thr/state_near_thr qua rong, can review thu cong')],
                   ['pairing_states.csv'])
review_phase(4)

# %% [markdown]
# ## Phase 5 — Geometry tùy chọn (VGGT + Depth Anything V2) — mặc định TẮT
#
# Chỉ dùng để **refine** (xác nhận/hạ bậc) `CONTACT_LIKE` bằng khoảng
# cách 3D khi `CFG['enable_geometry']=True`. Không bao giờ bật geometry
# để tự nới ngưỡng 2D ở Phase 4. Nếu tắt, phase này chỉ ghi chú rõ trong
# manifest và bỏ qua — không giả lập kết quả 3D.

# %%
phase_dir = begin_phase(5)
GEOMETRY_RESULT = None
if not CFG['enable_geometry']:
    phase_diagnostics(phase_dir, 'geometry_optional', dict(enabled=False), [
        ('WARN', 'geometry_tat', 'enable_geometry=False',
         'contact_like se khong duoc xac nhan bang khoang cach 3D, chi dua tren 2D'),
    ], [])
else:
    def vggt_reference_light(frames, folder: Path):
        from transformers import AutoModel
        model = AutoModel.from_pretrained('facebook/VGGT-1B', trust_remote_code=True).to(DEVICE).eval()
        model.point_head = None
        model.track_head = None
        idxs = np.linspace(0, len(frames) - 1, CFG['reference_frames']).astype(int)
        imgs = torch.stack([torch.from_numpy(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
                             for i in idxs]).unsqueeze(0).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu', dtype=torch.float16):
            agg = model.aggregator(imgs)
        with torch.no_grad(), torch.autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu', enabled=False):
            depth, conf = model.depth_head(agg, imgs)
            extr, intr = model.camera_head(agg)
        np.savez(folder / 'vggt_reference.npz', depth=depth.cpu().numpy(), conf=conf.cpu().numpy(),
                 K=intr.cpu().numpy(), extrinsics=extr.cpu().numpy(), frame_idx=idxs)
        return dict(depth=depth.cpu().numpy(), K=intr.cpu().numpy(), frame_idx=idxs)

    def depth_anything_v2_light(frames, folder: Path):
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download('depth-anything/Depth-Anything-V2-Metric-Hypersim-Small',
                                'depth_anything_v2_metric_hypersim_vits.pth')
        ensure_depth_anything_v2_repo()
        from depth_anything_v2.dpt import DepthAnythingV2
        model = DepthAnythingV2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
        model = model.to(DEVICE).eval()
        depths = []
        for f in frames:
            with torch.no_grad(), torch.autocast(device_type='cuda' if DEVICE == 'cuda' else 'cpu', dtype=torch.float16):
                d = model.infer_image(f, input_size=CFG['da_v2_input_size'])
            if not np.isfinite(d).all():
                raise RuntimeError('Depth Anything V2 tra ve gia tri khong huu han.')
            depths.append(d.astype(np.float16))
        np.save(folder / 'depth_anything_v2.npy', np.stack(depths))
        return np.stack(depths)

    try:
        vggt_out = vggt_reference_light(frames, phase_dir)
        dav2_out = depth_anything_v2_light(frames, phase_dir)
        GEOMETRY_RESULT = dict(vggt=vggt_out, dav2_shape=dav2_out.shape)
        phase_diagnostics(phase_dir, 'geometry_optional', dict(enabled=True), [
            ('PASS', 'vggt_dav2_loaded', 'ok', 'co the dung de refine CONTACT_LIKE o Phase 7'),
        ], ['vggt_reference.npz', 'depth_anything_v2.npy'])
    except Exception as e:
        phase_diagnostics(phase_dir, 'geometry_optional', dict(enabled=True, error=str(e)), [
            ('WARN', 'geometry_loi', str(e), 'fail-closed: tiep tuc chi voi 2D, khong chan pipeline'),
        ], [])
        GEOMETRY_RESULT = None
review_phase(5)

# %% [markdown]
# ## Phase 6 — Isolation Forest shadow scoring — mặc định chỉ chạy nếu model có sẵn
#
# Chỉ gắn thêm điểm bất thường chuyển động vào event để reviewer đối
# chiếu; **không bao giờ** dùng để gate hay thay đổi quyết định
# `CONTACT_LIKE`/`TOUCH_CANDIDATE`.

# %%
phase_dir = begin_phase(6)
IF_SCORES = None
if not CFG['enable_isolation_forest']:
    phase_diagnostics(phase_dir, 'isolation_forest_shadow', dict(enabled=False), [
        ('WARN', 'if_model_khong_co', 'enable_isolation_forest=False',
         'khong co diem shadow kem theo event, khong anh huong quyet dinh chinh'),
    ], [])
else:
    import warnings
    from collections import defaultdict

    import joblib

    IF_WINDOW_SECONDS = CFG['if_window_seconds']
    IF_N_STEPS = 16
    IF_MIN_VALID_RATIO = 0.80
    IF_MIN_VISIBILITY = 0.60
    IF_MIN_KP_CONF = 0.50
    IF_MAX_GAP_S = 0.75
    IF_KEYPOINT_NAMES = ['nose', 'eye_l', 'eye_r', 'ear_l', 'ear_r', 'shoulder_l', 'shoulder_r',
                          'elbow_l', 'elbow_r', 'wrist_l', 'wrist_r', 'hip_l', 'hip_r',
                          'knee_l', 'knee_r', 'ankle_l', 'ankle_r']
    IF_FEATURE_NAMES = (
        [f'pose_{a}_{kp}' for kp in IF_KEYPOINT_NAMES for a in ('x', 'y')]
        + [f'keypoint_confidence_{kp}' for kp in IF_KEYPOINT_NAMES]
        + [f'pose_velocity_{a}_{kp}' for kp in IF_KEYPOINT_NAMES for a in ('x', 'y')]
        + ['bbox_center_x', 'bbox_center_y', 'bbox_width', 'bbox_height', 'bbox_aspect',
           'hip_height', 'torso_angle', 'bbox_velocity_x', 'bbox_velocity_y']
    )
    assert len(IF_FEATURE_NAMES) == 94, f'Schema IF sai: ky vong 94 ten feature, duoc {len(IF_FEATURE_NAMES)}'

    def _if_row_features(rec, frame_w, frame_h):
        if rec.get('kp') is None or rec.get('kc') is None or rec.get('box') is None:
            return None
        coords = np.asarray(rec['kp'], dtype=np.float32)
        conf = np.asarray(rec['kc'], dtype=np.float32)
        box = np.asarray(rec['box'], dtype=np.float32)
        if coords.shape != (17, 2) or conf.shape != (17,) or not np.isfinite(box).all():
            return None
        x1, y1, x2, y2 = box
        bw, bh = max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
        visible = conf >= IF_MIN_KP_CONF
        hips, shoulders = [11, 12], [5, 6]
        origin = coords[hips].mean(axis=0) if visible[hips].all() else np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
        scale = max(bh, math.sqrt(bw * bh), 1.0)
        normalized = (coords - origin) / scale
        normalized[~visible] = np.nan
        hip_height, torso_angle = np.nan, np.nan
        if visible[hips].all():
            hc = coords[hips].mean(axis=0)
            hip_height = (hc[1] - y1) / bh
            if visible[shoulders].all():
                sc = coords[shoulders].mean(axis=0)
                vec = sc - hc
                torso_angle = math.atan2(float(vec[0]), float(-vec[1]))
        bbox = np.asarray([((x1 + x2) / 2) / frame_w, ((y1 + y2) / 2) / frame_h,
                            bw / frame_w, bh / frame_h, bw / bh, hip_height, torso_angle], np.float32)
        return normalized.reshape(-1), conf, bbox

    def _make_if_windows(by_track):
        windows = []
        for track_id, rows in by_track.items():
            rows.sort(key=lambda r: r[0])
            ts = np.asarray([r[0] for r in rows], dtype=np.float32)
            pose = np.asarray([r[1] for r in rows], dtype=np.float32)
            conf = np.asarray([r[2] for r in rows], dtype=np.float32)
            bbox = np.asarray([r[3] for r in rows], dtype=np.float32)
            if len(ts) < 2:
                continue
            start = ts[0]
            while start + IF_WINDOW_SECONDS <= ts[-1] + 1e-6:
                end = start + IF_WINDOW_SECONDS
                mask = (ts >= start) & (ts <= end)
                idx = np.nonzero(mask)[0]
                if len(idx) < 2:
                    start += 1.0
                    continue
                seg_t = ts[idx]
                if np.diff(seg_t).max() > IF_MAX_GAP_S:
                    start += 1.0
                    continue
                grid = np.linspace(start, end, IF_N_STEPS)
                pose_i = np.stack([np.interp(grid, seg_t, pose[idx, k]) for k in range(pose.shape[1])], axis=1)
                conf_i = np.stack([np.interp(grid, seg_t, conf[idx, k]) for k in range(conf.shape[1])], axis=1)
                bbox_i = np.stack([np.interp(grid, seg_t, bbox[idx, k]) for k in range(bbox.shape[1])], axis=1)
                valid_ratio = len(idx) / IF_N_STEPS
                visibility = float((conf_i >= IF_MIN_KP_CONF).mean())
                if valid_ratio < IF_MIN_VALID_RATIO or visibility < IF_MIN_VISIBILITY:
                    start += 1.0
                    continue
                dt = np.diff(grid)
                dt[dt == 0] = 1e-6
                pv = np.vstack([np.zeros((1, pose_i.shape[1]), np.float32),
                                 np.diff(pose_i, axis=0) / dt[:, None]])
                bv = np.vstack([np.zeros((1, bbox_i.shape[1]), np.float32),
                                 np.diff(bbox_i, axis=0) / dt[:, None]])
                x = np.concatenate([pose_i, conf_i, pv, bbox_i[:, :5], bbox_i[:, 5:7], bv[:, :2]], axis=1)
                if x.shape[1] != len(IF_FEATURE_NAMES):
                    raise ValueError('Schema live pose khac schema train.')
                windows.append(dict(track_id=track_id, start_s=float(start), end_s=float(end),
                                     features=x, valid_step_ratio=valid_ratio, keypoint_visibility=visibility))
                start += 1.0
        return windows

    def build_live_if_windows(frames_arr, tracks, fps):
        h, w = frames_arr[0].shape[:2]
        byid = defaultdict(list)
        for fi, recs in enumerate(tracks):
            for rec in recs:
                vals = _if_row_features(rec, w, h)
                if vals is not None:
                    byid[int(rec['id'])].append((fi / fps, *vals))
        return _make_if_windows(byid)

    def summarize_if_windows(features, feature_names):
        keep = np.asarray([not n.startswith('keypoint_confidence_') for n in feature_names])
        x = features[:, :, keep]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(x, axis=1)
            spread = np.nanpercentile(x, 75, axis=1) - np.nanpercentile(x, 25, axis=1)
            low = np.nanmin(x, axis=1)
            high = np.nanmax(x, axis=1)
        return np.concatenate([med, spread, low, high], axis=1).astype(np.float32)

    def if_percentile(reference, values):
        reference = np.sort(np.asarray(reference))
        return np.searchsorted(reference, values, side='right') / max(len(reference), 1)

    bundle = joblib.load(INPUT_PATHS['isolation_forest_normal_yolov8_pose.joblib'])
    pipeline = bundle['pipeline']
    source_feature_names = bundle.get('source_feature_names', bundle.get('feature_names'))
    assert len(source_feature_names) == 94, f'Schema IF thay doi: ky vong 94 feature, duoc {len(source_feature_names)}'
    camera_reference_scores = bundle.get('camera_reference_scores', {})

    live_windows = build_live_if_windows(frames, TRACKS, FPS)
    IF_SCORES = {}
    if live_windows:
        feats = np.stack([w['features'] for w in live_windows], axis=0)
        emb = summarize_if_windows(feats, IF_FEATURE_NAMES)
        raw = -pipeline.decision_function(emb)
        reference = camera_reference_scores.get(int(CLIP)) if camera_reference_scores else None
        pct = if_percentile(reference, raw) if reference is not None else np.full_like(raw, np.nan)
        for w, score, p in zip(live_windows, raw, pct):
            entry = IF_SCORES.setdefault(w['track_id'], [])
            entry.append(dict(start_s=w['start_s'], end_s=w['end_s'],
                               anomaly_score=float(score),
                               camera_percentile=None if np.isnan(p) else float(p)))
    phase_diagnostics(phase_dir, 'isolation_forest_shadow',
                       dict(enabled=True, n_tracks_scored=len(IF_SCORES), n_windows=len(live_windows)), [
        ('PASS' if live_windows else 'WARN', 'if_windows', len(live_windows),
         'khong co window hop le neu qua it pose lien tuc; van khong chan pipeline'),
    ], [])
review_phase(6)

# %% [markdown]
# ## Phase 7 — Fusion & Events: 2D state machine (bắt buộc) + geometry refine (nếu có) + IF annotate (nếu có)

# %%
def fuse_events(per_frame_states, geometry_result, if_scores, fps):
    """Nhãn quyết định chỉ dựa trên state machine 2D (T06). Dự án KHÔNG có bước xác
    minh tiếp xúc 3D theo từng event (xem manifest `not_implemented`:
    T12_open_research_3d_contact_verifier là bài toán nghiên cứu mở, ngoài scope MVP).
    Vì vậy mọi `CONTACT_LIKE` luôn ra `TOUCH_CANDIDATE_2D_ONLY`, bất kể Phase 5 (geometry)
    có chạy được hay không — tránh ngụ ý đã xác minh độ sâu cho riêng event này.
    `geometry_available` chỉ ghi lại việc dữ liệu geometry có tồn tại cho cả clip,
    không phải việc nó đã được dùng để xác minh event."""
    events = []
    open_events = {}
    geometry_available = geometry_result is not None
    for i, fs in enumerate(per_frame_states):
        active_keys = set()
        for s in fs:
            key = (s['track_id'], s['artifact_id'], s['hand'])
            active_keys.add(key)
            decision = s['state']
            if decision == 'CONTACT_LIKE':
                decision = 'TOUCH_CANDIDATE_2D_ONLY'
            if decision.startswith('TOUCH_CANDIDATE'):
                if key not in open_events:
                    open_events[key] = dict(track_id=s['track_id'], artifact_id=s['artifact_id'],
                                             hand=s['hand'], start_frame=i, end_frame=i, decision=decision,
                                             geometry_available=geometry_available)
                else:
                    open_events[key]['end_frame'] = i
            elif key in open_events:
                events.append(open_events.pop(key))
        for key in list(open_events):
            if key not in active_keys and (i - open_events[key]['end_frame']) > CFG['state_lost_grace_frames']:
                events.append(open_events.pop(key))
    events.extend(open_events.values())
    if if_scores:
        for e in events:
            windows = if_scores.get(e['track_id'])
            if not windows:
                e['motion_context'] = 'no_score'
                continue
            start_s, end_s = e['start_frame'] / fps, e['end_frame'] / fps
            nearby = [w for w in windows if w['start_s'] <= end_s and w['end_s'] >= start_s] or windows
            peak = max(nearby, key=lambda w: w['anomaly_score'])
            e['motion_anomaly_score'] = peak['anomaly_score']
            e['motion_anomaly_percentile'] = peak['camera_percentile']
            if peak['camera_percentile'] is None:
                e['motion_context'] = 'camera_unvalidated_score_only'
            elif peak['camera_percentile'] >= 0.99:
                e['motion_context'] = 'unusual_motion_outlier'
            else:
                e['motion_context'] = 'shadow_score_available'
    return events


def _lookup_state(per_frame_states, frame_idx, key):
    for s in per_frame_states[frame_idx]:
        if (s['track_id'], s['artifact_id'], s['hand']) == key:
            return s
    return None


def render_event_evidence(events, per_frame_states, frames, artifacts, artifacts_dir: Path, phase_dir: Path, fps):
    """Với mỗi event: ghép ảnh 3 keyframe (đầu/giữa/cuối) có bbox người, bbox+mask
    hiện vật, điểm neo tay và nhãn state/khoảng cách; nếu event dài hơn 1 frame thì
    còn ghi thêm 1 video ngắn cho đúng khoảng [start_frame, end_frame]. Chỉ là minh
    hoạ 2D — không suy ra hay vẽ bất kỳ thông tin độ sâu/3D nào chưa được xác minh."""
    masks = {a['id']: np.load(artifacts_dir / f"artifact_{a['id']}_mask.npy") for a in artifacts}
    boxes = {a['id']: a['box'] for a in artifacts}
    labels = {a['id']: a['label'] for a in artifacts}
    manifest = []
    for ei, e in enumerate(events):
        key = (e['track_id'], e['artifact_id'], e['hand'])
        lo, hi = e['start_frame'], min(e['end_frame'], len(frames) - 1)

        def draw(fidx):
            vis = frames[fidx].copy()
            aid = e['artifact_id']
            x1, y1, x2, y2 = map(int, boxes[aid])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis, labels[aid], (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            mask = masks.get(aid)
            if mask is not None and mask.shape[:2] == vis.shape[:2]:
                overlay = vis.copy()
                overlay[mask.astype(bool)] = (255, 0, 0)
                vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
            s = _lookup_state(per_frame_states, fidx, key)
            if s and s.get('anchor_xy'):
                ax, ay = map(int, s['anchor_xy'])
                cv2.circle(vis, (ax, ay), 6, (0, 0, 255), -1)
                cv2.putText(vis, f"{s['state']} d={s['norm_dist']:.2f}", (ax + 8, ay),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(vis, f"frame {fidx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return vis

        panels = [draw(fidx) for fidx in sorted(set([lo, (lo + hi) // 2, hi]))]
        h = min(p.shape[0] for p in panels)
        composite = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels])
        img_name = f"event_{ei:03d}_t{e['track_id']}_a{e['artifact_id']}_{e['hand']}.jpg"
        cv2.imwrite(str(phase_dir / img_name), composite)

        video_name = None
        if hi > lo:
            video_name = f"event_{ei:03d}_t{e['track_id']}_a{e['artifact_id']}_{e['hand']}.mp4"
            vh, vw = frames[lo].shape[:2]
            writer = cv2.VideoWriter(str(phase_dir / video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (vw, vh))
            for fidx in range(lo, hi + 1):
                writer.write(draw(fidx))
            writer.release()

        manifest.append(dict(event_index=ei, image=img_name, video=video_name,
                              track_id=e['track_id'], artifact_id=e['artifact_id'], hand=e['hand'],
                              start_frame=lo, end_frame=hi, decision=e['decision']))
    (phase_dir / 'evidence_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


phase_dir = begin_phase(7)
EVENTS = fuse_events(PER_FRAME_STATES, GEOMETRY_RESULT, IF_SCORES, FPS)
(phase_dir / 'events.json').write_text(json.dumps(EVENTS, indent=2, ensure_ascii=False))
with open(phase_dir / 'events.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['track_id', 'artifact_id', 'hand', 'start_frame', 'end_frame', 'decision', 'geometry_available'])
    for e in EVENTS:
        w.writerow([e['track_id'], e['artifact_id'], e['hand'], e['start_frame'], e['end_frame'],
                     e['decision'], e.get('geometry_available', False)])
n_touch = sum(1 for e in EVENTS if e['decision'].startswith('TOUCH_CANDIDATE'))
EVIDENCE = render_event_evidence(EVENTS, PER_FRAME_STATES, frames, ARTIFACTS, ARTIFACTS_DIR, phase_dir, FPS)
n_video = sum(1 for m in EVIDENCE if m['video'])
phase_diagnostics(phase_dir, 'fusion_events',
                   dict(n_events=len(EVENTS), n_touch_candidate=n_touch,
                        n_evidence_images=len(EVIDENCE), n_evidence_videos=n_video), [
    ('PASS' if n_touch > 0 else 'WARN', 'n_touch_candidate', n_touch,
     'khong co event nao dat CONTACT_LIKE — kiem tra lai nguong state machine hoac clip khong co tuong tac'),
    ('WARN', 'geometry_refine_note', 'khong dung de xac minh event',
     'du co geometry (VGGT/DAV2) hay khong, quyet dinh van la TOUCH_CANDIDATE_2D_ONLY — '
     'xac minh tiep xuc 3D theo tung event la T12_open_research_3d_contact_verifier, ngoai scope MVP'),
    ('PASS' if len(EVIDENCE) == len(EVENTS) else 'FAIL', 'so_anh_bang_chung', len(EVIDENCE),
     'moi event phai co dung 1 anh ghep minh hoa — thieu nghia la loi khi render'),
], ['events.json', 'events.csv', 'evidence_manifest.json'])
review_phase(7)

# %% [markdown]
# ## Phase 8 — Evaluation: so khớp với nhãn thật nếu có; nếu không, báo rõ chưa thể đánh giá
#
# T00 (gán nhãn ground-truth) **chưa triển khai** trong notebook này —
# cần annotator thủ công xem video và điền `annotations/events_ground_truth.csv`
# (schema: `gt_person_id,artifact_id,hand,start_frame,end_frame,ref_frame,ref_x1,ref_y1,ref_x2,ref_y2`).
# `gt_person_id` là nhãn người ổn định do annotator đặt — **không phải** track_id của
# model; `ref_frame`/`ref_box` là 1 bbox annotator khoanh tại `ref_frame` để xác nhận
# đúng người đó trong chính lần chạy này (xem PLAN_TOUCH_V2.md §5.2 + §11.2: không được
# giả định track_id của model trùng với số thứ tự người trong nhãn thật, track 3 trong
# clip pilot chỉ là case hồi quy để theo dõi, không phải quy tắc "track 3 luôn là người X").
# `ref_box` phải khoanh theo cùng độ phân giải ảnh làm việc đã resize (`frames`,
# `width=CFG['width']`) — đúng độ phân giải dùng trong ảnh bằng chứng ở Phase 7
# (`07_fusion_events/attempt_XX/event_*.jpg`), không phải `native_frames` gốc.
# Nếu file đó tồn tại trong input, phase này (1) ánh xạ từng `gt_person_id` sang track_id
# dự đoán bằng IoU bbox tại `ref_frame`, (2) match một-một theo overlap thời gian trên
# track đã ánh xạ; nếu không có file, ghi rõ giới hạn thay vì giả lập số liệu.

# %%
phase_dir = begin_phase(8)
GT_PATH = locate_input_file('events_ground_truth.csv', DATASET_DIR)
GT_MATCH_MIN_IOU = 0.3


def load_gt(path: Path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(dict(gt_person_id=r['gt_person_id'], artifact_id=int(r['artifact_id']), hand=r['hand'],
                              start_frame=int(r['start_frame']), end_frame=int(r['end_frame']),
                              ref_frame=int(r['ref_frame']),
                              ref_box=[float(r['ref_x1']), float(r['ref_y1']), float(r['ref_x2']), float(r['ref_y2'])]))
    return rows


def map_gt_person_to_track(g, tracks):
    """Không dùng ID bằng số làm căn cứ — khoanh 1 bbox tại ref_frame rồi tìm track
    dự đoán có IoU cao nhất với bbox đó. Nếu không track nào đạt GT_MATCH_MIN_IOU,
    coi gt_person_id này là không ánh xạ được trong lần chạy hiện tại (miss, không phải lỗi)."""
    if g['ref_frame'] >= len(tracks):
        return None, 0.0
    best_tid, best_iou = None, 0.0
    for rec in tracks[g['ref_frame']]:
        iou = box_iou(rec['box'], g['ref_box'])
        if iou > best_iou:
            best_tid, best_iou = rec['id'], iou
    if best_iou < GT_MATCH_MIN_IOU:
        return None, best_iou
    return best_tid, best_iou


def frame_overlap(a, b):
    lo = max(a['start_frame'], b['start_frame'])
    hi = min(a['end_frame'], b['end_frame'])
    return max(0, hi - lo)


if GT_PATH is None:
    phase_diagnostics(phase_dir, 'evaluation', dict(has_ground_truth=False), [
        ('WARN', 'thieu_nhan_thuc', 'khong co annotations/events_ground_truth.csv',
         'T00 (gan nhan) chua thuc hien — khong the tinh precision/recall, chi bao cao events du doan'),
    ], [str((RUN / '07_fusion_events').name)])
else:
    gt_rows = load_gt(GT_PATH)
    pred = [e for e in EVENTS if e['decision'].startswith('TOUCH_CANDIDATE')]

    gt = []
    n_unmapped = 0
    for g in gt_rows:
        track_id, iou = map_gt_person_to_track(g, TRACKS)
        if track_id is None:
            n_unmapped += 1
            continue
        gt.append(dict(g, track_id=track_id, match_iou=round(iou, 3)))

    candidates = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            if (p['track_id'], p['artifact_id'], p['hand']) == (g['track_id'], g['artifact_id'], g['hand']):
                ov = frame_overlap(p, g)
                if ov > 0:
                    candidates.append((ov, pi, gi))
    candidates.sort(reverse=True)
    matched_gt, matched_pred, matches = set(), set(), []
    for ov, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matches.append(dict(pred_index=pi, gt_person_id=gt[gi]['gt_person_id'], overlap_frames=ov))

    precision = len(matched_pred) / len(pred) if pred else 0.0
    recall = len(matched_gt) / len(gt) if gt else 0.0
    metrics = dict(has_ground_truth=True, n_pred=len(pred), n_gt_rows=len(gt_rows), n_gt_mapped=len(gt),
                    n_gt_unmapped=n_unmapped, precision=precision, recall=recall)
    (phase_dir / 'evaluation.json').write_text(json.dumps(dict(metrics=metrics, matches=matches,
                                                                 gt_mapping=[dict(gt_person_id=g['gt_person_id'],
                                                                                   mapped_track_id=g['track_id'],
                                                                                   match_iou=g['match_iou'])
                                                                             for g in gt]),
                                                            indent=2, ensure_ascii=False))
    phase_diagnostics(phase_dir, 'evaluation', metrics, [
        ('PASS' if recall > 0 or not gt else 'WARN', 'recall', round(recall, 3), 'ty le event thuc duoc phat hien'),
        ('PASS' if precision > 0 or not pred else 'WARN', 'precision', round(precision, 3), 'ty le du doan dung'),
        ('PASS' if n_unmapped == 0 else 'WARN', 'gt_khong_anh_xa_duoc', n_unmapped,
         f'co gt_person_id khong tim duoc track du doan nao dat IoU >= {GT_MATCH_MIN_IOU} tai ref_frame — '
         'co the nguoi khong xuat hien/bi che tai frame do, khong tinh la mismatch loi cua model'),
    ], ['evaluation.json'])
review_phase(8)

# %% [markdown]
# ## Đóng gói kết quả

# %%
manifest.update(completed=STATE['completed'],
                 implemented_scope='MVP Phase 0-8 (Section 10 PLAN_TOUCH_V2.md)',
                 not_implemented=['T00_manual_ground_truth_labeling',
                                   'T08_isolation_forest_v2_retrain_on_interaction_data',
                                   'T12_open_research_3d_contact_verifier'],
                 geometry_enabled=CFG['enable_geometry'],
                 isolation_forest_enabled=CFG['enable_isolation_forest'],
                 metric_calibrated=False, hand_landmarks=bool(HAS_HAND_POSE))
(RUN / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

zip_path = WORK / f'{RUN.name}_touch_v2_results.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for p in RUN.rglob('*'):
        if p.is_file():
            zf.write(p, p.relative_to(WORK))
print('Saved:', zip_path)

try:
    from IPython.display import FileLink, display
    display(FileLink(str(zip_path)))
except Exception:
    print('Mo file thu cong tai:', zip_path)
