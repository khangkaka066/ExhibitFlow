"""Cấu hình mặc định của pipeline ExhibitFlow.

Mọi hằng số nằm ở đây. Các giá trị kiểu dict / list được cập nhật TẠI CHỖ bởi `apply_overrides()` (file YAML / JSON,
cờ `--set` của main.py), nên module khác có thể `from .settings import CFG` mà vẫn thấy giá trị mới.
Các giá trị đơn (đường dẫn, chuỗi, số) phải đọc qua `settings.X` (không `from .settings import X`).
"""
import copy
import json
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent                      # thư mục exhibitflow_pipeline/
VERSION = '1.0.0'

# ---------------- Thư mục làm việc ----------------
# WORK : nơi để file tạm tải về (llama.cpp, GGUF cache, cache mask hiện vật). OUT: kết quả (main.py --out ghi đè).
WORK = Path(os.environ.get('EXHIBITFLOW_WORK') or PROJECT_DIR / 'work')
OUT = WORK / 'results'
CACHE = WORK / 'cache'
# Thư mục tìm trọng số (đệ quy, độ sâu hạn chế). main.py --weights-dir thêm vào đầu danh sách.
WEIGHT_ROOTS = [Path(p) for p in os.environ.get('EXHIBITFLOW_WEIGHTS', '').split(os.pathsep) if p]
WEIGHT_ROOTS += [PROJECT_DIR / 'weights', Path('/kaggle/input'), Path('/content'), Path('/content/drive/MyDrive')]
# Repo tracker (ByteTrack-DMA-LTC): cần cho detector 'yolox' và tracker 'bytetrack_dma_ltc'. None = tự tìm / clone.
TRACKER_REPO = Path(os.environ['EXHIBITFLOW_TRACKER_REPO']) if os.environ.get('EXHIBITFLOW_TRACKER_REPO') else None
TRACKER_REPO_URL = 'https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker'

try:
    import torch
    N_GPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
    GPU_NAMES = [torch.cuda.get_device_name(i) for i in range(N_GPU)]
except Exception:                                      # chưa cài torch (chỉ đọc cấu hình)
    N_GPU, GPU_NAMES = 0, []
CUDA_VISIBLE_DEVICES_0 = os.environ.get('CUDA_VISIBLE_DEVICES')   # Ultralytics sửa biến này khi predict(device=...)
GPU_A = 'cuda:0' if N_GPU >= 1 else 'cpu'              # luồng perception theo frame
GPU_B = 'cuda:1' if N_GPU >= 2 else GPU_A              # riêng VLM; chỉ 1 GPU thì dùng chung GPU_A

# ---------------- Model ----------------
# Detector + tracker = `make track-video-full-torch` của repo ByteTrack-DMA-LTC-Motion-Tracker: --fuse --fp16, conf 0.01,
# nms 0.7, chạy trên ảnh gốc của video, box đổi về ảnh làm việc 960 px. Tracker: ByteTrack + DMA (GBM) + LTC + ReID OSNet.
# ckpt: tên file (tự tìm trong WEIGHT_ROOTS) hoặc đường dẫn đầy đủ.
MODELS = dict(
    detector=dict(backend='yolox', device=GPU_A, conf=0.01, nms=0.7, fp16=True, fuse=True, gpu_preproc=True,
                  exp=None, depth=0.33, width=0.5, num_classes=1, test_size=(608, 1088), ckpt='best_ckpt.pth'),
    tracker=dict(backend='bytetrack_dma_ltc', device=GPU_A, with_reid=True,
                 track_thresh=0.6, track_buffer=30, match_thresh=0.9,
                 reid_model='osnet_x0_25', reid_ckpt='osnet_x0_25.pth', reid_weight=0.35, reid_thresh=0.7,
                 ltc_ckpt='ltc_motion_mot17.pth', ltc_min_history=16,
                 dma_ckpt='dma_gbm_mot17.gbm', min_box_area=100, max_aspect=1.6),
    pose=dict(backend='ultralytics_pose', device=GPU_A, weights='yolov8n-pose.pt', batch=True),
    hand=dict(backend='rtmw', device='cuda' if N_GPU else 'cpu', mode='lightweight'),          # | dict(backend='none')
    # segmenter chạy 1 lần / video TRƯỚC pipeline rồi được giải phóng (trả VRAM cho VLM). Mask được cache theo video.
    segmenter=dict(backend='gdino_sam', device=GPU_A, fp16=True,
                   dino='openmmlab-community/mm_grounding_dino_large_all', sam='sam2.1_b.pt'),
    # VLM sản phẩm = Qwen3-VL-8B Q4_K_M (unsloth GGUF) qua llama-server, cấu hình "8f_512_short" của benchmark:
    #   n_images=8 (lấy đều 8 trong 12 ảnh xuất ra) | max_side=512 (cạnh dài 512 px) | short_json (chỉ trả label + confidence)
    # Benchmark (19 đoạn): F1 0.69, ~4.3 s/đoạn trên T4, ~0.53 s trên RTX PRO 6000 (G4). Đổi cấu hình = sửa các khoá này.
    vlm=dict(backend='llama_cpp', repo_id='unsloth/Qwen3-VL-8B-Instruct-GGUF', quant_file='Q4_K_M', mmproj='F16',
             device=GPU_B, n_gpu_layers=99, ctx=16384, max_new_tokens=64, max_pixels=768 * 768, flash_attn='on',
             n_images=8, max_side=512, short_json=True),
    # vlm=dict(backend='dry_run', delay_s=0.0),         # không gọi model (kiểm thử pipeline)
)

# ---------------- Engine realtime ----------------
RT = dict(
    mode='parallel',            # 'parallel': các stage chạy song song (pipelining) | 'sequential': baseline như notebook gốc
    realtime=False,             # False: xử lý MỌI frame (dùng khi đo độ chính xác) | True: đọc đúng FPS như camera, không kịp thì bỏ frame
                                # (chỉ để đo latency/FPS: bỏ frame làm state machine mất chuỗi frame liên tục -> recall giảm mạnh)
    drop_frames=True,           # chỉ khi realtime=True: stage đầu còn bận -> bỏ frame thay vì trễ dần
    queue_size=8,               # độ dài queue giữa 2 stage
    pose_gate_norm_dist=1.5,    # chỉ chạy pose + hand cho người có bbox cách hiện vật <= 1.5 x chiều cao người (None = mọi người).
                                # State machine chỉ bật NEAR khi cổ tay cách hiện vật <= state_near_thr (0.9) -> người xa hơn không thể tạo event.
    artifact_person_filter=True,  # segmentation: chạy detector trên frame segment để loại vùng là người
    artifact_person_min_score=0.3,  # ... chỉ dùng box người có score >= 0.3 (detector chạy với conf 0.01 như make)
    seg_ref_default='auto',     # video không có trong SEG_REF_FRAMES: 'auto' = tự tách cảnh (cắt cảnh / đổi camera),
                                # segment 1 frame cho mỗi cảnh | 'first' = chỉ frame đầu cho cả video
    scene_cut_diff=25.0,        # 'auto': chênh lệch xám trung bình giữa 2 frame liền nhau (ảnh 64x36) >= ngưỡng -> cắt cảnh
                                # (frame thường ~0-2, cắt cảnh ~30-50)
    scene_step_sec=0.0,         # 'auto': so sánh mỗi N giây (0 = mọi frame, như benchmark). Dashboard đặt 0.2 để mở video nhanh hơn
    scene_min_sec=1.0,          # 'auto': cảnh ngắn hơn (vd vài frame chuyển cảnh) -> gộp vào cảnh sau
    scene_ref_offset_sec=0.5,   # 'auto': frame segment = đầu cảnh + 0.5s (tránh frame mờ lúc chuyển cảnh)
    artifact_cache=True,        # dùng lại mask hiện vật đã tính (cùng video + cùng segmenter)
    ring_seconds=20.0,          # ring buffer ảnh gốc để cắt ảnh cho VLM
    store_long=1280,            # cạnh dài ảnh lưu trong ring buffer (giới hạn RAM)
    vlm_early_sec=None,         # vd 2.0: gửi VLM ngay khi chạm liên tục 2s (báo động sớm), không đợi event kết thúc
    limit_cpu_threads=True,     # mỗi stage là 1 thread Python: torch / OpenCV chỉ dùng 1 thread nội bộ (Kaggle chỉ có 4 vCPU,
                                # để mặc định thì 7 stage x 4 thread tranh nhau CPU -> detect / track chậm hẳn)
    render_live=False,           # sink ghi video live (overlay + HUD FPS/latency)
    alarm_show_sec=4.0,         # video live: thời gian giữ khung đỏ sau khi VLM báo
    vlm_sync=True,              # mode parallel: khi gửi đoạn cho VLM thì dừng cả luồng video, VLM trả lời xong mới chạy tiếp
                                # (alert khớp với khung hình đang xem; thời gian dừng không tính vào FPS / latency)
    vlm_sync_timeout_s=180.0,   # đợi VLM tối đa bấy nhiêu giây mỗi lần dừng (VLM treo thì vẫn chạy tiếp)
)



# ---------------- Chạy ----------------
RUN = dict(
    max_frames=None,            # giới hạn số frame mỗi video (None = cả video)
    wait_vlm_loaded=True,       # đợi VLM nạp xong rồi mới chạy video
    wait_vlm_per_video=True,    # hết video -> đợi VLM xử lý xong các đoạn của video đó
    gpu_monitor=False,          # lấy mẫu nvidia-smi (util %, VRAM) trong lúc chạy
    gpu_monitor_interval=0.5,
)

# ---------------- Ngưỡng thuật toán (giữ nguyên notebook benchmark) ----------------
# ---------------- Ngưỡng thuật toán (giữ nguyên notebook benchmark) ----------------
CFG = dict(
    width=960,
    pose_crop_pad_frac=0.30, pose_min_crop_px=16, pose_conf_thr=0.30, pose_iou_match_thr=0.30, pose_margin_thr=0.15,
    hand_conf_thr=0.30, hand_min_bbox_diag_px=18, hand_small_use_center=True, hand_small_min_conf=0.5, hand_small_max_reach=0.6,
    pose_wide_retry=True, pose_wide_edge_frac=0.08, pose_wide_pad_x=0.7, pose_wide_pad_down=0.6,
    artifact_max_person_overlap=0.6, artifact_min_person_iou=0.4, artifact_max_motion=0.35, artifact_motion_samples=8,
    pairing_max_norm_dist=3.0,
    state_near_thr=0.9, state_contact_thr=0.35, state_hold_seconds=0.25, state_history_seconds=2.0,
    state_lost_grace_frames=5,
    state_min_person_h_px=80, state_reject_rest_arm=True, rest_arm_max_dx=0.10, rest_arm_min_dy=-0.12,
    self_touch_head_frac=0.10, state_ground_max_gap=0.30,
    foot_from_pose=True, foot_extend_min_frac=0.15, foot_extend_max_frac=0.8,
    event_merge_gap_seconds=0.5, event_min_frames=12, max_tasks_per_video=40,
)
SEGMENTATION_CFG = dict(
    sculpture_phrases=['statue', 'sculpture', 'bust'],
    painting_phrases=['painting', 'framed artwork'],
    pedestal_phrases=['pedestal'],
    full_thr=0.30, text_thr=0.25, mask_overlap=0.80, pedestal_merge=True,
    pedestal_point_frac=0.25, pedestal_max_area_ratio=4.0, pedestal_max_person_frac=0.30, min_mask_fill=0.15,
    merge_standalone_pedestal=True, pedestal_cover_merge=0.5,
    fallback_when_no_3d=True,
    fallback_object_phrases=['stone', 'rock', 'artifact', 'small object', 'vase'],
    fallback_pedestal_phrases=['pedestal', 'plinth', 'display stand', 'white box'],
    fallback_thr=0.20, fallback_text_thr=0.20, fallback_max_area_frac=0.20, fallback_max_keep=4,
)
EXCLUDE_ARTIFACTS = {}   # {'video_key': [12]} — id như trong artifact_review.jpg (cảnh k: id + k*10000)
MANUAL_ARTIFACTS = {}    # {'video_key': [[x1, y1, x2, y2]]} — toạ độ ảnh 960px, thêm vào cảnh đầu tiên
# Frame dùng để segment hiện vật, riêng cho từng video. Khoá: video_key ('video-youtube_7'), tên file ('7.mp4') hoặc stem ('7').
#   'auto'                  -> tự tách cảnh, mỗi cảnh segment 1 frame (mặc định cho video không khai báo: RT['seg_ref_default'])
#   'first' | 120           -> 1 cảnh cho cả video, segment frame 0 / frame 120
#   [(0, 600), (704, 720)]  -> nhiều cảnh: (frame bắt đầu cảnh, frame dùng để segment). Mask của cảnh dùng từ frame bắt đầu
#                              tới hết cảnh. Chọn frame hiện vật còn nguyên (TRƯỚC khi bị lấy) và ít bị người che.
# Xem cảnh đã dùng trong <OUT>/<exp>/<video>/artifact_review.jpg (mỗi cảnh 1 ảnh, ghi frame bắt đầu + frame segment).
SEG_REF_FRAMES = {
    # '9': [(0, 12), (537, 549), (720, 732), (1227, 1227)],
}

EXPORT = dict(pre_sec=1.0, post_sec=1.0, max_window_sec=8.0, n_sample_frames=12, crop_pad=0.4, crop_out_long=768)
MASK_REGIONS = [[0.855, 0.925, 0.995, 0.990]]   # xoá watermark (toạ độ tương đối); [] nếu không cần
VLM_IMAGE_SET = 'marked'                         # 'marked' | 'clean'

PROMPT_TEMPLATE = '''You are a museum security analyst. The {n} images are consecutive frames \
(about {dt:.2f}s apart, in order) cropped around ONE visitor marked with a RED box labeled "ID {tid}". \
BLUE outline(s) mark exhibit(s).
Classify ONLY the red-box person's behavior toward the exhibit:
- "normal": looking, walking, pointing, standing close without contact
- "touching": hand makes contact with the exhibit, exhibit stays in place
- "holding": person lifts / holds / moves the exhibit
- "stealing": person takes the exhibit away, hides it (bag/clothes), or the exhibit is missing afterwards
- "uncertain": view is occluded or too small to decide
Answer JSON only:
{{"label": "...", "confidence": 0.0, "contact_frames": [], "evidence": "<one sentence>"}}
"contact_frames" = the Frame numbers (1..{n}) in which the red-box person's hand is VISIBLY touching/holding the exhibit (empty list if never). Do not include frames where the hand is only approaching or near.'''
SYSTEM_PROMPT = ('You are a careful CCTV analyst for a museum. Judge only what is visible in the frames. '
                 'Touching means the hand visibly reaches the exhibit or its pedestal; being near is not touching. '
                 'Always answer with a single JSON object and nothing else.')

DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATE, SYSTEM_PROMPT   # bản gốc (dashboard ghi đè rồi khôi phục được)
PROMPT_FIELDS = ('n', 'dt', 'tid')                                                  # biến được {format}; dấu { } khác phải viết {{ }}

ANOMALY_LABELS = ('touching', 'holding', 'stealing')
VLM_MIN_CONF = 0.0
ALARM_ONSET = 'vlm'      # 'vlm' | 'contact' | 'event'  (như notebook gốc)


# ---------------- Ghi đè cấu hình ----------------
_DICTS = ('MODELS', 'RT', 'RUN', 'CFG', 'SEGMENTATION_CFG', 'EXPORT', 'EXCLUDE_ARTIFACTS', 'MANUAL_ARTIFACTS', 'SEG_REF_FRAMES')
_SCALARS = ('ALARM_ONSET', 'VLM_MIN_CONF', 'VLM_IMAGE_SET', 'MASK_REGIONS', 'ANOMALY_LABELS', 'PROMPT_TEMPLATE', 'SYSTEM_PROMPT')


def _merge_inplace(dst, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_inplace(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply_overrides(over):
    """over: dict {TÊN_HẰNG: giá trị}. Tên dict (MODELS, RT, CFG...) được gộp tại chỗ; tên đơn thì thay thế.
    Khoá thường (models / rt / cfg ...) cũng được chấp nhận (cho file YAML)."""
    g = globals()
    for key, val in (over or {}).items():
        name = key.upper()
        if name in _DICTS:
            if not isinstance(val, dict):
                raise TypeError(f'{key} phải là dict')
            _merge_inplace(g[name], val)
        elif name in _SCALARS:
            if name == 'MASK_REGIONS':
                g[name][:] = [list(r) for r in val]            # list: giữ nguyên đối tượng
            elif name == 'ANOMALY_LABELS':
                g[name] = tuple(val)
            else:
                g[name] = val
        elif name in ('WORK', 'OUT', 'CACHE'):
            g[name] = Path(val)
        elif name == 'WEIGHT_ROOTS':
            WEIGHT_ROOTS[:0] = [Path(p) for p in val]
        else:
            raise KeyError(f'Khoá cấu hình không hợp lệ: {key}. Hợp lệ: {list(_DICTS) + list(_SCALARS)}')


def load_config_file(path):
    """YAML (.yaml / .yml) hoặc JSON -> dict, rồi apply_overrides."""
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if p.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    apply_overrides(data)
    return data


def set_paths(work=None, out=None):
    global WORK, OUT, CACHE
    if work:
        WORK = Path(work)
        CACHE = WORK / 'cache'
        OUT = WORK / 'results'
    if out:
        OUT = Path(out)
    for d in (WORK, OUT, CACHE):
        d.mkdir(parents=True, exist_ok=True)

