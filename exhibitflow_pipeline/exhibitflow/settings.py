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
                  exp=None, depth=0.33, width=0.5, num_classes=1, test_size=(608, 1088), ckpt='best_ckpt.pth',
                  low_light_boost=dict(enabled=True, luma_thr=75.0, clip_limit=2.0, tile_grid=8)),
    tracker=dict(backend='bytetrack_dma_ltc', device=GPU_A, with_reid=True,
                 track_thresh=0.5, track_buffer=75, match_thresh=0.9,
                 reid_model='osnet_x0_25', reid_ckpt='osnet_x0_25.pth', reid_weight=0.5, reid_thresh=0.6,
                 ltc_ckpt='ltc_motion_mot17.pth', ltc_min_history=10,
                 dma_ckpt='dma_gbm_mot17.gbm', min_box_area=100, max_aspect=1.6,
                 keep_predicted_frames=8, attach_min_iou=0.30, attach_stable_min_iou=0.15,
                 attach_stable_frames=10, predicted_score_decay=0.9),
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
    pose_gate_norm_dist=2.0,    # chỉ chạy pose + hand cho người có bbox cách hiện vật <= 2.0 x chiều cao người (None = mọi người).
                                # State machine chỉ bật NEAR khi cổ tay cách hiện vật <= state_near_thr (0.9) -> người xa hơn không thể tạo event.
    artifact_person_filter=True,  # segmentation: chạy detector trên frame segment để loại vùng là người
    artifact_person_min_score=0.45,  # ... chỉ dùng box người có score >= 0.45 (detector chạy với conf 0.01 như make)
    person_filter=dict(
        enabled=True,              # lọc track/box giả sau tracker (không ảnh hưởng đầu vào ByteTrack)
        display_min_score=0.5,     # điểm detector tối thiểu để hiển thị / chạy pose / tạo event
        predicted_min_score=0.05,  # ngưỡng điểm tối thiểu cho track đã xác nhận khi đang dự đoán (score suy giảm)
        min_hits=3,                # số frame hợp lệ tối thiểu trước khi qua lọc
        min_box_area=600,          # diện tích bbox tối thiểu (px^2 trên ảnh làm việc 960px)
        aspect_range=(0.2, 1.0),   # khoảng tỷ lệ w/h hợp lệ của bbox người
        min_valid_kpts=5,          # số keypoint hợp lệ tối thiểu (kc >= pose_conf_thr) khi đã có pose
        pose_min_valid_frac=0.5,   # tỷ lệ frame pose gần nhất đạt min_valid_kpts tối thiểu (>= 50%)
        pose_window=10,            # số frame pose gần nhất dùng để kiểm tra tỷ lệ keypoint
    ),
    seg_ref_default='auto',     # video không có trong SEG_REF_FRAMES: 'auto' = tự tách cảnh (cắt cảnh / đổi camera),
                                # segment 1 frame cho mỗi cảnh | 'first' = chỉ frame đầu cho cả video
    seg_multi_frame=dict(
        enabled=True,           # segment nhiều frame mỗi cảnh và hợp nhất mask theo IoU
        step_sec=2.0,           # khoảng cách giữa các frame lấy mẫu trong cảnh (giây)
        max_frames_per_scene=4, # tối đa số frame lấy mẫu mỗi cảnh
    ),
    seg_merge_iou=0.5,          # ngưỡng IoU để hợp nhất mask cùng hiện vật giữa các frame trong cảnh
    seg_merge_min_votes=1,      # giữ mask xuất hiện >= N frame (mặc định 1 = union mọi frame mẫu)
    seg_merge_keep_ref=True,    # luôn giữ nguyên mọi mask ở frame tham chiếu đầu cảnh (superset bản 1 frame)
    seg_merge_min_votes_extra=1,  # số phiếu tối thiểu cho mask chỉ xuất hiện ở frame phụ (không thuộc frame tham chiếu)
    scene_cut_diff=25.0,        # 'auto': chênh lệch xám trung bình giữa 2 frame liền nhau (ảnh 64x36) >= ngưỡng -> cắt cảnh
                                # (frame thường ~0-2, cắt cảnh ~30-50)
    scene_color_cut=True,       # 'auto': bật tiêu chí cắt cảnh theo histogram màu (bắt đổi xám->màu / đổi sáng màu)
    scene_color_hist_thr=0.35,  # 'auto': ngưỡng khoảng cách L1/2 histogram màu (H, S) [0..1] để cắt cảnh
    scene_step_sec=0.0,         # 'auto': so sánh mỗi N giây (0 = mọi frame, như benchmark). Dashboard đặt 0.2 để mở video nhanh hơn
    scene_min_sec=1.0,          # 'auto': cảnh ngắn hơn (vd vài frame chuyển cảnh) -> gộp vào cảnh sau
    scene_ref_offset_sec=0.5,   # 'auto': frame segment = đầu cảnh + 0.5s (tránh frame mờ lúc chuyển cảnh)
    artifact_cache=True,        # dùng lại mask hiện vật đã tính (cùng video + cùng segmenter)
    ring_seconds=20.0,          # ring buffer ảnh gốc để cắt ảnh cho VLM
    store_long=1280,            # cạnh dài ảnh lưu trong ring buffer (giới hạn RAM)
    vlm_early_sec=None,         # vd 2.0: gửi VLM ngay khi chạm liên tục 2s (báo động sớm), không đợi event kết thúc
    periodic_vlm=dict(
        enabled=True,           # đường dự phòng: ở vùng NEAR liên tục >= min_near_sec mà chưa có event -> vẫn gửi VLM
        min_near_sec=1.0,       # thời gian tối thiểu ở vùng NEAR liên tục (giây)
        cooldown_sec=3.0,       # khoảng cách tối thiểu giữa 2 lần gửi dự phòng cho cùng track
        max_near_gap_frames=5,  # cho phép đứt quãng ngắn (pose/hand hụt vài frame) mà không reset bộ đếm NEAR
        flush_on_end=True,      # gửi VLM cuối khi track mất > max_gap hoặc hết video nếu đã NEAR >= flush_min_sec
        flush_min_sec=1.0,      # thời gian NEAR liên tục tối thiểu (giây) để flush cuối track / cuối video
        flush_max_per_track=2,  # số lần flush tối đa cho mỗi track
    ),
    missing_detect=dict(
        enabled=True,               # đường dự phòng bắt stealing: phát hiện hiện vật biến mất khỏi vùng mask / người mang vật đi
        check_sec=1.0,              # chu kỳ kiểm tra so với frame tham chiếu của cảnh (giây)
        sustain_sec=1.5,            # thời gian thay đổi > change_thr liên tục để tạo event suspect_missing (giây)
        max_side=160,               # thu nhỏ ảnh về cạnh dài tối đa 160px để chạy CPU nhẹ
        change_thr=0.35,            # ngưỡng chênh lệch kết hợp (cạnh + màu) trên vùng mask hiện vật [0..1]
        edge_weight=0.5,            # trọng số chênh lệch cạnh (Sobel) vs chênh lệch màu BGR
        edge_diff_scale=64.0,       # hệ số chuẩn hoá chênh lệch độ lớn gradient cạnh về [0..1]
        color_diff_scale=48.0,      # hệ số chuẩn hoá chênh lệch màu BGR về [0..1]
        max_person_occlude=0.25,    # bỏ qua kiểm tra nếu tỷ lệ mask bị bbox người che > ngưỡng
        person_range_norm=2.0,      # phải có người trong vòng <= person_range_norm x chiều cao người
        global_luma_diff=25.0,      # bỏ qua khi đổi độ sáng toàn cục >= ngưỡng (chống FP đổi ánh sáng / scene cut)
        global_hist_thr=0.35,       # bỏ qua khi histogram màu HSV toàn cục đổi >= ngưỡng
        cooldown_sec=4.0,           # khoảng cách tối thiểu giữa 2 event suspect_missing cho cùng hiện vật / track
        carry_detect=True,          # tín hiệu phụ: người tiếp cận rồi mang vật đi ra (khi mask bị che hoặc thiếu mask)
        carry_change_thr=0.42,      # ngưỡng thay đổi vùng mang vật so với lúc mới xuất hiện
        carry_sustain_sec=1.5,      # thời gian duy trì tín hiệu mang vật khi rời đi (giây)
    ),
    limit_cpu_threads=False,    # mỗi stage là 1 thread Python: torch / OpenCV chỉ dùng 1 thread nội bộ (Kaggle chỉ có 4 vCPU,
                                # để mặc định thì 7 stage x 4 thread tranh nhau CPU -> detect / track chậm hẳn)
    render_live=False,           # sink ghi video live (overlay + HUD FPS/latency)
    alarm_show_sec=4.0,         # video live: thời gian giữ khung đỏ sau khi VLM báo (khi alarm_persist=False)
    alarm_persist=True,         # video live: giữ khung đỏ suốt video cho người đã bị gắn cờ ANOMALY (False = dùng alarm_show_sec)
    flag_inherit=dict(
        enabled=True,              # kế thừa cờ ANOMALY qua đổi ID bằng ReID + vị trí
        reid_cos=0.7,              # ngưỡng cosine similarity tối thiểu (OSNet ReID hoặc fallback HSV)
        max_gap_sec=3.0,           # chỉ kế thừa khi track cũ đã mất <= max_gap_sec giây
        max_center_dist_norm=2.0,  # khoảng cách tâm bbox tối đa chuẩn hoá theo chiều cao người
    ),
    stream_max_sec=1800.0,      # camera (RTSP / webcam): thời lượng tối đa mỗi lần chạy (giây); bấm Dừng để kết thúc sớm hơn
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
    pose_crop_pad_frac=0.30, pose_predicted_expand=True, pose_min_crop_px=16, pose_conf_thr=0.30, pose_iou_match_thr=0.30, pose_margin_thr=0.15,
    hand_conf_thr=0.30, hand_min_bbox_diag_px=18, hand_small_use_center=True, hand_small_min_conf=0.5, hand_small_max_reach=0.6,
    pose_wide_retry=True, pose_wide_edge_frac=0.08, pose_wide_pad_x=0.7, pose_wide_pad_down=0.6,
    artifact_max_person_overlap=0.6, artifact_min_person_iou=0.4, artifact_max_motion=0.35, artifact_motion_samples=8,
    pairing_max_norm_dist=3.0,
    state_near_thr=0.9, state_contact_thr=0.35, state_hold_seconds=0.25, state_history_seconds=2.0,
    state_lost_grace_frames=12,
    state_hysteresis=True, state_hysteresis_mult=1.3,
    pose_smooth=True, pose_smooth_min_cutoff=1.5, pose_smooth_beta=0.02, pose_smooth_d_cutoff=1.0,
    pose_wrist_interp_max_gap=8,
    state_min_person_h_px=80, state_reject_rest_arm=True, rest_arm_max_dx=0.10, rest_arm_min_dy=-0.12,
    self_touch_head_frac=0.10, state_ground_max_gap=0.30,
    foot_from_pose=True, foot_extend_min_frac=0.15, foot_extend_max_frac=0.8,
    event_merge_gap_seconds=0.5, event_min_frames=12, max_tasks_per_video=40,
    ambiguous_policy='nearest', # 'nearest': chọn mask gần nhất khi 2 mask sát nhau | 'reject': loại (AMBIGUOUS)
    ambiguous_diff_thr=0.15,    # chênh lệch khoảng cách chuẩn hoá tối đa để coi 2 mask là nhập nhằng
    relaxed_gate=True,          # nới cổng chạm có kiểm soát (event ngắn / người nhỏ / bệ pedestal)
    relaxed_event_min_frames=8, relaxed_min_person_h_px=60,
    relaxed_hold_seconds=0.15, relaxed_pedestal_contact_thr=0.45,
)
SEGMENTATION_CFG = dict(
    sculpture_phrases=['statue', 'sculpture', 'bust'],
    painting_phrases=['painting', 'framed artwork', 'picture frame', 'artwork on wall', 'wall art', 'mounted painting'],
    pedestal_phrases=['pedestal'],
    full_thr=0.30, text_thr=0.25, mask_overlap=0.80, pedestal_merge=True,
    pedestal_point_frac=0.25, pedestal_max_area_ratio=4.0, pedestal_max_person_frac=0.30, min_mask_fill=0.15,
    merge_standalone_pedestal=True, pedestal_cover_merge=0.5,
    fallback_when_no_3d=True,
    fallback_object_phrases=['stone', 'rock', 'artifact', 'small object', 'vase', 'display case', 'glass cabinet', 'shelf'],
    fallback_pedestal_phrases=['pedestal', 'plinth', 'display stand', 'white box'],
    fallback_thr=0.20, fallback_empty_thr=0.15, fallback_lower_when_empty=True,
    fallback_text_thr=0.20, fallback_max_area_frac=0.20, fallback_max_keep=4,
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

EXPORT = dict(pre_sec=1.0, post_sec=3.0, max_window_sec=10.0, n_sample_frames=12, crop_pad=0.4, crop_out_long=768,
              before_ref=True)
MASK_REGIONS = [[0.855, 0.925, 0.995, 0.990]]   # xoá watermark (toạ độ tương đối); [] nếu không cần
VLM_IMAGE_SET = 'marked'                         # 'marked' | 'clean'

BEFORE_NOTE = ('Image 1 is a reference taken BEFORE (exhibit intact). '
               'If the exhibit visible in Image 1 is missing, moved or carried away in the later frames, answer stealing.\n')


class _PromptTemplate(str):
    """Chuỗi prompt hỗ trợ tuỳ chọn {before_note} mặc định rỗng khi caller cũ chỉ truyền (n, dt, tid)."""

    def format(self, *args, **kwargs):
        kwargs.setdefault('before_note', '')
        return super().format(*args, **kwargs)


PROMPT_TEMPLATE = _PromptTemplate('''You are a museum security analyst. The {n} images are consecutive frames \
(about {dt:.2f}s apart, in order) cropped around ONE visitor marked with a RED box labeled "ID {tid}". \
BLUE outline(s) mark exhibit(s).
{before_note}Classify ONLY the red-box person's behavior toward the exhibit:
- "normal": looking, walking, pointing, standing close without contact
- "touching": hand makes contact with the exhibit, exhibit stays in place
- "holding": person lifts / holds / moves the exhibit
- "stealing": person takes the exhibit away, hides it (bag/clothes), or the exhibit is missing afterwards
- "uncertain": view is occluded or too small to decide
Answer JSON only:
{{"label": "...", "confidence": 0.0, "contact_frames": [], "evidence": "<one sentence>"}}
"contact_frames" = the Frame numbers (1..{n}) in which the red-box person's hand is VISIBLY touching/holding the exhibit (empty list if never). Do not include frames where the hand is only approaching or near.''')
SYSTEM_PROMPT = ('You are a careful CCTV analyst for a museum. Judge only what is visible in the frames. '
                 'Touching means the hand visibly reaches the exhibit or its pedestal; being near is not touching. '
                 'Always answer with a single JSON object and nothing else.')

DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT = PROMPT_TEMPLATE, SYSTEM_PROMPT   # bản gốc (dashboard ghi đè rồi khôi phục được)
PROMPT_FIELDS = ('n', 'dt', 'tid', 'before_note')                                   # biến được {format}; dấu { } khác phải viết {{ }}

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

