"""Registry một nguồn cho các tham số tinh chỉnh (TASK-001..009).

Mọi cờ/ngưỡng tinh chỉnh được khai báo tại đây để cả dashboard cũ, web mới
và lớp backend (`runner.py`, `server.py`, `web_server.py`) dùng chung một schema.
Giá trị mặc định được đọc trực tiếp từ `exhibitflow.settings` lúc import.
"""
import copy
import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from . import settings as _settings


@dataclass(frozen=True)
class Param:
    key: str
    path: str
    type: str
    group: str
    label_vi: str
    help_vi: str
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    needs_reload: bool = False
    default: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_by_path(settings_mod: Any, path: str) -> Any:
    """Đọc giá trị từ `settings` theo đường dẫn dạng 'RT.person_filter.min_hits'."""
    parts = path.split('.')
    cur = getattr(settings_mod, parts[0])
    for part in parts[1:]:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f'Không tìm thấy đường dẫn cấu hình: {path}')
        cur = cur[part]
    return copy.deepcopy(cur)


GROUPS: list[dict[str, Any]] = [
    dict(
        id='alarm_id',
        title='Khung đỏ & ID',
        description='Giữ khung đỏ sau cảnh báo và kế thừa cờ ANOMALY khi tracker đổi ID.',
        needs_reload=False,
    ),
    dict(
        id='tracker',
        title='Tracker & Tiền xử lý (nạp lại model)',
        description='Cấu hình ByteTrack-DMA-LTC và tăng sáng vùng tối. Đổi nhóm này sẽ nạp lại model.',
        needs_reload=True,
    ),
    dict(
        id='person_filter',
        title='Lọc người (detector)',
        description='Lọc box/track giả sau detector conf=0.01 mà không làm giảm recall của tracker.',
        needs_reload=False,
    ),
    dict(
        id='pose',
        title='Pose & Làm mượt',
        description='Làm mượt keypoint theo thời gian, nội suy cổ tay và trễ chuyển trạng thái.',
        needs_reload=False,
    ),
    dict(
        id='contact_gate',
        title='Cổng chạm & Sự kiện',
        description='Ngưỡng khoảng cách/thời lượng để tạo sự kiện chạm gửi VLM.',
        needs_reload=False,
    ),
    dict(
        id='periodic_vlm',
        title='VLM dự phòng (NEAR kéo dài)',
        description='Đường dự phòng gửi VLM khi người đứng gần hiện vật lâu mà chưa kích hoạt chạm.',
        needs_reload=False,
    ),
    dict(
        id='missing_detect',
        title='Phát hiện biến mất (stealing)',
        description='Phát hiện hiện vật biến mất khỏi vùng mask hoặc người mang vật đi ra.',
        needs_reload=False,
    ),
    dict(
        id='segmentation',
        title='Segmentation & Tách cảnh',
        description='Segment đa frame theo cảnh, cắt cảnh đổi màu và ngưỡng Grounding-DINO.',
        needs_reload=False,
    ),
]


def _p(
    key: str,
    path: str,
    typ: str,
    group: str,
    label_vi: str,
    help_vi: str,
    *,
    min: Optional[float] = None,
    max: Optional[float] = None,
    step: Optional[float] = None,
    needs_reload: bool = False,
) -> Param:
    return Param(
        key=key,
        path=path,
        type=typ,
        group=group,
        label_vi=label_vi,
        help_vi=help_vi,
        min=min,
        max=max,
        step=step,
        needs_reload=needs_reload,
        default=get_by_path(_settings, path),
    )


PARAMS: list[Param] = [
    # ---- Khung đỏ & ID ----
    _p(
        'alarm_persist', 'RT.alarm_persist', 'bool', 'alarm_id',
        'Giữ khung đỏ suốt video sau cảnh báo',
        'Bật: người đã bị VLM báo ANOMALY giữ khung đỏ tới hết video. Tắt: khung đỏ quay về xanh sau alarm_show_sec.',
    ),
    _p(
        'flag_inherit_enabled', 'RT.flag_inherit.enabled', 'bool', 'alarm_id',
        'Kế thừa cờ đỏ khi đổi ID',
        'Bật: track mới gần vị trí/ReID giống người bị gắn cờ sẽ kế thừa khung đỏ. Tắt: đổi ID sẽ mất cờ đỏ.',
    ),
    _p(
        'flag_inherit_reid_cos', 'RT.flag_inherit.reid_cos', 'float', 'alarm_id',
        'Ngưỡng tương đồng ReID kế thừa cờ',
        'Tăng → chặt hơn, ít gộp nhầm người khác nhưng dễ mất cờ khi đổi góc nhìn. Giảm → dễ kế thừa hơn nhưng dễ nhầm người mặc giống.',
        min=0.0, max=1.0, step=0.05,
    ),
    _p(
        'flag_inherit_max_gap_sec', 'RT.flag_inherit.max_gap_sec', 'float', 'alarm_id',
        'Khoảng mất track tối đa kế thừa cờ (giây)',
        'Tăng → nối lại cờ sau khi mất dấu lâu hơn, nhưng tăng nguy cơ nối nhầm người mới đi qua. Giảm → chỉ nối khi mất rất ngắn.',
        min=0.0, max=30.0, step=0.5,
    ),
    _p(
        'flag_inherit_max_center_dist_norm', 'RT.flag_inherit.max_center_dist_norm', 'float', 'alarm_id',
        'Khoảng cách tâm tối đa kế thừa (× chiều cao)',
        'Tăng → cho phép kế thừa khi người dịch chuyển xa hơn lúc đứt track, nhưng dễ nhảy sang người bên cạnh. Giảm → chỉ nối rất gần.',
        min=0.1, max=10.0, step=0.1,
    ),

    # ---- Tracker & Tiền xử lý (nạp lại model) ----
    _p(
        'tracker_track_thresh', 'MODELS.tracker.track_thresh', 'float', 'tracker',
        'Ngưỡng tin cậy chính tracker (track_thresh)',
        'Tăng → ít track nhiễu hơn nhưng dễ mất người mờ/tối. Giảm → tăng recall cho người điểm thấp nhưng dễ tạo track giả.',
        min=0.05, max=0.95, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_track_buffer', 'MODELS.tracker.track_buffer', 'int', 'tracker',
        'Bộ đệm giữ track khi mất (frame)',
        'Tăng → giữ ID lâu hơn qua che khuất/hụt detect (giảm ID switch), nhưng có thể giữ nhầm ID cũ khi đông người. Giảm → giải phóng ID sớm.',
        min=5, max=300, step=5, needs_reload=True,
    ),
    _p(
        'tracker_match_thresh', 'MODELS.tracker.match_thresh', 'float', 'tracker',
        'Ngưỡng ghép cặp tracker (match_thresh)',
        'Tăng → nới lỏng điều kiện ghép box (dễ giữ track khi di chuyển nhanh, nhưng dễ trộn người gần nhau). Giảm → ghép chặt hơn.',
        min=0.1, max=0.99, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_reid_weight', 'MODELS.tracker.reid_weight', 'float', 'tracker',
        'Trọng số ReID trong ghép track',
        'Tăng → ưu tiên ngoại hình để giảm ID switch khi giao nhau, nhưng kém ổn định nếu ánh sáng đổi mạnh. Giảm → dựa nhiều vào vị trí/IoU.',
        min=0.0, max=1.0, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_reid_thresh', 'MODELS.tracker.reid_thresh', 'float', 'tracker',
        'Ngưỡng khớp đặc trưng ReID',
        'Tăng/giảm độ khắt khe khi đối sánh vector ReID (chặt quá dễ tạo ID mới, lỏng quá dễ ghép nhầm hai người).',
        min=0.0, max=1.0, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_ltc_min_history', 'MODELS.tracker.ltc_min_history', 'int', 'tracker',
        'Số frame lịch sử tối thiểu cho LTC',
        'Giảm → dự đoán quỹ đạo LTC bật sớm hơn cho track ngắn, nhưng quỹ đạo đầu kém ổn định. Tăng → chỉ bật với track đủ dài.',
        min=1, max=60, step=1, needs_reload=True,
    ),
    _p(
        'tracker_keep_predicted_frames', 'MODELS.tracker.keep_predicted_frames', 'int', 'tracker',
        'Số frame giữ track dự đoán khi hụt detect',
        'Tăng → giữ track và chạy pose qua khoảng hụt detect dài hơn (cứu cảnh mờ/tối), nhưng quá dài dễ lệch box tạo chạm giả. 0 = tắt.',
        min=0, max=60, step=1, needs_reload=True,
    ),
    _p(
        'tracker_attach_min_iou', 'MODELS.tracker.attach_min_iou', 'float', 'tracker',
        'IoU tối thiểu gắn điểm detector (track mới)',
        'Giảm → dễ gắn điểm detector khi box hơi lệch (tăng recall), nhưng dễ bám nhầm box người khác. Tăng → gắn chặt hơn.',
        min=0.01, max=0.95, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_attach_stable_min_iou', 'MODELS.tracker.attach_stable_min_iou', 'float', 'tracker',
        'IoU tối thiểu gắn điểm (track ổn định)',
        'Áp dụng cho track đã ổn định: giảm → ít rớt điểm khi người cúi/với nhanh; tăng → tránh bám nhầm detection lân cận.',
        min=0.01, max=0.95, step=0.05, needs_reload=True,
    ),
    _p(
        'tracker_attach_stable_frames', 'MODELS.tracker.attach_stable_frames', 'int', 'tracker',
        'Số frame để coi track là ổn định',
        'Tăng → track phải sống lâu hơn mới nới IoU gắn điểm (an toàn hơn, nhưng track ngắn ít được nới). Giảm → nới IoU sớm hơn.',
        min=1, max=100, step=1, needs_reload=True,
    ),
    _p(
        'tracker_predicted_score_decay', 'MODELS.tracker.predicted_score_decay', 'float', 'tracker',
        'Hệ số suy giảm điểm mỗi frame dự đoán',
        'Tăng (gần 1.0) → giữ điểm track lâu hơn qua frame dự đoán; giảm → điểm tụt nhanh hơn để sớm loại track mất dấu thật.',
        min=0.1, max=1.0, step=0.05, needs_reload=True,
    ),
    _p(
        'low_light_boost_enabled', 'MODELS.detector.low_light_boost.enabled', 'bool', 'tracker',
        'Tăng sáng tự động vùng tối (CLAHE)',
        'Bật: tăng tương phản kênh L ở ảnh tối trước khi detect để bắt người đồ tối. Tắt: giữ ảnh gốc (nhanh hơn chút, dễ sót cảnh tối).',
        needs_reload=True,
    ),
    _p(
        'low_light_boost_luma_thr', 'MODELS.detector.low_light_boost.luma_thr', 'float', 'tracker',
        'Ngưỡng độ sáng kích hoạt CLAHE',
        'Tăng → bật tăng sáng cho cả cảnh sáng vừa (dễ tăng nhiễu). Giảm → chỉ tăng sáng ở cảnh rất tối.',
        min=0.0, max=255.0, step=5.0, needs_reload=True,
    ),
    _p(
        'low_light_boost_clip_limit', 'MODELS.detector.low_light_boost.clip_limit', 'float', 'tracker',
        'Cường độ tăng tương phản (clip_limit)',
        'Tăng → đẩy sáng mạnh hơn ở vùng tối nhưng dễ khuếch đại nhiễu hạt. Giảm → tăng sáng nhẹ, tự nhiên hơn.',
        min=0.5, max=10.0, step=0.5, needs_reload=True,
    ),

    # ---- Lọc người (detector) ----
    _p(
        'person_filter_enabled', 'RT.person_filter.enabled', 'bool', 'person_filter',
        'Bật lọc track người giả',
        'Bật: loại box/track giả từ detector conf thấp (không vẽ, không chạy pose/tạo event). Tắt: giữ mọi box từ tracker.',
    ),
    _p(
        'person_filter_display_min_score', 'RT.person_filter.display_min_score', 'float', 'person_filter',
        'Điểm detector tối thiểu qua lọc',
        'Tăng → ít box giả hơn nhưng dễ bỏ sót người mờ hoặc mặc đồ tối. Giảm → giữ người điểm thấp nhưng dễ lọt vật thể giống người.',
        min=0.0, max=1.0, step=0.05,
    ),
    _p(
        'person_filter_min_hits', 'RT.person_filter.min_hits', 'int', 'person_filter',
        'Số frame hợp lệ tối thiểu (min_hits)',
        'Tăng → loại nhiễu thoáng qua tốt hơn nhưng chậm bắt người mới vào khung. Giảm → nhạy hơn nhưng dễ lọt nhiễu 1–2 frame.',
        min=1, max=30, step=1,
    ),
    _p(
        'person_filter_min_box_area', 'RT.person_filter.min_box_area', 'int', 'person_filter',
        'Diện tích bbox tối thiểu (px² @960px)',
        'Tăng → ít báo giả từ mảnh nhiễu nhỏ nhưng dễ bỏ sót người đứng xa/nhỏ. Giảm → giữ được người nhỏ nhưng dễ dính nhiễu.',
        min=0, max=50000, step=50,
    ),
    _p(
        'person_filter_min_valid_kpts', 'RT.person_filter.min_valid_kpts', 'int', 'person_filter',
        'Số keypoint pose hợp lệ tối thiểu',
        'Tăng → loại tượng/vật thể nhận nhầm hiệu quả hơn, nhưng dễ lọc nhầm người bị che khuất nhiều. Giảm → nới cho người bị che.',
        min=0, max=17, step=1,
    ),
    _p(
        'person_filter_pose_min_valid_frac', 'RT.person_filter.pose_min_valid_frac', 'float', 'person_filter',
        'Tỷ lệ frame đạt đủ keypoint tối thiểu',
        'Tăng → đòi hỏi pose ổn định hơn mới giữ track (giảm FP), nhưng dễ loại người cử động nhanh hoặc bị che chập chờn.',
        min=0.0, max=1.0, step=0.05,
    ),
    _p(
        'artifact_person_min_score', 'RT.artifact_person_min_score', 'float', 'person_filter',
        'Ngưỡng loại người khỏi mask hiện vật',
        'Tăng → ít loại nhầm tượng thành người lúc segment, nhưng người đứng chắn dễ bị giữ thành mask. Giảm → lọc người gắt hơn khi segment.',
        min=0.05, max=1.0, step=0.05,
    ),

    # ---- Pose & Làm mượt ----
    _p(
        'pose_smooth', 'CFG.pose_smooth', 'bool', 'pose',
        'Làm mượt keypoint pose (One-Euro)',
        'Bật: giảm rung keypoint để chuỗi chạm không bị đứt quãng. Tắt: dùng keypoint thô (không trễ nhưng dễ giật).',
    ),
    _p(
        'pose_smooth_min_cutoff', 'CFG.pose_smooth_min_cutoff', 'float', 'pose',
        'Tần số cắt tối thiểu (min_cutoff)',
        'Giảm → mượt hơn lúc đứng yên nhưng tăng độ trễ khi tay bắt đầu vươn. Tăng → bám sát frame thô hơn nhưng ít mượt.',
        min=0.1, max=10.0, step=0.1,
    ),
    _p(
        'pose_smooth_beta', 'CFG.pose_smooth_beta', 'float', 'pose',
        'Hệ số thích nghi tốc độ (beta)',
        'Tăng → giảm trễ khi tay vung nhanh nhưng kém mượt lúc chuyển động nhanh. Giảm → ưu tiên độ mượt.',
        min=0.0, max=1.0, step=0.005,
    ),
    _p(
        'pose_wrist_interp_max_gap', 'CFG.pose_wrist_interp_max_gap', 'int', 'pose',
        'Số frame tối đa nội suy cổ tay bị mất',
        'Tăng → nối khoảng mất cổ tay dài hơn để giữ chuỗi chạm, nhưng khoảng dài dễ đoán lệch vị trí tay. 0 = tắt nội suy.',
        min=0, max=30, step=1,
    ),
    _p(
        'pose_conf_thr', 'CFG.pose_conf_thr', 'float', 'pose',
        'Ngưỡng tin cậy keypoint pose',
        'Tăng → chỉ giữ khớp rõ nét (ít sai số), nhưng dễ mất cổ tay khi mờ/ngược sáng. Giảm → giữ khớp yếu nhưng dễ nhiễu.',
        min=0.05, max=0.95, step=0.05,
    ),
    _p(
        'hand_conf_thr', 'CFG.hand_conf_thr', 'float', 'pose',
        'Ngưỡng tin cậy keypoint bàn tay (RTMW)',
        'Tăng → ít điểm bàn tay nhiễu nhưng dễ mất tay nhỏ/xa. Giảm → dùng cả điểm tay tin cậy thấp (tăng recall, dễ lệch).',
        min=0.05, max=0.95, step=0.05,
    ),
    _p(
        'state_lost_grace_frames', 'CFG.state_lost_grace_frames', 'int', 'pose',
        'Số frame chờ khi mất tín hiệu chạm',
        'Tăng → event không bị xé nhỏ khi hụt tay tạm thời, nhưng có thể kéo dài event sau khi người đã rút tay.',
        min=0, max=60, step=1,
    ),
    _p(
        'state_hysteresis', 'CFG.state_hysteresis', 'bool', 'pose',
        'Bật trễ chuyển trạng thái (hysteresis)',
        'Bật: ngưỡng thoát NEAR/CONTACT nới rộng hơn ngưỡng vào để tránh nhấp nháy ở biên. Tắt: vào và ra cùng một ngưỡng.',
    ),
    _p(
        'state_hysteresis_mult', 'CFG.state_hysteresis_mult', 'float', 'pose',
        'Hệ số nhân ngưỡng thoát (hysteresis)',
        'Tăng → giữ NEAR/CONTACT ổn định hơn khi tay rung quanh biên, nhưng thoát trạng thái chậm hơn. 1.0 = không trễ.',
        min=1.0, max=3.0, step=0.05,
    ),

    # ---- Cổng chạm & Sự kiện ----
    _p(
        'relaxed_gate', 'CFG.relaxed_gate', 'bool', 'contact_gate',
        'Nới cổng chạm có kiểm soát',
        'Bật: dùng ngưỡng nới cho chạm ngắn, người nhỏ và chạm lên bệ (ưu tiên recall). Tắt: dùng ngưỡng chặt gốc.',
    ),
    _p(
        'relaxed_event_min_frames', 'CFG.relaxed_event_min_frames', 'int', 'contact_gate',
        'Số frame tối thiểu 1 event (chế độ nới)',
        'Giảm → bắt được chạm rất ngắn nhưng dễ tạo event nhiễu (tốn VLM). Tăng → chỉ lấy chạm rõ, dễ sót chạm lướt.',
        min=1, max=60, step=1,
    ),
    _p(
        'relaxed_min_person_h_px', 'CFG.relaxed_min_person_h_px', 'int', 'contact_gate',
        'Chiều cao người tối thiểu (px, chế độ nới)',
        'Giảm → bắt được người nhỏ/đứng xa nhưng pose kém ổn định. Tăng → ít báo giả hơn nhưng dễ bỏ sót người nhỏ.',
        min=10, max=400, step=5,
    ),
    _p(
        'relaxed_hold_seconds', 'CFG.relaxed_hold_seconds', 'float', 'contact_gate',
        'Thời gian giữ tối thiểu vào chạm (s, chế độ nới)',
        'Giảm → kích hoạt chạm nhanh hơn với động tác lướt, nhưng dễ dính khung rung tay. Tăng → đòi hỏi đưa tay giữ lâu hơn.',
        min=0.0, max=3.0, step=0.05,
    ),
    _p(
        'relaxed_pedestal_contact_thr', 'CFG.relaxed_pedestal_contact_thr', 'float', 'contact_gate',
        'Ngưỡng chạm bệ pedestal (chế độ nới)',
        'Tăng → bắt chạm lên bệ trưng bày nhạy hơn (recall cao), nhưng dễ báo chạm giả khi tay chỉ lướt gần bệ.',
        min=0.05, max=1.5, step=0.05,
    ),
    _p(
        'state_near_thr', 'CFG.state_near_thr', 'float', 'contact_gate',
        'Ngưỡng khoảng cách vùng NEAR (× chiều cao)',
        'Tăng → coi người đứng xa hơn đã ở vùng NEAR (mở rộng cơ hội bắt chạm & VLM dự phòng), nhưng tăng số lượt gửi VLM.',
        min=0.1, max=3.0, step=0.05,
    ),
    _p(
        'state_contact_thr', 'CFG.state_contact_thr', 'float', 'contact_gate',
        'Ngưỡng khoảng cách chạm CONTACT (× chiều cao)',
        'Tăng → tay cách hiện vật xa hơn vẫn tính là chạm (giảm sót nhưng tăng báo giả/tải VLM). Giảm → tay phải sát mask hơn.',
        min=0.05, max=1.5, step=0.05,
    ),
    _p(
        'state_min_person_h_px', 'CFG.state_min_person_h_px', 'int', 'contact_gate',
        'Chiều cao người tối thiểu (px, chế độ chuẩn)',
        'Dùng khi tắt relaxed_gate. Tăng → ít báo giả hơn nhưng dễ bỏ sót người nhỏ. Giảm → bắt thêm người đứng xa.',
        min=10, max=400, step=5,
    ),
    _p(
        'event_min_frames', 'CFG.event_min_frames', 'int', 'contact_gate',
        'Số frame tối thiểu 1 event (chế độ chuẩn)',
        'Dùng khi tắt relaxed_gate. Tăng → lọc event thoáng qua tốt hơn nhưng dễ sót chạm ngắn. Giảm → nhạy hơn.',
        min=1, max=60, step=1,
    ),
    _p(
        'ambiguous_diff_thr', 'CFG.ambiguous_diff_thr', 'float', 'contact_gate',
        'Ngưỡng chênh khoảng cách coi là nhập nhằng',
        'Khoảng cách chênh lệch tối đa giữa 2 mask gần nhất để coi là nhập nhằng (ambiguous). Khi ambiguous_policy=nearest sẽ chọn mask gần nhất.',
        min=0.0, max=1.0, step=0.05,
    ),

    # ---- VLM dự phòng ----
    _p(
        'periodic_vlm_enabled', 'RT.periodic_vlm.enabled', 'bool', 'periodic_vlm',
        'Bật gửi VLM dự phòng khi ở NEAR lâu',
        'Bật: đứng gần hiện vật liên tục chưa có event vẫn gửi VLM kiểm tra (cứu cảnh che tay). Tắt: chỉ gửi khi qua cổng chạm.',
    ),
    _p(
        'periodic_vlm_min_near_sec', 'RT.periodic_vlm.min_near_sec', 'float', 'periodic_vlm',
        'Thời gian ở vùng NEAR tối thiểu (giây)',
        'Giảm → gửi VLM dự phòng sớm hơn cho video ngắn (nhạy hơn nhưng tốn VLM). Tăng → chỉ kiểm tra khi người nán lại lâu.',
        min=0.5, max=30.0, step=0.5,
    ),
    _p(
        'periodic_vlm_cooldown_sec', 'RT.periodic_vlm.cooldown_sec', 'float', 'periodic_vlm',
        'Thời gian nghỉ giữa 2 lần gửi dự phòng (giây)',
        'Giảm → kiểm tra lại dày hơn khi người vẫn đứng gần (tốn VLM). Tăng → giảm tải VLM nhưng có thể lỡ khoảnh khắc giữa chừng.',
        min=0.5, max=60.0, step=0.5,
    ),
    _p(
        'periodic_vlm_flush_on_end', 'RT.periodic_vlm.flush_on_end', 'bool', 'periodic_vlm',
        'Gửi VLM khi mất track hoặc kết thúc video',
        'Bật: nếu người đã ở NEAR đủ lâu rồi mất dấu (> max_gap) hoặc hết video mà chưa có event/VLM, gửi 1 đoạn cuối. Tắt: bỏ qua.',
    ),
    _p(
        'periodic_vlm_flush_min_sec', 'RT.periodic_vlm.flush_min_sec', 'float', 'periodic_vlm',
        'Thời gian NEAR tối thiểu để flush cuối (giây)',
        'Giảm → dễ gửi đoạn cuối cho lượt tiếp cận ngắn (tốn VLM). Tăng → chỉ flush khi người đứng gần đủ lâu.',
        min=0.2, max=30.0, step=0.2,
    ),
    _p(
        'periodic_vlm_flush_max_per_track', 'RT.periodic_vlm.flush_max_per_track', 'int', 'periodic_vlm',
        'Số lần flush tối đa mỗi track',
        'Giới hạn số đoạn VLM gửi qua flush cho cùng một track_id để tránh trùng lặp khi track đứt/nối nhiều lần.',
        min=1, max=10, step=1,
    ),

    # ---- Phát hiện biến mất (stealing) ----
    _p(
        'missing_detect_enabled', 'RT.missing_detect.enabled', 'bool', 'missing_detect',
        'Bật phát hiện hiện vật biến mất',
        'Bật: so sánh vùng mask định kỳ để phát hiện trộm khi tay bị che. Tắt: chỉ dựa vào tín hiệu chạm cổ tay.',
    ),
    _p(
        'missing_detect_check_sec', 'RT.missing_detect.check_sec', 'float', 'missing_detect',
        'Chu kỳ kiểm tra biến mất (giây)',
        'Giảm → kiểm tra dày hơn, phát hiện sớm hơn nhưng tốn thêm CPU nhẹ. Tăng → kiểm tra thưa hơn.',
        min=0.1, max=10.0, step=0.1,
    ),
    _p(
        'missing_detect_sustain_sec', 'RT.missing_detect.sustain_sec', 'float', 'missing_detect',
        'Thời gian biến mất liên tục để báo (giây)',
        'Tăng → tránh báo giả do che khuất/chớp sáng ngắn, nhưng trễ lúc tạo event. Giảm → nhạy hơn nhưng dễ báo giả.',
        min=0.2, max=15.0, step=0.2,
    ),
    _p(
        'missing_detect_change_thr', 'RT.missing_detect.change_thr', 'float', 'missing_detect',
        'Ngưỡng chênh lệch vùng mask hiện vật',
        'Tăng → đòi hỏi hiện vật thay đổi rõ rệt mới nghi mất (ít FP hơn nhưng dễ sót vật nhỏ). Giảm → nhạy hơn với thay đổi nhỏ.',
        min=0.05, max=1.0, step=0.05,
    ),
    _p(
        'missing_detect_carry_detect', 'RT.missing_detect.carry_detect', 'bool', 'missing_detect',
        'Bật phát hiện người mang vật đi ra',
        'Bật: thêm tín hiệu khi người tiếp cận rồi mang vật rời đi (cứu cảnh thiếu mask). Tắt: chỉ kiểm tra vùng mask hiện vật.',
    ),
    _p(
        'missing_detect_carry_change_thr', 'RT.missing_detect.carry_change_thr', 'float', 'missing_detect',
        'Ngưỡng thay đổi vùng mang vật',
        'Tăng → ít báo nhầm người đi bộ bình thường, nhưng dễ sót vật nhỏ mang theo. Giảm → nhạy hơn nhưng dễ báo nhầm đổi tư thế.',
        min=0.05, max=1.0, step=0.05,
    ),
    _p(
        'missing_detect_carry_sustain_sec', 'RT.missing_detect.carry_sustain_sec', 'float', 'missing_detect',
        'Thời gian duy trì tín hiệu mang vật (giây)',
        'Tăng → cần mang vật rõ lâu hơn mới gửi VLM (giảm FP). Giảm → bắt được đoạn rời đi ngắn hơn.',
        min=0.2, max=15.0, step=0.2,
    ),
    _p(
        'missing_detect_max_person_occlude', 'RT.missing_detect.max_person_occlude', 'float', 'missing_detect',
        'Tỷ lệ bbox người che mask tối đa',
        'Giảm → bỏ qua kiểm tra sớm hơn khi người đứng chắn mask (ít FP nhưng dễ sót lúc đang lấy). Tăng → kiểm tra cả khi bị che một phần.',
        min=0.0, max=1.0, step=0.05,
    ),

    # ---- Segmentation & Tách cảnh ----
    _p(
        'seg_multi_frame_enabled', 'RT.seg_multi_frame.enabled', 'bool', 'segmentation',
        'Segment đa frame mỗi cảnh',
        'Bật: segment nhiều frame rồi hợp nhất (union) mask theo IoU để giữ đủ mask và bù hiện vật bị che. Tắt: chỉ segment 1 frame/cảnh (nhanh hơn).',
    ),
    _p(
        'seg_multi_frame_step_sec', 'RT.seg_multi_frame.step_sec', 'float', 'segmentation',
        'Khoảng cách giữa các frame mẫu (giây)',
        'Giảm → lấy mẫu dày hơn trong cảnh (ổn định hơn nhưng chuẩn bị chậm hơn). Tăng → lấy mẫu thưa hơn.',
        min=0.2, max=30.0, step=0.5,
    ),
    _p(
        'seg_multi_frame_max_frames_per_scene', 'RT.seg_multi_frame.max_frames_per_scene', 'int', 'segmentation',
        'Số frame mẫu tối đa mỗi cảnh',
        'Tăng → lấy mẫu nhiều hơn ở cảnh dài (tốn thời gian chuẩn bị). Giảm → chuẩn bị nhanh hơn.',
        min=1, max=20, step=1,
    ),
    _p(
        'seg_merge_iou', 'RT.seg_merge_iou', 'float', 'segmentation',
        'Ngưỡng IoU gộp mask trùng giữa các frame',
        'Giảm → dễ gộp hai mask hơi lệch thành một (ít mask trùng nhưng dễ gộp nhầm vật gần nhau). Tăng → tách mask chặt hơn.',
        min=0.1, max=0.95, step=0.05,
    ),
    _p(
        'seg_merge_min_votes', 'RT.seg_merge_min_votes', 'int', 'segmentation',
        'Số frame tối thiểu giữ mask (min_votes)',
        '1 = lấy hợp (union) mọi mask của các frame mẫu (khuyến nghị). Tăng lên 2+ (khi tắt giữ frame tham chiếu) để lọc mask thoáng qua.',
        min=1, max=20, step=1,
    ),
    _p(
        'seg_merge_keep_ref', 'RT.seg_merge_keep_ref', 'bool', 'segmentation',
        'Luôn giữ nguyên mask của frame tham chiếu',
        'Bật: mọi mask ở frame đầu cảnh luôn được giữ (tối thiểu bằng bản 1 frame), frame sau chỉ bổ sung mask mới. Tắt: lọc cả frame đầu theo min_votes.',
    ),
    _p(
        'seg_merge_min_votes_extra', 'RT.seg_merge_min_votes_extra', 'int', 'segmentation',
        'Số phiếu tối thiểu cho mask ở frame phụ',
        '1 = giữ mọi mask mới phát hiện thêm ở frame khác (bù mask bị che ở frame đầu). Tăng lên 2+ để chỉ bổ sung mask xuất hiện lặp lại.',
        min=1, max=20, step=1,
    ),
    _p(
        'scene_color_cut', 'RT.scene_color_cut', 'bool', 'segmentation',
        'Cắt cảnh theo histogram màu (xám → màu)',
        'Bật: tách cảnh mới khi video đổi tông màu/xám→màu dù độ sáng ít đổi. Tắt: chỉ cắt cảnh theo chênh lệch xám.',
    ),
    _p(
        'scene_color_hist_thr', 'RT.scene_color_hist_thr', 'float', 'segmentation',
        'Ngưỡng đổi histogram màu cắt cảnh',
        'Giảm → nhạy hơn với đổi màu/ánh sáng (dễ tách thêm cảnh giả). Tăng → chỉ cắt khi màu đổi rất mạnh.',
        min=0.05, max=1.0, step=0.05,
    ),
    _p(
        'scene_cut_diff', 'RT.scene_cut_diff', 'float', 'segmentation',
        'Ngưỡng chênh lệch độ sáng cắt cảnh',
        'Giảm → tách cảnh nhạy hơn (dễ cắt nhầm khi người lớn đi qua). Tăng → chỉ cắt khi chuyển cảnh rõ rệt.',
        min=1.0, max=100.0, step=1.0,
    ),
    _p(
        'seg_full_thr', 'SEGMENTATION_CFG.full_thr', 'float', 'segmentation',
        'Ngưỡng Grounding-DINO chính (full_thr)',
        'Tăng → ít mask hiện vật giả nhưng dễ bỏ sót hiện vật mờ/tranh trên tường. Giảm → bắt nhiều hiện vật hơn nhưng dễ dính nền.',
        min=0.05, max=0.95, step=0.05,
    ),
    _p(
        'seg_fallback_thr', 'SEGMENTATION_CFG.fallback_thr', 'float', 'segmentation',
        'Ngưỡng fallback hiện vật (fallback_thr)',
        'Tăng → ít mask dự phòng nhiễu hơn (kệ/tủ) nhưng dễ sót hiện vật lạ. Giảm → tăng recall cho kệ kính/đồ nhỏ.',
        min=0.05, max=0.95, step=0.05,
    ),
]

PARAMS_BY_KEY: dict[str, Param] = {p.key: p for p in PARAMS}
RELOAD_KEYS: tuple[str, ...] = tuple(p.key for p in PARAMS if p.needs_reload)


def _coerce_param(p: Param, raw_val: Any) -> Any:
    """Kiểm tra kiểu + khoảng [min, max] cho một tham số tinh chỉnh."""
    if p.type == 'bool':
        if not isinstance(raw_val, bool):
            raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) phải là kiểu boolean (true/false), nhận {raw_val!r}')
        return raw_val

    if isinstance(raw_val, bool):
        raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) phải là số ({p.type}), không nhận boolean {raw_val!r}')

    if p.type == 'int':
        if isinstance(raw_val, int):
            num = raw_val
        elif isinstance(raw_val, float) and math.isfinite(raw_val) and raw_val.is_integer():
            num = int(raw_val)
        else:
            raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) phải là số nguyên, nhận {raw_val!r}')
    elif p.type == 'float':
        if isinstance(raw_val, (int, float)) and math.isfinite(float(raw_val)):
            num = float(raw_val)
        else:
            raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) phải là số thực hữu hạn, nhận {raw_val!r}')
    else:
        raise ValueError(f'Kiểu tham số không hỗ trợ: {p.type}')

    if p.min is not None and num < p.min:
        raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) = {num} nhỏ hơn mức tối thiểu {p.min}')
    if p.max is not None and num > p.max:
        raise ValueError(f'Tham số "{p.key}" ({p.label_vi}) = {num} lớn hơn mức tối đa {p.max}')
    return num


def validate_tuning(raw: Any) -> dict[str, Any]:
    """Chuẩn hoá dict tuning từ UI/API: bỏ None/chuỗi rỗng (= mặc định), báo ValueError nếu sai kiểu/ngoài khoảng."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError('tuning phải là dict {key: value}')
    out: dict[str, Any] = {}
    for k, v in raw.items():
        p = PARAMS_BY_KEY.get(k)
        if p is None:
            continue
        if v is None or v == '':
            continue
        out[k] = _coerce_param(p, v)
    return out


def resolve_value(tuning_dict: Optional[dict[str, Any]], key: str) -> Any:
    """Lấy giá trị hiệu lực của một khoá tinh chỉnh (giá trị người dùng đặt hoặc mặc định settings.py)."""
    p = PARAMS_BY_KEY[key]
    if isinstance(tuning_dict, dict):
        v = tuning_dict.get(key)
        if v is not None and v != '':
            return _coerce_param(p, v)
    return copy.deepcopy(p.default)


def tuning_overrides(tuning_dict: Optional[dict[str, Any]] = None, *, include_defaults: bool = True) -> dict[str, Any]:
    """Chuyển dict tuning {key: value} thành dict override lồng ('rt'/'cfg'/'models'/'segmentation_cfg') cho apply_overrides."""
    clean = validate_tuning(tuning_dict)
    ov: dict[str, Any] = {}
    for p in PARAMS:
        if p.key in clean:
            val = clean[p.key]
        elif include_defaults:
            val = copy.deepcopy(p.default)
        else:
            continue
        parts = p.path.split('.')
        root = parts[0].lower()
        cur = ov.setdefault(root, {})
        for part in parts[1:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return ov


def schema() -> dict[str, Any]:
    """Schema JSON cho endpoint GET /api/tuning/schema."""
    return {
        'groups': copy.deepcopy(GROUPS),
        'params': [p.to_dict() for p in PARAMS],
    }
