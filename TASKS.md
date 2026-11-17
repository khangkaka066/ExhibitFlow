# TASKS — Tinh chỉnh pipeline ExhibitFlow (từ kết quả dashboard + GT)

**Ràng buộc chung cho mọi task (BẮT BUỘC):**
- Máy đang chạy pipeline khác → KHÔNG chạy video/GPU nặng. Mọi lệnh chạy bằng `nice -n 19`, đặt `CUDA_VISIBLE_DEVICES=""` (chỉ CPU) trừ khi task ghi khác, `OMP_NUM_THREADS=2`, tối đa 2 tiến trình song song.
- **CẤM TUYỆT ĐỐI động vào `external/ByteTrack-DMA-LTC-Motion-Tracker/`** (người dùng đang chạy ablation ở đó): không ghi, sửa, xoá, `git` (checkout/pull/stash/clean), không cài package vào `.venv` của nó, không chạy lệnh có cwd ở đó, không import repo đó khi chạy test (dùng mock). Luôn đặt `PYTHONDONTWRITEBYTECODE=1` để không sinh `__pycache__` ngoài dự án. Sau mỗi task chạy `git -C external/ByteTrack-DMA-LTC-Motion-Tracker status --porcelain` (nếu là repo riêng) hoặc `git status --porcelain -- external/` và đưa kết quả vào EVIDENCE (phải rỗng so với trước khi làm).
- Chỉ sửa trong `exhibitflow_pipeline/exhibitflow/` và `exhibitflow_pipeline/tests/`. Không động vào `results/`, `weights/`, `Colab/`.
- Mọi ngưỡng mới phải nằm trong `settings.py` (CFG/RT/MODELS) và có giá trị mặc định giữ hành vi cũ khi tắt (có cờ bật/tắt).
- Verification dùng test CPU/giả lập (`tests/`), không chạy Qwen/GDINO/SAM thật.

**Bảng đối chiếu GT ↔ pipeline (cơ sở của các task):**

| Video | GT | Pipeline | Chẩn đoán |
|---|---|---|---|
| 1,2,5 | touching | ĐÚNG (alert) | OK |
| 3 | touching f33-120 | alert nhưng 23 track, 2 ID (11,20) cùng hiện vật | ID switch / nhiều track |
| 4 | touching f26-86 | event chỉ f67-82, VLM=normal, không alert | event quá ngắn/muộn, chạm lên bệ; VLM bỏ sót |
| 6 | stealing f195-389 | 2 event NORMAL (f152-195, f363-386), 0 alert | VLM không thấy trộm; tranh trên tường mờ |
| 7 | stealing f590-795 | 0 event | mask hiện vật 4 cảnh, tranh bị gỡ ngoài khung; không có tín hiệu chạm |
| 8 | stealing f70-150 | 0 event, 19 mask | tay thò vào kệ kính không có mask; mask nhiễu |
| 9 | stealing f545-1484 | 6 alert nhưng 5 ID khác nhau | ID đổi liên tục, bbox đỏ ngắn |
| 10 | stealing f260-795 | alert 1 (f328-358), bbox đỏ ~f437 rồi xanh lại ở f562 khi đang mang hộp | **đúng lỗi bbox đỏ ngắn hạn** (alarm_show_sec=4s) |
| 11 | stealing f178-299 | 0 event | cảnh 3 (màu) có vẻ thiếu mask tranh trên tường (chưa xác nhận; kiểm tra lại cả khả năng giống video 12: track đứt) |
| 12 | stealing f85-287 | 0 event, 1 track | **Mask ĐÃ có** (6 mask, xác nhận qua live). Lỗi do TRACKING: người mặc đồ tối + mờ chuyển động, bbox chỉ có ~3/12 frame mẫu, mất đúng lúc với tay vào tranh → không pose/hand → không chạm. `_attach_scores` loại track không khớp detection |

---

### [x] VERIFIED TASK-001: Giữ bbox đỏ suốt video cho người đã bị gắn cờ
- **Task ID:** TASK-001
- **Mục tiêu:** Khi VLM trả ANOMALY cho một track, bbox của người đó phải đỏ từ lúc báo cho tới hết video (hoặc hết track), không quay về xanh sau `alarm_show_sec`.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/engine.py` (`AlarmBoard`, `on_vlm_result`, `render_live`)
  - `exhibitflow_pipeline/exhibitflow/settings.py` (`RT`)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Thêm `RT['alarm_persist']=True` (mặc định bật); `AlarmBoard.active()` trả tid của MỌI alarm đã phát khi bật. Giữ `alarm_show_sec` cho chế độ tắt.
  - Thêm `flagged` lưu `{tid: behavior, first_frame}`; HUD hiển thị nhãn hành vi bên cạnh ID đỏ.
  - Không đổi `events.csv`/`result.json` schema ngoài việc thêm trường tuỳ chọn `flagged_tracks`.
  - **Dependencies:** [TEMP] `pytest` vào `/tmp/venv_TASK-001`, xoá sau khi verify.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-001/bin/python -m pytest tests/test_pipeline_cpu.py -k "alarm" -v`
- **Trạng thái:** VERIFIED (3 passed -k alarm)

### [x] VERIFIED TASK-002: Kế thừa cờ ANOMALY qua đổi ID bằng ReID + vị trí
- **Task ID:** TASK-002
- **Mục tiêu:** Khi một track mới xuất hiện gần vị trí/ReID giống người đã bị gắn cờ (video 3, 9, 10), kế thừa cờ đỏ để theo dõi một người từ đầu đến cuối.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/engine.py`
  - `exhibitflow_pipeline/exhibitflow/perception.py` (lấy embedding OSNet từ tracker nếu có)
  - `exhibitflow_pipeline/exhibitflow/settings.py`
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Phụ thuộc TASK-001. Thêm `RT['flag_inherit']` = dict(enabled=True, reid_cos=0.7, max_gap_sec=3.0, max_center_dist_norm=2.0).
  - Nếu tracker không lộ embedding: dùng fallback màu histogram HSV của vùng thân + khoảng cách vị trí (ghi rõ trong code). Đánh đổi: có thể gộp nhầm 2 người mặc giống nhau → chỉ kế thừa khi track cũ đã mất ≤ `max_gap_sec`, mỗi cờ kế thừa tối đa 1 track tại một thời điểm.
  - Test giả lập: track A bị cờ biến mất ở frame 100, track B xuất hiện cùng chỗ frame 105 → B đỏ; track C ở xa → không đỏ.
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-002/bin/python -m pytest tests/test_pipeline_cpu.py -k "inherit" -v`
- **Trạng thái:** VERIFIED (3 passed)

### [x] VERIFIED TASK-003: Ổn định tracker (ID switch) bằng cấu hình + giữ track dự đoán
- **Task ID:** TASK-003
- **Mục tiêu:** Giảm ID switch (video 3: 23 track, video 9: 19 track) VÀ không để mất track khi detector bỏ sót/score thấp (video 12: người đồ tối, mờ, bbox mất đúng lúc với tay vào tranh → không pose → bỏ sót). ƯU TIÊN CAO NHẤT cho recall.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/settings.py` (`MODELS['tracker']`)
  - `exhibitflow_pipeline/exhibitflow/perception.py` (`_attach_scores`, tracker wrapper)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Đổi mặc định: `track_buffer` 30→75, `track_thresh` 0.6→0.5, `reid_weight` 0.35→0.5, `reid_thresh` 0.7→0.6, `ltc_min_history` 16→10. Mọi giá trị phải ghi đè được qua `--set`.
  - `_attach_scores`: thêm tham số `keep_predicted_frames` (mặc định 8): track không khớp detection được giữ với `predicted=True`, `score` suy giảm; pose/state machine được phép dùng nhưng không tạo event mới ở frame predicted dài > 8.
  - Video 12: khi không có detection khớp, dùng box DỰ ĐOÁN của Kalman/LTC của track (không bỏ track); tăng `_attach_scores.min_iou` 0.30→0.15 cho track đã ổn định (≥10 frame); chạy pose/hand trên box dự đoán (crop mở rộng thêm `pose_crop_pad_frac`) để vẫn có cổ tay ở frame detector hụt. Thêm `MODELS['detector'].low_light_boost` (CLAHE trên kênh L, chỉ khi độ sáng trung bình crop < ngưỡng) — bật/tắt được, chạy CPU nhẹ.
  - Test giả lập: chuỗi 100 frame, detection hụt frame 30–60 → track vẫn sống, `predicted=True`, pose được gọi; ID không đổi.
  - Trade-off: buffer dài làm tăng nhầm người khi đông (video 3, 9) → bù bằng reid_weight cao hơn.
  - Không chạy tracker thật; test bằng chuỗi detection tổng hợp (mock tracker).
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-003/bin/python -m pytest tests/test_pipeline_cpu.py -k "attach or predicted or tracker_cfg" -v`
- **Trạng thái:** VERIFIED (4 passed)

### [x] VERIFIED TASK-004: Lọc detector — chống nhận nhầm vật thể là người
- **Task ID:** TASK-004
- **Mục tiêu:** Thêm bước lọc track/box giả (sau detector conf=0.01) mà không phá recall của ByteTrack.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/perception.py`
  - `exhibitflow_pipeline/exhibitflow/settings.py`
  - `exhibitflow_pipeline/exhibitflow/engine.py`
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Giữ `detector.conf=0.01`. Thêm `RT['person_filter']` = dict(enabled=True, display_min_score=0.5, min_hits=3, min_box_area=600 (ảnh 960px), aspect_range=(0.2,1.0) w/h, min_valid_kpts=5 trong ≥50% frame gần nhất khi đã có pose).
  - Track chưa qua `min_hits`/lọc: không vẽ, không chạy pose, không tạo event. Đặt `artifact_person_min_score` 0.3→0.45 (chỉ dùng để loại vùng người khỏi mask hiện vật).
  - LƯU Ý: TASK-003 sẽ sửa cùng `perception.py` (`_attach_scores`) sau này → chỉ thêm hàm lọc MỚI tách riêng, KHÔNG sửa thân `_attach_scores` hay lớp tracker wrapper, để tránh xung đột.
  - Trade-off: người nhỏ/xa (video 5, 6, 12 người rất nhỏ) có thể bị lọc → `min_box_area` phải tính theo ảnh làm việc 960px và có cờ tắt.
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-004/bin/python -m pytest tests/test_pipeline_cpu.py -k "person_filter" -v`
- **Trạng thái:** VERIFIED (3 passed)

### [x] VERIFIED TASK-005: Làm mượt pose + nội suy cổ tay + hysteresis state machine
- **Task ID:** TASK-005
- **Mục tiêu:** Pose ổn định theo thời gian để chuỗi chạm không bị đứt (video 4: event chỉ 16 frame so với GT 60 frame).
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/perception.py` (đầu ra pose)
  - `exhibitflow_pipeline/exhibitflow/interaction.py` (`PairState`, `hand_anchor_point`)
  - `exhibitflow_pipeline/exhibitflow/settings.py` (`CFG`)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Thêm bộ lọc One-Euro (hoặc EMA) theo `track_id` cho keypoint, chỉ áp dụng khi `kc ≥ pose_conf_thr`; nội suy tuyến tính keypoint cổ tay cho khoảng mất ≤ 8 frame.
  - Đổi `state_lost_grace_frames` 5→12; thêm hysteresis: vào NEAR/CONTACT dùng `state_near_thr`/`state_contact_thr`, ra dùng ×1.3. Tăng `pose_gate_norm_dist` 1.5→2.0 (để pose không bật/tắt ở biên).
  - Thêm `CFG['pose_smooth']=True` để tắt được. Không đổi model pose trong task này (nâng yolov8s là việc riêng, cần GPU).
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-005/bin/python -m pytest tests/test_pipeline_cpu.py -k "smooth or hysteresis or interp" -v`
- **Trạng thái:** VERIFIED (4 passed)

### [x] VERIFIED TASK-006: Ghi log lý do loại ở contact gate / pairing
- **Task ID:** TASK-006
- **Mục tiêu:** Biết vì sao video 4, 7, 8, 11, 12 không có event/alert (too_small, depth_mismatch, arm_at_rest, self_touch, ambiguous, no_artifact_in_range, event_too_short...).
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/interaction.py` (`contact_gate`, `frame_states`, `_close`)
  - `exhibitflow_pipeline/exhibitflow/pipeline.py` (ghi `result.json`)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Đếm `gate_rejects` theo (track_id, lý do) và theo cảnh; ghi vào `result.json['diagnostics']` và `diagnostics.csv`. Không đổi quyết định, chỉ quan sát. Chi phí O(1) mỗi frame.
  - Ghi thêm `near_frames`/`min_norm_dist` của từng track (để thấy người có tới gần hiện vật không).
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-006/bin/python -m pytest tests/test_pipeline_cpu.py -k "diagnostics or gate_reject" -v`
- **Trạng thái:** VERIFIED (7 passed)

### [x] VERIFIED TASK-007: Nới cổng chạm có kiểm soát (event ngắn / người nhỏ)
- **Task ID:** TASK-007
- **Mục tiêu:** Bắt được chạm ngắn/người nhỏ mà không tăng báo động giả; dựa trên log TASK-006.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/settings.py` (`CFG`)
  - `exhibitflow_pipeline/exhibitflow/interaction.py`
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Phụ thuộc TASK-005, TASK-006. Thay đổi đề xuất (đặt sau cờ `CFG['relaxed_gate']=True`): `event_min_frames` 12→8, `state_min_person_h_px` 80→60 (khi `work` 960px), `state_hold_seconds` 0.25→0.15, `state_contact_thr` 0.35→0.45 khi mask là bệ (pedestal).
  - Thêm `RT['periodic_vlm']`: nếu người ở trong vùng NEAR liên tục ≥ 3s mà chưa có event → vẫn gửi một đoạn cho VLM (đường dự phòng cho video 4, 8).
  - Trade-off: FP tăng → VLM làm bộ lọc cuối; giữ `max_tasks_per_video=40`.
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-007/bin/python -m pytest tests/test_pipeline_cpu.py -k "relaxed or periodic" -v`
- **Trạng thái:** VERIFIED (3 passed)

### [x] VERIFIED TASK-008: Segmentation đa frame theo cảnh + từ khoá cho tranh trên tường / kệ kính
- **Task ID:** TASK-008
- **Mục tiêu:** Khắc phục mask thiếu: video 8 (kệ kính), video 6/7, và video 11 (cần xác nhận). KHÔNG áp dụng cho video 12 (mask đã đủ, lỗi nằm ở tracking → TASK-003).
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/segmenter.py`
  - `exhibitflow_pipeline/exhibitflow/settings.py` (`SEGMENTATION_CFG`, `RT`)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Thêm `RT['seg_multi_frame']` = dict(enabled=True, step_sec=2.0, max_frames_per_scene=4): segment nhiều frame mỗi cảnh, hợp nhất mask theo IoU ≥ 0.5 (giữ mask xuất hiện ≥ 2 lần hoặc điểm cao nhất).
  - Thêm phrase: `painting_phrases += ['picture frame','artwork on wall','wall art','mounted painting']`, `fallback_object_phrases += ['display case','glass cabinet','shelf']`; hạ `fallback_thr` 0.20→0.15 khi cảnh có 0 mask.
  - Phát hiện cảnh đổi sáng/màu (video 11: xám→màu): thêm tiêu chí cắt cảnh theo histogram màu bên cạnh chênh lệch xám.
  - Tăng cache key để mask cũ không bị dùng lại. Test dùng segmenter giả (không tải GDINO/SAM).
  - Giữ tương thích `MANUAL_ARTIFACTS`/`SEG_REF_FRAMES`.
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-008/bin/python -m pytest tests/test_pipeline_cpu.py -k "seg or scene or merge_masks" -v`
- **Trạng thái:** VERIFIED (5 passed)

### [x] VERIFIED TASK-009: Tín hiệu stealing: hiện vật biến mất / người mang vật đi (đường dự phòng không cần mask chạm)
- **Task ID:** TASK-009
- **Mục tiêu:** Bắt trộm khi cảnh gỡ tranh bị khuất hoặc mask thiếu (video 6, 7, 11, 12): phát hiện mask hiện vật biến mất so với frame tham chiếu, hoặc người đi ra với vật mang theo.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/interaction.py`
  - `exhibitflow_pipeline/exhibitflow/engine.py`
  - `exhibitflow_pipeline/exhibitflow/settings.py`
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - Phụ thuộc TASK-008. Mỗi ~1s so sánh vùng mask hiện vật (khi không bị người che) giữa frame hiện tại và tham chiếu bằng chênh lệch cạnh/màu; nếu thay đổi > ngưỡng ≥ 1.5s liên tục và có người trong vòng 2×chiều cao → tạo event `suspect_missing` gửi VLM (prompt giữ nguyên nhãn `stealing`).
  - Rủi ro FP khi ánh sáng đổi (video 11) → bỏ qua khi `scene_cut`/đổi độ sáng toàn cục. Cờ `RT['missing_detect']` mặc định True, chạy CPU nhẹ ở độ phân giải 160px.
  - **Dependencies:** [TEMP] `pytest`.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-009/bin/python -m pytest tests/test_pipeline_cpu.py -k "missing" -v`
- **Trạng thái:** VERIFIED (5 passed)

### [x] VERIFIED TASK-010: Công cụ đánh giá theo GT (recall / precision / số ID / đỏ-xuyên-suốt)
- **Task ID:** TASK-010
- **Mục tiêu:** Có metric lặp lại được để so sánh trước/sau mỗi task, dùng `benchmark anomaly/gt_videos.csv` và `results/*/result.json`.
- **Các file liên quan:**
  - `exhibitflow_pipeline/scripts/eval_gt.py` (mới)
  - `benchmark anomaly/gt_videos.csv`
  - `exhibitflow_pipeline/tests/test_eval_gt.py` (mới)
- **Lưu ý kiến trúc & Trade-offs:**
  - Đầu vào: thư mục results + gt csv (map `results/N` ↔ video `N`). Đầu ra: bảng mỗi video: có alert trong khoảng GT? (hit), độ trễ báo (frame), số alert ngoài GT (FP), số track_id khác nhau trong alert, % frame GT mà người bị gắn cờ còn đỏ.
  - Chỉ đọc JSON/CSV — không chạy model, nhẹ CPU. Đọc CSV bằng `utf-8-sig` (file có BOM).
  - **Dependencies:** [TEMP] `pytest`, `pandas` nếu chưa có trong môi trường.
- **Verification Command:** `cd exhibitflow_pipeline && nice -n 19 /tmp/venv_TASK-010/bin/python -m pytest tests/test_eval_gt.py -v && nice -n 19 python3 scripts/eval_gt.py --results results --gt "../benchmark anomaly/gt_videos.csv"`
- **Trạng thái:** VERIFIED (4 passed, eval_gt chạy OK)

### [x] VERIFIED TASK-011: Đưa các tham số tinh chỉnh mới lên menu Settings (dashboard cũ + web mới)
- **Task ID:** TASK-011
- **Mục tiêu:** Mọi cờ/ngưỡng thêm ở TASK-001..009 chỉnh được từ menu Settings (cả `dashboard/static/index.html` lẫn `web/`), lưu lại và áp dụng từ lần chạy kế tiếp, không phải sửa `settings.py`.
- **Các file liên quan:**
  - `exhibitflow_pipeline/exhibitflow/tuning.py` (mới: registry tham số)
  - `exhibitflow_pipeline/dashboard/runner.py` (`DEFAULT_SETTINGS`, `overrides_from_settings`, `MODEL_KEYS`, `save_settings`)
  - `exhibitflow_pipeline/dashboard/server.py` (thêm `GET /api/tuning/schema`)
  - `exhibitflow_pipeline/dashboard/static/index.html` (thẻ "Tinh chỉnh" dựng động từ schema, thêm tên khoá vào `FIELDS` hoặc xử lý riêng)
  - `exhibitflow_pipeline/server/web_server.py` (`PIPE_OPTS`, `pipe_of`, endpoint schema)
  - `web/index.html`, `web/js/app.js` (`PIPE_DEF`, `pipeOpts`)
  - `exhibitflow_pipeline/tests/test_pipeline_cpu.py`
- **Lưu ý kiến trúc & Trade-offs:**
  - **Registry một nguồn:** `tuning.py` khai báo list `Param(key, path, type, group, label_vi, help_vi, min, max, step, needs_reload)`; `path` kiểu `RT.person_filter.min_hits`, `CFG.state_contact_thr`, `MODELS.tracker.track_buffer`. Giá trị mặc định lấy từ `settings` lúc import (không hard-code lặp). Hai UI chỉ lấy schema qua API rồi dựng input (number/checkbox) theo `type`, nên thêm tham số sau này chỉ sửa một chỗ.
  - **Lưu:** dashboard cũ lưu trong `settings.json` dưới khoá `tuning` (dict `{key: value}`; `save_settings` phải chấp nhận khoá này, vì hiện chỉ nhận khoá có trong `DEFAULT_SETTINGS`). Web gửi `tuning` kèm options mỗi lần chạy (như `PIPE`, `localStorage` `ef_pipe2` giữ nguyên tương thích). Giá trị trống/None = dùng mặc định `settings.py`.
  - **Áp dụng:** `overrides_from_settings` chuyển `tuning` thành dict override lồng (`rt`/`cfg`/`models`) rồi `apply_overrides`. Phải kiểm tra kiểu và khoảng min/max phía server, giá trị sai thì HTTP 400 kèm tên khoá (không lưu). Không áp đè giá trị người dùng đặt tay ở các khoá cũ (`alarm_show_sec`, `pose_gate_norm_dist`).
  - **Tham số cần nạp lại model (`needs_reload=True`):** mọi `MODELS.tracker.*` và `MODELS.detector.low_light_boost.*` (tracker/detector nạp tham số lúc load) → đưa vào `model_key()`/`MODEL_KEYS` để đổi thì nạp lại, UI ghi chú rõ. Các khoá RT/CFG còn lại áp dụng ngay ở lần chạy sau, không nạp lại.
  - **Nhóm & khoá bắt buộc có (đúng tên trong settings.py):**
    - *Khung đỏ & ID:* `RT.alarm_persist`, `RT.flag_inherit.{enabled,reid_cos,max_gap_sec,max_center_dist_norm}`
    - *Tracker (nạp lại model):* `MODELS.tracker.{track_thresh,track_buffer,match_thresh,reid_weight,reid_thresh,ltc_min_history,keep_predicted_frames,attach_min_iou,attach_stable_min_iou,attach_stable_frames,predicted_score_decay}`; `MODELS.detector.low_light_boost.{enabled,luma_thr,clip_limit}`
    - *Lọc người (detector):* `RT.person_filter.{enabled,display_min_score,min_hits,min_box_area,min_valid_kpts,pose_min_valid_frac}`, `RT.artifact_person_min_score`
    - *Pose:* `CFG.pose_smooth`, `CFG.pose_smooth_min_cutoff`, `CFG.pose_smooth_beta`, `CFG.pose_wrist_interp_max_gap`, `CFG.pose_conf_thr`, `CFG.hand_conf_thr`, `CFG.state_lost_grace_frames`, `CFG.state_hysteresis`, `CFG.state_hysteresis_mult`
    - *Cổng chạm:* `CFG.relaxed_gate`, `CFG.relaxed_event_min_frames`, `CFG.relaxed_min_person_h_px`, `CFG.relaxed_hold_seconds`, `CFG.relaxed_pedestal_contact_thr`, `CFG.state_near_thr`, `CFG.state_contact_thr`, `CFG.state_min_person_h_px`, `CFG.event_min_frames`
    - *VLM dự phòng:* `RT.periodic_vlm.{enabled,min_near_sec,cooldown_sec}`
    - *Phát hiện biến mất (stealing):* `RT.missing_detect.{enabled,check_sec,sustain_sec,change_thr,carry_detect,carry_change_thr,carry_sustain_sec,max_person_occlude}`
    - *Segmentation/cảnh:* `RT.seg_multi_frame.{enabled,step_sec,max_frames_per_scene}`, `RT.scene_color_cut`, `RT.scene_color_hist_thr`, `RT.scene_cut_diff`, `SEGMENTATION_CFG.full_thr`, `SEGMENTATION_CFG.fallback_thr`
  - Có nút "Khôi phục mặc định" từng nhóm và toàn bộ; mỗi ô hiện giá trị mặc định cạnh nhãn. Nhãn/giải thích tiếng Việt, ghi rõ hậu quả (vd "tăng → ít báo giả hơn nhưng dễ bỏ sót người nhỏ").
  - Test (mock, không chạy model): (1) mọi `path` trong registry tồn tại thật trong `settings`; (2) `overrides_from_settings` với `tuning` mẫu sinh đúng override lồng và `apply_overrides` đổi được giá trị; (3) giá trị ngoài khoảng bị từ chối; (4) đổi khoá `needs_reload` làm `model_key` đổi, đổi khoá RT/CFG thì không; (5) `GET /api/tuning/schema` ở cả hai server trả đủ nhóm.
  - Không động vào `external/`. Không import tracker thật khi test. Không chạy trình duyệt/GPU; UI kiểm bằng test schema + `node --check` cho `app.js` nếu có node (nếu không có thì bỏ qua và ghi rõ trong EVIDENCE).
  - **Dependencies:** [TEMP] `pytest`, `fastapi`/`httpx` nếu chưa có (cài vào venv tạm `--system-site-packages`).
- **Verification Command:** `cd exhibitflow_pipeline && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" nice -n 19 /tmp/venv_TASK-011/bin/python -m pytest tests/test_pipeline_cpu.py -k "tuning" -v`
- **Trạng thái:** VERIFIED (4 passed -k tuning; 42 passed toàn bộ)

---
**Thứ tự thực hiện:** TASK-010 (XONG) → 001 → 006 → 003 (ưu tiên recall, hoãn nếu ablation tracker chưa xong) → 005 → 004 → 002 → 007 → 008 → 009 (tuần tự, không song song để tránh tranh CPU).
**Việc cần GPU/chạy lại video thật (làm sau, do người dùng quyết định khi GPU rảnh):** nâng pose lên yolov8s/m, tune VLM prompt cho video 4/6, chạy lại 12 video để lấy số liệu sau-sửa.
