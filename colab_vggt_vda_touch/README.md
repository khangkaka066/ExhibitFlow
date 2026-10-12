# Notebook VGGT + Depth Anything V2 Metric

## Kế hoạch nâng cấp Touch V2 — 09/10/2026

Xem [PLAN_TOUCH_V2.md](PLAN_TOUCH_V2.md) để triển khai nhánh tương tác bàn tay–hiện vật:
giữ ByteTrack-DMA-LTC, crop pose từ ảnh gốc, bổ sung pose bàn tay, temporal evidence,
geometry có điều kiện và đánh giá lại Isolation Forest. Plan có checklist T00–T12,
schema, phase preview, kiểm thử, đánh giá và gói upload Kaggle.

**Đây là kế hoạch, chưa phải code V2 đã được tích hợp.** Notebook và hướng dẫn
bên dưới vẫn mô tả phiên bản hiện tại (V1).

### Sửa fusion + ảnh/video bằng chứng (`touch_v2/notebook_source_v2.py`, Phase 7)

Theo thứ tự sửa đã thống nhất (checkpoint Kaggle → tọa độ/crop → pose bàn tay →
mask/reference hiện vật → temporal rule → **fusion và ảnh/video bằng chứng** → evaluation),
Phase 7 (`fuse_events`) trước đây tự động nâng `CONTACT_LIKE` lên nhãn "đã xác minh"
chỉ dựa vào việc Phase 5 (geometry) *có chạy được cho cả clip*, bất kể từng event có
thực sự được kiểm tra độ sâu hay không — gây hiểu nhầm là đã xác minh tiếp xúc 3D.
Đã sửa: mọi quyết định từ `CONTACT_LIKE` giờ luôn ra `TOUCH_CANDIDATE_2D_ONLY`; trường
`geometry_available` chỉ còn mang tính thông tin (có dữ liệu geometry cho clip hay không),
không phải tín hiệu đã xác minh. Xác minh tiếp xúc 3D theo từng event vẫn là bài toán
nghiên cứu mở `T12_open_research_3d_contact_verifier`, nằm ngoài scope MVP (đã ghi rõ
trong manifest `not_implemented`).

Đồng thời đã bổ sung bằng chứng hình ảnh/video: mỗi event giờ có 1 ảnh ghép 3 keyframe
(đầu/giữa/cuối) vẽ bbox + mask hiện vật, điểm neo tay và nhãn state/khoảng cách, cùng
1 clip `.mp4` ngắn cho event dài hơn 1 frame — lưu trong `07_fusion_events/attempt_XX/`
kèm `evidence_manifest.json`. Toạ độ dùng để vẽ (`anchor_xy`, box track/artifact) đều
ở cùng không gian ảnh resize (`frames`, không phải `native_frames`) nên không bị lệch vị trí.

### Sửa evaluation (`touch_v2/notebook_source_v2.py`, Phase 8)

Mục cuối trong thứ tự sửa đã thống nhất. Trước đây Phase 8 so khớp event dự đoán với
nhãn thật bằng cách so sánh thẳng `track_id` của model với một cột `track_id` trong
CSV nhãn — ngầm giả định track_id ổn định giữa các lần chạy, đúng điều PLAN_TOUCH_V2.md
§5.2/§11.2 cảnh báo không được làm (ByteTrack có thể đổi số track giữa các lần chạy
cho cùng một người). Việc khớp theo `set()` cũng không đảm bảo một-một — một event dự
đoán có thể "ăn" nhiều nhãn hoặc ngược lại.

Đã sửa: schema `annotations/events_ground_truth.csv` đổi `track_id` → `gt_person_id`
(nhãn người ổn định do annotator đặt) kèm `ref_frame,ref_x1,ref_y1,ref_x2,ref_y2` — một
bbox annotator khoanh tại một frame cụ thể để xác nhận đúng người đó **trong chính lần
chạy hiện tại**. `ref_box` phải khoanh theo độ phân giải ảnh làm việc đã resize (`frames`),
cùng không gian toạ độ với ảnh bằng chứng ở Phase 7. Mỗi `gt_person_id` được ánh xạ sang
track_id dự đoán của lần chạy đó bằng IoU lớn nhất giữa `ref_box` và box các track tại
`ref_frame` (dùng lại `box_iou()` đã có sẵn trong pipeline) — nếu không track nào đạt
IoU ≥ 0.3 thì coi là không ánh xạ được (ghi nhận trong `n_gt_unmapped`, không tính là
lỗi model). Track 3 trong clip pilot vẫn dùng làm case hồi quy, nhưng chỉ là nơi lấy
`ref_box` cho lần chạy đó — code không có bất kỳ điều kiện nào giả định "track 3 luôn
là người X".

Việc khớp precision/recall cũng được sửa thành một-một thật: tất cả cặp (dự đoán, nhãn)
có overlap thời gian > 0 trên track đã ánh xạ được sắp theo overlap giảm dần, rồi duyệt
tham lam — mỗi dự đoán và mỗi nhãn chỉ được dùng tối đa 1 lần, không còn tình trạng một
dự đoán khớp nhiều nhãn hay ngược lại.

## Chạy notebook hiện tại (V1)

Mở **touch_vggt_vda_bytetrack_kaggle.ipynb** bằng Kaggle Notebooks → bật GPU và Internet trong Notebook Settings → chạy các cell theo thứ tự.
Notebook tự chứa toàn bộ code, **không cần upload các file Python trong thư mục này**.

Nếu muốn đánh giá model trước, chạy notebook độc lập
[train_normal_pose_motion_isolation_forest.ipynb](train_normal_pose_motion_isolation_forest.ipynb).
Notebook đó chỉ trích pose bình thường bằng YOLOv8-Pose, train/evaluate Isolation Forest,
hiện các outlier holdout để audit và dừng; chưa chạy touch hay VGGT/Depth Anything.

Main notebook nạp model Isolation Forest đã train/evaluate bằng notebook riêng. Nó không train lại
hoặc cần mount MuseumVisitors mỗi lần chạy pipeline.

## Xem kết quả từng phase trước khi tiếp tục

Các cell đầu cài đặt và định nghĩa hàm. Tới **phase 0**, chọn `CLIP='1'` (hoặc `'2'`, `'3'`).
Notebook xử lý **một video mỗi lượt**, không còn vòng lặp chạy thẳng cả ba video.

| Phase | Kết quả hiện ra khi chạy |
|---|---|
| 0 — Input | Video, FPS, số frame/kích thước + kiểm tra đọc video |
| 1 — Tracking | Bbox/ID preview + detections, số ID, tỷ lệ frame có track; chưa chạy pose |
| 2 — Pose | Skeleton/cổ tay preview + tỷ lệ pose và cổ tay đủ confidence |
| 3 — Motion anomaly | Preview/CSV + cửa sổ hợp lệ, unusual candidates, cảnh báo camera holdout |
| 4 — Artifacts | Cấu hình `mmgdino-large-pedestal.yaml`: MM-Grounding-DINO large + tiles, SAM2.1 B; ảnh bbox, mask và candidate bị loại |
| 5 — VGGT | Depth/confidence, point cloud, độ phủ depth và intrinsics |
| 6 — Depth Anything V2 | RGB cạnh depth + tỷ lệ pixel hợp lệ và median depth ước lượng |
| 7 — Alignment | Biểu đồ lỗi, frame usable, anchor và pixel bề mặt hiện vật được hỗ trợ |
| 8 — Touch | Video/ảnh sự kiện + TOUCH/UNKNOWN/NO_TOUCH, tỷ lệ UNKNOWN và giới hạn hiệu chuẩn |

Cuối mỗi phase có bảng `PASS / WARN / FAIL`, giá trị quan sát được, gợi ý xử lý tiếp và danh sách file cần mở.
Notebook lưu `phase_diagnostic.json` và `.csv` ngay trong thư mục attempt của phase; ZIP cuối cũng giữ các báo cáo này.
`FAIL` nghĩa là nên dừng và xử lý trước khi tiếp tục; `WARN` cần xem kỹ preview. PASS tự động không thay thế review bằng mắt.

Không cần nhập `TIEP`/`DUNG`: phase tự hoàn tất và chuyển tiếp khi bạn chạy `Run all`.
Bạn vẫn có thể chạy từng cell để xem kết quả theo từng bước. Có `WARN` thì notebook ghi cảnh báo
nhưng vẫn chạy tiếp; lỗi thực thi nghiêm trọng sẽ khiến Kaggle dừng tại cell bị lỗi.
Chạy lại phase trước sẽ xóa trạng thái hoàn tất các phase sau và giữ kết quả cũ trong attempt riêng.
Nếu đổi CFG, bắt đầu lượt mới ở phase 0. Đổi video cũng bắt đầu từ phase 0; không cần cài lại thư viện.
Trạng thái chạy lưu trong `phase_review.json`. Sau reset runtime phải khởi tạo lại, chưa hỗ trợ resume từ file.

Tải [kaggle_upload_bundle.zip](../kaggle_upload_bundle.zip) lên một private Kaggle Dataset rồi gắn bằng **Add Input**.
Notebook tự giải nén bundle vào `/kaggle/working`; cũng có thể gắn dataset chứa 8 file đã bung sẵn:

- `1.mp4`, `2.mp4`, `3.mp4`
- `best_ckpt` (PyTorch checkpoint; không yêu cầu phần mở rộng), `dma_gbm_mot17.gbm`, `ltc_motion_mot17.pth`
- `isolation_forest_normal_yolov8_pose.joblib`, `evaluation.json`
- `sam2.1_b.pt` (SAM2.1 segmentation checkpoint)

Notebook tự tìm 8 file dưới `/kaggle/input`. Nếu có nhiều dataset chứa cùng tên file,
đổi `DATASET_DIR` trong cell input thành đúng thư mục `/kaggle/input/<dataset-slug>`.
Checkpoint SAM2.1 được đóng gói sẵn. Repository, tracker ReID, MM-Grounding-DINO, VGGT và Depth Anything V2
được tải lúc chạy, vì vậy vẫn cần bật **Internet**. Kết quả ghi trong `/kaggle/working/exhibit_vggt_depth_anything_v2/`;
cell cuối hiện link ZIP để tải xuống.

`best_ckpt` là checkpoint PyTorch, được nạp trực tiếp bằng `torch.load`. Bộ trọng số này dùng YOLOX nano một lớp người.
Lớp Focus nhận RGB rồi ghép thành 12 kênh nội bộ; không phải detector yêu cầu video 12 kênh.
Notebook dùng strict load và dừng nếu không khớp, không tự thay bằng YOLOv8 detector.
Checkpoint này đã được thử forward RGB và phát hiện trên frame video thật bằng CPU.
Tiền xử lý mặc định **BGR 0–255**: smoke test trên video 1 cho detections, còn chuẩn hóa ImageNet
trong demo cũ của repo cho 0 detections. Notebook ghi chế độ này vào manifest/tracker_report,
xuất `detector_smoke.jpg`, và dừng nếu cả clip không có detections; không tự đổi chế độ ngầm.

## Isolation Forest trong pipeline

Model được train/evaluate trước bằng [notebook độc lập](train_normal_pose_motion_isolation_forest.ipynb).
Main notebook nạp `.joblib` và báo cáo `evaluation.json`, rồi chấm các cửa sổ pose 2.5 giây sau phase YOLO-Pose.
Camera 1–2 được giữ ở mức candidate để review. Camera 3 có drift rõ trên normal holdout nên chỉ hiện điểm
tham khảo, không sinh candidate tự động. Mọi điểm cao chỉ có nghĩa chuyển động khác phân bố normal;
không xác nhận touch, trộm hay cướp.

## Luồng xử lý

1. YOLOX nano trong repo người dùng → ByteTrack + DMA GBM + LTC + OSNet ReID → ID.
2. YOLOv8n-pose trên crop theo ID → cổ tay. Đây là pose sau tracking.
   Touch V2 ([PLAN_TOUCH_V2.md](PLAN_TOUCH_V2.md)) thêm Phase 2b tùy chọn: RTMW
   wholebody (133 điểm) chạy trên đúng bbox track, cho tọa độ ngón tay thay vì
   chỉ cổ tay. Nếu `rtmlib`/backend không khởi tạo được, phase tự tắt và pipeline
   dùng lại cổ tay COCO17 như cũ. Nhánh này chưa được xác nhận trên GPU Kaggle thật.
3. Cấu hình `mmgdino-large-pedestal.yaml`: MM-Grounding-DINO large (full frame + 3 tiles, prompt dương/âm) + Ultralytics SAM `sam2.1_b.pt`; lọc mask thấp chất lượng, gộp mask trùng và ghép bệ sát tượng. Bật thêm material rescue theo độ sáng như config; candidate được đánh dấu `RESCUED` để dễ review vì heuristic màu có thể nhận nhầm.
   Touch V2 chọn frame tham chiếu cho bước này bằng `select_reference_frame`
   (frame có tổng diện tích bbox người che khung hình nhỏ nhất trong 30 frame
   đầu) thay vì luôn lấy frame đầu tiên — ghi `reference.frame_idx`/
   `person_cover_frac` vào `artifacts.json` để audit. Đây là một mask/reference
   duy nhất cho toàn scene (giả định camera tĩnh); phần đăng ký lại camera khi
   đổi góc, tách `visible_mask_t` theo từng frame, và `artifact_part_id` riêng
   tượng/bệ theo PLAN_TOUCH_V2.md T03 **chưa được triển khai**.
4. VGGT gốc → camera + depth tham chiếu; loại bbox người khỏi các điểm bề mặt hiện vật.
5. Depth Anything V2 **Metric Indoor Small** → depth từng frame, thang mét do model ước lượng.
6. Căn chỉnh scale trên nền tĩnh, kiểm tra pixel holdout và chuyển động camera.
7. Model Isolation Forest đã train → score pose/chuyển động; cảnh báo camera 3 bị hạ thành review-only.
8. Khoảng cách cổ tay–bề mặt + kiểm tra uncertainty + giữ gần theo thời gian → touch candidate độc lập.
   Touch V2 `PairState` ([PLAN_TOUCH_V2.md](PLAN_TOUCH_V2.md) T06) ghi thêm bằng chứng
   `approach_evidence`/`withdraw_evidence` (khoảng cách tay–hiện vật đang giảm rõ rệt / đã
   tăng lại sau cực tiểu) vào từng frame của `pairing_states.csv`, dùng để làm giàu nhãn
   state và hỗ trợ phân biệt Tap (chạm rồi rút) khỏi Hold. Theo đúng plan, motif Hold
   **không bắt buộc** phải có cả approach và withdraw (clip có thể bị cắt giữa lúc đang
   chạm) nên hai cờ này **không** được dùng để chặn `CONTACT_LIKE` — chỉ là tín hiệu diagnostic
   phụ. Những phần T06 **chưa triển khai**: phân loại state-sequence thành
   Tap/Hold/Slide/Pass-by cụ thể, cờ left/right-censoring khi clip bị cắt ở biên, ghi log
   timestamp dạng causal vs. refine riêng, reset lịch sử theo track-segment/đổi tay/cắt
   camera, và bộ mock test cho các kịch bản tap/hold/slide/pass-by/occlusion/wrist nhiễu/ID-switch.

Phase 3 xuất `motion_anomaly_scores.csv`; phase Touch ghi thêm `motion_anomaly_percentile` và
`motion_context` vào event/evidence để reviewer đối chiếu. Isolation Forest không thay đổi điều kiện
`TOUCH_CANDIDATE`. Camera 3 hiện được đánh dấu `camera_unvalidated_score_only`.

Các model lớn chạy tuần tự để giảm VRAM. Không cần UniDepth hoặc notebook cũ để chạy file mới.
Notebook training Isolation Forest là file riêng; main notebook chỉ nạp model đã train/evaluate.

## Kết quả

Mỗi video có `preview.mp4`, `artifact_review.jpg`, `tracks_pose.json`, `events.csv/json`,
`observations.csv`, `alignment.csv`, `summary.json`, depth và bề mặt 3D `.npy/.npz`.
Một ảnh tổng hợp cho mỗi sự kiện có mask hiện vật, ID người, bbox người nghi chạm.
Mỗi phase hiển thị kết quả ngay. Cell cuối tải ZIP cho video đang chọn, bao gồm mọi attempt
và `phase_review.json` ghi lại attempt/trạng thái các phase. ID chỉ có ý nghĩa trong từng video.

`TOUCH_CANDIDATE` là ứng viên review, **không phải xác nhận chạm**.
`UNKNOWN` dùng khi thiếu pose/depth, không khớp hình học, vùng bị che hoặc không đủ bằng chứng theo thời gian.
`NO_TOUCH` chỉ nghĩa không đạt điều kiện tại frame đủ quan sát, không phải ground truth.

Checkpoint Metric Indoor được fine-tune trên Hypersim tổng hợp. Thang mét có thể lệch khi chuyển
sang bảo tàng thật và chưa hiệu chuẩn tại đây. Depth Anything V2 dự đoán độc lập từng frame;
phase alignment kiểm tra độ ổn định trước khi dùng khoảng cách. COCO17 chỉ có cổ tay, không có ngón tay.
Không có góc nhìn mới từ camera cố định; vùng hiện vật luôn bị che không thể phục hồi chắc chắn.
Mask tự động có thể sai: kiểm tra ảnh review trước khi tin cảnh báo.
Ngưỡng mặc định chỉ phục vụ thử nghiệm. Cần đoạn âm tính và nhãn chạm thật để đánh giá/tinh chỉnh.

## Kiểm tra kỹ thuật

- Build: `python3 build_notebook.py`
- Kiểm tra CPU: `python3 test_notebook.py`
- Kiểm tra schema notebook, cú pháp từng cell, scale/holdout, intrinsics, che khuất, temporal reset và camera shift.
- Kiểm thử synthetic cả xuất video H.264, ảnh sự kiện, CSV và đóng sự kiện khi mất người.
- Chưa chạy toàn bộ inference GPU trên Kaggle từ môi trường local; không cam kết accuracy hoặc thời gian chạy.

Dùng Kaggle session mới nếu môi trường từng bị thay đổi thư viện.
Không cài lại PyTorch/CUDA, không hạ NumPy. Nếu OOM: giảm reference_frames xuống 2 hoặc da_v2_input_size xuống 280.
Nếu MM-Grounding-DINO/SAM2/mask hoặc checkpoint thất bại, notebook dừng rõ ràng thay vì tạo kết quả giả.

Nguồn: [VGGT](https://github.com/facebookresearch/vggt),
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[tracker người dùng](https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker),
[SAM2](https://github.com/facebookresearch/sam2).
