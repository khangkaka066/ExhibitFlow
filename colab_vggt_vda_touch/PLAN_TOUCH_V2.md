# ExhibitFlow — Kế hoạch triển khai pipeline Touch V2

Ngày lập: **09/10/2026 — Asia/Ho_Chi_Minh**.

Trạng thái: **kế hoạch triển khai, chưa thay đổi code inference, chưa có kết quả V2**.
Các số liệu baseline dưới đây đến từ lần chạy V1 đã kiểm tra. Các ngưỡng, cấu hình
và tên file V2 được ghi là đề xuất, không phải tính năng đã tồn tại.

### Cách đọc nhanh

- Kiến trúc và phạm vi: mục 1–4.
- Dữ liệu/cấu hình: mục 5; thứ tự thực hiện: mục 6.
- Việc cần code và tiêu chí đạt từng việc: **mục 7, T00–T12**.
- File/hàm cần thay đổi: mục 8; logic quyết định: mục 9.
- Những gì sẽ hiện trong notebook: **mục 10**.
- Cách chứng minh cải thiện và kiểm thử: **mục 11–12**.
- File upload, tài nguyên, rủi ro: mục 13–15.
- Thứ tự bàn giao và checklist sử dụng: **mục 16–17**; nguồn: mục 18.

Đây là tài liệu backlog đầy đủ, không phải yêu cầu hoàn thành mọi nhánh mới có
bản dùng thử. Đường bàn giao đầu tiên là **M0 → M1 → M2**; T07/T08/T12 không
được kéo dài hoặc chặn bản MVP thị giác nếu chưa chứng minh có ích.

## 1. Mục tiêu và quyết định kiến trúc

### 1.1. Kết quả cần đạt

Với video camera bảo tàng cố định, hệ thống phải trả lời được:

1. Người nào đang tương tác với hiện vật nào?
2. Bàn tay nào liên quan và sự kiện xảy ra khi nào?
3. Bằng chứng nào khiến hệ thống nghi có tiếp xúc?
4. Có quan sát được tiếp xúc hay đang thiếu thông tin do tay nhỏ, che khuất,
   mask lỗi, track lỗi hoặc depth không đáng tin?
5. Khi sai, xem output phase nào để tìm nguyên nhân?

Đầu ra vận hành ban đầu là **ứng viên chạm để review**, không phải chứng nhận
tiếp xúc vật lý hay xác định vi phạm. Mỗi sự kiện phải có video ngắn và **một ảnh
bằng chứng duy nhất** gồm mask hiện vật, ID hiện vật, mọi người có ID trong frame
và người nghi chạm được làm nổi bật.

### 1.2. Kiến trúc chọn triển khai

**Nhánh chính:** tracking → crop ảnh gốc → pose bàn tay → ghép cặp người–hiện vật
→ bằng chứng tiếp xúc theo thời gian → touch candidate.

**Nhánh hỗ trợ:** Depth Anything V2 + VGGT chỉ cung cấp thông tin khi có chất lượng
phù hợp. Thiếu 3D không được làm biến mất một candidate thị giác đang có bằng chứng.

**Nhánh học normal:** Isolation Forest chấm độ khác thường của chuyển động hoặc
tương tác; không phải điều kiện bắt buộc để touch candidate xuất hiện, không tự
đổi outlier thành touch.

### 1.3. Những thành phần giữ lại

- Detector **YOLOX-nano từ checkpoint của người dùng** nằm trong pipeline tracker.
- **ByteTrack-DMA-LTC**, cùng cấu hình association/ReID đang dùng; tracking trước pose.
- MM-Grounding-DINO large + SAM2.1 B, kế thừa cấu hình `mmgdino-large-pedestal.yaml`.
- VGGT và Depth Anything V2 Metric Indoor để làm nhánh thử nghiệm hỗ trợ.
- Isolation Forest đã train, nhưng giữ đúng hợp đồng đầu vào và giới hạn camera.
- Kaggle là môi trường chạy chính.
- Mỗi phase hiển thị kết quả rồi **tự chạy tiếp**, không có ô nhập duyệt/TIEP/DUNG.

### 1.4. Những việc không làm trong V2 đầu tiên

- Không thay tracker của người dùng bằng tracker mặc định của thư viện pose.
- Không huấn luyện detector/pose lớn từ đầu bằng ba video.
- Không đặt kích thước sàn, AB/AC hay khoảng cách thật bằng giả định tùy ý.
- Không coi chồng mask 2D là bằng chứng chạm vật lý.
- Không hạ ngưỡng depth/Isolation Forest chỉ để làm ID 3 hiện đỏ.
- Không dùng RGB sinh bởi AI, ảnh ghép hoặc điểm pose nội suy như ground truth tiếp xúc.
- Không mở thêm theft/fight/fall trong đợt này.
- Không xóa notebook, checkpoint, ZIP hoặc kết quả cũ trong quá trình triển khai.

### 1.5. Thứ tự ưu tiên

| Mức | Nội dung | Lý do |
|---|---|---|
| P0 | Ground truth pilot, giữ ảnh gốc, đúng tọa độ/thời gian/ID, chẩn đoán nguyên nhân | Không thể đánh giá model nếu dữ liệu đầu vào và nhãn sai |
| P1 | Pose bàn tay, mask ổn định, ghép cặp và chuỗi tương tác | Trực tiếp giải quyết khả năng quan sát touch |
| P2 | Fusion có trạng thái thiếu bằng chứng, preview và đánh giá sự kiện | Tránh bỏ sót im lặng và báo nhầm không giải thích được |
| P3 | Đo giá trị bổ sung của DA-V2/VGGT và IF | Chỉ giữ vai trò quyết định nếu thực nghiệm chứng minh có ích |
| P4 | Contact verifier học ít nhãn; hiệu chuẩn 3D ngoài hiện trường | Chỉ thực hiện khi baseline và dữ liệu đã đủ rõ |

## 2. Baseline đã biết và các điểm chưa được xác nhận

### 2.1. Dữ kiện đã kiểm tra

Nguồn baseline:

- Video: `../1.mp4`, 1820×1136, 121 frame, 24 FPS theo lần kiểm tra hiện tại.
- ZIP: `/Users/nguyenvokhang/Downloads/run_20261009_062025_596044_video_1_vggt_depth_anything_v2_touch_results.zip`.
- Source: [notebook_source.py](notebook_source.py).
- Kết quả đánh giá IF: [evaluation.json](../colab_iforest_normal_motion/evaluation.json).

| Quan sát V1 | Ý nghĩa cho V2 |
|---|---|
| Người dùng xác nhận người chạm là ID 3 trong bản preview baseline | Dùng làm ca regression; chưa tự suy ra toàn bộ nhãn sự kiện |
| Frame bị resize xuống 640×400 trước khi crop pose | V2 phải crop từ nguồn gốc, không phóng to ảnh đã mất chi tiết |
| ID 3 có track 121/121 frame; có pose 114/121 frame | Chưa có bằng chứng cần thay tracker để sửa ca này |
| Cổ tay phải ID 3 đạt confidence ≥ 0,55 ở 38/121 frame | Cần kiểm tra chất lượng bàn tay, không chỉ số frame có pose |
| 13/19 hiện vật có 0 pixel bề mặt 3D được hỗ trợ | Mask 2D và khả năng đo 3D phải được báo cáo riêng |
| `artifact_2` có mask nhưng 0 pixel bề mặt được hỗ trợ | Không thể tính khoảng cách với bề mặt không có dữ liệu |
| p90 bất đồng depth mỗi frame khoảng 1,36–2,03 m; gate hiện tại yêu cầu < 0,15 m | Gate hình học hiện tại không thể thông qua trong lần chạy này |
| IF của ID 3 cao nhất khoảng percentile 86,25 | Hạ ngưỡng 99 xuống 98 không giải quyết ca ID 3 |
| Điểm 98,75 thuộc ID 7 | Không dùng cực đại toàn video để giải thích riêng người chạm |
| 40.584 quan sát V1 đều UNKNOWN; 0 event | Không được trình bày kết quả này như video không có touch |

Con số depth ở trên là **bất đồng giữa các dự đoán**, không phải sai số đo được
so với khoảng cách thật. Hai model đồng ý cũng chưa chứng minh cả hai đúng.

### 2.2. Thử nghiệm chẩn đoán 2D đã làm

Với pose/mask đã lưu, chỉ dùng confidence cổ tay, khoảng cách ảnh ≤ 12 px và
ít nhất ba frame liên tiếp, không dùng depth hoặc tốc độ 3D:

- ID 3 tạo proposal với `artifact_2` ở khoảng 4,58–4,71 giây.
- Đồng thời có proposal cho ID 2, ID 6 và ID 8.

Đây là **screening trên output cũ**, không phải kết quả của pipeline V2 và không
phải nhãn thật. `artifact_2` và khoảng thời gian trên chỉ là giả thuyết cần đối chiếu.
Chưa gán các proposal còn lại thành false positive nếu chưa review đầy đủ.

### 2.3. Những thông tin cần bổ sung trong T00

- [ ] Xác định frame bắt đầu/kết thúc touch của ID 3 hoặc khoảng bất định của chúng.
- [ ] Xác nhận đúng hiện vật và bộ phận chạm: tượng, bệ, kính, khung hay phần khác.
- [ ] Đánh dấu bàn tay nếu nhìn rõ; cho phép `unknown` nếu bị che.
- [ ] Gán các sự kiện trong video 2 và 3; không mặc định chỉ có một touch mỗi video.
- [ ] Xác định ba clip có cùng camera/cùng đoạn nguồn hay không.
- [ ] Xác định phần nào của các clip đủ nhãn để tính precision và false alarm.

Không cần trả lời hết trước khi viết code hạ tầng. Nhưng thiếu các mục này thì
không được công bố recall/precision hoặc tuyên bố đã sửa đúng touch.

## 3. Định nghĩa đầu ra: tách đề xuất, bằng chứng và kết luận

### 3.1. Đơn vị xử lý

Một quan hệ được định danh bằng:

`(scene_id, clip_id, track_segment_id, hand_side, artifact_part_id)`.

- `track_id` chỉ có nghĩa trong một lần chạy/video; không phải danh tính khách.
- `track_segment_id` thay đổi khi mất association chắc chắn hoặc nghi ID switch.
- `artifact_id` ổn định trong scene; `artifact_part_id` phân biệt thân/bệ/kính.
- Hai tay được theo dõi riêng trước khi gộp event người–hiện vật.
- Không tự nối ID xuyên video/camera.

### 3.2. Các lớp trạng thái phải tách biệt

| Lớp | Trạng thái | Ý nghĩa |
|---|---|---|
| Đề xuất | `PAIR_PROPOSAL` | Cặp có khả năng tương tác; chỉ để chọn vùng/đoạn xử lý |
| Quyết định | `TOUCH_CANDIDATE` | Có bằng chứng tương tác đủ theo cấu hình để review |
| Quyết định | `NO_CONTACT_OBSERVED` | Trong đoạn đủ quan sát, không thấy bằng chứng tiếp xúc phù hợp |
| Quyết định | `UNKNOWN` | Không đủ hoặc mâu thuẫn bằng chứng; không đồng nghĩa không chạm |
| Hình học | `NOT_RUN / UNAVAILABLE / UNCALIBRATED / USABLE` | Chất lượng nhánh depth, không phải nhãn hành vi |
| Học normal | `SCORED / INELIGIBLE / DOMAIN_UNVALIDATED` | IF có thể chấm hợp lệ hay không |
| Review thủ công | `TOUCH / NO_TOUCH / UNCERTAIN / NOT_REVIEWED` | Nhãn người duyệt, độc lập với dự đoán |

Không đưa `TOUCH_CONFIRMED` vào output tự động của MVP. Không gán `NO_CONTACT_OBSERVED`
chỉ vì score thấp, không có proposal, thiếu model, mask rỗng hoặc phase lỗi.

### 3.3. Quy tắc hợp nhất bắt buộc

1. IF thấp không được xóa touch candidate có bằng chứng thị giác.
2. IF cao một mình không được tạo nhãn touch.
3. Depth không khả dụng không được chuyển candidate thành `NO_CONTACT_OBSERVED`.
4. Chỉ có chồng hình 2D thì giữ ở proposal/UNKNOWN nếu thiếu bằng chứng bổ sung.
5. Depth chưa hiệu chuẩn không được phủ quyết cứng bằng một ngưỡng mét cố định.
6. Bằng chứng mâu thuẫn phải được giữ trong event để review, không im lặng chọn một phía.
7. Mất track/tay/hiện vật không được giữ cảnh báo đỏ vô thời hạn.
8. Trạng thái của một người không bị chuyển thành UNKNOWN chỉ vì một hiện vật ở xa thiếu 3D.
9. Score quy tắc, confidence keypoint, anomaly percentile và xác suất contact là các đại lượng khác nhau.
   Không cộng trực tiếp rồi đặt tên tổng là xác suất.
10. Camera/hiện vật/pose chưa được kiểm tra miền dữ liệu phải có cờ `unvalidated`.

## 4. Sơ đồ hệ thống và quan hệ phụ thuộc

```text
Video gốc + timestamp + scene_id
    |
    +--> Luồng tracking: YOLOX-nano -> ByteTrack-DMA-LTC -> bbox + ID + chất lượng
    |                                                       |
    |                                               ánh xạ về ảnh gốc
    |                                                       |
    |                                               crop người / bàn tay
    |                                                       |
    |                                         pose cơ thể + tay + visibility
    |
    +--> Khung tham chiếu thật -> MM-GDINO + SAM2.1 -> danh mục hiện vật + mask
                                                            |
                                  ghép (ID, tay, hiện vật) <-+
                                             |
                              chuỗi ROI RGB + features + missing masks
                                             |
                                temporal evidence / state machine
                                             |
       DA-V2 + VGGT -> quality-gated geometry -+-> fusion -> events + evidence
                                             |
            normal-only IF -> context score -+

Ground truth/audit -> đánh giá offline; KHÔNG đi vào nhánh dự đoán
```

Ba nguyên tắc triển khai:

- Theo dõi người trước pose, nhưng chạy segmentation nền tĩnh độc lập với pose.
- Tạo bằng chứng thị giác trước nhánh depth để kiểm tra được MVP khi depth tắt.
- Model lớn chạy tuần tự; tái sử dụng cache phù hợp thay vì bắt chạy lại mọi phase.

## 5. Hợp đồng dữ liệu và cấu hình chung

### 5.1. Quy ước tọa độ và thời gian

- Tọa độ lưu canonical là pixel trên **frame nguồn sau khi đã xử lý orientation**.
- Bbox dùng `xyxy`, phải ghi rõ miền ảnh và quy ước biên; clamp trước crop.
- Mask lưu kèm độ phân giải và phép biến đổi về frame nguồn; resize mask bằng nearest-neighbor.
- Mỗi crop/letterbox/resize có ma trận biến đổi 3×3 và nghịch đảo được kiểm thử.
- Intrinsics phải đi cùng độ phân giải; đổi ảnh thì đổi intrinsics bằng đúng biến đổi ảnh.
- Dùng timestamp giải mã/PTS nếu có. Với CFR đã kiểm tra mới dùng `frame_idx / fps`.
- Không tự gán 24/30 FPS cho nguồn không đọc được thời gian.
- Video có VFR phải giữ timestamp từng frame; preview CFR nếu tạo lại phải có bảng ánh xạ.
- Đo vận tốc theo chênh lệch thời gian thật, không theo số thứ tự frame sau subsample.
- Mất keypoint/depth dùng `null`/mask hợp lệ; không dùng `(0,0)` làm điểm giả.

### 5.2. Các schema đề xuất

| Artifact dữ liệu | Các trường bắt buộc |
|---|---|
| `input_manifest.json` | run/version, clip/source hashes, scene/camera mapping, dimensions, timebase, config/model hashes |
| `frames_index.csv` | source frame index, timestamp, decode status, preview index nếu khác |
| `tracks.jsonl` | frame/time, track ID/segment, source bbox, detection confidence, measured/predicted, association quality |
| `poses.jsonl` | track reference, model/schema version, body keypoints, left/right hand, confidence, observed mask, crop transform |
| `artifacts.json` | scene/version, artifact/part ID, class, target policy, mask path, source reference frame, auto/manual provenance |
| `pairs.jsonl` | pair ID, track/tay/artifact, thời gian, proposal reason, distances, visibility, quality flags |
| `interaction_features.npz` | feature order/units, windows/timestamps, missing masks, metadata reference, schema version |
| `geometry_observations.jsonl` | pair/time, depth source, reference/time transform, support, calibration status, estimated gaps, failure reason |
| `anomaly_scores.csv` | model/feature version, eligible, score, calibration scope, percentile nếu hợp lệ, domain status |
| `events.jsonl` | event/pair/person/artifact IDs, interval, decision, evidence reasons, geometry/anomaly context, evidence paths |
| `phase_diagnostic.json` | execution/quality/observability status, metrics, reason counts, dependencies, duration, resources, output paths |

Định danh người ground truth là `gt_person_id` do người review tạo. Lưu ánh xạ
`baseline_run_id + baseline_track_id=3` sang `gt_person_id`; khi V2 đổi ID thì
đánh giá theo vị trí/quỹ đạo đã đối chiếu, không yêu cầu model phải xuất đúng số 3.

### 5.3. Trường tối thiểu của một sự kiện

```json
{
  "schema_version": "touch_events_v2",
  "event_id": "<run-scoped-event-id>",
  "scene_id": "<verified-scene>",
  "clip_id": "<clip>",
  "person_track_id": "<track>",
  "track_segment_id": "<track-segment>",
  "hand_side": "right",
  "artifact_id": "<artifact>",
  "artifact_part_id": "<part>",
  "start_s": null,
  "end_s": null,
  "first_alert_s": null,
  "decision": "TOUCH_CANDIDATE",
  "evidence_codes": [],
  "missing_evidence_codes": [],
  "geometry_status": "UNAVAILABLE",
  "estimated_gap_m": null,
  "anomaly_percentile": null,
  "review_label": "NOT_REVIEWED",
  "evidence_frame": "<relative-path>",
  "evidence_clip": "<relative-path>"
}
```

Đây là schema minh họa, không phải sự kiện thật. Event đã xuất phải có thời gian
hợp lệ và file bằng chứng tồn tại; `null` thời gian trong minh họa không được
chấp nhận ở validator vận hành.

### 5.4. Nhóm cấu hình

Tách cấu hình thành `runtime`, `input`, `tracking`, `pose`, `artifacts`, `pairing`,
`temporal`, `geometry`, `normal_model`, `fusion`, `evaluation`, `export`.

Các lựa chọn khởi đầu:

| Cấu hình | Lựa chọn đề xuất | Ghi chú |
|---|---|---|
| Tracking | YOLOX-nano + ByteTrack-DMA-LTC hiện có | Giữ detector preprocessing đã smoke-test |
| Pose baseline | YOLOv8n-pose trên crop ảnh gốc | Cô lập tác dụng của sửa độ phân giải trước khi đổi model |
| Pose thử nghiệm | RTMW whole-body 133 điểm | Dùng bbox ngoài; không chạy PoseTracker thay thế |
| Hand refinement | RTMPose hand 21 điểm, tùy chọn | Chỉ thêm nếu crop tay có đủ thông tin và cải thiện thực nghiệm |
| Segmentation | MM-GDINO-large + SAM2.1 B | Kế thừa config pedestal, ghi hash cấu hình thực thi |
| Geometry mode ban đầu | `shadow` hoặc `off` | Shadow vẫn tính/xuất nhưng không đổi quyết định |
| IF V1 | `legacy_shadow` | Chỉ chạy khi pipeline feature tương thích; không mặc định tương thích |
| IF V2 | `off` khi chưa có bundle đã đánh giá | Không chặn nhánh touch |
| Tiếp tục phase | Tự động | Không chờ người duyệt |
| Xử lý video | Một clip/lượt mặc định | Hỗ trợ batch ba clip là tiện ích sau, reset state từng clip |

Ngưỡng khoảng cách 2D nên chuẩn hóa theo kích thước người/tay và kiểm tra theo vùng
ảnh. Mọi giá trị ban đầu là hyperparameter trên tập development, không phải ngưỡng
vật lý. Không bê nguyên `12 px`, `0,15 m`, `0,12 s`, `0,8 m/s` sang tất cả model/
độ phân giải. Lưu toàn bộ cấu hình khi chạy và đóng băng trước đánh giá holdout.

## 6. Quy trình thực hiện tổng thể

Các mã **T00–T12 là hạng mục phát triển**, không phải số cell/phase của notebook.
Danh sách phase notebook cuối cùng được quy định ở mục 10.

```text
T00 Baseline + nhãn pilot
  -> T01 Runtime/input/coordinate contracts
  -> T02 Tracking + crop ảnh gốc
  -> T03 Hiện vật ổn định + T04 Pose/tay
  -> T05 Ghép cặp + features
  -> T06 Temporal evidence
  -> T09 Fusion/output bản thị giác
  -> T10 Đánh giá MVP

Sau khi bản thị giác kiểm tra được:
  T07 Depth/VGGT shadow -> T09/T10 ablation
  T08 IF normal-only   -> T09/T10 ablation
  T11 Notebook/package/resume -> bàn giao Kaggle

T12 Contact verifier/hiệu chuẩn 3D: nghiên cứu sau, không chặn MVP.
```

T03 và T04 độc lập về dữ liệu sau T02; có thể phát triển tách biệt nhưng chạy GPU
tuần tự để kiểm soát VRAM. Mọi checkbox triển khai dưới đây hiện là **chưa hoàn thành**.

## 7. Checklist triển khai chi tiết

### T00 — Đóng băng baseline và tạo nhãn đánh giá pilot

**Mục đích:** biết chính xác đang sửa lỗi gì và tránh chỉnh theo một frame duy nhất.

**Đầu vào:** ba video thật, ZIP V1, source/notebook hiện tại, xác nhận ID 3 của người dùng.

**Công việc:**

- [ ] Ghi hash video, ZIP, notebook, source, checkpoint và config baseline.
- [ ] Giữ bản V1 có thể tái chạy; workspace hiện tại không phải Git repository nên
  không dựa vào `git diff` như cơ chế backup duy nhất. Dùng snapshot có manifest,
  không ghi đè bản người dùng đang dùng.
- [ ] Trích các số liệu baseline ở mục 2 thành báo cáo máy đọc được; giữ đường dẫn
  thành viên trong ZIP để truy xuất nguồn từng số liệu.
- [ ] Tạo bảng scene/camera/clip mapping; không mặc định video `1.mp4` là camera 1.
- [ ] Tạo giao diện/bảng nhãn nhẹ: xem video, chọn người, hiện vật, interval và mức chắc chắn.
- [ ] Với ID 3, lưu ít nhất frame tham chiếu và bbox/quỹ đạo người thật để ánh xạ
  lại khi track ID của phiên bản mới thay đổi.
- [ ] Review toàn bộ ba clip, không chỉ top candidate; ghi cả các khoảng không có touch.
- [ ] Gán hard negative: đi ngang trước mask, đứng sau hiện vật, chỉ tay, chụp ảnh,
  tay bị che, tay gần nhưng không chạm, người chạm người, mất/đổi ID.
- [ ] Nếu onset/offset không rõ, ghi khoảng bất định và visibility; không ép thời gian chính xác giả.
- [ ] Nhãn chưa rõ để `UNCERTAIN`, không tự đẩy vào tập normal.
- [ ] Lập split theo scene/episode nguồn trước khi chỉnh ngưỡng; video 1 đã được dùng
  chẩn đoán nên coi là development, không coi là test mù.

**Đầu ra đề xuất:**

- `touch_v2/annotations/events.csv`: clip, gt_person, gt_artifact, thời gian,
  touch/no-touch/uncertain, bộ phận, tay, visibility, nguồn nhãn, reviewer.
- `touch_v2/annotations/person_anchors.json`: tham chiếu người thật và mapping baseline ID.
- `touch_v2/annotations/reviewed_intervals.csv`: phạm vi thời gian đã gán đầy đủ.
- `touch_v2/annotations/splits.json`: group/episode và vai trò train/calibration/test.
- `touch_v2/reports/baseline_v1.json` và `baseline_v1.md`.

**Đạt khi:** tất cả số liệu có nguồn; sự kiện ID 3 có định danh người ổn định;
không còn lẫn phỏng đoán `artifact_2 / 4,58–4,71 s` với ground truth. Những nhãn
còn thiếu được đánh dấu rõ, chưa sử dụng để công bố metric.

**Nếu chưa đủ nhãn:** vẫn làm T01–T06 và xuất preview, nhưng bảng evaluation phải
ghi `NOT_EVALUABLE` cho các metric cần ground truth.

### T01 — Runtime Kaggle, input và phép biến đổi ảnh

**Mục đích:** loại lỗi môi trường/đường dẫn/tọa độ trước khi đánh giá model.

**Đầu vào:** dataset Kaggle tại đường dẫn người dùng, video, các checkpoint hiện có.

**Công việc:**

- [ ] Giữ cell đầu in các file trong `/kaggle/input` như người dùng yêu cầu.
- [ ] Tạo một cell cấu hình rõ `DATASET_DIR`, `CLIP`, `SCENE_ID`, `RUN_MODE` và các nhánh bật/tắt.
- [ ] Resolve file theo logical key `detector_checkpoint`, không ràng buộc tên `best_ckpt.zip`.
- [ ] Hỗ trợ checkpoint là file `best_ckpt`, file `.pth/.pt/.zip` hoặc cây torch-save
  đã bung; khi dựng lại archive phải giữ đủ storage/metadata và kiểm tra strict load.
- [ ] Không truyền riêng `data.pkl` cho `torch.load`; không mở pickle/checkpoint không tin cậy.
- [ ] Nếu có nhiều checkpoint phù hợp, yêu cầu đường dẫn cấu hình chính xác; không tự lấy file đầu.
- [ ] Kiểm tra video decode, frame count, duration, FPS/timebase, orientation và frame hỏng.
- [ ] Tạo bộ đọc theo timestamp và truy xuất crop nguồn; không nạp mọi frame độ phân giải
  gốc của video dài vào RAM một lần.
- [ ] Lưu hai luồng: ảnh tracking/preview nhỏ và nguồn để crop. Kiểm thử round-trip tọa độ.
- [ ] Khóa commit/model revision, dependency đã smoke-test và checkpoint hash trong run manifest.
- [ ] Kiểm tra GPU, ONNX providers nếu dùng, precision, VRAM và dung lượng đĩa còn trống.
- [ ] Giữ PyTorch/CUDA/NumPy do runtime cung cấp khi tương thích; không downgrade mù
  hoặc cài package có thể phá ABI mà không có kiểm thử import/inference.
- [ ] Preflight cho từng model: import → load → inference một mẫu → shape/finite values.
- [ ] Nếu backend GPU không chạy, báo rõ; không âm thầm fallback CPU rồi báo đang tối ưu GPU.
- [ ] Tách dependency optional: `geometry=off` không cần tải VGGT/DA; `IF=off` không cần bundle IF.

**Đầu ra:** `environment.json`, `input_manifest.json`, `frames_index.csv`,
`input_preview.mp4`, `coordinate_roundtrip_report.json`, `model_preflight.json`.

**Đạt khi:** clip đọc được đầy đủ hoặc báo chính xác phạm vi lỗi; biến đổi ảnh/crop
khôi phục vị trí trong sai số kỹ thuật mục tiêu ≤ 1 pixel nguồn; đúng checkpoint
được load strict; không có package conflict chưa giải quyết; nguồn input read-only
không bị ghi sửa.

**Nếu fail:** dừng phần phụ thuộc input/model lỗi, vẫn lưu chẩn đoán và cho phép cell
export báo cáo thất bại. Không xuất kết quả hành vi giả để đủ số phase.

### T02 — Giữ tracking hiện tại, thêm chất lượng ID và crop ảnh gốc

**Mục đích:** mỗi pose/hand được gắn đúng người, ở độ phân giải nguồn.

**Đầu vào:** T01 và checkpoint YOLOX-nano + DMA + LTC + OSNet đang dùng.

**Công việc:**

- [ ] Tách detector/tracker adapter để trả schema chuẩn, không đổi association chỉ vì đổi pose.
- [ ] Giữ preprocessing BGR 0–255 đã được kiểm tra với checkpoint này; lưu chế độ thực dùng.
- [ ] Ghi model identity, strict-load report, detector size, score thresholds, DMA/LTC
  flags và checkpoint hashes; có flag bật nhưng không load phải báo lỗi/cảnh báo rõ.
- [ ] Phân biệt bbox đo từ detection với bbox dự đoán của tracker khi bị mất detection.
- [ ] Ghi tuổi track, khoảng mất detection và các chỉ báo nghi association sai nếu có.
- [ ] Cắt track segment khi continuity không đáng tin; không cho temporal evidence
  cộng dồn qua ID switch hoặc nối lại track không chắc chắn.
- [ ] Quy đổi bbox về ảnh gốc rồi crop với padding; giữ transform và padding trong record.
- [ ] Kiểm tra padding không cắt mất cánh tay duỗi ra; tránh crop quá rộng làm lẫn người bên cạnh.
- [ ] Cache crop theo source hash + frame + bbox/transform + phiên bản crop policy.
- [ ] Reset tracker/temporal state khi đổi clip hoặc bắt đầu run mới.
- [ ] Xuất preview toàn cảnh và bảng/crop theo ID, đặc biệt người tương ứng baseline ID 3.

**Đầu ra:** `detections.jsonl`, `tracks.jsonl`, `tracking_preview.mp4`,
`person_crop_preview.mp4`, `tracking_quality.json`, `crop_manifest.jsonl`.

**Đạt khi:** người chạm giữ association đúng trên khoảng sự kiện đã review; crop
thực sự đọc từ ảnh gốc; không gắn pose của người cạnh bên vào target ID; preview
vẫn nhận diện được rằng đang dùng tracker của người dùng.

**Không suy diễn:** 121/121 frame có ID không chứng minh không có ID switch; phải
review quỹ đạo/ảnh ở vùng tương tác. Chưa có nhãn MOT thì không báo HOTA/IDF1 giả.

### T03 — Danh mục hiện vật và mask ổn định theo scene

**Mục đích:** biết chính xác vùng/bộ phận nào đang được theo dõi, không phụ thuộc
một frame tham chiếu bị che nhiều.

**Đầu vào:** video nguồn, detections/tracks, segmentation config hiện có.

**Công việc:**

- [ ] Chạy MM-GDINO-large + SAM2.1 B trên các frame nguồn chọn lọc; ghi mọi resize/tile transform.
- [ ] Lưu bbox, mask, score, prompt/config hash, frame nguồn và nguyên nhân loại từng detection.
- [ ] Tạo gallery từng hiện vật để audit mask: tượng, bệ, tranh/khung, kính, đồ nội thất.
- [ ] Giữ `artifact_id` ổn định giữa các frame dựa trên mask/vị trí nền đã đăng ký.
- [ ] Phân biệt `artifact_id` với `artifact_part_id`; không gộp tượng và bệ đến mức
  mất thông tin người thực sự chạm phần nào.
- [ ] Có `target_policy`: bộ phận nào thuộc phạm vi cần review. Đề xuất ban đầu
  giữ thân hiện vật và bệ trong phạm vi candidate, gắn nhãn bộ phận riêng; đồ nội
  thất không tự trở thành mục tiêu. Kính/khung chưa rõ thì ghi chính sách chưa xác nhận.
- [ ] Chọn reference theo từng hiện vật: ưu tiên độ phân giải, mask rõ và ít che khuất
  tại vùng tương tác; không chỉ tối thiểu tổng diện tích bbox người toàn frame.
- [ ] Tách `reference_mask` với `visible_mask_t`; phần hiện vật bị che không được coi
  là bề mặt quan sát trực tiếp ở thời điểm đó.
- [ ] Nếu không có person mask đủ tốt, dùng bbox như vùng che khuất thận trọng và ghi
  uncertainty; không loại vĩnh viễn hiện vật khỏi registry vì bbox người chồng lên nó.
- [ ] So sánh background giữa các frame; đăng ký chuyển động ảnh nhỏ nếu đủ điểm tin cậy.
  Nếu camera đổi góc/cut, tạo scene mới hoặc đánh dấu cần dựng lại registry.
- [ ] Giữ heuristic material rescue nhưng đánh dấu nguồn `heuristic_rescue`; không
  coi màu sáng giống đá là xác nhận hiện vật.
- [ ] Cho phép sửa mask/target policy offline nếu cần, nhưng không bắt người dùng vẽ
  polygon mới hoặc duyệt từng phase ở mọi lần Run all.
- [ ] Không union mọi mask qua thời gian một cách mù vì sẽ phình sang người/nền.

**Đầu ra:** `artifacts.json`, `masks/`, `artifact_registry_preview.jpg`,
`artifact_visibility_preview.mp4`, `artifact_quality.csv`, `camera_registration.json`.

**Đạt khi:** hiện vật liên quan sự kiện thật có ID và mask được đối chiếu; biết mask
sai/đúng ở vùng tay tiếp cận; phần khuất và phần nhìn thấy không bị trộn; camera
dịch chuyển không làm giữ nguyên mask sai vị trí.

**Nếu fail:** xuất `MASK_UNAVAILABLE / MASK_UNVERIFIED / CAMERA_REGISTRATION_FAILED`
theo hiện vật/scene. Cho phép kiểm tra các hiện vật khác còn hợp lệ, không báo toàn
video không touch chỉ vì một hiện vật không segment được.

### T04 — Body pose và bàn tay trên crop nguồn

**Mục đích:** xác định tay thuộc ai và quan sát được điểm/vùng có khả năng tiếp xúc.

**Đầu vào:** tracks, crop nguồn, registry để chọn ROI cần xử lý kỹ.

**Thứ tự thử bắt buộc để biết cải thiện đến từ đâu:**

1. YOLOv8n-pose hiện có nhưng dùng crop ảnh gốc.
2. RTMW whole-body 133 điểm nhận bbox từ tracker.
3. Chỉ nếu cần: refinement bàn tay 21 điểm trên ROI bàn tay nguồn.

**Công việc:**

- [ ] Tạo `PoseBackend` chung với output body/hands, confidence, schema và transform.
- [ ] Dùng API pose mức thấp với bbox ngoài; không chạy tracker/detector thay thế ngầm.
- [ ] Smoke-test checkpoint RTMW cụ thể trên Kaggle, khóa revision/input size/backend
  sau khi chạy được; chưa ấn định số FPS hoặc bảo đảm vừa VRAM khi chưa đo.
- [ ] Lưu body 17 điểm chuẩn hóa nếu cần tương thích, đồng thời giữ nguyên toàn bộ
  schema 133 điểm và hand 21 điểm cho nhánh touch.
- [ ] Kiểm tra chỉ số trái/phải, thứ tự keypoint, BGR/RGB, letterbox và unflip.
- [ ] Xác định tay trái/phải theo người, không theo phía trái/phải màn hình.
- [ ] Dùng vai–khuỷu–cổ tay, vùng crop và continuity để loại bàn tay của người bên cạnh.
- [ ] Khi wrist thấp confidence, có thể chạy lại trên person/forearm ROI nguồn; không
  crop duy nhất quanh tọa độ cổ tay không đáng tin.
- [ ] Ước lượng visibility/occlusion và kích thước tay theo pixel nguồn; không dùng
  confidence cao như bảo đảm keypoint đúng.
- [ ] Ghi rõ tay quá nhỏ/blur/che; không dùng super-resolution để tạo bằng chứng tiếp xúc giả.
- [ ] Làm mượt ngắn hạn với timestamp; lưu cả raw và filtered. Không tạo positive
  observation ở frame chỉ có điểm dự đoán/nội suy.
- [ ] Xuất crop tay phóng to cho cả case tốt, xấu và camera/vùng ảnh khác nhau.
- [ ] Nếu đổi pose backend, đánh dấu bundle IF cũ `schema/domain incompatible` trừ khi
  giữ riêng pipeline legacy đúng đầu vào và đã revalidate.

**Đầu ra:** `poses.jsonl`, `pose_preview.mp4`, `hand_roi_preview.mp4`,
`pose_quality_by_track.csv`, `pose_backend_comparison.csv`.

**Đạt khi:** bàn tay của người chạm được gắn đúng ID/bên tay ở các frame có thể nhìn;
biết cụ thể frame nào không quan sát được; overlay source-coordinate đúng; có so
sánh YOLO ảnh nhỏ / YOLO crop gốc / RTMW crop gốc trên cùng mẫu đã review.

**Gate chọn model:** ưu tiên model tăng số observation tay đúng tại sự kiện và
hard negative, không chỉ tăng confidence hoặc số keypoint xuất ra. Nếu cả hai
không nhìn được tay, giữ UNKNOWN và xem lại giới hạn video thay vì đổi nhãn.

### T05 — Ghép cặp người–tay–hiện vật và trích đặc trưng tương tác

**Mục đích:** chuyển từ pose của một người sang quan hệ của người với một hiện vật.

**Đầu vào:** pose/hands, tracks, mask registry, timestamp.

**Công việc:**

- [ ] Tạo broad-phase pairing ưu tiên recall: vùng hoạt động bàn tay/cánh tay với
  bbox/mask hiện vật, có margin chuẩn hóa theo kích thước người.
- [ ] Nếu tay chưa quan sát được nhưng người ở vị trí có thể tương tác, giữ proposal
  chất lượng thấp để xem lại; không loại quá sớm chỉ vì wrist confidence thấp.
- [ ] Cho phép một tay gần nhiều hiện vật; lưu ambiguity thay vì ép chọn hiện vật gần nhất.
- [ ] Tính khoảng cách ảnh từ đầu ngón/lòng bàn tay/cổ tay đến mask và biên mask,
  phân biệt điểm nằm trong mask với điểm gần đường biên.
- [ ] Nếu có hand segmentation thì lưu vùng hand-mask; nếu chỉ keypoint/hull thì ghi
  đó là proxy, không gọi là mask bàn tay đo được.
- [ ] Giữ khoảng cách pixel nguồn và bản chuẩn hóa; ghi denominator để tái lập.
- [ ] Tính hướng cẳng tay, vận tốc tương đối tay–hiện vật trên ảnh, thay đổi khoảng
  cách, thời gian tiến tới/ở gần/rút ra và độ ổn định vùng tiếp cận.
- [ ] Tạo ROI RGB chứa tay và phần hiện vật liên quan, không chỉ crop tay tách ngữ cảnh.
- [ ] Ghi mức che khuất và visible surface tại điểm tiếp cận.
- [ ] Mọi feature có observed/valid mask; thiếu depth không điền khoảng cách bằng 0.
- [ ] Cache theo hash của pose, masks, pairing config; không tái dùng features của mask cũ.
- [ ] Audit cả cặp bị loại để đo candidate recall, không chỉ xem cặp được giữ.

**Feature groups dự kiến:**

| Nhóm | Ví dụ | Không được suy diễn |
|---|---|---|
| 2D relation | gap chuẩn hóa, khoảng cách đến biên, overlap proxy | Không đổi pixel sang mét bằng hệ số tùy ý |
| Pose | góc khuỷu/cẳng tay, vị trí tay tương đối người | Không suy ra ý định từ tư thế |
| Temporal | approach/withdraw, dwell, tốc độ ảnh, ổn định tương đối | Không bắt mọi touch phải dừng tay lâu |
| Appearance | ROI RGB vùng tiếp xúc và chuỗi crop | Không coi score model ảnh đơn là ground truth |
| Quality | kích thước tay, visibility, confidence, track/mask ambiguity | Chất lượng kém không tự là anomaly hành vi |
| Geometry optional | gap ước lượng, local support, calibration status | Không dùng nếu vùng tiếp xúc không quan sát được |

**Đầu ra:** `pairs.jsonl`, `pairing_preview.mp4`, `interaction_features.npz`,
`feature_schema.json`, `proposal_rejections.csv`.

**Đạt khi:** cặp GT ở sự kiện đủ quan sát đi được tới nhánh temporal; đo được
proposal recall riêng; số cặp/vùng xử lý tăng được kiểm soát và không loại candidate
chỉ để giảm thời gian chạy mà không ghi trade-off.

### T06 — Bằng chứng thời gian và state machine

**Mục đích:** phân biệt một frame gần/chồng hình với diễn biến tương tác đáng review.

**Đầu vào:** chuỗi cặp, ROI, features và quality masks; chưa cần depth/IF.

**Trạng thái nội bộ đề xuất:** `FAR → APPROACHING → NEAR → CONTACT_LIKE → WITHDRAWING`.
`OCCLUDED / LOST / AMBIGUOUS` là trạng thái chất lượng riêng, không phải contact.

**Đặc tả rule để bắt đầu cài đặt:**

- `Q`: track/tay/mask đủ quan sát, đúng hệ tọa độ và association không mơ hồ.
- `P`: khoảng cách ngón/lòng bàn tay tới mask đạt điều kiện proximity đã chuẩn hóa.
  Ví dụ feature `d_norm = d_px / max(person_bbox_height_px, 1)`; đây không phải mét.
- `A`: chuỗi khoảng cách giảm và hướng tay tiến về cùng vùng hiện vật.
- `H`: tay ở gần cùng vùng qua nhiều observation thật, hoặc trượt dọc vùng tiếp cận.
- `W`: khoảng cách tăng/rút tay sau một cực tiểu, có tính liên tục của cùng ID/tay.
- `C`: ảnh hưởng che khuất, đầu/cuối clip hoặc dữ liệu thiếu được ghi riêng.

| Motif | Bằng chứng cần thử | Điều không được bắt buộc |
|---|---|---|
| Tap ngắn | Q + P tại cực tiểu, A rồi W khi đủ context, tay hướng tới hiện vật | Dwell lâu hoặc vận tốc luôn thấp |
| Hold | Q + P + H trong một khoảng có observation hỗ trợ | Phải có cả approach và withdraw nếu clip đã cắt chúng |
| Slide | Q + P + diễn biến tay liên tục dọc vùng hiện vật | Tay đứng yên trên ảnh |
| Pass-by/overlap | Proximity không đi kèm quan hệ tay–hiện vật nhất quán | Không quyết định chỉ bằng một frame overlap |

Các tổ hợp này là **heuristic tạo candidate**, không chứng minh tiếp xúc vật lý.
Thời lượng, số observation hỗ trợ, threshold Q/P và cách nhận A/H/W phải nằm
trong config, được sweep trên development rồi khóa; không điều chỉnh theo ID.
Nếu Q không đủ thì giữ proposal/UNKNOWN. Nếu chỉ có P thì chưa tạo contact candidate.

Ở MVP, ROI RGB được lưu để review và phân tích lỗi; chưa có model contact RGB đã
huấn luyện riêng. Nếu thêm appearance score tự động, phải chỉ rõ model/đặc trưng
thực thi và đưa qua T12-A/T10, không để một biến `contact_score` không có nguồn.

**Công việc:**

- [ ] Cài baseline interpretable bằng rule/state machine trước, không gọi đây là model đã học touch.
- [ ] Tách motif chạm nhanh, giữ tay và trượt tay; không chỉ có một điều kiện tốc độ thấp + dwell.
- [ ] Proposal gần mask chỉ chuyển thành candidate khi có thêm bằng chứng temporal/
  hướng tay/appearance phù hợp theo policy đã hiệu chỉnh.
- [ ] Dùng thời gian thực cho duration, hysteresis, cooldown và gap tolerance.
- [ ] Chịu được gap ngắn bằng cách giữ context, nhưng không cộng frame thiếu quan sát
  vào số observation hỗ trợ contact.
- [ ] Không nối qua track segment, đổi bên tay, đổi artifact hoặc camera cut.
- [ ] Nếu clip bắt đầu/ kết thúc ngay trong tương tác, đánh dấu `left/right_censored`;
  không yêu cầu đủ approach và withdraw mới cho phép candidate.
- [ ] Nếu chuỗi chỉ cho thấy tay vào vùng che khuất rồi ra, lưu giả thuyết review và
  cờ thiếu bằng chứng, không gán đoạn khuất là contact đã quan sát.
- [ ] Phân biệt thời gian context clip với interval có bằng chứng; không mở rộng nhãn
  contact ra toàn bộ clip ± context.
- [ ] Chọn chế độ causal mặc định: chỉ dùng quá khứ/hiện tại để phát cảnh báo. Nếu chạy
  offline dùng frame tương lai để refine event thì lưu cả `first_alert_s` và thời gian
  refine, không báo độ trễ online từ kết quả nhìn trước tương lai.
- [ ] Log transition và rule fired/rejected cho từng cặp để giải thích UNKNOWN.
- [ ] Tạo mock tests cho tap, hold, slide, pass-by, occlusion, noisy wrist và ID switch.

**Đầu ra:** `temporal_observations.jsonl`, `visual_candidates.jsonl`,
`temporal_debug_preview.mp4`, `state_transitions.csv`.

**Đạt khi:** ca positive đã gán nhãn tạo candidate phù hợp hoặc báo được lý do quan
sát thất bại; không có stale alert; không dùng riêng overlap để báo chạm; các ca
hard negative đều có kết quả review và lý do rõ.

**Gate tiến độ:** hoàn thành bản T09/T10 chỉ bằng nhánh thị giác trước. Nếu không
thấy tay/mask đúng thì sửa T02–T05; chưa thêm depth để che lấp lỗi đầu vào.

### T07 — Tổ chức lại VGGT + Depth Anything V2 thành nhánh có điều kiện

**Mục đích:** đo giá trị thật của 3D và ngăn nhánh này làm mọi quan sát thành UNKNOWN.

**Đầu vào:** reference thật, mask/visibility, ROI tương tác, thông số camera/model.

**Công việc:**

- [ ] Có mode `off / shadow / advisory / validated`; khởi đầu `shadow`.
- [ ] Tách chất lượng model chạy được, có depth hợp lệ, có surface hỗ trợ, và đủ
  chính xác cho quyết định touch; không gộp thành một nhãn PASS.
- [ ] Chuẩn hóa cùng hệ tọa độ/camera/độ phân giải và thời điểm khi unproject.
- [ ] Thử baseline cùng nguồn depth cho tay và phần surface nhìn thấy trước khi
  ghép tay DA-V2 với mặt hiện vật VGGT; reference khác frame vẫn phải kiểm tra drift.
- [ ] Với reference bị người che, tìm observation sạch của chính hiện vật đó;
  không dùng depth người tại pixel che khuất làm bề mặt hiện vật.
- [ ] Thử DA-V2 độc lập và VGGT bổ sung trên cùng các ROI; giữ kết quả riêng để ablation.
- [ ] Thay bbox foreground thô bằng visibility/local masking phù hợp khi có dữ liệu;
  nếu phải dùng bbox thì báo tỷ lệ surface bị loại và chọn reference khác.
- [ ] Không lấy median depth patch chứa lẫn tay/nền/hiện vật rồi coi chắc là depth tay.
- [ ] Local quality gồm surface support đúng vùng, depth-edge ambiguity, temporal
  stability, alignment, reprojection/registration và độ nhỏ của tay.
- [ ] Hai model đồng thuận là feature chất lượng, không phải khoảng tin cậy đã hiệu chuẩn.
- [ ] Tách `surface_support_threshold`, `alignment_quality_threshold` và
  `contact_gap_threshold`; không dùng cùng 0,15 m cho mọi bước.
- [ ] Không lọc pixel bằng error rồi tính lại uncertainty trên chính phần đã chọn
  để tạo ra báo cáo tự chứng minh tốt; giữ tập audit/holdout và dropout count.
- [ ] Tính vận tốc 3D chỉ khi depth/timestamp ổn định; ghi uncertainty, không dùng
  jitter depth làm lý do mặc định bác bỏ chạm nhanh.
- [ ] Báo `UNAVAILABLE` theo từng cặp/hiện vật khi thiếu surface; không gộp vào
  `missing_pose_or_depth` nếu pose vẫn tốt.
- [ ] Cung cấp bảng ablation sự kiện bị mất/thêm khi bật geometry.
- [ ] Chỉ xem xét hard veto về khoảng cách khi có calibration và kiểm chứng phù
  hợp tại vùng tương tác; nếu chưa có thì chỉ advisory/review conflict.

**Đầu ra:** `geometry_quality_by_artifact.csv`, `geometry_observations.jsonl`,
`depth_preview.mp4`, `surface_support_preview.jpg`, `geometry_ablation.json`.

**Đạt khi:** bật/tắt geometry không phá nhánh thị giác; có báo cáo per-object,
per-pair; mọi gap mét đều gắn nhãn ước lượng/chưa hiệu chuẩn; chứng minh được phần
nào cải thiện bằng metric holdout trước khi nâng từ shadow sang advisory/validated.

**Nếu không có cải thiện:** vẫn có thể giữ visualization 3D cho nghiên cứu nhưng
tắt vai trò quyết định. Không tăng độ phức tạp chỉ vì đã tốn công tích hợp model.

### T08 — Đánh giá lại và nâng cấp Isolation Forest normal-only

**Mục đích:** giữ nhu cầu học không giám sát của người dùng, nhưng học đúng quan hệ
liên quan bài toán và không làm sai nghĩa touch.

**Đầu vào:** dữ liệu normal MuseumVisitors, metadata đã có, features/quality V2,
model IF legacy và evaluation của nó.

**Nhánh A — Kiểm tra tương thích IF cũ:**

- [ ] Ghi pose model/checkpoint, preprocessing, crop resolution, keypoint mapping,
  normalization, windowing, feature order, imputer/scaler và camera calibration scope.
- [ ] Không coi RTMW lấy ra 17 điểm là tự động tương đương YOLO 17 điểm đã train IF.
- [ ] Ngay cả cùng YOLO nhưng crop độ phân giải khác cũng phải đo drift lại.
- [ ] Nếu không tái lập đúng input legacy, tắt score hoặc đánh dấu incompatible;
  không nạp bundle rồi diễn giải percentile như cũ.
- [ ] Camera 3 chưa được tự gỡ cờ drift: evaluation cũ chỉ có 30 holdout windows
  với tỷ lệ vượt p99 khoảng 13,33%; chưa phải false-positive rate có nhãn.

**Nhánh B — Train IF V2 từ quan hệ tương tác normal:**

- [ ] Chọn các đoạn normal thực đã audit; loại đoạn positive/uncertain đã biết khỏi fit.
- [ ] Trích lại pose/hand theo đúng backend/crop policy V2, hoặc giữ nhánh body-only
  legacy riêng nếu chưa đủ ảnh tay để học relation.
- [ ] Muốn feature tay–hiện vật phải có registry phù hợp cho mỗi scene normal; không
  gán mask video touch sang ảnh MuseumVisitors khác góc nhìn.
- [ ] Kiểm tra sampling: MuseumVisitors có frame thưa/không đều; không nội suy thành
  chuyển động tần số cao rồi coi tương đương video touch 24 FPS.
- [ ] Dùng feature ổn định với sampling, hoặc thu thêm normal video cùng FPS/camera
  nếu cần học tap/slide ngắn. Tách metadata `temporal_resolution/domain`.
- [ ] Tạo window theo thời gian; không nối segment/gap/ID switch; giữ missing masks.
- [ ] Feature normal mới có pose tương đối, hướng/động học tay và quan hệ với hiện vật.
  Không đưa raw artifact/track ID như số thứ tự để model học thuộc đối tượng.
- [ ] Kiểm tra model không chỉ học người nhỏ/blur, visibility thấp hoặc camera ID.
- [ ] Chia train normal / calibration / holdout theo episode nguồn. Imputer, scaler
  và IF chỉ fit trên train; ngưỡng lấy trên calibration, không tune trên test.
- [ ] Nếu một camera không đủ episode để chia độc lập, dùng grouped evaluation
  phù hợp và ghi giới hạn; không cố chia các frame liền kề thành ba tập giả độc lập.
- [ ] Dùng hyperparameter search nhỏ, seed cố định và baseline đơn giản; không train
  mô hình sequence lớn chỉ vì tên model mạnh hơn.
- [ ] Audit top outlier cùng mẫu random/low-score; không chỉ chọn ví dụ đẹp.
- [ ] Đo drift, eligibility và alert rate theo camera/vùng/kích thước người.
- [ ] Sau khi khóa model, đo mức bổ sung cho touch trên nhãn event; IF thấp không veto.
- [ ] Lưu model bundle + feature schema + pose/config hashes + evaluation + train manifest.

**Đầu ra:** `normal_interaction_windows.npz`, `isolation_forest_interaction_v2.joblib`,
`feature_schema.json`, `evaluation.json`, `normal_outlier_review.html`, model card.

**Đạt khi:** train/inference dùng cùng feature extractor; không leakage; biết miền
nào đủ điều kiện chấm; kết quả trên normal chưa nhãn được gọi đúng là stability/
drift, không gọi là touch accuracy. Model có thể chạy shadow nếu chưa chứng minh
cải thiện việc xếp hạng review.

**Không có đủ normal relation data:** giữ nhánh touch hoạt động; bổ sung dữ liệu
normal có nguồn hợp lệ, ưu tiên camera/độ phân giải/FPS giống vận hành trước khi
tải dataset ngoài. Model học normal không tự học được tên lớp touch từ không nhãn.

### T09 — Fusion, sự kiện và bằng chứng xem lại

**Mục đích:** đầu ra có ý nghĩa, không trộn candidate và xác nhận.

**Đầu vào:** visual candidates, quality flags, optional geometry/IF scores.

**Công việc:**

- [ ] Tách `score_interactions()` khỏi `render_evidence()`; render không được thay đổi quyết định.
- [ ] Implement bảng quyết định ở mục 9, kèm test cho mỗi hàng.
- [ ] Lưu evidence riêng cho visual/temporal/geometry/anomaly; không cộng score khác
  đơn vị rồi gọi là probability.
- [ ] Tạo event theo cùng người–hiện vật; hai tay có thể được gộp hiển thị nhưng giữ
  observation riêng, không cộng observation trái/phải để vượt gate.
- [ ] Chỉ merge event qua gap ngắn nếu identity/pair còn chắc chắn; giữ đoạn UNKNOWN
  trong timeline, không gọi toàn gap là touch.
- [ ] Đóng event khi mất track, hết clip, scene đổi hoặc pair không còn; không kéo
  cảnh báo sang người khác tái sử dụng ID.
- [ ] Chọn frame bằng chứng có tay/mask/người rõ nhất; không bắt buộc là frame score cao nhất.
- [ ] Mỗi event tạo đúng một ảnh chính: mask + artifact ID, bbox/ID mọi người và highlight
  người nghi chạm. Clip cận cảnh tay là file bổ sung, không thay ảnh toàn cảnh chính.
- [ ] Mask đang bị người che không được tô lên người như thể đó là surface nhìn thấy;
  phần tham chiếu khuất phải dùng kiểu hiển thị khác hoặc chỉ contour.
- [ ] Ghi `CANDIDATE — cần review`, thời gian, lý do và chất lượng geometry; không vẽ
  khoảng cách mét khi nó null/không đủ hỗ trợ.
- [ ] Xuất preview toàn clip, từng event clip, CSV/JSONL, summary và trang HTML offline.
- [ ] Dù 0 candidate, vẫn xuất preview/quality và top ambiguous intervals để tìm false negative.
- [ ] Lưu nguồn clip/run/phase attempt trong mọi event để không mở nhầm kết quả cũ.

**Đầu ra:** `events.jsonl`, `events.csv`, `observations.csv`, `preview.mp4`,
`event_evidence/<event_id>.jpg`, `event_clips/<event_id>.mp4`, `review.html`, `summary.json`.

**Đạt khi:** chỉ nhìn một event là biết ai–hiện vật nào–khi nào–vì sao; video và JSON
khớp nhau; 0 event không bị diễn giải thành không có touch; người dùng nhìn được
UNKNOWN do lỗi nào mà không cần mở code.

### T10 — Evaluation, regression và quyết định giữ/bỏ thành phần

**Mục đích:** chứng minh cải thiện ở đúng sự kiện, không chỉ tạo nhiều cảnh báo hơn.

**Đầu vào:** nhãn T00, phiên bản/cấu hình frozen, output baseline và các biến thể.

**Công việc:**

- [ ] Xây event matcher một-một, khớp người thật và hiện vật trước khi khớp thời gian.
- [ ] Lưu mapping GT–prediction, matched/unmatched events và lý do mismatch.
- [ ] Ghi riêng proposal recall, candidate recall, precision trên vùng có nhãn đầy đủ,
  false alerts/phút, delay, unknown coverage và chất lượng identity.
- [ ] Positive không có candidate vì UNKNOWN vẫn tính là miss ở end-to-end recall;
  báo thêm recall trên subset đủ quan sát, không thay thế metric toàn bộ.
- [ ] Không tính unmatched candidate ngoài vùng đã review là false positive chắc chắn.
- [ ] Kiểm tra duplicate alerts; nhiều dự đoán cho cùng một GT không được tăng true positive.
- [ ] Báo cả outcome ID 3 và toàn bộ các ID khác; không chỉ chọn một ảnh đỏ làm nghiệm thu.
- [ ] Chạy ablation theo mục 11, khóa nhãn/splits/seed để so sánh công bằng.
- [ ] Phân loại lỗi: decode/ID/pose/mask/pairing/temporal/geometry/fusion/evaluation.
- [ ] Rà soát hiệu năng GPU, peak memory, thời gian từng phase và kích thước output.
- [ ] Nếu tập chỉ vài event, báo tử số/mẫu số; không tuyên bố tổng quát hóa bảo tàng
  hoặc khoảng tin cậy hẹp từ các frame tương quan.
- [ ] Mỗi vòng sửa có changelog: thay gì, kỳ vọng, kết quả, regression, quyết định.

**Đầu ra:** `metrics.json`, `event_matches.csv`, `error_cases.html`,
`ablation.csv`, `performance.json`, `acceptance_report.md`.

**Đạt khi:** có báo cáo tái lập được và cho biết chính xác V2 tốt/xấu hơn V1 ở đâu;
không rò nhãn test vào tune; mọi failure có test hoặc trường hợp review tương ứng.

### T11 — Notebook Kaggle, cache/resume và đóng gói

**Mục đích:** người dùng chạy được từ môi trường sạch, xem từng phase và tải đúng kết quả.

**Công việc:**

- [ ] Giữ source Python là nguồn chính, notebook được build; không sửa riêng notebook
  rồi bỏ quên source. Trong đợt đầu tạo target V2 riêng, không ghi đè V1.
- [ ] Dùng tên đề xuất `touch_interaction_v2_kaggle.ipynb`; tên folder cũ không bắt buộc đổi ngay.
- [ ] Cell đầu vẫn là cell Kaggle listing; cell tiếp theo là cấu hình dễ thấy.
- [ ] Một phase = gọi xử lý + hiện preview + metrics + diagnostic + tự chuyển tiếp.
- [ ] Không dùng `input()` hoặc chờ duyệt giữa phase; nhãn/audit là công việc offline.
- [ ] Artifact thiếu tùy chọn gây `SKIPPED/WARN`, không gây crash toàn notebook; lỗi
  core data khiến phase phụ thuộc blocked với thông báo rõ, không giả completed.
- [ ] Khi nhánh optional lỗi, lưu stack/error reason và tiếp tục nhánh visual nếu
  isolation đủ an toàn; không catch mọi exception rồi giả tạo score.
- [ ] Thêm cache manifest theo input/config/model/schema hashes; invalidate đúng descendants.
- [ ] Tái chạy pose không cần chạy lại segmentation tĩnh nếu dependency không đổi;
  đổi artifact mask phải invalidate pairing, features, geometry liên quan, events và evaluation.
- [ ] Resume sau reset đọc artifacts đã lưu và validate hash, không chỉ đọc cờ completed.
- [ ] Phase ghi output theo attempt mới và publish completion atomically; không dùng
  artifact đang viết dở hoặc trộn hai attempts.
- [ ] Check storage dựa trên dung lượng thực tế runtime; depth/crops nén tùy chọn,
  export mặc định ưu tiên preview/report thay vì tất cả tensor/debug frame.
- [ ] Đóng ZIP input với manifest/checksums, hướng dẫn file nào tải tự động và file nào bắt buộc upload.
- [ ] Đóng ZIP output với manifest, nhật ký attempts, outputs phase và review HTML; không
  kèm checkpoint lớn/private video gốc nếu không cần cho mục đích review đã chọn.
- [ ] Cell export vẫn dùng được khi một số phase failed/skipped; ghi `run_status=partial`
  và không dùng assert bắt buộc mọi phase optional completed mới tải được chẩn đoán.
- [ ] Smoke-test một clip trên Kaggle session sạch; sau đó cả ba clip, kiểm tra state reset.
- [ ] Cập nhật README đúng số lượng file; không lặp lại lỗi nói 8 file trong khi liệt kê 9.

**Đầu ra:** notebook V2 self-contained, source/builder/tests V2,
`kaggle_touch_v2_upload_bundle.zip`, `UPLOAD_MANIFEST.json`, `RUN_GUIDE.md`,
`KAGGLE_SMOKE_TEST.md` và output bundle mẫu.

**Đạt khi:** Add Input → chọn clip/scene → Run all → thấy đủ preview/diagnostic →
tải/mở HTML và event videos; có thể rerun hoặc resume mà không trộn dữ liệu;
không bắt cài/upload lại những file không thuộc nhánh đang bật.

### T12 — Các hướng nâng cấp sau MVP, không thực hiện mặc định

#### T12-A. Bộ xác minh contact chuyên biệt

- [ ] Chỉ mở nhánh khi T10 cho thấy hard negative còn nhiều dù tay/mask đã đúng.
- [ ] Thu tập ROI chuỗi người–hiện vật gồm positive và hard negative đã review.
- [ ] Khảo sát checkpoint contact có thể dùng, license, miền dữ liệu, input/output
  và chi phí; HOT là hướng nghiên cứu, không mặc định checkpoint đó hợp CCTV này.
- [ ] Thử frozen visual encoder + classifier nhỏ trước; chỉ thử TCN/GRU khi số
  episode độc lập và temporal labels đủ cho đánh giá.
- [ ] Input có ROI RGB, pose/hand relation và quality masks; tránh ép toàn frame vào
  model video lớn để học từ ba clip.
- [ ] Không dùng candidate rule hoặc pseudo-label chưa duyệt làm ground truth contact.
- [ ] Calibration và đánh giá trên held-out episode; giữ cơ chế abstain/UNKNOWN.
- [ ] Nếu không vượt baseline theo trade-off recall/false alert đã chốt, không tích hợp.

#### T12-B. Khoảng cách vật lý đáng tin hơn

- [ ] Chỉ mở khi yêu cầu thực sự là đo khoảng cách/thẩm định tiếp xúc, không chỉ review candidate.
- [ ] Thu thông số camera và đo đạc/kích thước thật có nguồn; không giả định mét.
- [ ] Với hiện vật có hình học đã biết, kiểm tra đăng ký model/scene vào camera và
  sai số ở chính vùng tay tiếp cận.
- [ ] Nếu có thêm camera đồng bộ/RGB-D, đánh giá calibration, synchronization và
  occlusion trước khi triangulate/đọc depth.
- [ ] Homography sàn chỉ dùng cho điểm trên mặt sàn; không dùng trực tiếp cho tay
  trên cao hoặc toàn bộ bề mặt tượng 3D.
- [ ] Báo error theo vùng/khoảng cách và tập giữ riêng; không suy ra độ chính xác
  centimet từ point cloud đẹp hoặc metric depth có đơn vị mét.

## 8. Bản đồ thay đổi code và cấu trúc file

### 8.1. Ánh xạ source hiện tại sang trách nhiệm mới

| Code hiện tại | Vấn đề/trách nhiệm mới | Hạng mục |
|---|---|---|
| `read_clip()` | Không làm mất nguồn gốc; reader streaming/timestamps + preview stream | T01 |
| `locate_input_file()`, `repack_extracted_torch_checkpoint()` | Logical checkpoint key, strict validation, nhánh optional | T01 |
| `track_people()` | Giữ detector/tracker, thêm canonical coords và track quality | T02 |
| `estimate_track_poses()` | Backend adapter + crop nguồn + body/hand schema | T04 |
| `artifact_masks()` | Per-artifact reference, registry, mask visibility, part IDs | T03 |
| `foreground_mask()` | Không đồng nhất bbox người với toàn bộ pixel bị che; phân biệt conservative exclusion | T03/T07 |
| `vggt_reference()` | Reference có provenance/visibility, geometry shadow | T07 |
| `depth_anything_v2_depth()` | Frame/time alignment, optional compute, quality + memory profiling | T07 |
| `fit_scale()`, `build_geometry()` | Tách support/alignment/contact thresholds; chất lượng per-pair | T07 |
| `wrist_point()` | Hand-surface sampling có quality, không trộn foreground/background | T07 |
| `temporal_gate()` | Nhiều motif temporal, missing-aware, không cộng dồn ID/tay khác nhau | T06 |
| `render_and_score()` | Tách scoring/fusion/events/render thành các phần test riêng | T09 |
| `_if_row_features()`, `_make_if_windows()` | Feature contract/version và normal relational extension | T08 |
| `score_live_if_windows()` | Compatibility checks và domain status trước scoring | T08 |
| `PHASE_NAMES`, `begin_phase()`, `review_phase()` | Dependency DAG, execution status, resume/cache, không chờ duyệt | T11 |
| `phase_diagnostics()` | Tách chạy thành công với quan sát đủ/thiếu và chất lượng task | T09/T11 |
| Cell export với assert đủ phase | Cho export partial run và optional SKIPPED | T11 |
| `test_notebook.py` | Bổ sung regression V2, giữ V1 tests cho baseline | T10 |
| `build_notebook.py` | Build target V2 từ source, không sửa notebook thủ công | T11 |

Các test V1 đang coi mọi chuyển động nhanh là không candidate phải được giữ trong
bộ baseline hoặc chỉnh giả thiết rõ: **fast pass-by** khác **fast tap**. Không biến
một test cũ thành ràng buộc kiến trúc sai cho V2.

### 8.2. Cấu trúc đề xuất

```text
colab_vggt_vda_touch/
├── PLAN_TOUCH_V2.md                    # tài liệu này — đã tạo
├── README.md                          # hướng dẫn V1 + đường dẫn plan V2
├── notebook_source.py                 # source V1, giữ làm baseline
├── touch_vggt_vda_bytetrack_kaggle.ipynb # notebook V1, chưa thay
└── touch_v2/                           # DỰ KIẾN, chưa tạo code ở lượt lập plan
    ├── notebook_source.py
    ├── build_notebook.py
    ├── touch_interaction_v2_kaggle.ipynb
    ├── training_notebook_source.py
    ├── train_normal_interaction_v2_kaggle.ipynb
    ├── configs/
    │   ├── default.yaml
    │   ├── model_registry.json
    │   └── scene_registry.json
    ├── annotations/
    │   ├── events.csv
    │   ├── person_anchors.json
    │   ├── reviewed_intervals.csv
    │   └── splits.json
    ├── tests/
    ├── reports/
    ├── UPLOAD_MANIFEST.json
    └── RUN_GUIDE.md
```

Trong source có thể phân nhóm hàm theo section; tránh sinh quá nhiều thư mục/package
khi chưa cần. Nếu tách module để test, builder phải đóng gói/nhúng đầy đủ để notebook
Kaggle vẫn self-contained. Không bắt người dùng tự sửa `sys.path` cho từng module.

Notebook inference chỉ nạp IF đã đánh giá. Notebook training riêng dùng khi đổi
feature/backend hoặc có dữ liệu normal mới; **không train lại IF mỗi lần đổi clip**.

## 9. Bảng quyết định fusion và chẩn đoán

### 9.1. Bảng quyết định tối thiểu

| Bằng chứng thị giác/thời gian | Geometry | IF | Đầu ra dự kiến |
|---|---|---|---|
| Tay/mask/ID đủ rõ, nhiều dấu hiệu contact-like theo policy | Thiếu/không tin cậy | Thấp hoặc thiếu | `TOUCH_CANDIDATE`, ghi thiếu 3D |
| Chỉ có gần/chồng mask 2D, không đủ dấu hiệu khác | Bất kỳ, chưa xác minh contact | Bất kỳ | Proposal/`UNKNOWN`, không tự nâng thành touch |
| Có bằng chứng tay tách rời hiện vật trong vùng đủ quan sát | Thiếu | Bất kỳ | `NO_CONTACT_OBSERVED` cho đoạn đó; IF cao là nhãn riêng |
| Visual candidate nhưng depth chưa hiệu chuẩn nói xa | Uncalibrated/conflict | Bất kỳ | Candidate gắn conflict hoặc `UNKNOWN` theo policy frozen; không hard veto bởi mét |
| Visual candidate nhưng geometry đã kiểm chứng cho thấy tách rời | Validated | Bất kỳ | Xử lý conflict theo policy đã đánh giá, có log; không tự coi model luôn đúng |
| Tay bị che/mask sai/ID lẫn | Bất kỳ | Cao | `UNKNOWN`; có thể thêm unusual-motion riêng nếu IF đủ điều kiện |
| Không có visual contact evidence | Không đủ | Cao | `UNUSUAL_MOTION_CANDIDATE` riêng hoặc score-only; không phải touch |
| Track đã mất/scene đã đổi | Bất kỳ | Bất kỳ | Đóng event, giữ evidence cũ; không chuyển cảnh báo sang ID khác |

Trong bản MVP, geometry `shadow` không thay đổi quyết định. Bảng trên chỉ mở
những hàng advisory/validated sau khi T10 xác nhận lợi ích và khóa policy.

### 9.2. Reason codes tối thiểu

- Input: `DECODE_FAILED`, `TIMESTAMP_INVALID`, `COORDINATE_MISMATCH`.
- Tracking: `TRACK_LOST`, `TRACK_PREDICTED_ONLY`, `IDENTITY_AMBIGUOUS`.
- Pose: `POSE_MISSING`, `HAND_TOO_SMALL`, `HAND_OCCLUDED`, `HAND_ASSOCIATION_AMBIGUOUS`.
- Hiện vật: `MASK_UNAVAILABLE`, `MASK_UNVERIFIED`, `PART_POLICY_UNRESOLVED`, `CAMERA_REGISTRATION_FAILED`.
- Pairing: `PAIR_AMBIGUOUS`, `ONLY_2D_OVERLAP`, `NO_RELIABLE_NEARBY_PAIR`.
- Temporal: `INSUFFICIENT_OBSERVED_SUPPORT`, `CONTEXT_CENSORED`, `LONG_GAP`, `PASS_BY_PATTERN`.
- Geometry: `SURFACE_MISSING`, `LOCAL_DEPTH_INVALID`, `DEPTH_EDGE_AMBIGUOUS`,
  `ALIGNMENT_FAILED`, `METRIC_UNCALIBRATED`, `VISUAL_GEOMETRY_CONFLICT`.
- IF: `FEATURE_SCHEMA_MISMATCH`, `POSE_DOMAIN_CHANGED`, `CAMERA_UNVALIDATED`, `WINDOW_INELIGIBLE`.

Một observation có thể có nhiều reason code; lưu nguyên nhân gốc và các hậu quả
riêng. Ví dụ pose tốt + surface rỗng phải là `SURFACE_MISSING`, không phải
`POSE_MISSING`. Reason counts cần mẫu số theo cặp đủ liên quan, tránh nhân với
mọi hiện vật xa rồi làm UNKNOWN rate gây hiểu nhầm.

### 9.3. Trạng thái phase tách ba chiều

| Trục | Giá trị minh họa | Câu hỏi trả lời |
|---|---|---|
| Execution | `SUCCESS / FAILED / SKIPPED / BLOCKED / PARTIAL` | Code có chạy và tạo output hợp lệ không? |
| Quality | `PASS / WARN / FAIL / NOT_EVALUATED` | Kiểm tra kỹ thuật/task đã đặt có đạt không? |
| Observability | `FULL / PARTIAL / NONE / NOT_APPLICABLE` | Những cặp cần xét có đủ dữ liệu quan sát không? |

Ví dụ phase geometry có thể `SUCCESS + FAIL + NONE`: code chạy xong nhưng không
có surface đủ dùng. Không hiển thị một chữ PASS làm người dùng hiểu nhầm đã đo
được touch. `review_status` của người dùng là field riêng, không được tự đổi
thành approved chỉ vì phase chạy xong.

## 10. Notebook V2: mỗi phase người dùng sẽ nhìn thấy gì?

Đây là **thứ tự chạy notebook đề xuất**, khác mã hạng mục T00–T12 ở mục 7.

| Phase | Nội dung | Kết quả hiển thị ngay | Nếu có vấn đề |
|---|---|---|---|
| 00 — Setup/input | Resolve input, models, clip, timebase, scene | Input inventory, video gốc/preview, resolution/FPS/GPU và preflight | Lỗi core thì dừng phụ thuộc, lưu diagnostic |
| 01 — Tracking | YOLOX-nano + ByteTrack-DMA-LTC | Video bbox/ID, track timeline, model/checkpoint identity | Chỉ rõ mất/đổi ID, không che bằng pose |
| 02 — Artifacts | Scene registry, per-object reference/mask | Gallery mask/part/ID, visibility và mask rejected | Hiện vật lỗi đánh dấu riêng |
| 03 — Pose/hands | Crop nguồn + body/hand pose | Toàn cảnh skeleton + crop tay, quality từng ID/bên tay | Tay nhỏ/khuất → reason, không keypoint giả |
| 04 — Pairing/features | Chọn người–tay–hiện vật | Video nối pair, gap ảnh, ambiguity, features plot | Xem ngay GT có bị loại từ bước này không |
| 05 — Visual temporal | Motif/state machine | Proposal/candidate timeline, transition reasons, clip minh họa | Chưa dùng depth/IF để người dùng thấy baseline riêng |
| 06 — Geometry optional | DA-V2, VGGT, local support | Depth, reference, surface coverage theo hiện vật, gaps có quality | Off/skipped/unavailable không làm mất visual output |
| 07 — IF optional | Model normal phù hợp schema | Score theo track/pair, eligible/domain status, top outliers | Không tương thích thì bỏ chấm, không tạo percentile giả |
| 08 — Fusion/events | Ghép evidence theo policy | Preview toàn video, event table, một ảnh/event, summary UNKNOWN | 0 event vẫn có chẩn đoán/top ambiguous clips |
| 09 — Evaluation | So với nhãn nếu đã có | Event matches, recall/precision/FP rate hợp lệ, missed events | Thiếu nhãn → NOT_EVALUABLE, không chặn inference |
| 10 — Export | HTML + ZIP + manifest | Đường dẫn file download và bảng nội dung bundle | Partial run vẫn export được báo cáo lỗi |

Mỗi phase phải hiển thị cùng một bố cục:

1. Mục tiêu phase và input attempt/hash đang dùng.
2. Preview trực quan; nếu không có thì lý do và artifact thay thế.
3. Bảng số liệu, execution/quality/observability.
4. Nguyên nhân lỗi phổ biến nhất kèm số lượng và mẫu số.
5. Danh sách file đã lưu và gợi ý phase cần sửa.
6. Thời gian chạy, peak resource khi đo được, trạng thái completed/skipped/failed.

Người dùng có thể Run all hoặc chạy từng cell; **không cần xác nhận thủ công để
chuyển phase**. Phase được phép tiếp tục khi WARN, không tiếp tục với dữ liệu core
hỏng. Một cell export độc lập phải luôn có thể chạy để tải diagnostic.

## 11. Thiết kế đánh giá và bảng ablation

### 11.1. Dataset tối thiểu để bắt đầu

- Ba clip touch thật hiện có: tập pilot, chưa đủ chứng minh khả năng tổng quát.
- Ground truth sự kiện và vùng thời gian đã review theo T00.
- Hard negative từ chính camera/scene: bắt buộc review, không chỉ lấy frame ngẫu nhiên
  mà không có người gần hiện vật.
- Normal MuseumVisitors cho nhánh học: phải kiểm tra sự phù hợp thời gian/scene/pose.

Nếu ba clip cùng một episode, giữ cùng group để tránh leakage. Nếu không đủ group
độc lập cho test, báo development/pilot và lên kế hoạch thu clip mới; không đổi
nhãn “validation” thành “test” để làm báo cáo đẹp.

Gợi ý mở rộng sau pilot: thu thêm nhiều đoạn normal và touch hợp lệ từ các lần
ghi khác nhau, ưu tiên đa dạng vị trí người, hiện vật, ánh sáng, tay trái/phải và
che khuất. Số lượng cần thiết phụ thuộc mục tiêu vận hành; không hứa một con số
clip cố định sẽ đủ train. Chỉ thu/chia sẻ dữ liệu khi có quyền phù hợp.

### 11.2. Event matching

1. Ánh xạ predicted track sang `gt_person_id` qua anchor/quỹ đạo/bbox đã kiểm tra.
2. Ánh xạ artifact/part đúng, không khớp chỉ vì cùng thời điểm.
3. Match một-một theo overlap thời gian và policy onset/offset đã khóa trên development.
4. Với tap ngắn, báo thêm hit trong khoảng thời gian GT có dung sai nhãn; không chỉ
   dùng temporal IoU có thể bất ổn khi event rất ngắn.
5. Dung sai phải xuất phát từ độ chính xác nhãn/FPS, không nới sau khi xem test misses.
6. Với event có onset/offset bất định, công bố policy strict/lenient hoặc báo riêng;
   không trộn vào một metric không giải thích được.
7. Prediction thừa sau match một-một là duplicate/unmatched và được audit theo vùng nhãn.

### 11.3. Metric bắt buộc

| Metric | Cách hiểu | Lưu ý |
|---|---|---|
| Proposal event recall | GT event có cặp tương ứng được đề xuất / GT events | Đánh giá riêng T05, chưa phải touch detector |
| Candidate event recall | GT events có candidate đúng người/hiện vật/thời gian / GT events | Positive bị UNKNOWN vẫn là miss end-to-end |
| Candidate precision | Candidate events đúng / candidate events trong vùng nhãn đầy đủ | Không dùng vùng chưa review làm negative |
| False alerts/phút hoặc giờ | False events / thời gian đánh giá có nhãn đầy đủ | Ghi thời lượng thật, không dùng số frame thay giờ |
| Normal false alerts | False events trong đoạn normal xác minh / phút normal | Báo riêng với alert sai người trong positive clip |
| Duplicate event rate | Các cảnh báo trùng cho cùng GT | Tránh mỗi frame một cảnh báo |
| Identity/artifact correctness | Tỷ lệ gắn đúng người/hiện vật trên các match | Không lấy recall thời gian thay thế |
| Alert delay | Thời gian phát cảnh báo đầu tiên − onset GT | Chế độ causal; offline refine báo riêng |
| Pair observability | Thời gian/cặp liên quan đủ tay/mask/track / tổng phạm vi xét | Không tính mọi người × mọi hiện vật xa |
| Unknown coverage | Tỷ lệ thiếu/mâu thuẫn evidence, phân theo reason | Không loại khỏi mẫu số recall toàn hệ thống |
| Hand quality | Điểm/tay đúng trên mẫu được review + coverage | Confidence không phải độ chính xác |
| Runtime/resources | Thời gian theo phase, peak VRAM/RAM, disk output | Đo trên đúng GPU và model phiên bản đã ghi |

Không đặt tên model score là confidence xác suất nếu chưa calibration. Không gọi
percentile 99 là “99% khả năng bất thường”. Với ít event, ưu tiên ghi `x/y` cùng
metric; các window/frame cùng episode không phải quan sát độc lập để tạo CI hẹp.

### 11.4. Các biến thể phải so sánh

| Mã | Cấu hình | Câu hỏi |
|---|---|---|
| A0 | V1 nguyên trạng | Baseline đang bỏ sót thế nào? |
| A1 | YOLO pose cũ + crop nguồn + cùng bộ screening 2D diagnostic | Sửa độ phân giải ảnh hưởng pose/proposal ra sao? |
| A2 | RTMW/crop nguồn + cùng screening như A1 | Pose tay mới cải thiện gì ngoài thay rule? |
| A3 | Pose đã chọn + registry tốt + pairing + temporal/fusion visual | Bản MVP giảm chồng hình giả mà giữ positive không? |
| A4 | A3 + DA-V2 shadow; sau đó advisory nếu đủ gate | Depth bổ sung có hữu ích không? |
| A5 | A4 + VGGT với cùng tập đánh giá | VGGT tăng giá trị hay chỉ tăng chi phí/dropout? |
| A6 | A3 + IF V2 shadow/review ranking | IF giúp ưu tiên review hay chỉ phản ánh quality? |
| A7 | Cấu hình kết hợp các thành phần đã vượt gate | Kết quả cuối tốt hơn ở trade-off đã chốt không? |

Trong A1/A2, screening là probe chung chứ không phải detector touch hoàn chỉnh.
Nếu thay cả model, input và threshold cùng lúc thì ghi là compound change, không
khẳng định improvement chỉ do model. Có thể thêm nhánh con khi cần cô lập một lỗi.

So sánh ở cùng test set, cách gán nhãn, event matcher và ngân sách false alert;
không chọn threshold trên test riêng cho mỗi cấu hình rồi gọi là so sánh công bằng.

### 11.5. Tiêu chí nghiệm thu theo mức

**Mức kỹ thuật — bắt buộc:**

- [ ] Tất cả tests core/schema/coordinate/state/cache/export qua.
- [ ] Notebook từ session Kaggle sạch chạy được clip đã chọn hoặc lỗi có chẩn đoán rõ.
- [ ] Tắt depth và IF vẫn có preview, visual candidates và export hợp lệ.
- [ ] Không có frame/event bị gắn sai do lỗi đổi tọa độ, timestamp hoặc ID reuse đã biết.

**Mức pilot — điều kiện để nói đã xử lý ca hiện tại:**

- [ ] Sự kiện ID 3 đã xác minh thời gian/hiện vật có candidate đúng người–hiện vật;
  nếu thiếu quan sát thì phải báo miss và giới hạn, không gọi hoàn thành nhận diện.
- [ ] Review đầy đủ các cảnh báo ID 2/6/8 và những ID khác, không chỉ xem ID 3.
- [ ] Có bảng sự kiện đúng/bỏ sót/báo nhầm/UNKNOWN trên cả ba clip có nhãn.
- [ ] Cải thiện không đến từ hardcode ID, artifact hoặc thời gian của video 1.
- [ ] Có so sánh cùng dữ liệu với V1 và báo giới hạn số mẫu.

**Mức vận hành rộng — chưa cam kết trong pilot:**

- Cần dữ liệu held-out độc lập và các mục tiêu được chốt trước:
  `TARGET_RECALL`, `MAX_FALSE_ALERTS_PER_HOUR`, `MAX_ALERT_DELAY`, `MIN_OBSERVABILITY`.
- Mục tiêu phải phù hợp camera và khả năng review thực tế; chưa có số đo thì không
  tự đặt “đạt 95%” hoặc tuyên bố sẵn sàng triển khai không người giám sát.
- Nếu không có tập test độc lập hoặc metric không đạt, tiếp tục ở chế độ nghiên cứu/review.

## 12. Danh sách kiểm thử cần bổ sung

### 12.1. Tests thuần CPU, không tải model

- [ ] Resize/letterbox/crop/rotation round-trip; ảnh ngang/dọc, bbox sát biên.
- [ ] BGR/RGB, normalize và decoder shape đúng cho từng adapter.
- [ ] CFR/VFR timestamps, frame drop, duplicate timestamp, clip kết thúc sớm.
- [ ] Mask resize không tạo soft-label giả; registry reset sau camera cut.
- [ ] Detector checkpoint dạng file, archive, folder đã bung và nhiều candidate ambiguous.
- [ ] Pose 17/133/21 schema mapping, trái/phải và missing-keypoint masks.
- [ ] Pose của người cạnh bên không được gắn vào target ID khi association mơ hồ.
- [ ] Pair ambiguity và no-surface không bị đổi thành no-contact.
- [ ] Fast tap có cơ chế được xét; fast pass-by không được loại/giữ chỉ bằng tốc độ.
- [ ] Giữ/trượt tay, context bị cắt đầu/cuối, một frame outlier và long gap.
- [ ] Hai tay/hiện vật/track khác nhau không cộng chung lịch sử.
- [ ] ID switch/ID reuse, scene change và reset video không giữ stale alert.
- [ ] IF score cao riêng không tạo touch; IF thấp không xóa touch candidate.
- [ ] Geometry off/missing/error không làm mất nhánh thị giác.
- [ ] Score/geometry NaN/null không biến thành 0 hoặc probability có vẻ hợp lệ.
- [ ] Feature schema mismatch làm IF ineligible, không chạy với feature order sai.
- [ ] Cache hash mismatch/incomplete output/upstream rerun invalidate đúng DAG.
- [ ] Export partial run và 0-event run vẫn tạo HTML/ZIP mở được.
- [ ] Event matcher một-một, duplicate prediction, GT uncertain và vùng chưa gán nhãn.
- [ ] Notebook schema/cú pháp mỗi cell hợp lệ, outputs lúc phát hành rỗng, không lộ đường dẫn riêng không cần thiết.

### 12.2. Tests tích hợp trên Kaggle GPU

- [ ] Model preflight cho YOLOX, DMA/LTC, pose, MM-GDINO/SAM và từng nhánh optional.
- [ ] Detector smoke-test có người và cảnh trống; ghi đầu vào/đầu ra và checkpoint.
- [ ] Batch crop hỗn hợp kích thước, tay nhỏ/người đông; GPU provider thực tế đúng.
- [ ] OOM được xử lý bằng giảm batch/chế độ đã định và log, không âm thầm đổi model.
- [ ] Một clip Run all, rerun một phase, đổi clip, reset runtime + resume.
- [ ] Mất network sau cache model vẫn chạy được những nhánh đã có đủ artifacts.
- [ ] Thiếu checkpoint optional và thiếu checkpoint core cho hành vi khác nhau đúng policy.
- [ ] Preview MP4 phát được, số frame/thời gian overlay khớp manifest, ảnh event tồn tại.
- [ ] Download bundle rồi mở `review.html` với relative paths không cần Kaggle session.

### 12.3. Regression thực trên video người dùng

- [ ] Người tương ứng ID 3 baseline: crop nguồn, hand association, artifact, event interval.
- [ ] Candidate chồng hình ID 2/6/8: đã gán nhãn trước khi kết luận TP/FP.
- [ ] `artifact_2` không có 3D: nhánh thị giác không bị chặn, output không giả gap mét.
- [ ] Trường hợp mask reference che lên người: overlay phân biệt visible/reference region.
- [ ] Frame wrist confidence thấp: không biến nội suy thành observation contact.
- [ ] So sánh full clip, không chỉ các crop được chọn sau khi thấy prediction.

Synthetic fixtures chỉ kiểm tra logic phần mềm, không phải dữ liệu chứng minh
độ chính xác nhận diện ngoài đời. Khi SciPy/NumPy local không tương thích, chạy
test trong môi trường được kiểm soát; không sửa toàn bộ môi trường máy để hoàn
thành một bài test hay bỏ test rồi đánh dấu PASS.

## 13. Dữ liệu/checkpoint cần upload và cách đóng gói

### 13.1. Notebook inference

Bundle V2 phải mô tả **logical inputs**, không ép mọi nhánh cần đủ mọi file.

| Thành phần | Nguồn hiện có/đề xuất | Bắt buộc khi nào? |
|---|---|---|
| Video chọn chạy | `1.mp4`, `2.mp4`, `3.mp4` ở workspace/dataset Kaggle | Cần clip được chọn; không cần cả ba để chạy một clip |
| Detector checkpoint | `best_ckpt`, archive hoặc folder torch-save hợp lệ | Luôn cần cho tracker hiện tại |
| DMA | `dma_gbm_mot17.gbm` | Khi dùng DMA theo cấu hình đã chốt |
| LTC | `ltc_motion_mot17.pth` | Khi dùng LTC theo cấu hình đã chốt |
| OSNet/ReID | Checkpoint đang tải trong pipeline | Khi tracker profile dùng ReID; pin/hash/cache rõ |
| SAM2.1 B | `sam2.1_b.pt` trong gói hiện tại | Khi dựng lại artifact registry |
| MM-GDINO large | Checkpoint/config được pipeline tải | Khi dựng registry; ghi revision và chế độ tải/offline |
| Pose | YOLO baseline hoặc RTMW checkpoint đã chọn | Cần backend đã cấu hình; không bắt cả hai trong inference cuối |
| Artifact registry | Masks + metadata của đúng scene | Tùy chọn cache; thay segmentation nếu đã hợp lệ |
| DA-V2 Metric Indoor | Checkpoint đúng phiên bản | Chỉ khi geometry bật |
| VGGT | Checkpoint đúng phiên bản/license | Chỉ khi nhánh VGGT bật |
| IF model + evaluation + schema | Bundle legacy tương thích hoặc IF V2 | Chỉ khi chấm anomaly |
| Nhãn events/splits | T00 | Chỉ cần evaluation, không được dùng làm input dự đoán |

Mỗi mục có trong `UPLOAD_MANIFEST.json`: logical name, filename/path, SHA256,
size, model/schema version, required-if, cách tải nếu thiếu, license/provenance.

Đường dẫn Kaggle do người dùng cung cấp hiện là
`/kaggle/input/datasets/nguyenvohoangkhang/anything1`. Notebook dùng cell cấu hình
và inventory để kiểm tra mount thực tế, không giả định mọi session giữ nguyên cấu trúc.

MVP nên ưu tiên chế độ Internet-on có checkpoint cache. Nếu muốn offline hoàn
toàn phải đóng đủ repos/dependencies/checkpoints theo manifest riêng; không tuyên
bố chạy offline chỉ vì đã kèm ba trọng số tracking.

### 13.2. Notebook training normal

Cần dataset normal gốc/frames có quyền sử dụng; `manifest.csv` và
`annotations.jsonl` chỉ là metadata, không thay thế ảnh. Nhánh relation còn cần
registry hiện vật phù hợp với từng scene và sampling đủ cho loại feature muốn học.

Notebook training xuất IF bundle + evaluation + schema. Notebook inference nạp
các file này sau khi train đạt gate; không mount toàn dataset normal mỗi lần suy luận.

### 13.3. Bundle output

```text
run_<id>_touch_v2_results.zip
├── run_manifest.json
├── phase_state.json
├── summary.json
├── review.html
├── phases/<phase>/<attempt>/...
├── events.jsonl
├── events.csv
├── event_evidence/...
├── event_clips/...
├── evaluation/...          # nếu có nhãn
└── diagnostics/...
```

Bundle phải có thể mở độc lập sau download; link tương đối không trỏ ngược vào
`/kaggle/working` của session đã mất. Không nhét checkpoint lớn hoặc video gốc
vào output mặc định khi chỉ cần review; export đầy đủ phục vụ tái lập là tùy chọn.
Clip/ảnh chứa người cần lưu/chia sẻ theo quyền dữ liệu, ưu tiên private dataset.

## 14. Hiệu năng, tài nguyên và khả năng chạy lại

### 14.1. Tối ưu theo phép đo

- Tracking có thể dùng ảnh nhỏ nhưng crop pose phải lấy ảnh gốc.
- Chạy pose/hand theo batch ROI vừa VRAM; batch size là cấu hình, ghi giá trị thực dùng.
- Segmentation nền tĩnh theo reference, không mặc định chạy MM-GDINO mọi frame.
- Chỉ tính depth cho tập thời gian/ROI đủ context cần thiết khi đã kiểm tra sampling;
  không bỏ frame làm mất touch ngắn để lấy FPS đẹp.
- Giải phóng model nặng giữa phase, không giữ MM-GDINO/SAM/RTMW/VGGT cùng trên GPU
  nếu không có nhu cầu và ngân sách VRAM đã đo.
- Không cố bật xFormers nếu môi trường không tương thích; dùng đường attention
  mà model hỗ trợ và smoke-test trên GPU thực tế, ghi hiệu năng.
- Lưu source frame/crop cache có giới hạn; ưu tiên streaming/chunking cho video dài.
- Nếu chạy chunk, overlap context phải đủ và merger event không duplicate/mất event
  ở biên chunk; reset chỉ theo scene/clip, không vô tình reset giữa một sự kiện.
- Đo wall time của từng phase; đo GPU latency có synchronization phù hợp và phân biệt
  cold start/download với inference. Không suy ra realtime từ thời gian một forward.

### 14.2. Cache và provenance

Cache key tối thiểu gồm input hash, frame/time selection, model/checkpoint hash,
config subset, schema/code version, dependency artifact hashes và geometry coordinate space.

Không cache bằng mỗi tên `1.mp4` hoặc `video_1`; file cùng tên khác nội dung là run khác.
Không xem `phase_state.completed=true` là đủ nếu output mất/hỏng hoặc config đã đổi.

Mục tiêu thay đổi ít nhất cần thiết:

- Đổi visualization: render lại, không infer model lại.
- Đổi temporal policy: reuse pose/mask/features nếu schema không đổi.
- Đổi mask: chạy lại pairing/geometry/events, không bắt buộc tracking lại.
- Đổi pose: chạy lại features/temporal/IF/events, không mặc định segmentation lại.
- Đổi input/scene hoặc coordinate policy: invalidate toàn bộ artifacts bị ảnh hưởng.

## 15. Rủi ro và cách xử lý

| Rủi ro | Biểu hiện | Hành động |
|---|---|---|
| Tay quá nhỏ ngay trong nguồn | Keypoint nhảy hoặc tự tin sai dù đã crop gốc | Đánh dấu hạn chế quan sát; cải thiện camera/dữ liệu nếu cần, không tạo chi tiết giả |
| Che khuất đúng lúc chạm | Có approach/withdraw nhưng không thấy tiếp xúc | Candidate review có cờ occlusion hoặc UNKNOWN; không tự xác nhận |
| Tượng/bệ bị nhận thành người | Bbox foreground xóa hiện vật, track tĩnh giả | Audit detector/registry, tách exclusion policy; không loại vật chỉ vì bbox |
| Mask dư/mask sai bộ phận | Người đi ngang cũng overlap | Sửa scene registry/part policy và đánh giá hard negatives |
| Pose/hand gắn nhầm người | Người cạnh bên bị tô nghi chạm | Association quality, track segments, reject ambiguity |
| Sai scale/tọa độ/camera | Overlay lệch, depth vô nghĩa | Contracts/tests trước mọi model comparison |
| 3D đồng thuận giả | Hai model sai cùng hướng | Không dùng agreement làm ground-truth uncertainty; calibration/ablation |
| Đổi pose phá IF | Score tăng khắp video | Compatibility gate, trích/train lại normal, shadow trước |
| Normal dataset khác FPS/scene | Model học sampling/camera chứ không tương tác | Group/domain evaluation và thêm normal cùng miền |
| Ít nhãn | Một thay đổi trông tốt trên một clip | Pilot labels, hard negatives, held-out episode, không tuyên bố accuracy lớn |
| Dependency Kaggle đổi | NumPy ABI/xFormers/ONNX provider lỗi | Pin cấu hình đã test, preflight, session sạch, không downgrade mù |
| Lỗi optional làm chết toàn run | Không tải được kết quả visual đã xong | Dependency isolation, partial export, lỗi giữ nguyên trong diagnostics |
| Chạy lại dùng cache cũ | Preview không phản ánh cấu hình hiện tại | Hash/DAG/attempt provenance, show active attempt |
| Nhiều candidate nhưng thiếu review | Không biết tăng recall hay tăng báo nhầm | Event budget, xếp hạng riêng, review lấy mẫu + đầy đủ vùng đánh giá |

## 16. Các mốc bàn giao và tiêu chí chuyển bước

### M0 — Baseline và khả năng đánh giá

Phạm vi: T00 + thiết kế schema/splits.

Nhận được: báo cáo V1, nhãn pilot/unknowns, anchor ID 3, kế hoạch split và metric.
Được chuyển sang code hạ tầng ngay; chưa công bố accuracy khi nhãn chưa đủ.

### M1 — Đầu vào, người, tay và hiện vật nhìn đúng

Phạm vi: T01–T04.

Nhận được: notebook/dev preview tracking, crop nguồn, masks/parts, body/hands, quality.
Gate: tọa độ/ID đúng; biết rõ tay và mask quan sát được tới đâu.

Nếu fail ở đây, sửa đầu vào/model/segmentation; chưa tune distance hoặc train IF
để bù cho dữ liệu sai.

### M2 — MVP phát hiện tương tác bằng hình ảnh

Phạm vi: T05–T06 + phần visual của T09/T10/T11.

Nhận được: notebook V2 chạy visual end-to-end, candidate/UNKNOWN, một ảnh/event,
HTML review, đánh giá pilot và error cases. Geometry/IF được phép off.
Gate: kết quả ID 3 và các hard negative được đối chiếu đầy đủ; không hardcode ca demo.

Đây là **mốc ưu tiên bàn giao đầu tiên để người dùng chạy Kaggle**.

### M3 — Kiểm chứng hai nhánh bổ sung

Phạm vi: T07 và T08, lặp lại T09/T10.

Nhận được: geometry shadow/ablation, IF compatible hoặc IF V2 bundle, so sánh
with/without trên cùng tập; mỗi thành phần có quyết định giữ/tắt và lý do.
Gate: không giảm recall ngoài trade-off đã chốt; không biến mọi case thành UNKNOWN.

### M4 — Đóng gói ổn định

Phạm vi: T11 hoàn chỉnh, regression T10 và danh sách kiểm thử ở mục 12.

Nhận được: notebook inference, notebook training nếu đã làm T08, bundle upload,
hướng dẫn, smoke-test Kaggle sạch, output mẫu và cache/resume được kiểm tra.
Gate: user chạy được không chỉnh code nội bộ và xem được phase nào có vấn đề.

### M5 — Nâng cấp sau pilot

Phạm vi: T12-A hoặc T12-B khi có lý do thực nghiệm/dữ liệu mới.

Không đưa vào đường găng của M2. Không hứa ngày hoàn thành hoặc thời gian GPU
trước khi có smoke-test và khối lượng dữ liệu thực tế.

## 17. Checklist sẵn sàng sử dụng

- [ ] Có artifact/person mapping và nhãn pilot đáng tin.
- [ ] Tracking đúng bộ YOLOX-nano + ByteTrack-DMA-LTC của người dùng.
- [ ] Pose/hand lấy ảnh gốc; transform, ID và bên tay đã kiểm tra.
- [ ] Scene registry/part policy có provenance, mask/visibility rõ.
- [ ] Proposal và candidate không bị đồng nhất.
- [ ] Temporal model xử lý tap/hold/slide/gap/censored context có test.
- [ ] Depth và IF không còn là hai cổng bắt buộc cho mọi touch.
- [ ] IF tương thích input hoặc được tắt với lý do rõ; camera chưa validate vẫn bị gắn cờ.
- [ ] UNKNOWN được bảo toàn, không biến thành normal để làm metric đẹp.
- [ ] Mỗi event có một ảnh tổng hợp đúng yêu cầu và clip xem lại.
- [ ] Mỗi phase có preview, quality, observability, reason và output path.
- [ ] Run all không chờ duyệt; rerun/resume không dùng artifacts cũ sai hash.
- [ ] Có kiểm tra positive và hard negative, không chỉ ID 3.
- [ ] Có split nguồn độc lập hoặc công bố rõ chỉ là development/pilot.
- [ ] Kaggle clean-session smoke-test qua; export partial/0-event hoạt động.
- [ ] Model/checkpoint/config/version/input hashes đầy đủ, bundle upload không thiếu file bắt buộc.
- [ ] README chỉ rõ những tính năng đã làm và còn planned; không tuyên bố đo touch vật lý đã hiệu chuẩn.

## 18. Nguồn kỹ thuật và ghi chú phạm vi

Nguồn đã đối chiếu trong lần nghiên cứu kiến trúc trước; khi triển khai phải kiểm
tra lại revision/checkpoint/license cụ thể và smoke-test môi trường Kaggle:

- [Tracker người dùng — ByteTrack-DMA-LTC](https://github.com/khangkaka066/ByteTrack-DMA-LTC-Motion-Tracker).
- [RTMLib](https://github.com/Tau-J/rtmlib): có API pose riêng, RTMW 133 điểm,
  RTMPose hand 21 điểm và backend ONNX Runtime; không bảo đảm sẵn hiệu năng trên clip này.
- [RTMW paper](https://arxiv.org/abs/2407.08634): hướng whole-body pose dùng để lựa
  chọn ứng viên thử nghiệm, không phải bằng chứng đã cải thiện CCTV bảo tàng.
- [VGGT](https://github.com/facebookresearch/vggt): camera/depth/geometry, có single-view
  inference; không phải chứng nhận scale hay độ chính xác centimet cho camera này.
- [Depth Anything V2 Metric](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth):
  checkpoint indoor/outdoor riêng; output metric do model dự đoán cần được đánh giá trong miền sử dụng.
- [HOT — Human-Object Contact](https://hot.is.tue.mpg.de/): hướng nghiên cứu contact
  ảnh; feasibility trên video này phải thử riêng trước khi đưa vào pipeline.

Tài liệu liên quan trong dự án:

- [README notebook hiện tại](README.md).
- [Plan lịch sử pose/rule/normal-only](../Museum/pose_rule_anomaly/PLAN.md).
- [Source notebook inference V1](notebook_source.py).
- [Source training IF hiện tại](train_normal_motion_if_notebook.py).
- [Tests V1](test_notebook.py).

**Kết luận triển khai:** làm M0 → M1 → M2 trước. Chỉ sau khi nhìn thấy và đánh giá
được đúng tương tác tay–hiện vật mới đo giá trị bổ sung của VGGT, Depth Anything
và IF ở M3. Mục tiêu là giảm bỏ sót mà không đánh đổi bằng những cảnh báo chồng
hình vô nghĩa, không phải làm một ID cố định xuất hiện màu đỏ.
