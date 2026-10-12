# Plan theo task — Pose, rule và học hành vi bình thường

> **Cập nhật 09/10/2026:** kế hoạch triển khai nhánh touch mới nằm tại
> [PLAN_TOUCH_V2.md](../../colab_vggt_vda_touch/PLAN_TOUCH_V2.md).
> Plan V2 ưu tiên tương tác tay–hiện vật trên crop ảnh gốc; depth/IF là nhánh hỗ trợ
> có điều kiện, kèm kiểm thử và đánh giá trên video touch thật. Đây là kế hoạch,
> chưa thay đổi notebook inference. Nội dung dưới đây được giữ làm lịch sử;
> các nhận định cũ về chưa có video positive và thử nghiệm UniDepth/VDA không mô tả
> đầy đủ trạng thái dữ liệu/pipeline hiện nay. Khi triển khai touch mới, ưu tiên plan V2.

**Mục tiêu:** chạy trên cả 3 camera MuseumVisitors, học mẫu chuyển động thường
gặp từ dữ liệu hiện có và kết hợp với rule không gian để sinh **ứng viên bất
thường có bằng chứng xem lại**. Hiện **không có dataset hành vi bất thường được
gán nhãn**; plan này không phụ thuộc vào positive dàn dựng hoặc ảnh AI.

> **Phạm vi vận hành hiện tại:** xuất `touch_candidate` từ rule không gian và
> `unusual_motion_candidate` từ baseline normal-only T06. Cả hai đều là ứng
> viên để review; T06 không tự gọi tên hay xác nhận một loại vi phạm cụ thể.
> `near_artwork` chỉ là trạng thái nội bộ. Các loại hành vi cụ thể khác chỉ
> được bật bằng rule có bằng chứng rõ hoặc model giám sát theo từng lớp sau khi
> có dataset phù hợp.

**Nguồn:** `Museum/dataset/MuseumVisitors/`. Plan tổng quan về dữ liệu nằm ở
`Museum/PLAN_BUILD_DATASET.md`; phần này là backlog triển khai phát hiện.

## 1. Phạm vi và ý nghĩa đầu ra

| Đầu ra | Cơ sở | Mức khẳng định |
|---|---|---|
| `cross_barrier_candidate` | Track đi từ phía được phép sang vùng cấm | Có bằng chứng hình học; kiểm tra lại nếu chân/dây bị che |
| `near_artwork` | Người/tay tiến sát ROI | Chưa chứng minh chạm |
| `touch_candidate` | Cổ tay gần ROI + tư thế tiếp cận + tín hiệu quan hệ ảnh nếu dùng RAM | Chỉ là ứng viên, cần xem clip |
| `unusual_motion_candidate` | Pose/quỹ đạo khác phân bố normal-only theo camera | Outlier cần review; không tự gọi tên chạy, ngã hay đánh nhau |
| `fall_candidate` | Chuyển động lạ + hông/thân hạ thấp và nằm kéo dài | Ứng viên để kiểm tra |
| `insufficient_evidence` | Pose, ROI hoặc track không đủ chất lượng | Không ép thành normal/abnormal |

Trong bảng trên, `touch_candidate` và `unusual_motion_candidate` là output
ứng viên cần review. Các loại khác vẫn là trạng thái trung gian hoặc hạng mục
tương lai.

Mỗi candidate sau khi hợp nhất phải có thêm **một ảnh bằng chứng duy nhất**
(`evidence_frame.jpg`). Ảnh này dùng đúng một frame đại diện và chồng đồng thời:

- mask hiện vật đã được duyệt cùng `artwork_id`;
- bbox và `track_id` của những người xuất hiện trong frame;
- `track_id` bị nghi có hành vi bất thường được làm nổi bật, kèm `event_id`,
  loại candidate và score chính.

`track_id` chỉ là mã theo dõi trong video, không phải danh tính thật của khách.
Nếu mask hiện vật chưa được duyệt hoặc người nghi vấn không có track đáng tin cậy,
ảnh phải ghi `insufficient_evidence`, không vẽ thông tin như thể đã xác nhận.

Không xuất `touch_confirmed` từ cổ tay 2D hoặc từ một ảnh RAM. Không dùng
`artwork_id`/hướng nhìn gốc làm nhãn chạm. `PostIncidentTrace`, ReID xuyên
camera và phân loại nhiều hành vi cụ thể là hạng mục sau MVP.

## 2. Những gì đã biết về gói dữ liệu

- 4.820 frame JPG 1280×800: Camera 1 = 3.977, Camera 2 = 473, Camera 3 = 370.
  `annotations.dat` có 53.631 dòng, 16 cột.
- Đường dẫn Camera 3 trong annotation ghi `4/...`, ảnh thật ở `frames/3/...`.
  Adapter sửa khi đọc; giữ nguyên dữ liệu gốc.
- Có 11 đoạn `frame_id` liên tục: 7/2/2 theo Camera 1/2/3. Hai đoạn Camera 1
  dài 3 và 4 frame không dùng để học chuỗi.
- Frame cùng giây xuất hiện ở cả ba camera trong hai khoảng ngắn, nhưng
  timestamp chỉ chính xác đến giây và camera không nhất thiết nhìn cùng khu vực.
- Số frame/giây biến thiên. Mọi cửa sổ thời gian phải nằm trong một segment;
  không nối qua gap và không coi 30 frame là 1 giây.
- Không có ground truth `touch`, `fall`, `run`, `fight` hay `cross_barrier`.
  Các ảnh ghép 2D trước đây đã bị reject (`use_for_training=false`).

## 3. Kiến trúc chốt

```text
Frame + camera_id
  → bbox người + track_id + pose 17 điểm + confidence
  → quality gate
  ├─ ROI/rule riêng từng camera
  │    ├─ barrier state machine
  │    └─ hand-to-artwork candidate state machine
  └─ cửa sổ pose + quỹ đạo của một track
       → mô hình normal-only dùng chung
       → anomaly_score
  → hợp nhất theo loại sự kiện + chất lượng bằng chứng
  → events.jsonl + một evidence_frame.jpg tổng hợp cho mỗi event

Tùy chọn: RAM nhận crop + vùng người/hiện vật của touch_candidate
          → relation_score trên ảnh → thêm vào bằng chứng, không tự xác nhận
```

Model normal-only dùng chung ba camera; ROI và ngưỡng vận hành theo camera.
Nếu model pose hoạt động kém trên người nhỏ/che khuất, tiếp tục dùng bbox và
track cho những rule còn đủ bằng chứng. Không train ba LSTM riêng trên số
frame rất ít của Camera 2/3.

## 4. Backlog theo phần và thứ tự phụ thuộc

### T01 — Chuẩn hóa dữ liệu gốc

**Làm:** parser `annotations.dat` 16 cột, sửa path Camera 3 khi đọc; manifest
`camera_id, segment_id, frame_id, timestamp_source, image_path, visitor_id`;
kiểm tra ảnh tham chiếu, bbox và trùng ID trong từng segment.

**Đầu ra:** `outputs/manifest.csv`, `outputs/annotations.jsonl`, báo cáo
validation. **Đạt khi:** mọi annotation trỏ tới ảnh tồn tại; không mất cột
gốc; segment và gap được ghi rõ.

### T02 — Pose + tracking + kiểm tra chất lượng

**Phụ thuộc:** T01.

**Làm:** chạy pose pretrained và gắn với track người trong annotation gốc
hoặc tracker mới; xuất 17 keypoint `(x,y,confidence)` cùng bbox. Tạo overlay
cho mỗi camera, lấy mẫu người gần/xa, đông/thưa, rõ/bị che. Đo tỉ lệ thấy
được cổ tay, cổ chân và tỉ lệ track bị đứt/đổi ID.

**Đầu ra:** `outputs/poses.jsonl`, preview, `reports/pose_quality.md`.
**Đạt khi:** biết rõ camera/vùng nào dùng được rule cổ tay, cổ chân; frame
thiếu pose có `quality_status` thay vì tọa độ `(0,0)` giả.

### T03 — Cấu hình không gian theo camera

**Phụ thuộc:** T01; có thể tiến hành song song với T02.

**Làm:** gán polygon `artwork_roi`, `pedestal_roi`, `restricted_zone`, đường
`barrier_line` và vùng sàn quan sát được trên keyframe thật từng camera.
Ghi `source_frame`, kích thước ảnh, người gán và chú thích vùng bị che.

**Đầu ra:** `configs/cam1_roi.json`, `cam2_roi.json`, `cam3_roi.json` cùng
overlay kiểm tra. **Đạt khi:** mọi ROI bám đúng background gốc và có ít
nhất một khu vực có thể thực sự đánh giá cho từng rule được bật.

### T04 — Rule theo trạng thái, có mức không chắc chắn

**Phụ thuộc:** T02, T03.

**Vượt rào:** xác định phía của track trước/sau đường chắn; kết hợp vị trí
chân khi đủ confidence, tâm/bottom bbox, vùng sàn và dwell/hysteresis.
Chuỗi `allowed → approaching → crossing → restricted` mới sinh candidate.

**Gần hiện vật:** khoảng cách cổ tay tới **bề mặt khả dụng** của ROI; xét
hướng tay, thay đổi khoảng cách theo thời gian, chất lượng pose và che khuất.
Chuỗi `far → approach → near_artwork → touch_candidate/uncertain → withdraw`.
Một điểm cổ tay nằm trong polygon không đủ để gọi là chạm.

**Đầu ra:** `outputs/rule_candidates.jsonl`, preview trước/sau sự kiện.
**Đạt khi:** review được các hard negative như chỉ tay, chụp ảnh, người đứng
sau rào và trường hợp chồng hình do phối cảnh.

### T05 — Xây tập học hành vi thường gặp, không cần nhãn bất thường

**Phụ thuộc:** T02.

**Làm:** tạo cửa sổ 2–3 giây trong từng track/segment, resample về trục thời
gian chung, lưu confidence và mask cho keypoint thiếu. Loại cửa sổ có gap,
ID switch hoặc pose không đủ chất lượng. Chọn train từ các đoạn quan sát
khách tham quan thường ngày; xem nhanh mẫu ngẫu nhiên và các cửa sổ ngoại
lai để giảm nhiễm sự kiện bất thường. Đây là **kiểm tra dữ liệu tối thiểu**,
không yêu cầu gán nhãn toàn bộ dataset.

**Đặc trưng:** pose tương đối với tâm hông/tâm bbox, chuẩn hóa kích thước;
góc thân, độ cao hông, tỉ lệ bbox; vận tốc và đổi hướng của tâm bbox/tay/chân.
Giữ tọa độ/zone tuyệt đối riêng cho rule. Khi confidence thấp, bỏ qua hoặc
mask keypoint, không thay bằng số 0.

**Đầu ra:** `outputs/windows.parquet` hoặc `.npz`, mô tả feature và danh sách
cửa sổ bị loại. **Đạt khi:** không có cửa sổ vượt segment/gap; mẫu của ba
camera ở cùng schema và cùng đơn vị thời gian.

### T06 — Baseline học không giám sát/normal-only

**Phụ thuộc:** T05.

**Làm:** train một model nhỏ trên đặc trưng cửa sổ (ưu tiên Isolation Forest
hoặc khoảng cách kNN tới ngân hàng normal). Tập MuseumVisitors chưa được
gán nhãn hoàn chỉnh nên coi training set là **mostly normal, có thể nhiễm**;
kiểm tra top outlier và train lại nếu thấy lỗi pose/ID switch chiếm ưu thế.
Chỉ thử LSTM-Autoencoder, pose prediction hoặc STG-NF sau khi baseline nhỏ
có lỗi cụ thể cần khắc phục và lượng chuỗi hợp lệ đủ dùng.

**Đầu ra:** model, `outputs/anomaly_scores.jsonl`, đồ thị score theo track.
**Đạt khi:** điểm cao không chỉ phản ánh pose hỏng, người ở xa hay camera ID;
người xem có thể mở các cửa sổ top score và hiểu nguyên nhân.

### T07 — Hợp nhất quyết định và ngưỡng theo camera

**Phụ thuộc:** T04, T06.

**Làm:** rule crossing tạo `cross_barrier_candidate`; rule tay tạo
`near_artwork`/`touch_candidate`; model tạo `unusual_motion_candidate`; hình học
thân/hông kéo dài hỗ trợ `fall_candidate`. Áp dụng debounce và cooldown
theo track. Hiệu chỉnh ngưỡng từ phân bố score mỗi camera và khả năng
review cảnh báo thực tế. Không dùng phép `rule OR anomaly` chung cho mọi lớp.
Trong bản hiện tại, T06 dùng percentile theo camera (mặc định 0.99) để tạo
ứng viên chuyển động lạ; score chỉ bổ sung ngữ cảnh cho `touch_candidate`,
không tự nâng nó thành xác nhận chạm hay vi phạm.

**Đầu ra:** `outputs/events.jsonl` gồm `camera_id, segment_id, track_id,
start/end, event_type, rule_score, anomaly_score, pose_quality, evidence,
review_status`; và `outputs/event_evidence/<event_id>.jpg`. Mỗi JPG là **một
ảnh tổng hợp duy nhất**, hiển thị mask + `artwork_id`, mọi người + `track_id`,
và làm nổi bật `track_id` nghi bất thường cùng loại event/score. Không tạo ba
ảnh rời bắt người xem tự đối chiếu. **Đạt khi:** mỗi event có ảnh bằng chứng
tự đủ nghĩa và giải thích rule/score nào đã kích hoạt; mask chưa duyệt phải
được đánh dấu rõ và không được trình bày như ground truth.

### T08 — Review và đánh giá khi chưa có ground truth bất thường

**Phụ thuộc:** T07.

**Làm:** chọn mẫu ngẫu nhiên từ thời gian có frame và review toàn bộ top
candidate. Ghi `valid_candidate`, `false_alarm`, `uncertain`, `pose_failure`
cho **tập audit nhỏ**; không giả vờ đó là nhãn gốc của dataset. Báo:

- số candidate/giờ quan sát theo từng camera, kèm số phút thật có frame;
- tỉ lệ false alarm **trên tập candidate đã review**;
- độ phủ: % track/cửa sổ đủ pose, % `insufficient_evidence`;
- các ví dụ lỗi do che khuất, FPS thấp, ROI hoặc ID switch.

Giao diện review ưu tiên hiển thị trực tiếp `event_evidence/<event_id>.jpg` để
người duyệt thấy hiện vật, người liên quan và lý do cảnh báo trong cùng một ảnh.
Nếu một event có nhiều frame, chọn frame có tổng `rule_score`, chất lượng pose
và độ rõ mask tốt nhất; thời gian bắt đầu/kết thúc vẫn giữ trong `events.jsonl`.

Không báo event recall/precision tổng thể hay khả năng phát hiện `touch`
chắc chắn khi chưa có positive ground truth. Split theo segment/episode,
không chia ngẫu nhiên theo frame. Với số segment ít, gọi kết quả là pilot.

### T09 — Thử RAM (tùy chọn, chỉ cho quan hệ ảnh)

**Phụ thuộc:** T03, T04; không chặn T05–T08.

Ở đây RAM là [Relate-Anything-Model của EvolvingLMMs-Lab](https://github.com/EvolvingLMMs-Lab/RelateAnything),
kết hợp SAM để lấy mask và dự đoán quan hệ giữa cặp vùng trong **một ảnh**.
Kiểm tra checkpoint, cách chạy, nhãn quan hệ thực có, license và tài nguyên
máy trước khi tích hợp. Có dự án khác cùng tên
[RelateAnything open-vocabulary](https://github.com/Maelic/RelateAnything)
với API và điều kiện sử dụng khác; không trộn hai model dưới một tên `RAM`.

**Thử nghiệm:** lấy frame từ `touch_candidate` và các hard negative được
review (`pointing`, `near_without_contact`, `occluded`). Đưa crop ảnh, vùng
người và mask/box tác phẩm vào RAM; lưu top quan hệ và score nếu API có.
So sánh **cùng tập frame** khi không/có RAM:

1. RAM có nhận ra quan hệ liên quan `touching/contact` trong vocabulary và
   tại độ phân giải người thực tế của từng camera không?
2. Nó có giảm false alarm trên hard negative mà không làm mất candidate
   hợp lệ không? Nếu không có positive xác minh, chỉ báo cáo tính khả thi
   và ví dụ, không tuyên bố cải thiện recall.
3. Thời gian chạy/memory có phù hợp khi chỉ chạy trên candidate không?

**Đầu ra:** `reports/ram_feasibility.md` và score/quan hệ theo frame.
**Gate:** nếu checkpoint khó chạy, predicate không phù hợp hoặc kết quả
không đáng tin trên ảnh CCTV, bỏ RAM khỏi pipeline vận hành. Dù đạt gate,
RAM vẫn là **bằng chứng phụ**; quan hệ dự đoán trên ảnh tĩnh không xác nhận
tiếp xúc vật lý hay diễn biến thời gian.

### T10 — Mở rộng khi sau này có ít nhãn được xác minh

Đây là nhánh **chưa cần làm hiện tại**. Khi có video positive thật hoặc một
tập candidate do người xác minh, giữ nguyên feature/encoder và thêm classifier
nhỏ hoặc phương pháp semi-supervised dùng ít nhãn + nhiều cửa sổ chưa nhãn.
Không dùng rule cổ tay, RAM hoặc ảnh ghép AI làm pseudo-label `touch=true`
chưa qua review. Giữ mọi biến thể một episode trong cùng split.

### T11 — Hành vi mới và dữ liệu ngoài (supervised theo từng lớp)

Không dùng một điểm anomaly chung để tự đặt tên hành vi. Mỗi nhánh mới có gate
riêng và chỉ được bật sau đánh giá chéo miền trên camera bảo tàng:

- `fall_candidate`: ưu tiên dữ liệu fall chuyên biệt; CHAD bổ sung pose CCTV.
- `fight_candidate`: RWF-2000 và phần fighting của CHAD/UCF-Crime.
- `running_candidate`: CHAD hoặc VIRAT, sau đó hiệu chỉnh tốc độ theo phối cảnh.
- `theft_or_vandalism_candidate`: UCF-Crime làm pretrain yếu; bắt buộc có mẫu
  xác minh tại bảo tàng vì nhãn video-level và lệch miền lớn.
- `cross_barrier_candidate` và `loitering_candidate`: ưu tiên state machine
  theo ROI/thời gian; không cần học anomaly chung.
- `object_removed_candidate`: so sánh mask/background hiện vật theo thời gian;
  cần cảnh quay trước/sau hoặc positive dàn dựng được cho phép.

Chi tiết nguồn, license và giới hạn nằm ở
`reports/external_behavior_datasets.md`. Không tải toàn bộ dataset lớn vào máy
local khi chưa kiểm tra dung lượng và điều kiện sử dụng.

### T12 — Thử VGGT + Video Depth Anything cho hình học touch

Notebook thử nghiệm độc lập:
`touch_vggt_video_depth_anything_experiment_colab.ipynb`.
Video Depth Anything metric ViT-S tạo depth liên tục; VGGT ước lượng camera,
depth và point map từ các khung hình lấy mẫu. Trước mắt chỉ xuất NPZ và depth
preview trên ba video touch để kiểm tra độ ổn định, chưa thay nhánh UniDepth
và chưa được dùng để khẳng định `touch_candidate`.

**Gate tích hợp:** (1) xác nhận thứ tự/thời điểm depth map trùng frame nguồn;
(2) duyệt trực quan mask hiện vật, pose cổ tay và điểm 3D; (3) so sánh khoảng
cách tay–mask của VGGT/VDA với UniDepth trên cùng frame và các đoạn touch thật;
(4) hiệu chuẩn scale bằng kích thước/đo đạc thật tại camera. Chỉ khi qua các
gate này mới dùng khoảng cách 3D làm một feature cho state machine touch.
Depth đơn camera là ước lượng; không đặt ngưỡng mét như sự thật nếu chưa đo.
Tuân thủ license checkpoint VGGT; model gốc không dành cho mục đích thương mại.

## 5. Thứ tự chạy gọn

```text
T01 → T02 ───────→ T05 → T06 ──┐
  └→ T03 → T04 ────────────────┴→ T07 → T08
             └→ T09 (thử riêng, tùy chọn)
T10: sau T08, khi có nhãn thật
T12: thử depth/geometries độc lập; chỉ tích hợp sau gate đo kiểm
```

T01–T08 là đường hoàn thành MVP. T09 chỉ được giữ lại nếu có bằng chứng
thực nghiệm trên ảnh MuseumVisitors. T10 là bước nâng cấp, không phải điều
kiện để demo hệ thống hiện tại. T12 là thử nghiệm độc lập; chưa làm thay đổi
đầu ra touch cho đến khi qua các gate hiệu chuẩn và đánh giá.

## 6. Cấu trúc triển khai dự kiến

```text
Museum/pose_rule_anomaly/
├── PLAN.md
├── configs/          # ROI và ngưỡng cam1/cam2/cam3
├── src/              # ingest, pose, rules, features, anomaly, fusion
├── outputs/          # manifest, poses, windows, scores, events, preview
└── reports/          # pose quality, candidate audit, RAM feasibility
```

Giữ nguyên `Museum/dataset/MuseumVisitors/`. Mọi file xuất phải ghi rõ
`source=real_museum` hoặc `source=synthetic` và không kéo các ảnh synthetic
đã reject vào train, calibration hay đánh giá.
