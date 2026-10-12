# Kế hoạch xây dựng dataset Museum Interaction & Anomaly

**Cập nhật:** 02/10/2026  
**Trạng thái:** Draft để triển khai pilot  
**Dataset nền:** MuseumVisitors — MICC, National Museum of Bargello

## 1. Mục tiêu

Xây dựng một dataset video đa camera phục vụ các bài toán:

1. Phát hiện và theo dõi khách trong từng camera.
2. Liên kết tracklet của cùng một người giữa các camera khi có đủ bằng chứng.
3. Xác định khách đang quan sát tác phẩm nào.
4. Phát hiện chuỗi tương tác với hiện vật:
   `normal → approach → near_touch → touch → withdraw`.
5. Phát hiện các sự cố có định nghĩa rõ: vượt rào, leo trèo, chạy, ngã,
   tương tác hung hăng và tụ tập đông.
6. Ghi lại lộ trình của khách sau sự cố qua camera/zone cho đến khi xác nhận
   rời bảo tàng hoặc mất dấu.

Dataset không được dùng để tự động kết luận một người đã phá hoại hiện vật.
Nhãn và dự đoán an ninh phải kèm bằng chứng, độ tin cậy và trạng thái review.

## 2. Nguyên tắc bắt buộc

- Giữ nguyên dữ liệu và annotation gốc; mọi bổ sung nằm trong một annotation
  layer có version riêng.
- Không suy luận rằng file tải về có đầy đủ nhãn chỉ từ mô tả trong bài báo.
- Không dùng `artwork_under_observation` thay cho nhãn `touch`.
- Không gán `normal` khi hình ảnh không đủ quan sát; dùng `uncertain`.
- Chia train/validation/test theo capture episode hoặc sequence, không chia
  ngẫu nhiên theo frame.
- Các camera ghi cùng một sự kiện phải nằm trong cùng một split.
- Tập test chính chỉ chứa video thật và phải được khóa trước khi huấn luyện.
- AI được dùng để đề xuất/propagate nhãn; con người chịu trách nhiệm xác nhận
  nhãn chuẩn.
- Mọi dữ liệu dàn dựng hoặc tổng hợp phải có trường `data_origin` riêng.

## 3. Phạm vi nguồn dữ liệu

### 3.1. MuseumVisitors

Thông tin được mô tả trong tài liệu công bố:

- 3 camera IP.
- Độ phân giải 1280 × 800.
- Tốc độ trung bình khoảng 5 FPS.
- Nhãn được mô tả gồm bbox người, ID, vùng che khuất, nhóm, hướng cơ thể,
  hướng nhìn và tác phẩm đang quan sát.

Các thông tin phải xác minh trực tiếp sau khi nhận dữ liệu:

- License và quyền tạo/phát hành annotation mở rộng.
- Số video, sequence, frame và dung lượng thực tế.
- FPS thực của từng video, frame bị thiếu và timestamp.
- Mức đồng bộ giữa ba camera.
- Phạm vi của person ID: theo camera, theo sequence hay xuyên camera.
- Trường annotation thật sự tồn tại và tỷ lệ bị thiếu.
- Video gốc còn nguyên hay chỉ có frame đã trích xuất.

### 3.2. Dữ liệu dàn dựng

Dùng khi video gốc thiếu ca dương. Quay tại không gian mô phỏng, dùng tranh in,
khung, tủ hoặc mặt phẳng giả thay cho hiện vật thật.

Mỗi kịch bản cần biến thiên theo:

- Camera/góc nhìn và khoảng cách.
- Ánh sáng, trang phục, vóc dáng và mật độ người.
- Tay trái, tay phải và cả hai tay.
- Tốc độ chậm, bình thường và rất nhanh.
- Không che, che một phần và che nhiều.
- Một người, nhóm người và nhân viên được phép thao tác.
- Tiếp cận nhưng dừng lại, chạm bề mặt giả và tay đi ngang qua do phối cảnh.

Ưu tiên quay 25–30 FPS để tạo ground truth thời gian, sau đó xuất thêm bản
downsample 5 FPS mô phỏng MuseumVisitors. Không yêu cầu diễn viên thực hiện
hành động nguy hiểm.

### 3.3. Dữ liệu tổng hợp bằng AI hoặc mô phỏng

Có thể dùng cho train, pretraining và stress test:

- Render 3D để sinh pose, quỹ đạo, che khuất và mask chính xác.
- Compositing người/đạo cụ vào background camera thật.
- Generative image/video để tăng biến thiên ngoại hình hoặc tạo clip ứng viên.
- Pseudo-anomaly từ video bình thường để pretrain anomaly model.

Không đưa dữ liệu AI-generated vào test set chính. Clip sinh bằng generative
model không mặc nhiên có ground truth tiếp xúc vật lý chính xác; chỉ dùng nếu
nhãn được suy ra từ trạng thái mô phỏng hoặc được người kiểm duyệt.

## 4. Cấu trúc thư mục đề xuất

```text
Museum/dataset/
├── raw/
│   ├── museumvisitors/       # dữ liệu gốc, read-only
│   ├── staged/               # video dàn dựng
│   └── synthetic/            # dữ liệu render/generative
├── manifests/                # inventory, checksum, license, provenance
├── annotations/
│   ├── original/             # bản sao annotation nguồn
│   ├── spatial/              # artwork, plinth, barrier, forbidden zone
│   ├── tracks/               # detections, tracklets, identity links
│   ├── interactions/         # approach/near_touch/touch/withdraw
│   ├── incidents/            # taxonomy bất thường
│   └── reviews/              # adjudication và audit log
├── derived/
│   ├── clips/
│   ├── frames/
│   ├── overlays/
│   └── embeddings/
├── splits/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── reports/
    ├── audit/
    ├── quality/
    └── dataset_card.md
```

Không commit video, ảnh nhận diện khuôn mặt hoặc embedding sinh trắc học vào
Git. Chỉ commit manifest, schema, script, split ID và báo cáo không nhạy cảm.

## 5. Schema dữ liệu chung

Mỗi bản ghi có các trường chung:

```text
id
schema_version
dataset_version
data_origin = original | staged | synthetic
annotation_source = original | manual | model_assisted | imported
created_at
annotator_id
review_status = unreviewed | accepted | rejected | needs_adjudication
```

### 5.1. Các entity chính

| Entity | Trường tối thiểu |
| --- | --- |
| `Scene` | `scene_id`, địa điểm giả danh, thời gian, license, split |
| `CaptureEpisode` | các camera/sequence ghi cùng một khoảng thời gian hoặc sự kiện |
| `Camera` | `camera_id`, resolution, FPS, timezone, calibration, zone coverage |
| `Frame` | camera, episode, frame index, timestamp, media path, checksum |
| `Artwork` | `artwork_id`, bbox/mask, plinth, protection type, protected ROI |
| `Zone` | polygon, loại zone, camera coverage, adjacent zones |
| `PersonDetection` | bbox, keypoints, occlusion, local person ID, quality |
| `Tracklet` | camera, local ID, start/end, detection IDs, quality |
| `IdentityLink` | hai tracklet, quan hệ, score, evidence, review status |
| `Attention` | person/tracklet, artwork, start/end, gaze/body evidence |
| `Interaction` | actor, artwork, state, hand, time range, visibility, evidence |
| `Incident` | type, actors, zone/artwork, severity, time range, evidence |
| `PostIncidentTrace` | incident, subject, chuỗi camera/zone/tracklet/timestamp |

### 5.2. Schema `Interaction`

```json
{
  "interaction_id": "int_000001",
  "episode_id": "episode_001",
  "camera_id": "cam_01",
  "person_id": "person_or_tracklet_id",
  "artwork_id": "artwork_01",
  "state": "near_touch",
  "hand": "right",
  "start_frame": 1200,
  "end_frame": 1206,
  "start_time_ms": 240000,
  "end_time_ms": 241200,
  "contact_target": "artwork",
  "visibility": "partial",
  "evidence": ["hand_artwork_distance", "roi_entry"],
  "confidence": 0.82,
  "authorized": "unknown",
  "data_origin": "original",
  "annotation_source": "model_assisted",
  "review_status": "accepted"
}
```

### 5.3. Schema `IdentityLink`

```text
tracklet_a
tracklet_b
relation = same | different | uncertain
confidence
evidence = appearance | timing | topology | manual
review_status
```

Không ép mọi tracklet vào một global ID. Match yếu được giữ là `uncertain`.

### 5.4. Schema `PostIncidentTrace`

```text
trace_id
incident_id
subject_identity_id
steps[] = {camera_id, zone_id, tracklet_id, enter_time, exit_time, confidence}
exit_status = confirmed_exit | probable_exit | lost | still_inside | unknown
exit_camera_id
exit_time
```

Chỉ dùng `confirmed_exit` khi camera quan sát người đó đi qua một exit zone.
Không còn xuất hiện trong camera không đồng nghĩa với đã rời bảo tàng.

## 6. Quy tắc gán nhãn tương tác

| Nhãn | Định nghĩa vận hành |
| --- | --- |
| `normal` | Không có động tác tay hoặc cơ thể hướng tới vùng bảo vệ |
| `approach` | Người/tay đang tiến gần artwork hoặc protected ROI |
| `near_touch` | Tay rất gần biên mục tiêu nhưng chưa có bằng chứng chạm |
| `touch` | Có bằng chứng hình ảnh rõ tay tiếp xúc/giao biên mục tiêu |
| `withdraw` | Tay rút khỏi mục tiêu sau `near_touch` hoặc `touch` |
| `uncertain` | Che khuất, blur, thiếu frame hoặc góc nhìn không đủ kết luận |

Ở 5 FPS, hai frame cách nhau xấp xỉ 200 ms. Nếu tiếp xúc có thể xảy ra giữa
hai frame nhưng không quan sát được trực tiếp, dùng `uncertain` và
`evidence=inferred_between_frames`, không gán chắc chắn là `touch`.

### 6.1. Hard negatives bắt buộc

- Chỉ tay vào hiện vật.
- Đứng rất gần nhưng không đưa tay ra.
- Tay đi ngang ROI do phối cảnh.
- Chạm kính bảo vệ nhưng không chạm hiện vật.
- Chạm điện thoại, túi, quần áo hoặc người đi cùng.
- Nhân viên được phép thao tác.
- Tay bị che bởi người hoặc vật khác.
- Bóng, phản chiếu và motion blur giống bàn tay.
- Góc camera khác của cùng sự kiện không đủ bằng chứng.

Lưu `negative_subtype`; không gom tất cả thành một nhãn `normal` duy nhất.

## 7. Taxonomy sự cố

| Loại | Điều kiện gợi ý | Hard negative |
| --- | --- | --- |
| `barrier_crossing` | Cơ thể/chân vượt polygon hàng rào | đứng sát, tay qua rào |
| `climbing` | đặt chân/cơ thể lên bệ hoặc cấu trúc cấm | cúi, ngồi ghế hợp lệ |
| `running` | tốc độ vượt ngưỡng trong nhiều frame | camera rung, trẻ đi nhanh |
| `fall` | mất thăng bằng và kết thúc ở tư thế thấp/nằm | cúi, quỳ, ngồi |
| `aggressive_interaction` | chuyển động tấn công/xô đẩy có actor-target | ôm, chơi, chen lấn nhẹ |
| `crowding` | mật độ vượt ngưỡng theo zone và kéo dài đủ lâu | đoàn khách hợp lệ |
| `suspected_vandalism` | tác động lên hiện vật kèm hành vi phá hoại quan sát được | chạm vô ý, nhân viên |

Mỗi incident có thể có nhiều tag. Ngưỡng cho `running` và `crowding` phải cấu
hình riêng theo camera/zone, không dùng một ngưỡng chung cho mọi không gian.

## 8. Pipeline AI-assisted annotation

1. Chạy person detector để tạo bbox ứng viên.
2. Chạy tracker để tạo tracklet trong từng camera.
3. Chạy pose/hand model để lấy keypoint tay, chân và thân người.
4. Gán mask artwork/barrier trên keyframe và dùng video segmentation để
   propagate; người gán nhãn sửa điểm drift.
5. Dùng khoảng cách tay–mask, ROI crossing, vận tốc và hướng cơ thể để tìm
   clip ứng viên.
6. Dùng action/VLM model để xếp hạng clip bất thường; không dùng kết quả này
   làm ground truth cuối.
7. Annotator xác nhận state, frame bắt đầu/kết thúc, actor, artwork và mức
   visibility.
8. Reviewer kiểm tra toàn bộ positive, uncertain và một mẫu negative.
9. Sau seed set, huấn luyện model đầu tiên và dùng active learning để chọn
   clip không chắc chắn hoặc có model disagreement.
10. Lặp lại đến khi validation không còn cải thiện đáng kể hoặc đạt mục tiêu
    coverage/quality đã khóa.

## 9. Quy trình QA

### 9.1. Kiểm tra tự động

- ID duy nhất và mọi foreign key đều tồn tại.
- `start_frame <= end_frame`; timestamp tăng đơn điệu.
- Frame nằm trong phạm vi video.
- Interaction tham chiếu tới person và artwork hợp lệ.
- Tracklet chỉ thuộc một camera.
- Các camera của cùng episode không bị tách sang split khác.
- Dữ liệu synthetic không lọt vào real-only test.
- Không có checksum trùng giữa train và test.

### 9.2. Kiểm tra con người

- Hai annotator gán độc lập tối thiểu 10–20% pilot set.
- Review 100% nhãn `touch`, `fall`, `aggressive_interaction`,
  `suspected_vandalism` và `uncertain`.
- Bất đồng được một reviewer thứ ba adjudicate.
- Ghi agreement cho class, actor/artwork association và ranh giới thời gian.
- Mỗi release có changelog và audit trail.

## 10. Chiến lược split

Đơn vị split là `CaptureEpisode`, không phải frame.

Một episode gồm toàn bộ camera ghi cùng một khoảng thời gian/sự kiện, tất cả
phiên bản FPS, clip crop và dữ liệu dẫn xuất từ khoảng thời gian đó.

Các split đánh giá:

1. **Standard:** train/val/test theo episode.
2. **Cross-camera:** giữ riêng một camera hoặc góc nhìn cho test nếu quy mô cho phép.
3. **Cross-domain:** train dùng real + staged + synthetic; test chỉ real.
4. **Cross-actor:** diễn viên trong staged test không xuất hiện trong train.
5. **Cross-site:** thêm khi thu được dữ liệu ở bảo tàng/không gian thứ hai.

Tỷ lệ khởi điểm có thể dùng 70/15/15, nhưng ưu tiên đủ số positive event trong
validation/test hơn là cố giữ tỷ lệ tuyệt đối.

## 11. Chỉ số và báo cáo

| Hạng mục | Chỉ số |
| --- | --- |
| Detection | AP, precision/recall theo kích thước và occlusion |
| Tracking | HOTA, IDF1, ID switches |
| Identity linking | precision/recall/F1, unresolved rate |
| Attention | event F1, sai số start/end, dwell-time MAE |
| Interaction | event precision/recall/F1, temporal IoU, false alerts/hour |
| Incident | event AUPRC/F1, time-to-detect, false alerts/hour theo class |
| Trace | ID switch rate, track retention, exit detection accuracy/latency |
| Annotation | inter-annotator agreement và tỷ lệ bị reviewer sửa |

Không dùng frame accuracy làm chỉ số chính vì lớp bình thường áp đảo.

## 12. Kế hoạch triển khai

### Phase 0 — Audit và khóa phạm vi

**Đầu ra:** `dataset_inventory`, license record, schema mapping, báo cáo đồng bộ.

- Kiểm kê toàn bộ file và checksum.
- Mở mẫu ở mọi camera/sequence.
- Xác minh FPS, timestamp và annotation.
- Xác minh ID có xuyên camera hay không.
- Xác định các camera/ROI có khả năng quan sát bàn tay.
- Khóa định nghĩa nhãn và quy tắc privacy/retention.

**Gate:** chưa qua Phase 0 thì không thiết kế split hoặc tuyên bố có GT ReID.

### Phase 1 — Pilot không gian và tracking

**Đầu ra:** 1 camera, 2–3 artwork, ROI/mask, tracklet và overlay đã QA.

- Chuẩn hóa dữ liệu về schema chung.
- Gán artwork, plinth, barrier và forbidden zone.
- Chạy person detection/tracking và sửa track lỗi.
- Xây annotation UI/project và export validator.

**Gate:** overlay khớp video; lỗi schema bằng 0; sample track đạt chất lượng đủ
để gán interaction.

### Phase 2 — Seed set tương tác

**Đầu ra:** tập interaction thật + staged đầu tiên và baseline rule.

- Tìm/gán clip `approach`, `near_touch`, `touch`, `withdraw`, `uncertain`.
- Thu dữ liệu dàn dựng cho các ca còn thiếu.
- Thu tối thiểu 3–5 hard negative gần giống cho mỗi ca positive trong pilot.
- Chạy baseline hand–ROI và đo false alerts/hour.

**Gate:** hai annotator đạt agreement chấp nhận được; test episode được khóa.

### Phase 3 — Taxonomy bất thường

**Đầu ra:** guideline, positive/negative theo class và camera, rule baseline.

- Thu staged data an toàn cho từng loại sự cố.
- Gán actor, target, zone, thời gian, severity và authorization.
- Đánh giá riêng từng camera/zone và mức che khuất.
- Không huấn luyện class chưa có đủ positive và hard negative đại diện.

### Phase 4 — Active learning và mở rộng

**Đầu ra:** dataset version 1.0 và model-assisted labeling loop.

- Huấn luyện baseline trên seed set.
- Chọn clip uncertainty/disagreement/diversity cao.
- Annotate, review và huấn luyện lại theo vòng.
- Theo dõi class balance, camera coverage và tỷ lệ lỗi nhãn.

### Phase 5 — Release nội bộ

**Đầu ra:** immutable release, dataset card, split, metrics và audit report.

- Freeze media manifest và annotations.
- Chạy validator, leakage check và privacy review.
- Sinh thống kê theo origin/camera/class/occlusion.
- Ghi rõ limitation: 5 FPS, góc khuất, ca synthetic và phạm vi license.

## 13. Backlog có thể đưa vào tracker

| ID | Công việc | Ưu tiên | Phụ thuộc | Definition of Done |
| --- | --- | --- | --- | --- |
| DS-001 | Audit file và license MuseumVisitors | P0 | — | Có manifest, checksum, license status |
| DS-002 | Map annotation gốc sang schema chung | P0 | DS-001 | Parse được toàn bộ file, không mất trường |
| DS-003 | Kiểm FPS/timestamp/sync/ID scope | P0 | DS-001 | Có báo cáo cho từng sequence/camera |
| DS-004 | Viết labeling guideline v1 | P0 | DS-002 | Có ví dụ positive, negative, uncertain |
| DS-005 | Gán artwork/barrier/zone pilot | P0 | DS-003 | ROI đã review trên 1 camera, 2–3 artwork |
| DS-006 | Person/track pre-annotation | P1 | DS-002 | Tracklet export và overlay kiểm chứng được |
| DS-007 | Hand/pose/mask propagation | P1 | DS-005 | Candidate clips được sinh tự động |
| DS-008 | Gán seed interaction set | P0 | DS-004, DS-007 | Positive, hard negative, uncertain đã QA |
| DS-009 | Thu staged interaction data | P1 | DS-004 | Có nhiều tốc độ, occlusion, actor và camera |
| DS-010 | Định nghĩa và thu incident taxonomy | P1 | DS-004 | Mỗi class có positive/negative guideline |
| DS-011 | Double annotation và adjudication | P0 | DS-008, DS-010 | Có agreement report và nhãn đã chốt |
| DS-012 | Tạo split chống leakage | P0 | DS-003, DS-011 | Episode không xuất hiện ở nhiều split |
| DS-013 | Chạy baseline và error analysis | P1 | DS-012 | Có event metrics và false alerts/hour |
| DS-014 | Active learning loop | P2 | DS-013 | Có log selection và label yield mỗi vòng |
| DS-015 | Dataset card và release v1.0 | P0 | DS-001–014 | Validator/QA/privacy đều đạt |

## 14. Tiêu chí nghiệm thu dataset v1.0

- License và quyền sử dụng được ghi rõ; trường hợp chưa rõ phải đánh dấu blocked.
- 100% file có checksum và provenance.
- Schema validator không có lỗi nghiêm trọng.
- Không có episode hoặc media dẫn xuất bị rò rỉ giữa các split.
- Tập test chỉ chứa dữ liệu thật, trừ khi có một synthetic benchmark tách biệt.
- 100% positive và uncertain của các lớp an ninh đã được reviewer xác nhận.
- Có hard negatives theo từng class và từng camera trong phạm vi đánh giá.
- Có thống kê class/camera/zone/origin/occlusion và danh sách failure modes.
- Có dataset card mô tả giới hạn 5 FPS, privacy, license và intended use.
- Có baseline reproducible để chứng minh dataset đọc được và nhãn có thể chấm.

## 15. Rủi ro chính và cách giảm thiểu

| Rủi ro | Giảm thiểu |
| --- | --- |
| Không rõ license | Không phát hành lại; liên hệ chủ dataset; lưu license snapshot |
| Không có ID xuyên camera | Đổi thành identity-link candidate; tự gán subset có review |
| 5 FPS bỏ lỡ chạm nhanh | Nhãn uncertain; staged high-FPS; đánh giá riêng bản 5 FPS |
| Tay quá nhỏ hoặc bị che | Chọn ROI/camera phù hợp; không ép nhãn; báo theo visibility |
| Synthetic domain gap | Dùng chủ yếu cho train; fine-tune và test trên real-only |
| Model tạo sai pseudo-label | Human-in-the-loop; review positive/uncertain; audit model version |
| Leakage từ clip/frame | Split theo episode trước khi trích clip hoặc augmentation |
| Class imbalance | Hard-negative mining, active learning và báo AUPRC/FAR-hour |
| Sai IdentityLink | Kết hợp appearance + time + topology; cho phép unresolved |
| Quyền riêng tư | Pseudonymize ID, hạn chế retention/access, không public face embedding |

## 16. Quyết định cần khóa sau pilot

- Camera và artwork nào đủ độ phân giải để phát hiện tay–hiện vật.
- Ngưỡng vận hành giữa `near_touch` và `touch`.
- Mức agreement tối thiểu giữa annotator.
- Số positive event tối thiểu cho từng anomaly class.
- Ngưỡng false alerts/hour chấp nhận được.
- Khi nào incident được dùng để kích hoạt `PostIncidentTrace`.
- Dataset được phát hành công khai, nội bộ hay chỉ phát hành annotation/schema.

