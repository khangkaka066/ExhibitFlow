# ExhibitFlow Museum — kế hoạch phân tích khách tham quan và cảnh báo bảo vệ hiện vật

Cập nhật: 02/10/2026.

## 1. Mục tiêu và phạm vi

Xây dựng pipeline đọc video camera cố định trong bảo tàng để:

1. Phát hiện và theo dõi khách trong từng camera và, khi dữ liệu cho phép, nối cùng một người giữa các camera.
2. Ước lượng hướng cơ thể, hướng nhìn, nhóm khách và tác phẩm đang được quan sát.
3. Đo lượt tiếp cận, thời gian dừng và mức độ quan tâm theo tác phẩm hoặc khu trưng bày.
4. Tạo cảnh báo ứng viên cho hành vi `near_touch`, `touch`, vượt vùng bảo vệ, leo trèo hoặc tác động lên hiện vật.
5. Phát hiện hành vi bất thường theo ngữ cảnh và luôn xuất clip bằng chứng để con người kiểm tra.
6. Khi một khách bị gắn cờ hành vi bất thường, tiếp tục bám theo (re-ID/track liên tục) người đó xuyên camera để dựng lại **lộ trình sau sự kiện** — họ đi đâu, dừng ở khu/tác phẩm nào, có rời khỏi bảo tàng hay không — phục vụ điều tra và cảnh báo sớm nếu quay lại khu vực nhạy cảm.

Hệ thống không tự kết luận khách đã phá hoại hiện vật. Mọi đầu ra an ninh là cảnh báo có độ tin cậy và trạng thái `pending/confirmed/rejected`.

## 2. Dataset chính và giới hạn

Nguồn: [MuseumVisitors](https://www.micc.unifi.it/resources/datasets/museumvisitors/), thu tại National Museum of Bargello.

| Thuộc tính | Giá trị/vai trò |
| --- | --- |
| Camera | 3 IP camera |
| Độ phân giải | 1280 × 800 |
| Tốc độ | trung bình khoảng 5 FPS |
| Kịch bản | khách cá nhân xem các tác phẩm; nhóm khách xem cùng tác phẩm |
| Nhãn công bố | bbox người, vùng che khuất, ID, nhóm, hướng cơ thể, hướng nhìn, tác phẩm đang quan sát |
| Phù hợp để chấm | person detection, tracking, ReID, group detection, gaze/body orientation, artwork attention |
| Chưa đủ để chấm | tiếp xúc tay–hiện vật, vượt hàng rào, phá hoại, ngã, đánh nhau và bất thường tổng quát |

Dataset chỉ khoảng 5 FPS nên có thể bỏ lỡ tiếp xúc rất nhanh. Trước khi triển khai phải kiểm tra file tải thực tế, license, số sequence, cách đồng bộ camera, schema annotation và sự hiện diện của ID xuyên camera; không suy từ mô tả bài báo rằng mọi nhãn đều có trong gói tải.

Không dùng nhãn `artwork under observation` như nhãn `touch`. Nhìn một tác phẩm, đứng gần và chạm là ba mức bằng chứng khác nhau.

## 3. Kết quả cần bàn giao

| Kết quả | Nội dung tối thiểu |
| --- | --- |
| Dataset manifest | Sequence, camera, FPS/timestamp, resolution, số frame, nhãn có thật, license và checksum |
| Camera/zone map | ROI tác phẩm, vùng bảo vệ, lối đi, vùng khuất và quan hệ camera |
| Timeline khách | local/global ID, camera, thời gian, vị trí, nhóm, hướng nhìn, tác phẩm đang xem và đoạn mất dấu |
| Báo cáo tham quan | lượt xem, dwell time, tỷ lệ nhìn, số nhóm, chuyển tiếp giữa tác phẩm/khu vực |
| Cảnh báo hiện vật | `near_touch/touch/barrier_crossing/...`, actor, artwork, thời gian, confidence và clip trước/sau |
| Cảnh báo bất thường | loại sự kiện hoặc anomaly score, vùng/camera, bằng chứng và trạng thái người kiểm tra |
| Lộ trình sau bất thường | chuỗi vị trí/camera/thời gian của actor sau thời điểm bị gắn cờ, đến khi rời bảo tàng hoặc mất dấu, kèm độ tin cậy liên kết track |
| Báo cáo đánh giá | chỉ số riêng theo từng mô-đun; ghi rõ nhãn gốc, nhãn bổ sung và dữ liệu dàn dựng |

## 4. Chuẩn dữ liệu chung

| Bản ghi | Trường tối thiểu |
| --- | --- |
| `Scene` | `dataset_id`, `scene_id`, split, license, thời gian chuẩn |
| `Camera` | `camera_id`, FPS/timestamp, resolution, calibration/homography nếu có |
| `Frame` | scene, camera, frame index, timestamp, đường dẫn ảnh/video |
| `Artwork` | `artwork_id`, tên nếu có, bbox/mask/ROI, vùng bảo vệ, nguồn annotation |
| `PersonDetection` | person/local ID, bbox, occlusion, score, pose/keypoints |
| `Tracklet` | camera, local ID, start/end, detection sequence, embedding, chất lượng |
| `IdentityLink` | hai tracklet, global ID, score, bằng chứng, trạng thái |
| `Attention` | person, artwork, gaze/body orientation, start/end, confidence |
| `Interaction` | person, artwork, `near_touch/touch/other`, start/end, hand/object evidence |
| `Incident` | loại, actor candidate, artwork/zone, start/end, confidence, clip, review status |
| `PostIncidentTrace` | `incident_id`, global ID của actor, chuỗi `(camera, zone, timestamp)` theo sau sự kiện, trạng thái (`tracking/exited/lost`), thời điểm/cổng rời bảo tàng nếu có, độ tin cậy liên kết track |

Giữ nguyên annotation gốc và lưu annotation bổ sung ở lớp riêng, có annotator, phiên bản và quy tắc gán nhãn. Chia train/validation/test theo sequence hoặc buổi quay, không chia ngẫu nhiên từng frame vì sẽ gây rò rỉ gần như cùng một hình ảnh.

## 5. Pipeline đề xuất

1. **Ingest và kiểm dữ liệu:** đọc video/frame/annotation, xác minh FPS thực và đồng bộ, tạo manifest; giữ timestamp gốc.
2. **Định nghĩa không gian:** gán ROI/mask cho tác phẩm, bệ, dây chắn và vùng cấm. Nếu dataset không có mask hiện vật thì gán thủ công trên keyframe và propagate có kiểm tra.
3. **Phát hiện và tracking:** detector người kết hợp tracker; lưu che khuất, footpoint và clip tracklet.
4. **ReID đa camera:** kết hợp embedding ngoại hình, thời gian và topology camera; match yếu giữ `candidate` hoặc `unresolved`.
5. **Hướng nhìn và mức độ quan tâm:** dùng nhãn head/body orientation và artwork-under-observation; tổng hợp thành visit/dwell theo tác phẩm.
6. **Tay–hiện vật:** chạy pose/hand detector và segmentation hiện vật; tính khoảng cách/chồng lấp tay–mask theo nhiều frame.
7. **Quan hệ ảnh:** có thể dùng Relate Anything Model như tín hiệu phụ cho quan hệ `touching`, nhưng không dùng một mình vì RAM xử lý ảnh tĩnh, chịu domain shift và không xác nhận diễn biến thời gian.
8. **Phân loại sự kiện thời gian:** hợp nhất pose, khoảng cách, optical/motion feature, RAM score và track history để phân biệt `approach → near_touch → touch → withdraw`.
9. **Anomaly detection:** ưu tiên các luật có nghĩa nghiệp vụ (vượt vùng, tay vào ROI, chạy, ngã, tụ tập bất thường). Mô hình one-class chỉ tạo anomaly candidate, không tự đặt tên hành vi.
10. **Cảnh báo và review:** lưu clip trước/sau, actor/artwork, các tín hiệu đóng góp và cho phép xác nhận/bác bỏ.
11. **Bám lộ trình sau sự kiện:** khi `Incident` được xác nhận (hoặc pending), duy trì global ID của actor qua ReID/tracking liên camera; ghi `PostIncidentTrace` theo thời gian thực cho đến khi actor rời khỏi phạm vi camera, rời bảo tàng (qua camera lối ra/cổng soát vé), hoặc track bị mất dấu quá ngưỡng thời gian; nếu actor tiếp cận lại một ROI nhạy cảm khác, sinh cảnh báo ưu tiên cao kèm liên kết tới `incident_id` gốc.

## 6. Chiến lược dữ liệu cho chạm và bất thường

MuseumVisitors không công bố nhãn chạm hoặc bất thường, vì vậy cần tạo extension dataset:

- Chọn sequence có đủ độ phân giải tay và hiện vật; loại hoặc đánh dấu frame không thể quan sát tiếp xúc.
- Gán `normal`, `approach`, `near_touch`, `touch`, `withdraw`, `uncertain` theo đoạn thời gian, không chỉ từng frame.
- Với mỗi sự kiện, gắn `person_id`, `artwork_id`, bàn tay trái/phải, frame bắt đầu/kết thúc, mức che khuất và độ chắc chắn.
- Gán thêm hard negatives: chỉ tay, đứng sát, tay đi qua trước hiện vật do phối cảnh, chạm điện thoại/túi và nhân viên được phép thao tác.
- Nếu dataset không có đủ ca dương, quay dữ liệu dàn dựng có kiểm soát tại mô hình trưng bày; không yêu cầu người thật chạm hiện vật quý.
- Với anomaly detection, định nghĩa trước danh sách sự kiện mục tiêu. “Bất thường tổng quát” phải được báo dưới dạng anomaly score và đánh giá false alarms/hour.

## 7. Backlog triển khai

### P0 — Xác minh dataset và khóa bài toán

| Task | Việc làm | Điều kiện qua |
| --- | --- | --- |
| P0.1 | Tải/kiểm file list, license, dung lượng, video/frame và annotation thực nhận | Manifest đầy đủ; mở được mẫu từ cả 3 camera |
| P0.2 | Đối chiếu schema với bài báo: ID, group, bbox, occlusion, body/head orientation, artwork | Bảng trường có/thiếu và ít nhất một overlay kiểm chứng |
| P0.3 | Xác minh ID xuyên camera, timestamp và mức đồng bộ | Báo phạm vi ID; không hứa ReID GT nếu thiếu |
| P0.4 | Chọn năng lực MVP: tracking + attention; chạm là extension cần nhãn mới | Tài liệu scope và tiêu chí đánh giá được khóa |
| P0.5 | Chia train/val/test theo sequence | Không có cùng track/person sequence ở nhiều split |

### P1 — Baseline MuseumVisitors

| Task | Việc làm | Điều kiện qua |
| --- | --- | --- |
| P1.1 | Viết adapter về schema chung | Đọc ổn định mọi sequence và giữ annotation gốc |
| P1.2 | Render bbox, ID, group, body/head orientation, artwork target | Overlay khớp annotation trên mẫu ngẫu nhiên |
| P1.3 | Detector + single-camera tracking | Báo AP/precision-recall và HOTA/IDF1/ID switches nếu GT cho phép |
| P1.4 | ReID/cross-camera linking | Báo F1 link/IDF1 và tỷ lệ unresolved; chỉ chấm khi có GT |
| P1.5 | Group và artwork-attention baseline | Báo group F1, sai số góc và attention F1 |
| P1.6 | Tổng hợp visit/dwell theo tác phẩm | So sánh với annotation thủ công theo event và thời lượng |

### P2 — Phát hiện chạm hiện vật

| Task | Việc làm | Điều kiện qua |
| --- | --- | --- |
| P2.1 | Tạo artwork mask/protected ROI và hand keypoints | QC liên annotator; ghi ca không quan sát được |
| P2.2 | Gán nhãn temporal `near_touch/touch` và hard negatives | Có train/val/test theo sequence, thống kê class balance |
| P2.3 | Baseline hình học: hand–artwork distance/overlap + tracking | Có event precision/recall/F1 và false alerts/hour |
| P2.4 | Chạy RAM như tín hiệu phụ và hiệu chỉnh threshold trên validation | Ablation cho biết RAM cải thiện hay làm giảm chỉ số nào |
| P2.5 | Huấn luyện temporal classifier trên chuỗi đặc trưng | Giảm nhầm near-touch/touch và chịu được frame bị che |
| P2.6 | Xuất cảnh báo kèm clip và evidence | Mỗi alert truy được actor, artwork và frame chứng cứ |

### P3 — Hành vi bất thường

| Task | Việc làm | Điều kiện qua |
| --- | --- | --- |
| P3.1 | Định nghĩa taxonomy: barrier crossing, climbing, running, fall, aggressive interaction, crowding... | Mỗi lớp có hướng dẫn nhãn và hành động vận hành |
| P3.2 | Thu/gán dữ liệu dàn dựng an toàn và normal baseline | Có ca dương/âm theo camera, gồm hard negatives |
| P3.3 | Rule baseline dựa trên zone, pose, tốc độ và nhóm | Báo false alerts/hour và độ trễ cảnh báo |
| P3.4 | Temporal action/anomaly model khi đủ dữ liệu | So sánh với rule baseline trên test khóa kín |
| P3.5 | Human review và feedback loop | Lưu quyết định, lý do và phiên bản model; không tự động kết tội |
| P3.6 | Bám lộ trình sau sự kiện: duy trì ReID actor qua camera sau khi bị gắn cờ, phát hiện rời bảo tàng, cảnh báo khi quay lại ROI nhạy cảm | Báo tỷ lệ giữ track (ID switch rate), độ trễ phát hiện rời bảo tàng, recall của cảnh báo tái tiếp cận |

### P4 — Đánh giá tích hợp và triển khai thử

| Task | Việc làm | Điều kiện qua |
| --- | --- | --- |
| P4.1 | Ablation: tracking → pose/hand → ROI → RAM → temporal model | Bảng đóng góp và chi phí từng thành phần |
| P4.2 | Kiểm theo camera, khoảng cách, che khuất và nhóm đông | Báo failure modes thay vì chỉ điểm trung bình |
| P4.3 | Chạy thử offline trên video liên tục | Đo tốc độ, VRAM, latency và false alerts/hour |
| P4.4 | Kiểm quyền riêng tư và retention | Làm mờ/xử lý ID theo chính sách; giới hạn lưu clip |
| P4.5 | Pilot có người giám sát tại một khu | Không kích hoạt xử lý tự động; mọi alert được review |

## 8. Thước đo

| Năng lực | Thước đo chính |
| --- | --- |
| Person detection | AP hoặc precision/recall theo bbox và mức che khuất |
| Tracking | HOTA, IDF1, MOTA và ID switches |
| ReID | tracklet-link precision/recall/F1, global IDF1, unresolved rate |
| Group | group membership F1 và group detection precision/recall |
| Gaze/body | sai số góc trung bình, accuracy theo bin 5°, attention F1 |
| Dwell | event F1, sai số start/end và MAE thời lượng |
| Touch | event precision/recall/F1, false alerts/hour, latency; chấm riêng near-touch |
| Anomaly | event AUPRC/F1, false alerts/hour, time-to-detect theo từng loại |
| Lộ trình sau bất thường | ID switch rate khi bám track sau incident, độ trễ/độ chính xác phát hiện rời bảo tàng, recall cảnh báo tái tiếp cận ROI nhạy cảm |

Không dùng frame accuracy làm chỉ số duy nhất vì lớp bình thường áp đảo. Báo confidence interval theo sequence/camera nếu số lượng sự kiện đủ.

## 9. Mốc MVP và tiêu chí công bố

MVP gốc hoàn thành khi adapter đọc được MuseumVisitors, overlay đúng annotation, theo dõi khách, nhận diện nhóm/hướng nhìn/tác phẩm được quan sát và xuất báo cáo visit/dwell có đánh giá định lượng.

MVP chạm chỉ hoàn thành sau khi có nhãn temporal độc lập và hệ thống đạt ngưỡng đã khóa trên test sequence, đồng thời báo false alerts/hour. RAM chỉ được mô tả là tín hiệu quan hệ phụ, không phải bằng chứng vật lý duy nhất.

Chỉ công bố năng lực “phát hiện hành vi bất thường” cho các lớp đã định nghĩa và có test dương/âm. Khi dùng one-class model, mô tả đúng là “phát hiện lệch khỏi mẫu bình thường”, không đồng nhất với nguy hiểm hay phá hoại.
