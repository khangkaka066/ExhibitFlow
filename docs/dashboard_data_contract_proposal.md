# Dashboard / analytics contract — đề xuất v0.1

Trạng thái: **Draft để review chung**, không phải API đã triển khai. Không thay
đổi [tracker data contract v0.1](data_contract_v0.md).

## Phân chia trách nhiệm

| Module | Input | Output cần cho dashboard |
| --- | --- | --- |
| Tracker | Detection/video | Track JSONL v0.1 hiện có |
| Mapping | `point_image`, camera calibration, zone polygons | Điểm floor map + zone hoặc lý do không ánh xạ được |
| Analytics | Chuỗi điểm có thời gian, vùng và khoảng thiếu | Visits, KPI, heatmap, transitions, journey |
| Dashboard | Bản chụp analytics theo bộ lọc | Hiển thị và điều khiển xem lại; không tự suy ra calibration |

## Envelope chung

Mỗi bản chụp cần `analytics_schema_version`, `source_type` (`synthetic` hoặc
`tracker`), `sequence_id`, `camera_id`, `floorplan_id`, `calibration_id`,
`zone_definition_version`, `window_start_ms`, `window_end_ms`, `generated_at`
(UTC ISO 8601), `status` (`ready`, `no_data`, `uncalibrated`, `error`).
Khoảng lọc là `[window_start_ms, window_end_ms)` từ đầu video; khác với đồng hồ UTC.
Mọi ID track tham chiếu đủ `(sequence_id, camera_id, track_id)`.

## Các đối tượng đề xuất

| Đối tượng | Trường | Đơn vị / quy ước |
| --- | --- | --- |
| Floorplan | `floorplan_id`, `asset_url`, `width`, `height`, `unit`, `origin` | Dùng tọa độ mét, gốc trên trái, x sang phải, y xuống; nếu chưa có tỷ lệ vật lý phải khai báo `unit: normalized`, không ghi m² |
| Calibration | `calibration_id`, `camera_id`, `image_size`, `homography`, `valid_polygon`, `status` | Ma trận 3×3 từ pixel ảnh sang mặt bằng; phiên bản gắn với độ phân giải ảnh nguồn |
| Zone | `zone_id`, `name`, `polygon`, `color` | Polygon ít nhất 3 điểm, cùng hệ tọa độ floorplan; zone dùng để tính visits không chồng lấn; biên chung có quy tắc ưu tiên cố định |
| Mapped observation | `track_id`, `timestamp_ms`, `point_floor`, `zone_id`, `mapping_status`, `score` | `point_floor: [x,y]` hoặc null; `zone_id` null khi ngoài zone; score là detection confidence, không phải độ tin cậy của mapping |
| Visit | `track_id`, `zone_id`, `enter_ms`, `exit_ms`, `observed_dwell_ms`, `left_censored`, `right_censored` | Một lượt ghé liên tục; cờ censored cho lượt bị cắt bởi khoảng lọc / đầu cuối dữ liệu |
| Zone stats | `zone_id`, `unique_track_count`, `visit_count`, `total_observed_dwell_ms`, `mean_observed_dwell_ms`, `interest_label` | Mean là null khi visit_count=0; label và ngưỡng đi kèm `rules_version` |
| Heatmap | `grid_origin`, `cell_size`, `cols`, `rows`, `values`, `metric`, `normalization` | Ô theo hàng trước; `metric: observed_person_seconds`; giá trị không âm; trả giá trị thô, legend ghi rõ thang màu |
| Transition | `from_zone_id`, `to_zone_id`, `count` | Có hướng; nguồn ≠ đích; thời điểm vào đích nằm trong khoảng lọc |
| Journey | `track_id`, `segments`, `visits` | Mỗi segment là chuỗi `{timestamp_ms, point_floor, zone_id}` liên tục; không nối qua khoảng mất track |
| Quality | `observation_count`, `mapped_count`, `unmapped_count`, `gap_count`, `max_gap_ms` | Giúp giải thích dữ liệu thưa; không diễn giải là chỉ số accuracy của mô hình |

## Ví dụ một observation sau mapping

```json
{
  "analytics_schema_version": "0.1-draft",
  "source_type": "synthetic",
  "sequence_id": "demo_gallery",
  "camera_id": "cam_01",
  "floorplan_id": "gallery_demo",
  "calibration_id": "synthetic_mapping_v1",
  "track_id": 1,
  "timestamp_ms": 35000,
  "point_floor": [6.2, 2.4],
  "zone_id": "C",
  "mapping_status": "mapped",
  "score": 0.95
}
```

Ví dụ chỉ minh họa shape; không phải calibration hoặc kết quả từ video thật.
Prototype hiện dùng tọa độ SVG nội bộ, không có adapter đọc contract đề xuất này.

## Quy tắc cần thống nhất trước khi nối dữ liệu thật

1. Dùng homography đã kiểm tra trên camera cố định. Điểm không hợp lệ được trả
   null với trạng thái rõ ràng, không ép về `[0,0]` hoặc gán bừa vào zone.
2. Chỉ cộng dwell giữa hai quan sát hợp lệ, cùng zone, có khoảng cách thời gian
   ≤ `max_gap_ms` cấu hình. Khoảng mất track không tự động được xem là ở lại zone.
   Với hai quan sát khác zone, cần chính sách mốc chuyển rõ ràng; bản đề xuất dùng
   thời điểm quan sát đầu tiên ở zone đích, không cộng khoảng chưa biết cho zone nguồn.
3. Không nối transition qua khoảng mất quan sát hoặc vùng không xác định.
   Nếu dùng làm mượt biên zone, cấu hình `min_zone_residence_ms` và ghi
   `rules_version`; prototype dùng chuyển zone mẫu đã xác định sẵn.
4. Chỉ đếm ID trong phạm vi phiên/camera. Không cộng tổng unique của nhiều zone
   để suy ra tổng khách; không gộp ID giữa camera bằng cùng một số `track_id`.
5. Lọc window cắt visits và heatmap; transition dựa trên thời điểm vào zone đích.
   Dwell trung bình phải ghi rõ trên lượt ghé, không lẫn với trung bình trên người.
6. Phân biệt 0 quan sát, thiếu dữ liệu, chưa calibration và lỗi backend.
   Dữ liệu thưa hoặc mất track không đủ bằng chứng để gọi một khu vực là dead zone.
7. Nếu heatmap dùng thời gian, tích lũy theo delta timestamp hợp lệ để tránh
   phụ thuộc FPS. Chuẩn hóa được khai báo; muốn so sánh hai phiên phải dùng cùng thang.

Việc chốt trường, đơn vị, ngưỡng và định dạng trao đổi (file JSON hay HTTP API)
là dependency cho task triển khai dashboard production. Wireframe có thể review
ngay mà không cần các module đó đã hoàn thành.
