# ExhibitFlow — Dashboard wireframe v0.1

## Mục tiêu và phạm vi

Deliverable cho task **Product & Visualization — Thiết kế dashboard layout**.
Wireframe thể hiện đủ floor map, heatmap, zone stats, transition và journey trong
một màn hình tổng quan có các vùng chi tiết. Đối tượng sử dụng: người vận hành
triển lãm và nhóm trình bày demo đồ án.

Thiết kế dựa trên đề cương ExhibitFlow trong repo và
[tracker data contract v0.1](data_contract_v0.md). Bộ dữ liệu trong giao diện là
**8 track, 5 zone, 120 giây được tạo để minh họa**, không phải kết quả chạy mô hình
hay mặt bằng thật của CAVIAR. Không gọi API và không yêu cầu cài thư viện.

## Mở wireframe

Mở `dashboard/index.html` bằng trình duyệt; trên macOS, từ thư mục repo:

```bash
open dashboard/index.html
```

Hoặc phục vụ từ thư mục gốc repo:

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

Truy cập `http://127.0.0.1:8080/dashboard/`. Dừng server bằng Ctrl+C.

## Bố cục màn hình demo

```text
┌ Điều hướng ─┬ Phiên / camera · Thời gian · Zone ─────────────┐
│ Tổng quan  │ Track quan sát | Dừng TB | Chuyển tiếp | Top zone│
│ Floor map  ├───────────────────────────────┬─────────────────┤
│ Zone stats │ Floor map / Heatmap / Journey │ Chi tiết zone   │
│ Transition │ Mặt bằng + các vùng tương tác │ Lượt ghé, dwell │
│ Journey    ├───────────────────────────────┴─────────────────┤
│            │ Bảng thống kê zone  │ Top cạnh + ma trận chuyển │
│            ├─────────────────────┴───────────────────────────┤
│            │ Chọn track | Timeline zone | Phát lại / tua     │
└────────────┴─────────────────────────────────────────────────┘
```

| Khu vực | Câu hỏi người dùng | Thành phần / tương tác |
| --- | --- | --- |
| Bộ lọc chung | Đang xem dữ liệu nào? | Phiên / camera cố định ở demo; đổi toàn phiên, nửa đầu, nửa sau; chọn zone; đặt lại |
| Chỉ số tổng quan | Có bao nhiêu track và hoạt động? | Số ID duy nhất, dwell trung bình / lượt ghé, tổng chuyển tiếp, zone có tổng dwell cao nhất |
| Floor map | Khách ở khu vực nào? | Mặt bằng SVG gồm 5 zone A–E, tên zone và lối đi; chọn bằng chuột hoặc Enter/Space |
| Heatmap | Khách tập trung ở đâu? | Lớp mật độ thời gian trên mặt bằng, thang ít–nhiều; đổi bộ lọc sẽ tính lại |
| Zone stats | Zone nào giữ chân khách? | Số ID ghé, dwell TB / lượt ghé, nhãn nóng / ít ghé; nhấn dòng để chọn zone |
| Transition | Khách đi từ đâu đến đâu? | 3 cạnh phổ biến nhất, số lần, ma trận có hướng A–E; hàng là nguồn, cột là đích |
| Journey | Một track di chuyển thế nào? | Chọn ID ẩn danh, chuỗi zone và thời gian, xem đường đi, nút phát / tạm dừng, thanh tua |

Desktop có sidebar cố định, mặt bằng ở cột chính và zone spotlight bên phải.
Tablet/mobile chuyển thành một cột, thanh điều hướng ngang; bảng và timeline
cuộn ngang khi cần. Spotlight phụ được ẩn trên màn hình hẹp vì cùng số liệu đã
có trong bảng zone. Năm nội dung bắt buộc vẫn hiện diện.

## Quy tắc dữ liệu của wireframe

- Khoảng thời gian dùng `[start, end)` theo giây từ đầu video. Dwell được cắt
  theo giao của lượt ghé và bộ lọc, không lấy cả lượt ngoài khoảng đã chọn.
- `Track quan sát` là số ID có quan sát ở zone / khoảng thời gian đã chọn.
  Không khẳng định đó là số người thật: một người bị đổi ID có thể bị đếm lại.
- Một lượt ghé là một đoạn liên tục trong một zone. Dwell TB = tổng thời gian
  các đoạn sau khi cắt / số đoạn còn lại, đơn vị giây. ID tái ghé vẫn chỉ được
  đếm một lần ở cột `ID ghé`, nhưng tạo thêm một lượt ghé.
- Zone nóng trong demo: có trên 2 ID và dwell TB ≥ 30 giây. Zone ít ghé: 1–2 ID.
  Không có lượt ghé hiển thị `Chưa có dữ liệu`, không suy ra dead zone. Đây là
  quy tắc minh họa; ngưỡng dùng thực tế cần thống nhất với nhóm analytics.
- Transition ghi nhận tại thời điểm vào zone đích trong khoảng lọc. Chỉ đếm
  hai zone khác nhau liên tiếp. Lọc một zone giữ cạnh đi hoặc đến zone đó.
  Không cộng đường chéo. Trạng thái trước đầu khoảng được dùng để biết nguồn.
- Heatmap minh họa lấy một quan sát / giây / ID, gom theo ô mặt bằng. Độ đậm
  biểu diễn thời gian quan sát tích lũy, chuẩn hóa tương đối và làm mượt bằng SVG.
  Không phải mật độ người/m² đã hiệu chuẩn; không so trực tiếp độ đậm giữa hai
  khoảng có chuẩn hóa khác nhau. Chọn zone làm mờ các zone còn lại.
- Lọc zone giới hạn danh sách track đi qua zone đó. Journey vẫn giữ toàn bộ
  các zone của track trong khoảng thời gian đã chọn để thấy nguồn và đích.
- Timeline và thanh tua dùng thời gian của phiên. Nét liền nằm trong zone;
  nét đứt nối hai zone chỉ thể hiện thứ tự, chưa phải đường đi hợp lệ qua cửa.
- Phát lại chạy theo dữ liệu minh họa, không phát video và không chạy tracker.
- Track ID luôn nằm trong phạm vi `(sequence_id, camera_id, track_id)`;
  không ghép người qua nhiều camera và không hiển thị danh tính cá nhân.

## Shared data contract và phần còn cần triển khai

Contract v0.1 hiện có `timestamp_ms`, `track_id`, `point_image`, `bbox`,
`sequence_id`, `camera_id` và `image_size`. Chưa có floor map, homography, zone,
dwell hay transition. Vì vậy không thể nạp trực tiếp JSONL hiện tại vào dashboard
để có các chỉ số thật.

[Đề xuất contract analytics](dashboard_data_contract_proposal.md) mô tả các
trường bổ sung, đơn vị, quy tắc thiếu dữ liệu và trách nhiệm của module.
Đây là **đề xuất để nhóm review**, chưa thay đổi contract tracker đã có.

Luồng tích hợp dự kiến:

```text
Tracker JSONL v0.1
  → Mapping (hiệu chỉnh camera → floor map, polygon zone)
  → Analytics (visits, dwell, heatmap, transitions)
  → Dashboard (bộ lọc, trình bày, phát lại journey)
```

Không yêu cầu sửa C++/CMake hoặc đổi mô hình trong task thiết kế này. API thật,
hiệu chỉnh mặt bằng, tính analytics trên video, streaming video và xác thực
người dùng là các task triển khai tiếp theo.

## Trạng thái cần có khi nối backend

| Trạng thái | Cách trình bày đã xác định |
| --- | --- |
| Đang tải | Skeleton cho KPI/map/bảng; không hiển thị số 0 như kết quả |
| Chưa hiệu chỉnh camera | Giải thích thiếu homography; vô hiệu hóa analytics theo mặt bằng |
| Không có quan sát | Map nền, thông báo trống; dwell là `—`, không gán nhãn ít ghé |
| Track bị mất / điểm ngoài map | Khoảng đứt trên journey; không cộng dwell hoặc nối transition qua khoảng chưa biết |
| Backend lỗi | Thông báo lỗi + thử lại; nếu giữ kết quả cũ phải ghi rõ thời điểm dữ liệu |
| Dữ liệu mẫu | Banner mẫu luôn hiển thị như wireframe hiện tại |

Các trạng thái backend ở bảng này là đặc tả thiết kế; prototype chỉ triển khai
dữ liệu mẫu và cách render danh sách/transition rỗng.

## Kịch bản nghiệm thu

1. Mở dashboard: đủ năm nội dung bắt buộc, banner dữ liệu mẫu và bộ lọc chung.
2. Toàn phiên, tất cả zone: 8 ID, 18 chuyển tiếp, dwell TB làm tròn 25 giây;
   zone C có tổng dwell lớn nhất (288 giây).
3. Chọn zone C trên map: KPI còn 7 ID, 11 chuyển tiếp đi/đến; bảng chỉ giữ C;
   danh sách hành trình còn các track ghé C.
4. Chọn nửa sau: số liệu tính trên 01:00–02:00; timeline cắt tại đầu khoảng.
5. Chọn track, bấm xem đường đi: lớp Journey bật và map có đường đi.
6. Tua về đầu, phát / tạm dừng: marker và đường đi đổi theo mốc thời gian.
7. Đặt lại bộ lọc: trở về toàn phiên, tất cả zone, lớp heatmap.
8. Kiểm tra ở desktop và mobile: map, bảng zone, transition, journey đều dùng được.

Kết luận phạm vi: deliverable **dashboard wireframe** hoàn thành khi các bước
trên đạt; đây không phải nghiệm thu chất lượng tracking hay analytics thật.

## Kết quả kiểm tra bản bàn giao

- Chrome headless: đạt 18 tổ hợp bộ lọc thời gian / zone; số liệu mặc định và
  zone C khớp kịch bản nghiệm thu.
- Đạt thao tác chọn zone bằng chuột / bàn phím, trạng thái rỗng (zone E ở nửa
  đầu), chọn track, xem đường đi, tua, phát / tạm dừng và đặt lại.
- Viewport 390px, 768px, 1440px: đủ các phần bắt buộc, không tràn ngang toàn trang.
- Không có JavaScript runtime error; `node --check dashboard/app.js` và
  `git diff --check` đạt.
- Ảnh bàn giao: [desktop](../dashboard/preview-desktop.png),
  [mobile](../dashboard/preview-mobile.png).
