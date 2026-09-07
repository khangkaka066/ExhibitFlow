# Kế hoạch hai task Systems & Deployment — Khang

## 1. Hiểu dự án và vị trí của hai task

Nguồn: `FA26AI15_ExhibitFlow_Long_Term_Context_And_Dynamic_Motion_Appearance_Weighting_For_Visitor_Journey_Analytics_AnhVH54.docx` trong thư mục dự án.

ExhibitFlow phân tích cách khách di chuyển và tương tác với các khu vực trong bảo tàng, triển lãm hoặc showroom. Pipeline được mô tả trong tài liệu:

`Video → Person detection → Multi-object tracking → Track history → Floor-plan mapping → Visitor analytics → Dashboard`

- Tracking dùng ID ẩn danh; tài liệu nhắc đến ByteTrack và một phương pháp tracking được lựa chọn.
- Track history gồm frame ID, track ID, bounding box, timestamp và thông tin vị trí.
- Mapping dùng các điểm tham chiếu và phép biến đổi hình học như homography để đưa vị trí từ ảnh lên mặt bằng 2D.
- Analytics tính số khách theo zone, dwell time, heatmap mật độ, chuyển zone, hành trình, hot zone và dead zone.
- Không nhận diện khuôn mặt hoặc lưu danh tính cá nhân. ID tracking không phải định danh con người xuyên mọi video/camera.
- Đánh giá và demo cuối dự án cần ít nhất hai video/dataset hoặc môi trường triển lãm tự ghi hình; đây không phải tiêu chí hoàn thành riêng của hai task nền tảng.

Tên đề tài có “Long-Term Context” và “Dynamic Motion–Appearance Weighting”, nhưng nội dung chưa định nghĩa công thức, dữ liệu đầu vào, cách lưu lịch sử hay thuật toán cụ thể. Interface nên cho phép bổ sung backend tracking mà không tự coi những chi tiết này là yêu cầu đã chốt.

**Vai trò của Khang trong hai task:** xây nền tảng C++ có thể build lặp lại và một ranh giới tích hợp rõ ràng để nhóm tracking, mapping và analytics phát triển độc lập.

## 2. Phạm vi và các quyết định đề xuất

Các lựa chọn dưới đây là đề xuất triển khai, chưa phải quyết định có sẵn trong tài liệu:

| Hạng mục | Đề xuất cho milestone này | Lý do |
| --- | --- | --- |
| Ngôn ngữ | C++17 | Đủ cho interface, CLI và quản lý tài nguyên; chưa cần phần C riêng |
| Build | CMake tối thiểu 3.20; kiểm chứng trên Ubuntu 22.04 hoặc 24.04 | Có mốc môi trường nghiệm thu rõ ràng |
| Cấu trúc | Thư viện `exhibitflow_core` và executable `exhibitflow_tracker` | Thuật toán không phụ thuộc cách parse CLI |
| Input CLI v0 | Detections theo frame, đọc từ JSONL | Chạy skeleton không cần detector, video decoder, model hoặc GPU |
| Config | JSON, có `schema_version` và tên backend | Dễ kiểm tra và mở rộng |
| Output v0 | Track history JSONL trong tọa độ ảnh | Phục vụ trực tiếp module mapping |
| Backend đầu tiên | `mock` với dữ liệu mẫu xác định trước | Kiểm chứng tích hợp; không phải tracker có chất lượng đo được |
| Mapping | Nằm sau tracker | Tránh đưa homography, zone và dashboard vào core tracking |

Chưa đưa vào hai task: tích hợp ByteTrack thật, detection từ video, ReID/model appearance, trọng số động, long-term memory, multi-camera association, Docker deployment, dashboard và benchmark HOTA/IDF1/MOTA. Đây là công việc tiếp theo; skeleton chỉ chuẩn bị điểm tích hợp.

Nếu nhóm yêu cầu CLI nhận video ngay trong milestone này, cần bổ sung task video/detection adapter và dependency tương ứng; không chỉ đổi tên tham số `--input`.

## 3. Task 1 — Setup C/C++ project structure và CMake

**Mục tiêu:** từ bản clone sạch trên Linux, người khác có thể configure, build và chạy executable theo README.

**Dependency:** không phụ thuộc data contract. Cần repository và compiler/CMake trên máy nghiệm thu.

### Cấu trúc dự kiến sau cả hai task

```text
ExhibitFlow/
├── CMakeLists.txt
├── README.md
├── .gitignore
├── include/exhibitflow/
│   ├── types.hpp
│   └── tracker.hpp
├── src/
│   ├── CMakeLists.txt
│   ├── core/
│   └── trackers/mock_tracker.cpp
├── apps/
│   ├── CMakeLists.txt
│   └── tracker_main.cpp
├── configs/mock.json
├── examples/
│   ├── detections.jsonl
│   └── expected_tracks.jsonl
├── docs/data_contract_v0.md
├── tests/
│   ├── CMakeLists.txt
│   └── fixtures/
└── third_party/                 # Chỉ thêm nếu chọn dependency vendored
```

Task 1 chỉ tạo cấu trúc tối thiểu và executable có `--help`/`--version`; các file contract, parser và mock được bổ sung trong task 2. Không cần tạo sẵn hàng loạt module trống cho tương lai.

### Thứ tự thực hiện

1. **Chốt build baseline:** C++17, CMake >= 3.20, GCC hoặc Clang; Linux là môi trường nghiệm thu, macOS là kiểm tra bổ sung nếu dùng để phát triển.
2. **Tạo core library và CLI target:** CLI liên kết core; public headers nằm trong `include/exhibitflow`; tránh để core phụ thuộc `main` hoặc xử lý terminal.
3. **Cấu hình CMake theo target:** dùng `target_compile_features`, `target_include_directories`, `target_link_libraries`; warning flags phù hợp compiler, tránh đường dẫn tuyệt đối và cờ toàn cục.
4. **Giữ build độc lập model/GPU:** chưa yêu cầu OpenCV, CUDA, TensorRT hoặc model weights. Dependency JSON ở task 2 cần được ghim phiên bản và mô tả rõ cách cung cấp; ưu tiên vendored nếu nhóm cần build offline.
5. **Viết hướng dẫn:** prerequisites, configure/build/run, Debug/Release và vị trí binary. Bỏ qua thư mục build/output trong Git.
6. **Kiểm chứng:** build từ checkout sạch trên Linux và lưu compiler/CMake version cùng kết quả. CI Linux là lựa chọn tốt nếu repository đã có nền tảng CI, không phải điều kiện để bắt đầu task.

### Cách nghiệm thu dự kiến

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/bin/exhibitflow_tracker --help
./build/bin/exhibitflow_tracker --version
```

CMake cần cấu hình output directory phù hợp với các đường dẫn trên.

**Definition of Done:**

- Configure/build thành công trên môi trường Linux đã chọn từ checkout sạch.
- `exhibitflow_core` và `exhibitflow_tracker` là hai target tách biệt.
- `--help` và `--version` trả exit code 0.
- Không dựa vào đường dẫn hoặc file cục bộ chỉ có trên máy Khang.
- README đủ để thành viên khác tái hiện; không cần GPU/model để chạy.

**Bàn giao:** source tree, CMake files, executable tối thiểu, README và bằng chứng build Linux.

## 4. Task 2 — Thiết kế tracker interface và CLI skeleton

**Mục tiêu:** CLI nhận input/config/output, gọi backend qua interface chung và xuất kết quả mẫu đúng contract.

**Dependency:** task 1 cung cấp nền tảng build; data contract cần được thống nhất với người phụ trách detector/tracker và mapping/analytics trước khi nghiệm thu tích hợp.

### 4.1. Chốt data contract v0 trước khi code parser

Khang soạn `docs/data_contract_v0.md`; nhóm xác nhận các quy ước sau:

| Dữ liệu | Quy ước đề xuất |
| --- | --- |
| `schema_version` | `"0.1"`; từ chối phiên bản chưa hỗ trợ |
| Phạm vi chạy | Một video/sequence và một camera mỗi lần chạy CLI v0 |
| `sequence_id`, `camera_id` | Bắt buộc để phân biệt nguồn; không chứa danh tính cá nhân |
| `frame_id` | Số nguyên không âm, tăng nghiêm ngặt; cho phép nhảy frame |
| `timestamp_ms` | Millisecond từ đầu sequence, số hữu hạn không âm, tăng theo thời gian; lấy từ input, không dùng thời gian chạy máy |
| Thiếu timestamp | V0 báo lỗi. Adapter video sau này cung cấp timestamp; nếu suy từ FPS phải ghi rõ giả định |
| `image_width`, `image_height` | Số nguyên dương, kích thước khung hình gốc |
| `bbox` | `[x, y, width, height]`, pixel, gốc góc trên trái, x sang phải, y xuống dưới; x/y không âm, width/height dương, box trong ảnh |
| Detections | Chỉ người; `score` hữu hạn thuộc [0, 1]; tọa độ đã đưa về ảnh gốc, không phải ảnh resize của detector |
| `track_id` | Số nguyên không âm, ổn định trong sequence/camera; khóa downstream là `(sequence_id, camera_id, track_id)` |
| `point_image` | Điểm giữa đáy bbox: `[x + width / 2, y + height]`; tọa độ ảnh, chưa phải tọa độ mặt bằng |
| Không có detection | Vẫn có record của frame với `detections: []` |
| Không có track | Vẫn có record output với `tracks: []` để downstream biết frame đã được xử lý |
| Output track v0 | Chỉ observation có detection hỗ trợ; chưa xuất vị trí dự đoán của lost tracks; nếu mở rộng phải định nghĩa trạng thái để analytics tránh đếm nhầm |
| JSONL | Mỗi dòng không rỗng là một JSON object; xử lý tuần tự, không nạp toàn bộ video vào RAM |

Contract mẫu input, hai dòng tương ứng hai frame:

```jsonl
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","frame_id":0,"timestamp_ms":0,"image_width":1920,"image_height":1080,"detections":[{"bbox":[100,200,60,160],"score":0.95}]}
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","frame_id":1,"timestamp_ms":40,"image_width":1920,"image_height":1080,"detections":[]}
```

Config mẫu:

```json
{
  "schema_version": "0.1",
  "tracker": "mock"
}
```

Output tương ứng:

```jsonl
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","tracker_backend":"mock","frame_id":0,"timestamp_ms":0,"image_width":1920,"image_height":1080,"tracks":[{"track_id":1,"bbox":[100,200,60,160],"point_image":[130,360],"score":0.95}]}
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","tracker_backend":"mock","frame_id":1,"timestamp_ms":40,"image_width":1920,"image_height":1080,"tracks":[]}
```

Ở v0, `score` output là confidence của detection hỗ trợ observation, không phải độ chắc chắn của danh tính. Mock cấp ID theo quy tắc xác định, được ghi rõ trong README; có thể cấp ID mới cho mỗi detection. Không dùng mock để suy ra chất lượng association, đếm unique visitors hoặc dwell time thực tế. Fixture nhiều frame có ID lặp để kiểm thử analytics nên được chuẩn bị riêng và gắn nhãn dữ liệu giả lập.

### 4.2. Thiết kế interface C++ độc lập CLI

Các kiểu public dự kiến:

- `FrameContext`: sequence/camera, frame ID, timestamp, kích thước ảnh.
- `Detection`: bbox và confidence.
- `TrackObservation`: ID, bbox và confidence.
- `ITracker`: `update(context, detections)` trả danh sách observation; `reset()` xóa trạng thái; virtual destructor để quản lý qua smart pointer.
- Factory tạo backend theo config; tên backend không hỗ trợ phải báo lỗi, không tự chuyển sang mock.

CLI chịu trách nhiệm đọc config, parse/validate input, gọi tracker và serialize output. Tracker không tự đọc file, in JSON hoặc xử lý tham số dòng lệnh. Writer tính `point_image` theo quy ước chung.

Backend thật được phép giữ history/state bên trong object. Trước khi tích hợp phương pháp motion–appearance, nhóm cần chốt ai tính embedding và backend có cần ảnh frame hay không; khi đó bổ sung trường kiểu dữ liệu hoặc adapter có kiểm soát. Không khóa interface vào đường dẫn ảnh và không dùng một JSON object tùy ý thay cho typed API.

### 4.3. Xây CLI skeleton

```bash
./build/bin/exhibitflow_tracker \
  --input examples/detections.jsonl \
  --config configs/mock.json \
  --output build/tracks.jsonl
```

- Hỗ trợ `--help`, `--version` và ba tham số bắt buộc `--input`, `--config`, `--output`.
- Output ghi file; log và thông báo lỗi ghi `stderr`. `--help`/`--version` có thể dùng `stdout`.
- Không ghi đè input/config. Nếu output đã tồn tại, báo lỗi; tùy chọn ghi đè có thể bổ sung khi có nhu cầu.
- Đọc tuần tự từng frame, giữ đúng thứ tự và metadata; kiểm tra sequence/camera không thay đổi trong cùng lần chạy.
- Parse lỗi phải ghi rõ file, dòng và trường liên quan khi xác định được.
- Ghi qua file tạm cạnh output; chỉ công bố file kết quả sau khi chạy thành công, tránh để downstream đọc kết quả dở dang như output hoàn chỉnh.
- Quy ước exit code: `0` thành công; `2` tham số/config/input/schema không hợp lệ; `3` lỗi I/O; `4` lỗi backend/runtime.

### 4.4. Kiểm chứng có ý nghĩa

1. **Luồng hợp lệ:** CLI đọc fixture có detection và frame trống, xuất đúng metadata/bbox/điểm chân/schema. So sánh JSON đã parse, không so raw text theo thứ tự key.
2. **Tính lặp lại:** cùng input/config, mock cho cùng kết quả ngữ nghĩa.
3. **Input lỗi:** thiếu field, bbox sai, timestamp/frame đảo thứ tự, JSON hỏng hoặc schema version không hỗ trợ → exit code 2, lỗi dễ xác định và không công bố output hoàn chỉnh.
4. **CLI/I/O lỗi:** thiếu tham số, không tìm thấy input, output không ghi được, output trùng input → mã lỗi phù hợp và bảo toàn dữ liệu gốc.
5. **Config lỗi:** backend không tồn tại → báo lỗi, không âm thầm chạy mock.
6. **Ranh giới interface:** mock chạy qua `ITracker`; gọi `reset()` trả state về ban đầu theo quy ước đã ghi.
7. **Bàn giao downstream:** người phụ trách mapping xác nhận đọc được output mẫu và hiểu đúng timestamp, bbox, điểm chân, ID scope và frame trống.

Đăng ký những kiểm tra phù hợp vào CTest để chạy cùng build:

```bash
ctest --test-dir build --output-on-failure
```

**Definition of Done:**

- Contract v0 đã được nhóm thống nhất và có ví dụ input/config/output.
- CLI nhận đủ ba đường dẫn và chạy được từ fixture sạch trên Linux.
- Kết quả mock đúng schema, có `tracker_backend: "mock"` để nhận biết dữ liệu mô phỏng.
- Interface tách khỏi CLI; có lifecycle/state reset rõ ràng.
- Những ca kiểm tra nêu trên đạt; README có lệnh chạy và bảng exit code.
- Downstream đọc được file mẫu mà không phải đoán đơn vị hoặc hệ tọa độ.

**Bàn giao:** `types.hpp`, `tracker.hpp`, mock backend, CLI parser/runner, config, fixtures, contract, hướng dẫn và kiểm tra tích hợp.

## 5. Thứ tự làm, ước lượng và phụ thuộc

Ước lượng theo ngày làm việc tập trung của Khang, chưa tính thời gian chờ phản hồi nhóm hoặc thiết lập máy Linux:

| Bước | Công việc | Ước lượng | Kết quả |
| --- | --- | --- | --- |
| 1 | Cấu trúc project, core/CLI targets, CMake | 0.5–1 ngày | Build và executable tối thiểu |
| 2 | README và kiểm chứng checkout sạch trên Linux | 0.5 ngày | Hoàn tất task 1 |
| 3 | Soạn và thống nhất contract v0 với nhóm | 0.5 ngày + thời gian chờ | Schema và fixture được chấp nhận |
| 4 | Typed interface, mock, config/input/output handling | 1 ngày | CLI prototype chạy xuyên suốt |
| 5 | Kiểm tra lỗi, CTest, hướng dẫn, downstream smoke test | 0.5–1 ngày | Hoàn tất task 2 |

**Tổng dự kiến: 3–4 ngày làm việc, cộng thời gian chờ thống nhất contract.** Có thể gửi draft contract ngay trong task 1 để giảm thời gian chờ. Trước khi contract được chốt, vẫn có thể làm interface lifecycle và CLI argument handling; chưa coi schema output là ổn định để downstream tích hợp.

Các điểm cần nhóm thống nhất trong bước 3:

- Producer của detections là module nào và có cung cấp được timestamp/kích thước ảnh theo contract không?
- Milestone đầu dùng detections JSONL hay bắt buộc nhận video trực tiếp?
- Người phụ trách mapping đồng ý dùng điểm giữa đáy bbox, đơn vị pixel/ms và phạm vi ID nêu trên chưa?
- Backend nghiên cứu cần embeddings hay ảnh frame, và thành phần nào chịu trách nhiệm cung cấp?

## 6. Trạng thái hiện tại

Đã đọc tài liệu đề xuất và kiểm tra thư mục: hiện chỉ có DOCX, chưa có source code, CMake hoặc data contract để kế thừa. Tài liệu này là kế hoạch; chưa triển khai hai task và chưa có kết quả build/test Linux.
