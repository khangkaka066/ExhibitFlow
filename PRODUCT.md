# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Người xem dashboard là người theo dõi hành vi khách tham quan trong phòng trưng bày và xử lý cảnh báo. **Chưa xác nhận** đó là nhân viên an ninh, quản lý bảo tàng hay hội đồng chấm đồ án (người dùng chưa trả lời); coi là suy luận từ code cho đến khi có xác nhận.

## Product Purpose

ExhibitFlow phân tích video camera bảo tàng: theo dõi từng khách (Visitor ID), phát hiện chạm / cầm / nghi lấy cắp hiện vật bằng VLM, và (dự kiến) vẽ bản đồ 2D. Dashboard là một trang: chọn nguồn video, chạy pipeline trên backend, xem video có khung người, danh sách cảnh báo hành vi và trạng thái hệ thống. Về chức năng, người dùng xác nhận dashboard hiện đã ổn.

## Operating Context

- Pipeline chạy ở backend (Kaggle T4×2 qua cloudflared, hoặc server 2×H200); web chỉ là giao diện gọi API (`/api/jobs`, `/live`...).
- Chế độ **Video**: upload hoặc nhập đường dẫn, kết quả hiện khi chạy xong. Chế độ **Realtime** (camera): giao diện có, backend chưa đọc được camera (HTTP 501).
- Thư mục `web/` (backend thật, `demo:false`) và `web_demo/` (cùng giao diện, `demo:true`, tự mô phỏng).

## Capabilities and Constraints

- HTML/CSS/JS thuần, không framework, không bước build; chạy bằng `python serve.py`. Giao diện sinh từ nguồn dashboard (`dash/`) rồi ghi ra cả `web/` và `web_demo/`.
- Ngôn ngữ giao diện: tiếng Việt.
- Phần chưa có dữ liệu thật (Map 2D, heatmap, vệt đi, vùng gần) **vẫn phải hiện nguyên** trên giao diện, không che hay ẩn.
- Không đưa key/token (ngrok, cloudflared URL...) vào file; chỉ nhập lúc chạy ở Settings.
- Không đổi chi tiết giao diện khi chưa được người dùng yêu cầu.

## Evidence on Hand

- Kết quả pipeline mẫu: `exhibitflow_results/4/` (video 4: 1 sự kiện NORMAL, 5 file).
- Chưa có dữ liệu bản đồ 2D thật, chưa chạy thử trên Kaggle/GPU thật với bản giao diện hiện tại. Không bịa số liệu hay ảnh minh hoạ.

## Product Principles

1. Trung thực về trạng thái: phần chưa có dữ liệu để nguyên chỗ, không giả vờ có.
2. Chức năng ổn rồi: thay đổi bề ngoài không được làm hỏng luồng upload → phân tích → xem cảnh báo.
3. Nhìn như một công cụ vận hành thật, không như sản phẩm sinh tự động (hướng người dùng vừa nêu: giao diện còn "AI hoá").
4. Không bao giờ để lộ thông tin bí mật trong mã nguồn.

## Accessibility & Inclusion

Chưa xác lập yêu cầu riêng. Giữ chữ đủ tương phản và vùng bấm đủ lớn.
