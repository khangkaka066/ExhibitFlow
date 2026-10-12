# Dataset ngoài cho các hành vi tương lai

## Kết luận

Không tìm thấy dataset CCTV công khai có nhãn trực tiếp cho hành vi **chạm
hiện vật bảo tàng**. MuseumVisitors hiện tại phù hợp tracking/group/gaze nhưng
không có ground truth touch. Vì vậy `touch_candidate` tiếp tục dùng mask hiện
vật + cổ tay + tiếp cận theo thời gian và bắt buộc human review.

Không dùng lại anomaly model normal-only để tự gọi tên hành vi. Hướng mở rộng
là một detector/classifier riêng cho từng hành vi, dùng dữ liệu ngoài để
pretrain rồi kiểm tra trên camera bảo tàng.

## Nguồn ưu tiên

| Nguồn | Dữ liệu hữu ích | Dùng cho | Giới hạn |
|---|---|---|---|
| [CHAD](https://github.com/TeCSAR-UNCC/CHAD) | 412 video, khoảng 1.15M frame; bbox, Re-ID, pose và nhãn anomaly theo frame; 4 camera cố định | pose CCTV, fall/run/fight như dữ liệu phụ | Cảnh bãi đỗ xe; nhãn phát hành chính là normal/anomaly, không đủ một mình để phân loại tên hành vi |
| [RWF-2000](https://arxiv.org/abs/1911.05913) | 2.000 clip CCTV, fight/non-fight | `fight_candidate` | Chỉ hai lớp, khác miền bảo tàng |
| [Le2i Fall](http://le2i.cnrs.fr/Fall-detection-Dataset) | video fall và hoạt động thường ngày trong nhiều bối cảnh | `fall_candidate` | Quy mô nhỏ; cần kiểm tra lại link/license trước tải |
| [VIRAT](https://viratdata.org/) | camera cố định, track người/vật và activity annotation; có running, carrying, pickup, interaction | chạy, mang/lấy vật, tương tác vật thể | Chủ yếu ngoài trời; ground data có thỏa thuận sử dụng |
| [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/) | 1.900 video/128 giờ; stealing, shoplifting, vandalism, fighting và các lớp khác | pretrain yếu cho trộm/phá hoại/đánh nhau | Nhãn train chủ yếu cấp video, domain rất rộng; không dùng trực tiếp làm xác nhận bảo tàng |
| [100 Days of Hands](https://fouheylab.eecs.umich.edu/~dandans/projects/100DOH/) | 100K frame có hand box, contact state và object-in-contact | bằng chứng phụ cho tiếp xúc tay–vật | Tay thường lớn/gần; không tương đồng CCTV xa và đông người |
| [UBnormal](https://github.com/lilygeorgescu/UBnormal) | anomaly synthetic có mask mức pixel | thử open-set/localization | Synthetic, license CC BY-NC-ND; không dùng làm ground truth bảo tàng |

## Thứ tự triển khai đề nghị

1. Giữ production chỉ có `touch_candidate`.
2. Tải **metadata CHAD trước**, không tải video đầy đủ lên máy local vì chỉ còn
   khoảng 23 GiB trống và kích thước video chưa được công bố rõ trên trang dự án.
3. Chọn một hành vi cụ thể để làm tiếp: ưu tiên `fall_candidate`, sau đó
   `fight_candidate`. Hai bài toán có dataset chuyên biệt và nhãn rõ hơn.
4. Dùng external data để pretrain; validation/test cuối phải gồm clip thật từ
   góc camera bảo tàng. Nếu chưa có positive thật, output vẫn là candidate.
5. Theft/vandalism và object removal chỉ bật sau khi có dữ liệu site-specific
   hoặc buổi dàn dựng được cho phép; UCF-Crime một mình không đủ.

## Liên kết tải CHAD đã xác minh

- Metadata: `https://drive.google.com/file/d/1g01Ay86cuPmKGsrfxddxwA1uG_7bFFmu/view`
- Video: `https://drive.google.com/file/d/13am4hfhicErcozAYgkmQm02_K-cCmtkQ/view`

Các link Google Drive có thể yêu cầu đăng nhập/xác nhận. Không tự động tải full
video cho đến khi biết kích thước thực và có đủ dung lượng.
