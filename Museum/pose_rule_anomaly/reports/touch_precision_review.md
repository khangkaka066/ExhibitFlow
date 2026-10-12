# Touch precision correction

## Vấn đề

Mask hiện vật là mask nền tĩnh. Khi một người đi ngang phía trước, mask vẫn
chồng lên ảnh người; khoảng cách cổ tay 2D có thể bằng 0 dù không có tiếp xúc.
Ví dụ phản hồi `track_id=3`, frame 3868 là lỗi phối cảnh/occlusion, không phải
touch.

## Gate mới

Một `touch_candidate` chỉ được sinh khi đồng thời đạt:

- cổ tay đủ gần **biên** mask, không lấy mọi điểm bên trong mask là khoảng cách 0;
- cổ tay không nằm sâu trong mask;
- khuỷu tay và vai nằm ngoài mask, có khoảng hở tối thiểu;
- cổ tay gần hiện vật hơn khuỷu tay;
- cổ tay vươn ra ngoài dải hai vai ít nhất 0.12 chiều cao người;
- phần mask nằm trong person bbox không vượt 8%;
- tốc độ thân quanh thời điểm tiếp cận không vượt 0.12 chiều cao người/giây;
- đủ số quan sát, dwell và approach delta như rule trước.

## Kết quả

- Trước sửa: 163 candidate.
- Sau gate occlusion/arm/motion: 12 candidate.
- Sau gate tay vươn khỏi thân: 1 candidate.
- Trường hợp `track_id=3` do người đi ngang đã bị loại.

Candidate còn lại vẫn mang nhãn `CANDIDATE, NOT CONFIRMED`. Dữ liệu không có
ground truth touch và có thể thực tế không chứa positive. Không báo precision/
recall hoặc đổi candidate thành touch xác nhận nếu chưa có video positive thật
hay cảm biến/đa góc nhìn hỗ trợ.
