# Quy tắc vẽ pipeline trong Draw.io

## Mục tiêu

Sơ đồ phải đọc được khi thu nhỏ, thể hiện đúng luồng dữ liệu và không để đường nối che chữ hoặc xuyên qua khối. Mỗi đường nối phải có điểm bắt đầu/kết thúc rõ ràng và tuyến đi có chủ đích.

## 1. Bố cục

- Chia pipeline thành các stage theo chiều dọc; trong mỗi stage, đặt các bước theo chiều trái → phải.
- Giữ khoảng trống giữa các stage để đặt connector; không đặt khối xử lý đè lên tiêu đề stage.
- Dành lề ngoài sơ đồ làm “hành lang” cho luồng quay lại hoặc đi qua nhiều stage. Không cho connector xuyên ngang một stage chỉ vì đường thẳng ngắn hơn.
- Căn khối theo lưới; giữ khoảng cách đủ để mũi tên và nhãn không chạm khối khác.
- Khối là nơi xử lý hoặc tạo dữ liệu; tiêu đề stage, ghi chú và chú giải không phải endpoint của luồng.

## 2. Quy tắc nối khối

- Nối vào **viền** khối, không nối vào tâm. Luồng trái → phải dùng cổng phải → trái; luồng trên → dưới dùng cổng dưới → trên.
- Bật kiểu connector vuông góc (orthogonal/elbow); tránh đường chéo, đường tự động cắt qua khối, và đầu mũi tên chồng lên chữ.
- Nếu tuyến tự động đi sai, thêm waypoint để điều khiển đường đi qua khoảng trống hoặc hành lang ngoài.
- Bắt buộc đặt connector ở lớp dưới block: trong XML, các `mxCell edge="1"` phải đứng trước các vertex nội dung trong thứ tự vẽ. Block có nền đặc để che phần connector nằm dưới diện tích của block; đầu nối chỉ lộ ra tới viền.
- Không đi xuyên qua bất kỳ khối nào, kể cả khối không liên quan; không để đường chạy phía sau chữ bên trong khối.
- Ưu tiên đổi vị trí/cổng nối trước khi cho hai luồng cắt nhau. Chỉ cho phép giao cắt khi không thể tránh; giao cắt không có chấm nối nghĩa là hai luồng không kết hợp.
- Luồng đi qua nhiều stage không được dùng router tự động. Mỗi luồng phải có cổng nối và waypoint tường minh, đi trong làn riêng ngoài vùng nội dung; các làn song song cách nhau ít nhất 24 px và không dùng chung đoạn tuyến.
- Không cho đường dài đi xuyên qua thanh tiêu đề stage. Nếu không thể bố trí lối đi gọn trong trang, dùng connector cùng mã/nhãn ở hai đầu thay cho một đường vòng xuyên sơ đồ.
- Không cho phép waypoint vượt khỏi khổ trang hoặc làm đường đi vòng sát mép như khung bao quanh sơ đồ.
- Tránh đường vòng dài không có nhãn. Với luồng quay lại, dùng một hành lang riêng và gắn nhãn ngắn nếu hướng đi chưa rõ.

## 3. Mũi tên và ý nghĩa

- Mỗi cạnh biểu diễn một quan hệ dữ liệu/điều khiển có thật; không thêm mũi tên chỉ để trang trí.
- Dùng một đầu mũi tên ở endpoint đích; không dùng connector hai chiều trừ khi dữ liệu thật sự hai chiều.
- Chỉ bật `flowAnimation=1` cho connector ngắn trong cùng một stage. Không animate luồng liên-stage, đường vòng, nhánh thử nghiệm hoặc đoạn đang giao/cắt với luồng khác. Hiệu ứng chạy trong Draw.io, còn ảnh/PDF xuất tĩnh không giữ chuyển động.
- Màu và nét phải nhất quán: luồng đang dùng, luồng thử nghiệm, touch/rule, anomaly, và review.
- Nhánh thử nghiệm dùng nét đứt; không nhập nhánh đó vào luồng hiện hành nếu chưa được kiểm chứng.
- Nếu hai nhánh hợp nhất, vẽ chúng tới một khối hợp nhất rõ ràng, không chồng hai mũi tên lên cùng một cạnh.

## 4. Chữ và khả năng đọc

- Tên khối ngắn, mô tả đầu vào/đầu ra rõ; chi tiết dài chuyển sang ghi chú hoặc tài liệu.
- Không đặt nhãn cạnh lên khối, mũi tên, hoặc giao cắt. Đặt nhãn gần đoạn thẳng trống nhất.
- Dùng cỡ chữ đủ đọc ở mức zoom toàn trang; kiểm tra cả khi xuất PNG/PDF.
- Tương phản màu chữ/nền phải đủ; không chỉ dựa vào màu để phân biệt nhánh.

## 5. Checklist trước khi bàn giao

- [ ] Mọi cạnh bắt đầu/kết thúc đúng khối và chạm viền khối.
- [ ] Không cạnh nào xuyên khối, chữ, tiêu đề stage hoặc ghi chú.
- [ ] Các connector nhiều stage đi trong hành lang đã định; các giao cắt còn lại có chủ ý.
- [ ] Mũi tên chỉ đúng chiều và mỗi màu/nét có ý nghĩa nhất quán.
- [ ] Đã kiểm tra toàn trang và từng vùng ở mức zoom lớn.
- [ ] Đã render/preview trong Draw.io; XML hợp lệ thôi chưa đủ để bàn giao.
- [ ] File `.drawio` còn chỉnh sửa được; bản xuất xem trước khớp với file nguồn.
