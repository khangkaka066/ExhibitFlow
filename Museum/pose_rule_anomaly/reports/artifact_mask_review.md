# Review mask hiện vật — vòng 1

Nguồn: `artifact_mask_results.zip`, chạy bằng Grounding DINO Base + SAM 2
Hiera Small trên GPU. Validator cấu trúc đã đạt: 8/8 proposal, kích thước mask
1280×800, không proposal nào được tự động approve.

## Kết quả xem trực quan

| Camera | Artifact | Đánh giá | Ghi chú |
|---|---|---|---|
| 1 | `center_sculpture_left` | đề nghị duyệt | Mask bám tượng và phần bệ liên quan. |
| 1 | `center_sculpture_right` | đề nghị duyệt sau khi cập nhật ROI | Mask bám tượng thật; ROI cũ lệch sang trái, thể hiện qua prior IoU thấp 0.074. |
| 2 | `left_bronze_statue` | đề nghị duyệt | Mask bám tượng đồng bên trái. |
| 2 | `foreground_stone` | đề nghị duyệt | Mask bám khối đá tiền cảnh, biên rõ. |
| 2 | `right_tomb` | đề nghị duyệt | Mask bám quan tài/khối đá bên phải. |
| 3 | `rear_left_bust` | đề nghị duyệt có ghi chú | Mask bám phần tượng nhỏ; cờ area xuất hiện vì prior cũ gồm cả bệ. |
| 3 | `rear_center_relief` | đề nghị duyệt | Mask bám phù điêu giữa phía sau. |
| 3 | `rear_right_bust` | **loại vòng 1** | Detector chọn trùng phù điêu giữa; không phải hiện vật bên phải. |

Các trạng thái trên là khuyến nghị kỹ thuật, chưa phải xác nhận nghiệp vụ. Chỉ
những mask được người duyệt xác nhận mới được chuyển thành `approved` và dùng
để tạo khoảng cách người–hiện vật.

## Hành động tiếp theo

Chạy gói retry Camera 3 với artifact đổi tên thành `rear_right_relief`, prompt
đúng loại hiện vật và spatial prior hẹp hơn. Nếu Grounding DINO vẫn không tìm
được box đạt IoU tối thiểu 0.20, pipeline phải dùng prior box cho SAM 2 và ghi
fallback; tuyệt đối không nhận lại mask trùng ở giữa.

## Kết quả retry và tích hợp

Retry đã đạt: `rear_right_relief` bám đúng hiện vật bên phải, spatial-prior IoU
0.518, SAM 2 score 0.880 và không có quality flag. Proposal sai
`rear_right_bust` đã bị loại khỏi bộ cuối.

Bộ `outputs/artifact_masks_v1/` hiện có 8 mask đã qua kiểm tra trực quan. Rule
khoảng cách cổ tay–hiện vật sử dụng distance transform từ mask này thay cho
polygon bao rộng; interaction zone theo camera vẫn là gate bắt buộc. Phạm vi
áp dụng là camera cố định với cùng registration. Nếu camera dịch chuyển, mask
phải trở về trạng thái cần review và tạo lại trước khi sinh cảnh báo.
