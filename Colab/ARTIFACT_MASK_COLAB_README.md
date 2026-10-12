# Chạy phát hiện và tách mask hiện vật trên Colab

## Hai file cần dùng

- `T03b_artifact_masks_colab.ipynb`
- `artifact_mask_payload.zip`

## Cách chạy

1. Mở Google Colab và upload notebook.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chạy lần lượt tất cả cell.
4. Khi notebook yêu cầu file, upload `artifact_mask_payload.zip`.
5. Chờ model Grounding DINO và SAM 2 chạy xong.
6. Colab sẽ tải về `artifact_mask_results.zip`.
7. Gửi file kết quả đó lại để duyệt mask trước khi tích hợp vào rule.

## Ý nghĩa kết quả

Grounding DINO tìm box theo tên hiện vật, có giới hạn bằng vùng dự kiến của
từng camera. SAM 2 tách mask chính xác hơn từ box đó. Nếu detector không tìm
được box phù hợp, hệ thống dùng box ROI cũ để hỏi SAM 2 và đánh dấu rõ là
fallback. Mọi mask đều mang trạng thái `needs_review`; chưa mask nào được phép
dùng vận hành trước khi xem overlay.
