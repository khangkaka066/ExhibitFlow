# Gói upload Colab — tracking lại

Thư mục này gom notebook, 3 video đầu vào và 2 checkpoint cần cho lần chạy lại.

1. Mở `touch_detection_bytetrack_dma_ltc_colab.ipynb` bằng Google Colab và bật GPU để chạy trọn pipeline phát hiện touch.
2. Chạy các cell theo thứ tự.
3. Ở hộp thoại upload, chọn `1.mp4`, `2.mp4`, `3.mp4`, `dma_gbm_mot17.gbm`, `ltc_motion_mot17.pth`, `isolation_forest.joblib` và `anomaly_scores.jsonl` trong thư mục này.

`touch_detection_bytetrack_dma_ltc_colab.ipynb` là notebook tổng hợp: YOLOv8n phát hiện bbox người → ByteTrack-DMA-LTC gán track ID → YOLOv8n-Pose ước lượng pose trên crop của từng track hiện diện; UniDepth + mask hiện vật tạo touch candidates, và Isolation Forest T06 chấm điểm chuyển động lạ. YOLO nano tải tự động trong Colab; `dma_gbm_mot17.gbm` và `ltc_motion_mot17.pth` là checkpoint riêng cho tracker. `isolation_forest.joblib` cùng `anomaly_scores.jsonl` dùng để chấm điểm và hiệu chuẩn percentile theo camera. `best_ckpt.zip` không nằm trong gói vì detector bên trong nhận ảnh 12 kênh, không tương thích với frame RGB thông thường. Kết quả gồm touch candidates, unusual-motion candidates, CSV và video preview; anomaly không tự xác nhận vi phạm.
