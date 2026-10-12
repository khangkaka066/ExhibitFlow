# Train Isolation Forest cho pose/chuyển động bình thường

Thư mục gồm notebook train, metadata đầu vào, checkpoint đã train và báo cáo đánh giá:

- `train_normal_pose_motion_isolation_forest.ipynb`
- `manifest.csv`
- `annotations.jsonl`
- `isolation_forest_normal_yolov8_pose.joblib`
- `evaluation.json` (camera 3 chưa ổn định trên normal holdout)

Để chạy notebook train/evaluate, chọn GPU và upload hai file CSV/JSONL khi notebook yêu cầu. Dataset ảnh MuseumVisitors
(thư mục `frames/1`, `frames/2`, `frames/3`) vẫn cần được giải nén trong Google Drive tại
`MyDrive/MuseumVisitors/` hoặc sửa `NORMAL_ROOT` trong notebook.

Để dùng checkpoint hiện có trong pipeline VGGT + Depth Anything, upload `.joblib` và `evaluation.json`
cùng các video và trọng số tracker.
