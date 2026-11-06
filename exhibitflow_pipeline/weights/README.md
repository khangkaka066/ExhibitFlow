# Trọng số (không đưa vào git)

Đặt các file sau vào thư mục này (hoặc dùng `--weights-dir <thư mục>`; trên Kaggle chỉ cần Add Input một dataset chứa chúng):

| File | Dùng cho |
|---|---|
| `best_ckpt.pth` | detector YOLOX-s (1 lớp: person) fine-tune CCTV / bảo tàng |
| `osnet_x0_25.pth` | ReID của tracker |
| `ltc_motion_mot17.pth` | LTC motion của tracker |
| `dma_gbm_mot17.gbm` | DMA (gradient boosting) của tracker — **giữ đúng tên này** |

Tự tải ở lần chạy đầu (không cần đặt tay): `yolov8n-pose.pt`, `sam2.1_b.pt` (ultralytics), Grounding-DINO (Hugging Face),
GGUF `Qwen3-VL-8B-Instruct-Q4_K_M` + `mmproj-F16` (~6 GB, Hugging Face), llama.cpp prebuilt (GitHub).

Kiểm tra: `python main.py --check`
