# ExhibitFlow Web + Backend: hướng dẫn chạy

Giao diện web (thư mục `web/`) gửi video lên **backend** chạy engine của dashboard (model nạp một lần, chạy nhiều video) rồi hiện video có khung người, danh sách cảnh báo (chạm / cầm / nghi lấy cắp) và trạng thái máy.

```
 Máy người dùng                         Máy có GPU (Kaggle / Vast.ai / server riêng)
┌─────────────────┐   URL (cloudflared)  ┌────────────────────────────────────────┐
│ web/ (serve.py) │ ───────────────────► │ server/web_launch.py                   │
│ trình duyệt     │ ◄─────────────────── │  └─ server/web_server.py (FastAPI)     │
└─────────────────┘   API /api/jobs ...  │      └─ dashboard.runner + Engine      │
                                          └────────────────────────────────────────┘
```

## File nằm ở đâu (nhánh `Thuan`)

| Đường dẫn | Việc |
|---|---|
| `web/` | Giao diện đen. Chạy trên **máy người dùng** (`python serve.py`). |
| `exhibitflow_pipeline/server/web_server.py` | Backend nối web với engine của dashboard (`State`, `PipelineRunner`). Không sửa file nào của `dashboard/`. |
| `exhibitflow_pipeline/server/web_launch.py` | Chạy `web_server.py` + mở URL công khai (cloudflared), in URL để dán vào web. |
| `exhibitflow_pipeline/notebooks/exhibitflow_kaggle_server.ipynb` | Notebook Kaggle: clone, cài thư viện, tìm trọng số, chạy backend. |

## Cách 1: Thử nhanh không cần GPU (dữ liệu giả)

Dùng để xem giao diện và luồng hoạt động. Không chạy pipeline thật.

```bash
pip install fastapi uvicorn python-multipart psutil
cd exhibitflow_pipeline
python server/web_server.py --port 8000 --demo
```

Cửa sổ khác, chạy web:

```bash
cd web
python serve.py
```

Mở link `serve.py` in ra (`http://127.0.0.1:8765/index.html`), vào **Settings → URL backend**, dán `http://127.0.0.1:8000` → **Kết nối** → Upload một video bất kỳ → **Phân tích**.

## Cách 2: Kaggle (GPU thật)

1. Kaggle → New Notebook → **File → Import Notebook** → chọn `exhibitflow_kaggle_server.ipynb`.
2. Settings: **Accelerator = GPU T4 x2**, **Internet = On**. **Add Input**: dataset chứa 4 file trọng số `best_ckpt.pth`, `osnet_x0_25.pth`, `ltc_motion_mot17.pth`, `dma_gbm_mot17.gbm`.
3. Ô 1: `BRANCH = 'Thuan'` (đổi nếu code nằm ở nhánh khác).
4. Chạy lần lượt các ô. Ô 5 hỏi `HF_TOKEN` (Enter để bỏ qua), nạp model rồi in **URL BACKEND**.
5. Giữ ô 5 chạy. Dán URL vào web (như Cách 1) → **Kết nối**.

Ô 5 không bao giờ "chạy xong"; nó chạy mãi để giữ server sống. URL chỉ in sau dòng `Model đã sẵn sàng.`

## Cách 3 (khuyến nghị): chạy tất cả trên Vast.ai, không cần URL backend

`web_server.py` phục vụ luôn giao diện `web/` ở `/` (cùng cổng với API), nên mở một địa chỉ là dùng được.

```bash
# trên instance Vast (template mở port 8000, xem VAST_SETUP.md)
BRANCH=Hung HF_TOKEN=hf_xxx START=1 bash setup.sh     # hoặc: make run
make url                                              # in http://<IP>:<port-ngoài>
```

`make run` chạy `server/web_server.py` nền (log: `make logs`). Mở `http://<IP>:<port-ngoài>/` trên trình duyệt, đợi chip trạng thái báo **Đã kết nối**
(lần đầu nạp model vài phút), rồi Upload video hoặc nhập đường dẫn trên server (vd `/workspace/...`) và bấm Phân tích.
Ô URL trong Settings để trống. Không cần tunnel, Kaggle hay `serve.py`.  Chưa mở được port ngoài: thêm `-p 8000:8000` vào Docker options của template.
Chỉ dùng cách cũ (`serve.py` + dán URL) khi muốn mở web ở máy khác.

## Dùng giao diện

1. Menu bên trái (nút ở góc trên) → **Settings** → **URL backend** → **Kết nối**. Chip đầu trang đổi thành "Đã kết nối".
2. **Dashboard**: chọn **Video**, bấm **Upload** (hoặc nhập đường dẫn trên server, vd. `/kaggle/input/<dataset>/4.mp4`) → **Phân tích**. **Dừng** để hủy.
3. Dòng thông báo cho biết bước đang chạy (`[1/5] ... [4/5] Phân tích video`). Xong hiện "Hoàn tất", cảnh báo, KPI.
4. Danh sách cảnh báo: lọc Tất cả / Bất thường / Cần xác nhận / Đã loại, **Xem lại / Xác nhận / Bỏ qua**, **Xuất CSV**.

## Giới hạn hiện tại

- **Chỉ chế độ Video.** Realtime bằng camera (RTSP) chưa chạy: engine mới đọc file video, server trả lỗi 501.
- Lúc đang chạy, giao diện chỉ hiện cảnh báo VLM đã kết luận **bất thường**; danh sách đầy đủ hiện khi chạy xong.
- Map 2D của dashboard chưa được web này dùng; khối Map 2D chỉ giữ chỗ.
- Server chỉ chạy **một job một lúc**. Gửi job khi model đang nạp hoặc đang có job khác → lỗi 409 kèm lý do.
- Đường dẫn trên server chỉ lấy trong `/kaggle/input`, `/kaggle/working`, `/content` (đổi bằng biến `EXHIBITFLOW_SOURCE_ROOTS`).
- Backend này **chưa được thử** với GPU thật trên Kaggle / Vast.ai tại thời điểm viết; mới thử với `--demo`.

## Biến môi trường (tuỳ chọn)

| Biến | Tác dụng |
|---|---|
| `EF_WEIGHTS_DIR` | Thư mục trọng số (notebook tự tìm trong `/kaggle/input`) |
| `HF_TOKEN` | Token Hugging Face để tải model nhanh hơn. **Không ghi vào file.** |
| `EXHIBITFLOW_SOURCE_ROOTS` | Thư mục server được phép đọc video |
| `EXHIBITFLOW_MAX_MB` | Giới hạn upload (mặc định 600) |
| `EXHIBITFLOW_SAVE_VIDEO=0` | Không ghi `live.mp4` (video có khung) |

## Gặp lỗi?

| Hiện tượng | Nguyên nhân / cách xử lý |
|---|---|
| Ô 2 Kaggle báo "Không thấy server/web_server.py" | Nhánh trong `BRANCH` chưa có file này (chưa push). Kiểm tra tên nhánh. |
| Chip "Mất kết nối" | URL sai hoặc đã đổi (URL cloudflared đổi mỗi lần chạy lại ô cuối); ô 5 đã dừng. |
| Lỗi 409 khi Phân tích | Model đang nạp hoặc đang có job khác: đợi hoặc bấm Dừng. |
| Lỗi 501 | Nguồn camera (RTSP/RTMP) chưa hỗ trợ; dùng file video. |
| Log có `CUDNN_STATUS_NOT_SUPPORTED`, `Falling back to CPUExecutionProvider` | Model bàn tay (RTMW) rơi về CPU vì cuDNN không khớp trên máy này (hay gặp ở Kaggle T4). Vẫn chạy, chỉ **chậm hơn**. |
| Video xem lại sau khi chạy xong không có khung người | Server không có `live.mp4` để phát (đang `--demo`, hoặc `EXHIBITFLOW_SAVE_VIDEO=0`, hoặc ghi video lỗi) nên web phát video gốc. |

## An toàn

- URL backend **không có mật khẩu**: ai có URL đều upload và chạy được trên máy của bạn. Không đăng công khai; tắt máy thuê khi dùng xong.
- Không ghi key / token (HF, cloudflared...) vào file rồi push.
