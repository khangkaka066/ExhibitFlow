# ExhibitFlow Web Dashboard

> **Hướng dẫn chạy web + backend đầy đủ (demo, Kaggle, Vast.ai): xem [`../WEB_SETUP.md`](../WEB_SETUP.md).** File này là ghi chú chi tiết về giao diện.

Dashboard một trang để chạy pipeline và xem kết quả (cảnh báo, video, map 2D). Thuần HTML/CSS/JS, không cần build.

## Chạy web

Bấm đúp **`start.bat`**: chạy server và tự mở trình duyệt vào http://127.0.0.1:8765/index.html. Đóng cửa sổ đen để tắt.

Hoặc chạy tay:

```bash
cd web
python -m http.server 8765 --bind 127.0.0.1
```

## Dùng dashboard

| Khu vực | Việc |
|---|---|
| **Video phân tích** | Nhập **đường dẫn file / thư mục (trên server) hoặc URL video**, hoặc bấm **Upload** để chọn file trên máy. Bấm **Phân tích** để chạy, **Dừng** để hủy. |
| **Header** | Chip trạng thái kết nối (bấm để mở Settings), đồng hồ, **Refresh Data** (kiểm tra lại kết nối và trạng thái máy). |
| **Cảnh báo hành vi** | Mọi hành vi dạng thẻ: ảnh, nhãn hành vi, Visitor → hiện vật, thời gian, độ tin cậy và lý do của VLM, nút **Xem lại / Xác nhận / Bỏ qua**; lọc Tất cả / Bất thường / Cần xác nhận / Đã loại; **Xuất CSV**. |
| **System status** | CPU / RAM / GPU / VRAM của máy chạy backend. |
| **Thông báo** | Một dòng dưới hàng nguồn video hiện bước đang chạy hoặc lỗi gần nhất (không còn khung nhật ký). |
| **Settings** | Nhập URL backend, danh sách camera (cho Realtime), ngưỡng tin cậy VLM. |

Map 2D, heatmap, hành trình chưa có dữ liệu thật (chờ module map): với kết quả thật khối map hiện thông báo "Chưa có dữ liệu bản đồ".

Thư mục `web_demo/` là cùng giao diện nhưng **chế độ demo** (`js/config.js`: `demo:true`): chưa kết nối backend thì "Phân tích" chạy mô phỏng và hiện dữ liệu mẫu. Thư mục này (`demo:false`) bắt buộc kết nối backend.

## Chế độ Realtime

> Ghi chú: phần phát khung hình / cảnh báo từng phần mô tả dưới đây dựa trên server cũ (`server/server.py`) và `publisher.py` (đã bỏ khỏi nhánh mới). Với `web_server.py` (engine của dashboard), chế độ **Video** dùng được; Realtime bằng camera vẫn chưa chạy (RTSP chưa hỗ trợ).

Hàng nguồn đầu vào **đổi theo chế độ**:

| Chế độ | Nguồn | Nút |
|---|---|---|
| **Video** | đường dẫn / URL video, hoặc **Upload** file | Phân tích |
| **Realtime** | **camera**: chọn trong danh sách (hoặc nhập URL `rtsp://...`); quản lý danh sách ở **Settings → Camera**. Không có Upload | Bắt đầu |

Chế độ **Realtime (camera)** mới có giao diện: backend hiện **trả lỗi 501 "chưa hỗ trợ camera trực tiếp"** vì pipeline mới đọc file video (cần mở rộng bộ đọc khung hình để nhận luồng RTSP). Ở `web_demo/` chế độ này chạy mô phỏng.

Chọn **Realtime** (thay vì Video), chọn camera rồi bấm **Bắt đầu**:
- Pipeline đọc video **đúng nhịp fps** như camera (không chờ hết video), khung hình đã vẽ khung người / hiện vật hiện **trực tiếp** trên dashboard.
- Thanh trạng thái: khung hiện tại, FPS thực, số khung bị bỏ, hàng đợi VLM.
- Bảng cảnh báo **hiện dần**: dòng "Đang chạm" xuất hiện ngay khi bắt đầu chạm, rồi được thay bằng kết luận của VLM khi VLM trả lời. Khi video kết thúc, danh sách tạm được thay bằng kết quả cuối (`result.json`).
- Nút **Dừng** hủy job.

Cách hoạt động: backend chạy `main.py` với `rt.realtime=true` và `rt.publish_dir=<thư mục>`. Pipeline (mô-đun `exhibitflow/publisher.py`) ghi `live.jpg`, `status.json`, `events.jsonl` ở đó; server phát lại qua `GET /api/jobs/:id/live` và `/live/events`. Mặc định `publish_dir` tắt nên chạy theo lô không đổi.

Giới hạn hiện tại:
- Nguồn camera (RTSP / webcam) chưa được backend hỗ trợ (xem bảng trên). Phần phát kết quả từng phần của pipeline đã sẵn sàng và đã được kiểm thử với file video chạy đúng nhịp fps.
- Khi máy xử lý không kịp, pipeline **bỏ khung** (`rt.drop_frames`), và cấu hình pipeline cảnh báo việc này làm giảm recall mạnh. Với server mạnh (H200) thường không cần bỏ khung; xem số "bỏ N khung" trên thanh trạng thái.
- Cảnh báo luôn **trễ** bằng độ trễ của VLM (vài giây).
- Phần sửa pipeline (`publisher.py`, `engine.py`, `settings.py`, `pipeline.py`) phải có trong bản code mà server chạy. Notebook Kaggle clone từ GitHub nhánh `Hung`, nên cần commit và push các file này thì mới dùng realtime được trên Kaggle.

## Chạy backend trên server riêng (ví dụ 2 GPU H200)

Không cần đường hầm nếu trình duyệt truy cập được máy đó:

```bash
pip install fastapi uvicorn python-multipart
python exhibitflow_pipeline/scripts/install_deps.py
python exhibitflow_pipeline/server/web_server.py --port 8000
```

Mở cổng 8000 trên tường lửa, rồi trong web: **Settings → URL backend** nhập `http://<ip-server>:8000`. Đặt `EXHIBITFLOW_WEIGHTS` trỏ tới thư mục chứa 4 file trọng số nếu chúng không nằm trong `exhibitflow_pipeline/weights/`. Nếu trang web chạy trên `https://` thì backend cũng phải là `https://` (trình duyệt chặn nội dung hỗn hợp).

## Backend trên Kaggle (engine của dashboard, model nạp sẵn một lần)

Web nói chuyện với `exhibitflow_pipeline/server/web_server.py`: server này dùng lại `State` + `PipelineRunner` của `dashboard/` (model nạp một lần lúc khởi động, chạy nhiều video không phải nạp lại) và trả lời bằng API "job" mà web dùng. Không sửa file nào của `dashboard/`.

1. Push nhánh có các file `server/web_server.py`, `server/web_launch.py` và notebook. Notebook clone code từ GitHub nhánh `Hung`.
2. Import `exhibitflow_pipeline/notebooks/exhibitflow_kaggle_server.ipynb` lên Kaggle. Settings: **GPU T4 x2**, **Internet On**, Add Input dataset trọng số.
3. Chạy lần lượt các ô. Ô cuối nạp model (lần đầu vài phút) rồi in **URL BACKEND** (cloudflared). Giữ ô đó chạy.
4. Trên web: **Settings → URL backend → Kết nối**. Chip đầu trang đổi thành "Đã kết nối".
5. Dashboard: Upload (hoặc nhập đường dẫn, vd. `/kaggle/input/<dataset>/4.mp4`) → **Phân tích**.

Lưu ý:
- URL cloudflared đổi mỗi lần chạy lại ô cuối.
- Server chỉ chạy một job một lúc. Nếu model đang nạp hoặc đang có job khác, web nhận lỗi 409 kèm lý do. **Dừng** gọi `POST /api/jobs/:id/cancel`.
- Đường dẫn trên server chỉ lấy trong `/kaggle/input`, `/kaggle/working`, `/content` (đổi bằng `EXHIBITFLOW_SOURCE_ROOTS`).
- Chế độ **Video** chạy mọi khung hình; kết quả (danh sách cảnh báo, video có khung) hiện khi xong. Cảnh báo ngay lúc chạy chỉ gồm các cảnh báo VLM đã kết luận **bất thường**.
- Nguồn camera (RTSP) vẫn chưa được hỗ trợ (server trả 501): engine mới đọc file video.
- Map 2D của dashboard (`/api/map-image`, vị trí người trên map) chưa được web này dùng; khối Map 2D vẫn giữ nguyên.
- Phiên Kaggle bị ngắt khi nghỉ lâu hoặc hết giờ GPU.

Thử không cần GPU (dữ liệu giả của dashboard): `python exhibitflow_pipeline/server/web_server.py --port 8000 --demo`.

(`server/server.py` là bản server cũ chạy `main.py` cho từng video, nạp lại model mỗi lần; giữ để tham khảo.)

## Cấu trúc

```
web/
├─ index.html      dashboard + Settings
├─ css/style.css
└─ js/
   ├─ config.js    demo true/false
   ├─ app.js       dữ liệu mẫu, chuyển result.json, vẽ video / map, nối backend, bảng cảnh báo
   └─ hex.js       họa tiết lục giác nền
exhibitflow_pipeline/
├─ server/server.py                           API FastAPI bọc main.py
└─ notebooks/exhibitflow_kaggle_server.ipynb  notebook Kaggle (server + đường hầm)
```

Sửa `server/server.py` xong, tạo lại notebook để hai bản khớp (notebook nhúng bản copy của file này).

## API (server ↔ web)

| Endpoint | Việc |
|---|---|
| `GET /api/health` | kiểm tra sống |
| `GET /api/system` | CPU / RAM / GPU / VRAM |
| `POST /api/jobs` | multipart `video` **hoặc** `source` (đường dẫn trên server / URL) + `options` → `{job_id}` |
| `GET /api/jobs/:id` | `{status: queued/running/done/error/cancelled, stage, stages, progress, log, key, error}` |
| `POST /api/jobs/:id/cancel` | dừng job |
| `GET /api/jobs/:id/result` | `result.json` của pipeline |
| `GET /api/jobs/:id/video` | `live.mp4` (có khung); `?kind=original` là video gốc |
| `GET /api/jobs/:id/files/<path>` | file trong thư mục kết quả (ảnh bằng chứng) |

## Chưa làm

- Phát kết quả theo thời gian thực từng phần (pipeline hiện chỉ trả kết quả khi xong).
- Khung người, skeleton do web tự vẽ: cần pipeline lưu hộp người theo khung hình. Hiện web phát `live.mp4` do pipeline vẽ sẵn.
- Map 2D, heatmap, hành trình: chờ module map 2D.
- Trang "Hành trình khách" (đã bỏ khỏi dashboard mới cho tới khi có dữ liệu map).
