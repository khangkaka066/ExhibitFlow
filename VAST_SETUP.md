# Rút ngắn thời gian khởi động ExhibitFlow trên Vast.ai

## Sự thật quan trọng về chi phí (đọc trước)

`PROVISIONING_SCRIPT` **chạy bên trong instance, sau khi container boot** — tức là trên máy GPU bạn đã thuê.
Vì vậy **không có cách nào để một provisioning script chạy "miễn phí" trước khi tính phí**: nó chỉ chuyển việc
cài đặt từ thao tác tay sang tự động (bạn không phải ngồi chờ rồi gõ lệnh), nhưng thời gian vẫn bị tính tiền.
Docs Vast chỉ nói script chạy "on first boot", không mô tả chi tiết thời điểm bắt đầu tính phí (trang billing không truy cập được
lúc kiểm tra) — hãy tự xác minh trên giao diện: xem cột trạng thái (`loading` → `running`) và mục *Billing* của instance.
Đừng giả định phần `loading` / provisioning là miễn phí.

| Giải pháp | Cài đặt chạy ở đâu | Tốn tiền GPU? |
|---|---|---|
| `make` tay (hiện tại) | trên instance, sau boot | Có, và phải ngồi chờ |
| **`PROVISIONING_SCRIPT` = `setup.sh`** | trên instance, tự động lúc boot | Có (nhưng tự chạy, không cần can thiệp) |
| **Docker image dựng sẵn** (khuyến nghị để giảm thật sự) | GitHub Actions / máy bạn, **trước** khi thuê | Chỉ còn thời gian pull image |
| Volume / instance đã cài sẵn rồi `stop` | lần thuê đầu | Chỉ trả phí lưu trữ khi dừng; start lại không cài lại |

## Phần nào làm trước / sau khi GPU bắt đầu tính phí

**Trước (miễn phí, làm trên máy bạn / GitHub Actions):**
- Build + push Docker image từ `exhibitflow_pipeline/docker/Dockerfile` (đã có sẵn: llama-server build sẵn, deps, code, model bản đồ ~200 MB).
- Push code lên nhánh `Hung`, tạo template Vast.

**Sau khi tạo instance (tính phí):**
- Pull image / boot container, chạy `PROVISIONING_SCRIPT` (`setup.sh`): clone, `make setup` (deps, tải weights HF, build llama.cpp nếu thiếu, model bản đồ).
- Runtime: `make run` → dashboard.

Với `setup.sh` + image `vastai/pytorch` (giữ nguyên) thì mọi thứ trong cột "sau" vẫn tính phí; lần build llama.cpp (~10–20 phút trên Ubuntu 22.04) là phần tốn nhất.
Muốn giảm thật sự, dùng image dựng sẵn (xem cuối).

## Cấu hình Template (vastai/pytorch, giữ nguyên image/port/Jupyter/SSH)

1. Vast.ai → **Templates** → chọn template `vastai/pytorch` hiện tại → **Edit** (hoặc Clone).
2. Giữ nguyên Image, port (vd `-p 8000:8000`), launch mode Jupyter/SSH, env hiện có.
3. Thêm **Environment Variables**:
   - `PROVISIONING_SCRIPT` = `https://raw.githubusercontent.com/khangkaka066/ExhibitFlow/Hung/setup.sh`
   - `HF_TOKEN` = token của bạn (đặt trong ô env của Vast, **không** commit vào repo)
   - (tuỳ chọn) `START=1` để tự chạy dashboard sau setup; `MODE=full`, `PORT=8000`
4. **Save / Create and Use**. Chọn instance, thuê như bình thường.
5. Theo dõi tiến độ: Jupyter/SSH vào xem log provisioning (thường `/var/log/portal/provisioning.log`; nếu không có: `ls /var/log/portal/`), hoặc `tail -f` log của instance trên giao diện.
6. Khi xong (log in `[setup] XONG`): `cd /workspace/ExhibitFlow && make run` (không cần nếu `START=1`).

Nếu muốn giữ `entrypoint.sh` của Dockerfile: file đó chỉ dùng khi bạn dùng image tự build (ENTRYPOINT), không áp dụng với `vastai/pytorch`; không cần sửa.

## Lệnh CLI Vast.ai

```bash
pip install vastai && vastai set api-key <VAST_API_KEY>

# tìm máy (>= 24 GB VRAM)
vastai search offers 'gpu_ram>=24 num_gpus=1 reliability>0.98 cuda_vers>=12.4' -o 'dph+'

# tạo instance từ template đã lưu (hash lấy ở `vastai search templates` hoặc trang Templates)
vastai create instance <OFFER_ID> --template_hash <TEMPLATE_HASH> --disk 80

# hoặc không dùng template
vastai create instance <OFFER_ID> --image vastai/pytorch:latest --disk 80 --ssh --direct \
  --env '-p 8000:8000 -e PROVISIONING_SCRIPT=https://raw.githubusercontent.com/khangkaka066/ExhibitFlow/Hung/setup.sh -e HF_TOKEN=hf_xxx -e START=1'

vastai show instances            # trạng thái loading / running
vastai logs <INSTANCE_ID>        # xem log provisioning
vastai stop instance <ID>        # dừng GPU (vẫn tính phí đĩa); `destroy instance` để xoá hẳn
```
(Tên cờ có thể khác theo phiên bản CLI: kiểm tra bằng `vastai create instance --help`.)

## Giảm chi phí thực sự: image dựng sẵn

`exhibitflow_pipeline/docker/Dockerfile` đã cài sẵn llama-server, deps, code, model bản đồ. Build trên GitHub Actions rồi:

```bash
docker build -t <user>/exhibitflow exhibitflow_pipeline/docker/   # context cần đủ code; xem comment trong Dockerfile
docker push <user>/exhibitflow
```
Trong template đặt Image = `<user>/exhibitflow`, env `HF_TOKEN` (trọng số tải ở lần chạy đầu vào `/workspace/weights`, ~vài GB).
Không nên nấu `HF_TOKEN` hoặc weights private vào image public.
