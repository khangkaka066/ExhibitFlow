# ExhibitFlow — pipeline phát hiện hành vi bất thường trong video bảo tàng

Đầu vào: **video**. Đầu ra: **file JSON** liệt kê các cảnh báo `touching` / `holding` / `stealing` (ai, hiện vật nào, lúc nào, độ tin cậy).

```
video ─► detector (YOLOX-s) ─► tracker (ByteTrack + DMA + LTC + ReID) ─► pose (YOLOv8-pose) ─► hand (RTMW)
                                                                              │
 hiện vật (Grounding-DINO + SAM, 1 lần/cảnh) ───────────────────────────────► state machine chạm / gom event
                                                                              │ đoạn ứng viên (8 ảnh)
                                                                              ▼
                                            VLM Qwen3-VL-8B Q4_K_M (llama.cpp) ─► result.json
```

Các stage chạy song song (thread + queue). VLM chạy ở tiến trình `llama-server` riêng, trên GPU thứ hai nếu máy có 2 GPU.

## Chạy nhanh

```bash
python scripts/install_deps.py          # cài phụ thuộc (giữ torch có sẵn)
python main.py --check                  # kiểm tra GPU, thư viện, trọng số
python main.py --videos a.mp4 b.mp4 --out results
python main.py --video-dir /data/videos --out results --weights-dir /data/weights
```

Kaggle / Colab: mở `notebooks/exhibitflow_kaggle.ipynb` hoặc `notebooks/exhibitflow_colab.ipynb` (chỉ clone repo nhánh `Hung`, cài, chạy `main.py`).

Trọng số: xem `weights/README.md` (4 file bắt buộc: `best_ckpt.pth`, `osnet_x0_25.pth`, `ltc_motion_mot17.pth`, `dma_gbm_mot17.gbm`).
Lần chạy đầu tự tải: GGUF Qwen3-VL-8B Q4_K_M + mmproj (~6 GB), Grounding-DINO, SAM2, YOLOv8-pose, llama.cpp bản dựng sẵn (cần driver CUDA ≥ 12.8,
nếu không sẽ tự build từ source, thêm 10-20 phút).

### Tham số `main.py`

| Cờ | Ý nghĩa |
|---|---|
| `--videos F...` / `--video-dir D` | video đầu vào (mp4/avi/mov/mkv/webm) |
| `--out D` | thư mục kết quả (mặc định `work/results`) |
| `--work-dir D` | cache: llama.cpp, mask hiện vật (mặc định `./work`) |
| `--weights-dir D` | thư mục trọng số (lặp lại được) |
| `--config F.yaml` | ghi đè cấu hình (xem `configs/default.yaml`) |
| `--set khoá=giá_trị` | ghi đè 1 giá trị, vd `models.vlm.max_side=768`, `rt.queue_size=4` |
| `--max-frames N` | chỉ chạy N frame đầu (thử nhanh) |
| `--save-video` | ghi `live.mp4` (khung người / hiện vật / cảnh báo) |
| `--dry-run-vlm` | không nạp VLM (nhãn giả) |
| `--clean-vlm-inputs` | xoá ảnh gửi VLM sau khi xong |
| `--cpu` | ép CPU |
| `--check` | chỉ kiểm tra môi trường rồi thoát |

Mã thoát: `0` ổn, `1` có video lỗi, `2` tham số / môi trường sai.

## Đầu ra

```
<out>/summary.json                       tổng hợp mọi video (n_alerts, lỗi, thời gian)
<out>/<video>/result.json                kết quả 1 video (schema bên dưới)
<out>/<video>/events.csv                 các event chạm
<out>/<video>/vlm_inputs/seg_*/...       ảnh đã gửi VLM (bằng chứng): marked/, marked_512/, prompt.txt, sheet.jpg
<out>/<video>/live.mp4                   chỉ khi --save-video
```

`result.json` (`schema_version` = `"1.0"`):

| Khoá | Nội dung |
|---|---|
| `video` | `path, name, key, fps, n_frames, width, height` |
| `summary` | `n_people_tracks, n_events, n_alerts, alerts_by_behavior, n_vlm_tasks, has_anomaly` |
| **`alerts`** | mỗi cảnh báo: `event_id, track_id, artifact_id, behavior, confidence, evidence, alarm_start_frame/_time_s, alarm_end_frame/_time_s, onset_source, task` |
| `events` | mọi event chạm (kể cả `NORMAL` / `CANDIDATE`): `start/end_frame, start/end_time_s, decision, behavior, vlm_confidence, alarm_spans, ...` |
| `vlm_tasks` | từng lần gọi VLM: `label, confidence, raw, latency_s, infer_s, prefill_s, queue_wait_s, status, error` |
| `artifacts`, `scenes` | hiện vật tìm thấy (id, nhãn, box) và các cảnh |
| `performance` | FPS pipeline, ms / stage, e2e, thời gian VLM, VRAM đỉnh |
| `warnings` | vd segmentation lỗi (khi đó video không có hiện vật → không có event) |
| `config` | phiên bản, model, spec VLM, ngưỡng, danh sách GPU |

`decision`: `ANOMALY` (VLM trả touching / holding / stealing), `NORMAL`, hoặc `CANDIDATE` (VLM chưa trả lời / không chắc / lỗi).
Mốc thời gian theo giây của video = `frame / fps`.

## Cấu hình VLM hiện tại (`8f_512_short`)

Qwen3-VL-8B Q4_K_M, 8 ảnh × cạnh dài 512 px, prompt chỉ hỏi `label` + `confidence`, `max_new_tokens=64`, `-fa on`.
Đo trên 19 đoạn có nhãn (13 bất thường, 6 bình thường):

| GPU | F1 | Latency / đoạn |
|---|---|---|
| Kaggle T4 | 0.690 | 4.3 s |
| Colab G4 (RTX PRO 6000 Blackwell) | 0.690 | 0.53 s |

Đổi cấu hình = sửa `models.vlm` (`n_images`, `max_side`, `short_json`, `max_new_tokens`) trong YAML hoặc bằng `--set`.
Cấu hình đủ ảnh nhất (12 × 768, JSON đầy đủ) cho F1 0.74-0.79 nhưng chậm hơn khoảng 3-4 lần. Lưu ý: không cấu hình nào nhận đúng `stealing`
(behavior accuracy 5-21 %) — cảnh báo nên hiểu là "có chạm / cầm hiện vật", chưa phân biệt được lấy cắp.

## Cấu trúc mã

```
main.py                  CLI
exhibitflow/
  settings.py            mọi hằng số + ghi đè (YAML / --set)
  common.py              tiện ích ảnh / GPU, registry model, tìm trọng số
  perception.py          detector, tracker, pose, hand
  segmenter.py           hiện vật (Grounding-DINO + SAM), tách cảnh, cache
  interaction.py         pairing, state machine chạm, gom event, cắt ảnh cho VLM
  vlm.py                 llama-server (tải / chạy / gọi), VLMWorker, adapt_task, parse
  engine.py              các stage song song, ring buffer, live video
  pipeline.py            điều phối + ghi JSON
configs/default.yaml     ví dụ ghi đè
notebooks/               Kaggle + Colab
tests/                   test CPU (không cần GPU / trọng số)
```

Thêm / đổi model: viết class, gắn `@register('detector' | 'tracker' | 'pose' | 'hand' | 'segmenter' | 'vlm', 'tên')`, rồi đổi `backend`
trong `MODELS`.

## Kiểm thử

```bash
CUDA_VISIBLE_DEVICES= python tests/test_pipeline_cpu.py
```

3 test chạy CPU trong vài giây, dùng video tổng hợp + detector / pose / segmenter giả + `llama-server` giả. Chúng kiểm tra toàn bộ đường đi thật của
engine, state machine, cắt ảnh, `n_images=8`, `max_side=512`, và schema `result.json`. Chúng **không** kiểm tra model thật.

## Hạn chế đã biết

- Chưa chạy end-to-end với model thật trên GPU (segmenter Grounding-DINO + SAM, RTMW, llama-server thật) bằng package này. Từng thành phần đã được
  chạy trong notebook benchmark trên Kaggle (T4 × 2) và Colab (G4); package giữ nguyên code và ngưỡng. Lần chạy đầu trên GPU mới nên đi qua `--check` và `--max-frames 200`.
- Detector + tracker + ReID thật đã được chạy thử trên CPU với 1 video thật (40 frame): nạp trọng số và chạy ổn.
- Máy deploy chưa xác định (OS, GPU, driver). Bản llama.cpp dựng sẵn là Ubuntu x64 CUDA 12.8; máy khác sẽ phải build từ source.
- Perception chưa được đo FPS trên Colab; số FPS tham khảo là từ Kaggle (tracker `original`: khoảng 22.8 FPS ở chế độ standalone).
