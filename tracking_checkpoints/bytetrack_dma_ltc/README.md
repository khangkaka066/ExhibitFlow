# Checkpoint ByteTrack + DMA + LTC

- `best_ckpt.zip`: preserved but not loaded by the current RGB pipeline; its detector input is 12-channel and does not match RGB video frames.
- `dma_gbm_mot17.gbm`: DMA GBM checkpoint trained on MOT17.
- `ltc_motion_mot17.pth`: LTC motion checkpoint trained on MOT17.

The active Colab touch pipeline uses `yolov8n.pt` (nano) for person boxes, then ByteTrack-DMA-LTC assigns IDs using the GBM and LTC files below, and finally `yolov8n-pose.pt` estimates pose on crops for currently detected tracks. YOLO nano weights are downloaded by Ultralytics in Colab; they are distinct from tracker weights. It also loads pretrained OSNet appearance weights so DMA can use appearance features. DMA and LTC are separate because the tracker loads them through different options.
