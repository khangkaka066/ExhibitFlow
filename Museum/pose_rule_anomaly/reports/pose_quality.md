# T02 pose quality report

Model: torchvision `keypointrcnn_resnet50_fpn` (COCO, 17 keypoints), one forward pass per unique frame; each annotation row matched to the best-IoU detection (threshold 0.3). `track_id` reuses the dataset's `visitor_id` — no new tracker was introduced, so there are no ID-switch events to count. The 'track break' metric below is instead a frame-coverage continuity measure per (camera, segment, visitor_id).

## Camera 1
Total annotation rows: 49819
- ok: 36616 (73.5%)
- low_confidence: 10715 (21.5%)
- no_match: 2488 (5.0%)
- no_person_detected: 0 (0.0%)

Wrist visibility rate (of ok/low_confidence rows): 40449/47331 (85.5%)
Ankle visibility rate (of ok/low_confidence rows): 39452/47331 (83.4%)

## Camera 2
Total annotation rows: 1707
- ok: 1488 (87.2%)
- low_confidence: 130 (7.6%)
- no_match: 89 (5.2%)
- no_person_detected: 0 (0.0%)

Wrist visibility rate (of ok/low_confidence rows): 1579/1618 (97.6%)
Ankle visibility rate (of ok/low_confidence rows): 1480/1618 (91.5%)

## Camera 3
Total annotation rows: 2105
- ok: 1978 (94.0%)
- low_confidence: 113 (5.4%)
- no_match: 14 (0.7%)
- no_person_detected: 0 (0.0%)

Wrist visibility rate (of ok/low_confidence rows): 2062/2091 (98.6%)
Ankle visibility rate (of ok/low_confidence rows): 1983/2091 (94.8%)

## Track frame-coverage continuity
- Camera 1: 102 tracks, mean coverage 0.96, min 0.44
- Camera 2: 14 tracks, mean coverage 0.99, min 0.89
- Camera 3: 16 tracks, mean coverage 1.00, min 1.00

## Reliability verdict per camera
- Camera 1: reliable (pose-match rate 0.95, wrist visibility 0.85)
- Camera 2: reliable (pose-match rate 0.95, wrist visibility 0.98)
- Camera 3: reliable (pose-match rate 0.99, wrist visibility 0.99)
