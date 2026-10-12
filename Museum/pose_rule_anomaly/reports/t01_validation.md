# T01 validation report
Total annotation rows: 53631

## Image existence
Unique referenced images: 4820
Missing images: 0

## Bbox sanity (bounds 0..1280 x 0..800)
Rows with invalid bbox: 13
Rows (visitor_id, camera_id, frame_id, bb_x, bb_y, bbV_x, bbV_y):
- v2 cam1 f7367 bb=(2575,290) bbV=(2575,290)
- v21 cam1 f7418 bb=(755,94) bbV=(987650,94)
- v16 cam1 f7430 bb=(754,96) bbV=(63161227,96)
- v13 cam1 f7448 bb=(836,100) bbV=(828,63161120)
- v21 cam1 f7463 bb=(736,99) bbV=(63161227,95)
- v3 cam1 f7962 bb=(572,116) bbV=(63161227,63161120)
- v11 cam1 f9574 bb=(701,108) bbV=(12417197,108)
- v26 cam1 f14429 bb=(688,76) bbV=(243,63161120)
- v1 cam2 f2803 bb=(1151,166) bbV=(63161227,169)
- v1 cam2 f2856 bb=(873,134) bbV=(63161227,63161120)
- v4 cam2 f2972 bb=(0,334) bbV=(-3,334)
- v4 cam2 f2973 bb=(0,334) bbV=(-3,334)
- v11 cam3 f5934 bb=(816,85) bbV=(816,63161120)

## Duplicate (visitor_id, frame_id) within camera+segment
Duplicate rows: 0

## Segments

### Camera 1
- seg01: frame 3820..4424 (605 frames)
- seg02: frame 7246..8200 (955 frames)
- seg03: frame 8237..8239 (3 frames) SHORT (exclude from sequence/window learning)
- seg04: frame 8243..8246 (4 frames) SHORT (exclude from sequence/window learning)
- seg05: frame 9574..10573 (1000 frames)
- seg06: frame 14135..14544 (410 frames)
- seg07: frame 15973..16972 (1000 frames)

### Camera 2
- seg01: frame 2801..3051 (251 frames)
- seg02: frame 6568..6789 (222 frames)

### Camera 3
- seg01: frame 5850..6100 (251 frames)
- seg02: frame 15546..15664 (119 frames)

Total segments across all cameras: 11 (PLAN.md expects 11)
