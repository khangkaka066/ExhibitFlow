# ExhibitFlow Tracker Data Contract v0.1

The tracker CLI processes one sequence and one camera per run. It does not
store personal identity, face crops, or face recognition fields.

## Detection Input JSONL

Each line is one frame:

```json
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","frame_id":0,"timestamp_ms":0,"image_size":{"width":1920,"height":1080},"detections":[{"bbox":[100,200,60,160],"score":0.95}]}
```

Required fields:

- `schema_version`: must be `"0.1"`
- `sequence_id`: non-empty string, fixed for the run
- `camera_id`: non-empty string, fixed for the run
- `frame_id`: non-negative integer, strictly increasing
- `timestamp_ms`: non-negative integer milliseconds from sequence start, strictly increasing
- `image_size.width` and `image_size.height`: positive integer source-frame dimensions
- `detections`: array, present even when empty
- `bbox`: `[x, y, width, height]` in original image pixels, top-left origin, inside bounds
- `score`: finite number in `[0, 1]`

## Track Output JSONL

Each output line keeps the frame context and replaces detections with tracks:

```json
{"schema_version":"0.1","sequence_id":"demo_01","camera_id":"cam_01","frame_id":0,"timestamp_ms":0,"image_size":{"width":1920,"height":1080},"tracks":[{"track_id":1,"bbox":[100,200,60,160],"point_image":[130,360],"score":0.95}]}
```

Output fields:

- `track_id`: non-negative integer scoped to `(sequence_id, camera_id)`
- `bbox`: image-space tracking box
- `point_image`: `[x + width / 2, y + height]`, used later for floorplan mapping
- `score`: supporting detection score

The v0.1 CLI emits only detection-supported observations. It does not emit
predicted lost tracks or floorplan coordinates.
