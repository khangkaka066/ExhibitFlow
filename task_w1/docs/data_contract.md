# ExhibitFlow — Data Contract (`schema_v1`)

Status: **DRAFT — for team review at Checkpoint C0**

Single agreement between the three workstreams — **Tracking**, **Spatial & Analytics**,
**Dashboard**. Every module reads/writes exactly these files with exactly these columns.
Any change = bump to `schema_v2` + notify the team.

### What already exists vs. what is spec-only

| Exists now (`exhibitflow/`) | Spec only — built in Week 2 |
|---|---|
| `docs/data_contract.md` (this file) | `tracks_mapped.csv` |
| `data/zones.json`, `data/floor_plan.png` | `zone_stats.csv`, `transitions.csv`, `journeys.csv` |
| `data/mock_tracks.csv` (+ `_truth.csv`) | `analytics/metrics.py`, `viz/floor_map.py` |
| `data/tracks_raw_museumvisitors.csv` | |
| `analytics/geometry.py` (homography + zone test) | |

---

## 1. Pipeline & file flow

```
video / image frames
        │
        ▼  Tracking: YOLOX + ByteTrack            ── OR ──  ground-truth annotations
   tracks_raw.csv                                           (MuseumVisitors → converter)
        │
        ▼  Spatial & Analytics: geometry.py  (homography: img px → m, + zone test)
   tracks_mapped.csv
        │
        ├─▼  metrics.py   zone_stats.csv
        ├─▼  metrics.py   transitions.csv
        ├─▼  metrics.py   journeys.csv
        │
        ▼  Dashboard   reads tracks_mapped.csv + *_stats + floor_plan.png + zones.json
```

Until the real tracker is ready, **`mock_tracks.csv`** (synthetic) and
**`tracks_raw_museumvisitors.csv`** (real, hand-labelled) both follow the `tracks_raw.csv`
schema, so analytics and dashboard are built in parallel. The real tracker output later
drops in unchanged.

---

## 2. Coordinate conventions (read this first)

| Thing | Decision |
|---|---|
| **Image space** | pixels of the source video frame. Origin top-left, `+x` right, `+y` down. |
| **Foot point** | where a person stands = `(x + w/2, y + h)` of the bbox (bottom-centre). All mapping uses the foot point, never the box centre. |
| **Floor space** | metres on the floor plan. Origin `(0,0)` = **top-left corner of `floor_plan.png`**, `+x` right, `+y` down (same orientation as the image — no axis flip). |
| **Floor → plan pixels** | `plan_px = floor_m * px_per_m` (`px_per_m` from `zones.json → floor_plan`). |
| **Time** | `timestamp` = seconds since the first frame of the clip (float). `t = 0.0` at frame 1. |
| **FPS** | per source. MuseumVisitors: nominal `5.0` (`timestamp = (frame_id - 1) / 5`). Synthetic mock: `10.0`. |
| **Missing zone** | foot point inside no zone polygon → `zone_id = ""` (empty string). |
| **Units** | distance = m, speed = m/s, dwell = s, area = m². |

---

## 3. `tracks_raw.csv` — tracker output (Tracking → Spatial & Analytics)

One row = one person in one frame.

| col | type | unit | notes |
|---|---|---|---|
| `frame_id` | int | — | 1-based, increasing; `track_id` may skip frames but `frame_id` itself has no gaps |
| `track_id` | int | — | stable per person for the whole clip; never reused |
| `x` | float | px | bbox top-left x (image space, original resolution) |
| `y` | float | px | bbox top-left y |
| `w` | float | px | bbox width |
| `h` | float | px | bbox height |
| `score` | float | 0–1 | detection confidence. Hand-labelled data uses `1.0` |
| `timestamp` | float | s | seconds since first frame |

Example:
```csv
frame_id,track_id,x,y,w,h,score,timestamp
1,1,812.4,540.2,44.0,110.0,0.93,0.000
1,2,410.1,380.5,40.0,100.0,0.88,0.000
2,1,813.0,540.9,44.0,110.0,0.92,0.100
```

Rules:
- Rows sorted by `(frame_id, track_id)`.
- A `track_id` may skip frames (occlusion / lost) — allowed; analytics handles gaps.
- Coordinates in the **original** frame resolution (MuseumVisitors: 1280×800). A tracker
  running on a resized frame must scale boxes back before writing.

---

## 4. `tracks_mapped.csv` — analytics input (produced by Spatial & Analytics)

`tracks_raw.csv` + 5 columns:

| col | type | unit | notes |
|---|---|---|---|
| …all `tracks_raw` columns… | | | unchanged |
| `img_foot_x` | float | px | `x + w/2` |
| `img_foot_y` | float | px | `y + h` |
| `floor_x` | float | m | foot point mapped through the homography |
| `floor_y` | float | m | |
| `zone_id` | str | — | zone polygon containing `(floor_x, floor_y)`, else `""` |

---

## 5. `zone_stats.csv` — per-zone summary

| col | type | unit | notes |
|---|---|---|---|
| `zone_id` | str | — | from `zones.json` |
| `zone_name` | str | — | human label |
| `area_m2` | float | m² | polygon area |
| `visitor_count` | int | — | distinct `track_id` that ever had a foot point in the zone |
| `n_visits` | int | — | number of contiguous stays (one track entering twice = 2) |
| `total_dwell_s` | float | s | summed duration of all stays across all visitors |
| `avg_dwell_s` | float | s | `total_dwell_s / n_visits` (0 if `n_visits = 0`) |
| `density_per_m2` | float | 1/m² | `visitor_count / area_m2` — hot/dead-zone ranking |

---

## 6. `transitions.csv` — zone-to-zone flow

| col | type | notes |
|---|---|---|
| `from_zone` | str | zone left |
| `to_zone` | str | next zone entered (consecutive in the track's zone sequence, `""` excluded) |
| `count` | int | number of tracks that made this transition |

---

## 7. `journeys.csv` — per-visitor path

One row per contiguous zone visit, ordered by time.

| col | type | unit | notes |
|---|---|---|---|
| `track_id` | int | — | |
| `visit_index` | int | — | 0,1,2… within this track |
| `zone_id` | str | — | |
| `enter_t` | float | s | timestamp entering the zone |
| `exit_t` | float | s | timestamp leaving |
| `dwell_s` | float | s | `exit_t - enter_t` |

The point-by-point polyline for drawing a trajectory is read straight from
`tracks_mapped.csv` filtered by `track_id`; `journeys.csv` is the zone-level summary.

---

## 8. `zones.json` — space definition (`data/zones.json`)

```jsonc
{
  "schema": "schema_v1",
  "floor_plan": { "image": "floor_plan.png", "width_m": 24.0, "height_m": 13.0, "px_per_m": 40 },
  "source":     { "dataset": "MuseumVisitors", "camera": 1,
                  "frame_ref": "frames/1/24-03-14/16h21m46s_3820.jpg",
                  "frame_size_px": [1280, 800], "fps": 5.0 },

  "reference_points": [        // >= 4, spread across the floor; used to build the homography
    { "id": "rp1",
      "landmark": "far wall meets floor, left side, near the doorway",
      "image_px": [305, 118],
      "image_px_status": "ESTIMATE - refine with scripts/pick_reference_points.py",
      "floor_m": [3.0, 0.5] }
    // ... rp2..rp6
  ],

  "zones": [                   // polygons in METRES; first match wins (priority = array order)
    { "id": "Z1", "name": "Cửa ra vào / lối vào",                     "polygon_m": [[0.0,1.0],[3.5,1.0],[3.5,6.0],[0.0,6.0]] },
    { "id": "Z2", "name": "Hiện vật tường xa (dãy tượng + tranh)",    "polygon_m": [[3.5,0.0],[22.0,0.0],[22.0,2.6],[3.5,2.6]] },
    { "id": "Z3", "name": "Tượng tường cửa sổ",                      "polygon_m": [[9.0,10.4],[23.0,10.4],[23.0,13.0],[9.0,13.0]] },
    { "id": "Z4", "name": "Tủ trưng bày bên trái",                   "polygon_m": [[0.0,6.0],[2.6,6.0],[2.6,12.5],[0.0,12.5]] },
    { "id": "Z5", "name": "Khu đoàn khách đứng",                     "polygon_m": [[12.5,5.5],[19.5,5.5],[19.5,10.0],[12.5,10.0]] },
    { "id": "Z6", "name": "Sảnh giữa (sàn trống)",                   "polygon_m": [[3.5,2.6],[22.0,2.6],[22.0,10.4],[3.5,10.4]] }
  ]
}
```

Current state / rules:
- `reference_points`: the 6 `image_px` values are **rough estimates** eyeballed from the
  reference frame — good enough to run, replace with `scripts/pick_reference_points.py` for
  an accurate homography. Minimum 4, must **not** be collinear, must be on the floor plane
  (never on a wall / pedestal / person), spread near *and* far from the camera.
- `floor_m` values are also estimates (the Bargello hall was not measured). Fine for the
  demo; measure or scale from a known object for final numbers.
- `zones`: 6 draft polygons in metres. Overlap resolved by array order (first zone that
  contains the point wins — `Z5` is listed before `Z6` on purpose). Point in no polygon → `""`.
- `floor_plan.png` is **generated** from this file by `scripts/make_floor_plan.py` — edit
  `zones.json`, re-run the script, never hand-edit the PNG (unless you replace it entirely
  and keep `width_m/height_m/px_per_m` in sync).

---

## 9. Directory layout

```
exhibitflow/
  docs/data_contract.md            <- this file
  data/
    zones.json                     <- space definition (committed)
    floor_plan.png                 <- schematic, generated (committed)
    mock_tracks.csv                <- synthetic, tracks_raw schema (committed)
    mock_tracks_truth.csv          <- ground-truth floor_x/floor_y/zone_id per row (local)
    zones_mock.json                <- zones.json + synthetic homography, for full-chain test (local)
    tracks_raw_museumvisitors.csv  <- real hand-labelled tracks, tracks_raw schema (local)
    tracks_mapped.csv              <- Week 2, generated
    zone_stats.csv  transitions.csv  journeys.csv   <- Week 2, generated
  analytics/
    geometry.py                    <- Space(): homography + zone_of() + floor_to_planpx()
    metrics.py                     <- Week 2: count / dwell / transition / journey
  viz/
    floor_map.py                   <- Week 2: live dots + trails on floor_plan.png
  scripts/
    make_floor_plan.py             zones.json      -> floor_plan.png
    gen_mock_tracks.py             zones.json      -> mock_tracks.csv + _truth + zones_mock.json
    museumvisitors_to_raw.py       annotations.dat -> tracks_raw_museumvisitors.csv
    pick_reference_points.py       click frame     -> fills zones.json reference_points.image_px
```

`scripts/`, `analytics/`, `viz/` and the three "(local)" data files are kept out of the
shared repo for now (see `exhibitflow/.gitignore`); the committed set is the C0 hand-off:
`data_contract.md`, `zones.json`, `floor_plan.png`, `mock_tracks.csv`.

---

## 10. MuseumVisitors → `tracks_raw.csv` mapping (reference)

`annotations.dat` — comma-separated, 16 fields per row:

```
visitorid, cameraid, frameid, bb_x, bb_y, bb_w, bb_h,
bbV_x, bbV_y, bbV_w, bbV_h, gaze_x, gaze_y, filename, operaid, groupid
```
(`bbV_*` = visible-part box, `bb_*` = full box incl. occluded part.)

Converter (`scripts/museumvisitors_to_raw.py`):

| tracks_raw | from |
|---|---|
| `track_id` | `visitorid` |
| `frame_id` | `frameid`, reindexed to 1..N within the chosen recording session |
| `x,y,w,h` | `bb_x, bb_y, bb_w, bb_h` (full box) |
| `score` | `1.0` |
| `timestamp` | `(reindexed frame_id - 1) / 5.0` — filename wall clock is only 1 s resolution, so it is used only as a sanity check, not as the timestamp |

Camera 1 = ~49.8k of ~53.6k rows (43 visitors). `frameid` has large gaps → the converter
splits into recording sessions on gaps `> 50` frames:

| session | frameid range | rows |
|---|---|---|
| 0 | 3820–4424 | 5 022 |
| 1 | 7246–8246 | 10 866 |
| 2 | 9574–10573 | 13 198 |
| 3 | 14135–14544 | 3 460 |
| 4 | 15973–16972 | 17 273 |

Default working clip = **session 4** (`--session largest`): 29 tracks, 1000 frames,
`timestamp` 0–200 s.
