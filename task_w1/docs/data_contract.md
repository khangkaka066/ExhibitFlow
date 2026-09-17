# ExhibitFlow — Spatial & Analytics Data Contract

Status: aligned to the team's real, committed contract — **`../docs/data_contract_v0.md`**
(schema_version `"0.1"`, JSONL) written by Khang. This file does **not** redefine that
contract; it documents the downstream extension that Spatial & Analytics adds on top of it,
which `docs/data_contract_v0.md` intentionally leaves open ("Mapping nằm sau tracker" —
`PLAN_CPP_TRACKER.md`).

> Supersedes the earlier CSV-based `tracks_raw.csv` draft written in Week 1 before Khang's
> C++ skeleton and real contract existed. That draft is obsolete — nothing in this repo should
> read/write CSV track data any more, only JSONL.

---

## 1. Upstream contract (owned by Khang, not repeated here)

Read `../docs/data_contract_v0.md` for the authoritative tracker input/output schema. Key
points this module relies on:

- One JSON object per frame: `schema_version, sequence_id, camera_id, frame_id,
  timestamp_ms, image_size:{width,height}, tracks:[...]`
- Each track: `track_id, bbox:[x,y,w,h], point_image:[x+w/2, y+h], score`
- `point_image` (the foot point) is **already computed upstream** — this module never
  recomputes it, only maps it.
- ID scope: `(sequence_id, camera_id, track_id)`.
- `tracker_backend` labels the source (`"mock"`, `"synthetic_fixture"`, or a real backend
  name). **Never treat `"mock"` output as ground truth for count/dwell testing** — the mock
  backend may assign a new ID per detection (see `PLAN_CPP_TRACKER.md` §4.1).

## 2. Downstream extension (owned by Spatial & Analytics — this module)

### 2.1 `tracks_mapped.jsonl` — output of `tools/map_tracks.py`

Same envelope and same `tracks[]` array as the input JSONL, each track object extended with:

| field | type | unit | notes |
|---|---|---|---|
| `point_floor` | `[float, float]` | m | `point_image` mapped through the homography |
| `zone_id` | string | — | id of the zone polygon containing `point_floor`, else `""` |

### 2.2 `zone_stats.csv` — output of `tools/metrics.py`

| col | type | unit | notes |
|---|---|---|---|
| `zone_id` | str | — | from `zones.json` |
| `zone_name` | str | — | human label |
| `area_m2` | float | m² | polygon area |
| `visitor_count` | int | — | distinct `track_id` ever seen in the zone |
| `n_visits` | int | — | contiguous stays (leave + return = 2 visits) |
| `total_dwell_s` | float | s | summed duration of all stays |
| `avg_dwell_s` | float | s | `total_dwell_s / n_visits` |
| `density_per_m2` | float | 1/m² | `visitor_count / area_m2` |

### 2.3 `transitions.csv`

`from_zone, to_zone, count` — consecutive zone-to-zone moves per track (`""` excluded).

### 2.4 `journeys.csv`

`track_id, visit_index, zone_id, enter_t, exit_t, dwell_s` — one row per contiguous zone
visit, `enter_t`/`exit_t` in seconds (`timestamp_ms / 1000`).

## 3. Coordinate conventions

| Thing | Decision |
|---|---|
| Image space | pixels, origin top-left, `+x` right, `+y` down — same as `data_contract_v0.md` |
| Floor space | metres, origin `(0,0)` = top-left of `floor_plan.png`, `+x` right, `+y` down (no axis flip) |
| Floor → plan pixels | `plan_px = floor_m * px_per_m` (`zones.json → floor_plan.px_per_m`) |
| Time | seconds = `timestamp_ms / 1000` |
| Missing zone | `zone_id = ""` |

## 4. `zones.json` — space definition (`data/zones.json`)

```jsonc
{
  "floor_plan": { "image": "floor_plan.png", "width_m": 24.0, "height_m": 13.0, "px_per_m": 40 },
  "source": { "dataset": "MuseumVisitors", "camera": 1,
              "frame_ref": "frames/1/24-03-14/16h21m46s_3820.jpg",
              "frame_size_px": [1280, 800], "fps": 5.0 },
  "reference_points": [ /* >=4, image_px x floor_m pairs, used to build the homography */ ],
  "zones": [ /* Z1..Z6 polygons in metres, first-match-wins */ ]
}
```

**Current status:** dataset confirmed = **MuseumVisitors** (Bargello, Donatello hall).
`reference_points.image_px` are **rough estimates**, not yet clicked on the real reference
frame (`tools/pick_reference_points.py` exists for that but has not been run) — homography
works and gives plausible numbers, accuracy not yet verified. Per team decision: leave as-is
for now, revisit only if mapped output looks clearly wrong once real tracks arrive.

## 5. Where things live (see `task_w1/README.md` for the full map)

```
task_w1/
  docs/data_contract.md        <- this file
  data/zones.json  floor_plan.png
  data/synthetic_tracks.jsonl  <- dedicated synthetic fixture (NOT the C++ mock backend)
  data/tracks_mapped.jsonl  zone_stats.csv  transitions.csv  journeys.csv  floor_map_summary.png
  analytics/geometry.py        <- Space: homography + zone_of()
  tools/map_tracks.py  metrics.py  floor_map_viz.py  make_floor_plan.py  gen_synthetic_tracks.py
```

## 6. Open items with the rest of the team

- **Khang:** where should `tracks_mapped.jsonl` / `zone_stats.csv` etc. live so the (future)
  dashboard can read them? Not yet specified in `data_contract_v0.md`.
- **Khang:** a real `tracks.jsonl` run on MuseumVisitors (he said he will re-run the real
  pipeline against MuseumVisitors instead of CAVIAR) — needed to replace the synthetic
  fixture with real tracker output for validation.
