# DMA feature extraction C++ core

The selected tracking bottleneck is the DMA association feature path. The
Python implementation evaluates every track/detection pair and performs a
small covariance solve for each pair. `exhibitflow_core` now contains an
initial C++17 port of that path in:

- `include/exhibitflow/dma_features.hpp`
- `src/core/dma_features.cpp`

## Contract

The six output columns intentionally match
`external/ByteTrack-DMA-LTC-Motion-Tracker/yolox/DMA/features.py`:

| Index | Feature | Definition |
| ---: | --- | --- |
| 0 | `motion_cost` | `1 - IoU(predicted_bbox, detection_bbox)` |
| 1 | `mahalanobis_norm` | Four-dimensional innovation distance divided by `9.4877`, clipped to `[0, 1]` |
| 2 | `cov_trace_log` | `log1p(trace(innovation_covariance))` |
| 3 | `cosine_dist` | Cosine distance between smoothed track and current detection embeddings |
| 4 | `bbox_area_log` | `log1p(max(width, 1) * max(height, 1)) / 15` |
| 5 | `tracklet_len_norm` | `min(tracklet_len / 30, 1)` |

`DmaTrackState::innovation_covariance` is the already projected 4x4
innovation covariance for `[center_x, center_y, aspect_ratio, height]`, stored
row-major. Keeping Kalman projection outside this core makes the boundary
explicit and avoids tying the initial C++ library to a particular tracker
implementation.

`extract_dma_batch_features()` returns row-major `(track, detection)` pairs.
It factorises each track covariance once and reuses that factor for every
detection in the row. Empty appearance embeddings use the neutral cosine
distance `0.5`; invalid covariance uses the Python path's fallback
Mahalanobis value `1.0`.

This milestone ports feature extraction only. ReID inference, LTC prediction,
the LightGBM weight model, and assignment remain behind the existing Python
ByteTrack integration boundary until their data/ABI contracts are selected.

## Test

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

`exhibitflow_dma_feature_tests` checks the feature definitions, scalar/batch
equivalence, fallbacks, and empty-batch behavior.
