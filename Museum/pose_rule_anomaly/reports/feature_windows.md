# T05 temporal feature windows

- Accepted windows: 9563
- Excluded windows: 4082
- Window: 2.5s, stride: 1.0s, resampled steps: 16
- Feature tensor: `(9563, 16, 94)` (`float32`; missing values are NaN)
- Keypoint mask tensor: `(9563, 16, 17)`
- Time base: source filename seconds with observed frames distributed within each second; no fixed FPS assumption.
- Normalization: keypoints relative to hip center (bbox center fallback), scaled by person size.
- Quality gates: interpolation gap <= 0.75s, >=80% valid time steps, >=60% visible keypoints.
- Windows never cross a segment or track boundary.

## Accepted by camera
- Camera 1: 8764
- Camera 2: 477
- Camera 3: 322

## Exclusion reasons
- insufficient_keypoint_visibility: 2316
- insufficient_temporal_coverage: 1766

## Feature groups

Relative pose (34), keypoint confidence (17), pose velocity (34), bbox/shape/body geometry (7), and bbox-center velocity (2). Absolute image position is retained only as normalized bbox trajectory for motion modeling; ROI membership remains separate for rules.
