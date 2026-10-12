# T06 normal-only behavior model

- `isolation_forest.joblib`: trained shared Isolation Forest on pose/trajectory windows from real MuseumVisitors footage.
- `anomaly_scores.jsonl`: scored training windows used to calibrate camera-specific percentiles.

The Colab touch pipeline scores 2.5-second windows from each tracked person and uses the corresponding camera's training-score distribution. The default review threshold is the 99th percentile. The output is `unusual_motion_candidate`, not a semantic violation label; pose-model/domain mismatch and the lack of labeled abnormal examples mean every result needs review.
