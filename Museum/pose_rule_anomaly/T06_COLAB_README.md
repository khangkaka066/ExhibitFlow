# T06 training on Google Colab

Upload `t06_training_payload_v2.zip` to the root of Google Drive, then open and
run `T06_anomaly_training_colab.ipynb` from top to bottom with a CPU runtime.
Isolation Forest does not benefit materially from a GPU.

The notebook validates the payload before fitting, uses whole held-out segments
for validation, trains one shared camera-balanced model, and downloads
`t06_results.zip`. Do not interpret top scores as confirmed abnormal behavior;
they are review candidates for T07/T08.

V2 scores only windows with keypoint visibility >= 0.80 and valid-step ratio
>= 0.90. Other windows remain in the score file as `insufficient_evidence`
with null scores. Pose-confidence and missingness signals are excluded from the
Isolation Forest embedding so image quality is not itself treated as behavior.
