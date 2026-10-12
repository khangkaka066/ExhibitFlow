"""Train the T06 shared normal-only Isolation Forest baseline.

Run from Museum/pose_rule_anomaly/:
    python -m src.anomaly.train_baseline

Use ``--validate-only`` to verify the payload without fitting a model.
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "windows.npz"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "anomaly"
RANDOM_STATE = 42


def load_windows(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict], list[str]]:
    data = np.load(path)
    features = data["features"].astype(np.float32, copy=False)
    masks = data["keypoint_mask"].astype(bool, copy=False)
    metadata = [json.loads(value) for value in data["metadata_json"]]
    names = data["feature_names"].tolist()
    if features.ndim != 3 or features.shape[:2] != masks.shape[:2]:
        raise ValueError(f"incompatible feature/mask shapes: {features.shape}, {masks.shape}")
    if len(features) != len(metadata) or features.shape[2] != len(names):
        raise ValueError("metadata or feature-name count does not match tensor")
    if np.isinf(features).any():
        raise ValueError("feature tensor contains infinity")
    if any(row.get("source") != "real_museum" for row in metadata):
        raise ValueError("training payload contains a non-real_museum source")
    return features, masks, metadata, names


def summarize_windows(features: np.ndarray, names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build kinematic summaries without exposing pose-quality signals to the model."""
    keep = np.asarray([not name.startswith("keypoint_confidence_") for name in names])
    selected = features[:, :, keep]
    selected_names = [name for name, include in zip(names, keep) if include]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(selected, axis=1)
        spread = np.nanpercentile(selected, 75, axis=1) - np.nanpercentile(selected, 25, axis=1)
        minimum = np.nanmin(selected, axis=1)
        maximum = np.nanmax(selected, axis=1)
    embedding = np.concatenate([median, spread, minimum, maximum], axis=1).astype(np.float32)
    embedding_names = (
        [f"median:{name}" for name in selected_names]
        + [f"iqr:{name}" for name in selected_names]
        + [f"min:{name}" for name in selected_names]
        + [f"max:{name}" for name in selected_names]
    )
    return embedding, embedding_names


def segment_split(metadata: list[dict]) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Hold out the last segment of each camera; never split windows by frame."""
    by_camera: dict[int, set[str]] = defaultdict(set)
    for row in metadata:
        by_camera[int(row["camera_id"])].add(row["segment_id"])
    held_out = {camera: sorted(segments)[-1] for camera, segments in by_camera.items() if len(segments) > 1}
    validation = np.asarray([
        held_out.get(int(row["camera_id"])) == row["segment_id"] for row in metadata
    ], dtype=bool)
    train = ~validation
    if not train.any() or not validation.any():
        raise ValueError("segment split produced an empty train or validation set")
    return train, validation, held_out


def balanced_training_indices(metadata: list[dict], train_mask: np.ndarray, cap: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE)
    by_camera: dict[int, list[int]] = defaultdict(list)
    for index, (row, eligible) in enumerate(zip(metadata, train_mask)):
        if eligible:
            by_camera[int(row["camera_id"])].append(index)
    selected = []
    target = min(cap, *(len(indices) for indices in by_camera.values()))
    for camera in sorted(by_camera):
        indices = np.asarray(by_camera[camera])
        if len(indices) > target:
            indices = rng.choice(indices, target, replace=False)
        selected.extend(indices.tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=np.int64)


def percentile_against(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / max(len(ordered), 1)


def save_score_plots(scores: np.ndarray, metadata: list[dict], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for camera, axis in zip((1, 2, 3), axes):
        values = scores[[int(row["camera_id"]) == camera for row in metadata]]
        values = values[np.isfinite(values)]
        axis.hist(values, bins=50, color="#3b82f6", alpha=0.85)
        axis.set_title(f"Camera {camera} (n={len(values)})")
        axis.set_xlabel("anomaly score (higher = stranger)")
    fig.suptitle("T06 Isolation Forest score distributions")
    fig.savefig(output_dir / "score_distribution.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--max-train-per-camera", type=int, default=2500)
    parser.add_argument("--min-visibility", type=float, default=0.80)
    parser.add_argument("--min-valid-steps", type=float, default=0.90)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    features, masks, metadata, names = load_windows(args.input)
    embedding, embedding_names = summarize_windows(features, names)
    train_mask, validation_mask, held_out = segment_split(metadata)
    eligible = np.asarray([
        row["keypoint_visibility"] >= args.min_visibility
        and row["valid_step_ratio"] >= args.min_valid_steps
        for row in metadata
    ], dtype=bool)
    train_mask &= eligible
    validation_mask &= eligible
    selected = balanced_training_indices(metadata, train_mask, args.max_train_per_camera)
    print(f"windows={len(features)}, temporal_shape={features.shape[1:]}, embedding={embedding.shape}")
    print(f"eligible={int(eligible.sum())}, insufficient_evidence={int((~eligible).sum())}")
    print(f"eligible_train_pool={int(train_mask.sum())}, balanced_fit={len(selected)}, eligible_validation={int(validation_mask.sum())}")
    print(f"held_out_segments={held_out}")
    print(f"fit_by_camera={dict(sorted(Counter(metadata[i]['camera_id'] for i in selected).items()))}")
    if args.validate_only:
        print("validation passed; model was not fitted")
        return

    import joblib
    from sklearn.ensemble import IsolationForest
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
        ("scaler", RobustScaler(quantile_range=(10.0, 90.0))),
        ("model", IsolationForest(
            n_estimators=args.trees,
            contamination="auto",
            max_samples="auto",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    print("fitting shared Isolation Forest...")
    pipeline.fit(embedding[selected])
    scores = np.full(len(embedding), np.nan, dtype=np.float64)
    scores[eligible] = -pipeline.decision_function(embedding[eligible])

    percentiles = np.full(len(scores), np.nan, dtype=np.float32)
    for camera in (1, 2, 3):
        camera_all = np.asarray([int(row["camera_id"]) == camera for row in metadata]) & eligible
        camera_ref = camera_all & train_mask
        percentiles[camera_all] = percentile_against(scores[camera_ref], scores[camera_all])

    model_bundle = {
        "pipeline": pipeline,
        "embedding_names": embedding_names,
        "source_feature_names": names,
        "held_out_segments": held_out,
        "random_state": RANDOM_STATE,
        "training_indices": selected,
    }
    joblib.dump(model_bundle, args.output_dir / "isolation_forest.joblib")

    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[np.argsort(-scores[eligible_indices])]
    rank = np.zeros(len(scores), dtype=np.int64)
    rank[order] = np.arange(1, len(order) + 1)
    with (args.output_dir / "anomaly_scores.jsonl").open("w") as f:
        for index, row in enumerate(metadata):
            record = dict(row)
            record.update({
                "anomaly_score": float(scores[index]) if eligible[index] else None,
                "camera_percentile": float(percentiles[index]) if eligible[index] else None,
                "global_rank": int(rank[index]) if eligible[index] else None,
                "evidence_status": "scored" if eligible[index] else "insufficient_evidence",
                "split": "validation" if validation_mask[index] else ("train_pool" if train_mask[index] else "not_scored"),
                "model": "isolation_forest_shared_normal_only",
            })
            f.write(json.dumps(record) + "\n")

    save_score_plots(scores, metadata, args.output_dir)
    top_camera = Counter(int(metadata[i]["camera_id"]) for i in order[:100])
    lines = [
        "# T06 Isolation Forest baseline\n\n",
        "This is a normal-only/mixed-normal pilot, not a labeled abnormal-behavior classifier.\n\n",
        f"- Windows: {len(features)}\n",
        f"- Scored windows: {int(eligible.sum())}; insufficient evidence: {int((~eligible).sum())}\n",
        f"- Fit sample: {len(selected)} (equal count per camera, capped at {args.max_train_per_camera})\n",
        f"- Validation windows: {int(validation_mask.sum())}\n",
        f"- Temporal input: `{features.shape}`; summary embedding: `{embedding.shape}`\n",
        f"- Held-out segments: `{held_out}`\n",
        f"- Top-100 outliers by camera: `{dict(sorted(top_camera.items()))}`\n",
        f"- Quality gate: visibility >= {args.min_visibility:.2f}, valid-step ratio >= {args.min_valid_steps:.2f}.\n",
        "- Ineligible windows have `evidence_status=insufficient_evidence` and null scores.\n",
        "- Keypoint confidence, missing-rate, visibility, and imputer missing indicators are excluded from the model embedding.\n",
        "- Higher `anomaly_score` and `camera_percentile` mean more unusual among scored windows.\n",
        "- Thresholds are not event labels; review top windows before choosing operating thresholds.\n\n",
        "## Leakage controls\n\n",
        "Validation uses whole held-out segments, never a random frame/window split. One shared model is fitted across cameras with per-camera sampling caps; camera-specific percentiles are calibrated from each camera's training pool.\n\n",
        "## Required next review\n\n",
        "Inspect top-ranked windows for pose failures, occlusion, distant people and ID issues. If those dominate, clean the affected windows and retrain before T07 fusion.\n",
    ]
    (args.output_dir / "training_report.md").write_text("".join(lines))
    print(f"wrote model and scores -> {args.output_dir}")


if __name__ == "__main__":
    main()
