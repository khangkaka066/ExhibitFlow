#!/usr/bin/env python3
"""Run ByteTrack-DMA-LTC on one video and write MOT-style tracks.

This is an ExhibitFlow-owned video entry point. The upstream demo_track.py
supports basic video inference, but its video loop does not pass the raw frame
into BYTETracker.update(). ReID and DMA need that frame for appearance crops, so
this runner keeps the model loading path and calls update(..., frame=frame).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ByteTrack-DMA-LTC video inference.")
    parser.add_argument("--model-repo", required=True, type=Path)
    parser.add_argument("--exp-file", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--result-txt", required=True, type=Path)
    parser.add_argument("--annotated-video", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--tsize", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--nms", type=float)
    parser.add_argument("--track-thresh", type=float, default=0.15)
    parser.add_argument("--track-buffer", type=int, default=30)
    parser.add_argument("--match-thresh", type=float, default=0.8)
    parser.add_argument("--min-box-area", type=float, default=1.0)
    parser.add_argument("--aspect-ratio-thresh", type=float, default=1.6)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--show-scale", type=float, default=1.0)
    parser.add_argument("--window-name", default="ExhibitFlow Tracking")
    parser.add_argument("--ltc-motion-ckpt", type=Path)
    parser.add_argument("--ltc-device", default="cpu")
    parser.add_argument("--ltc-history-len", type=int, default=16)
    parser.add_argument("--ltc-min-history", type=int, default=8)
    parser.add_argument("--ltc-covariance-scale", type=float, default=1.0)
    parser.add_argument("--ltc-max-abs-residual", type=float, default=256.0)
    parser.add_argument("--with-reid", action="store_true")
    parser.add_argument("--reid-backend", choices=("deep", "fast"), default="fast")
    parser.add_argument("--reid-device", default="cpu")
    parser.add_argument("--reid-weight", type=float, default=0.35)
    parser.add_argument("--reid-thresh", type=float, default=0.7)
    parser.add_argument("--reid-alpha", type=float, default=0.9)
    parser.add_argument("--reid-model", default="osnet_x1_0", help="torchreid model name (deep backend)")
    parser.add_argument("--reid-model-path", default="")
    parser.add_argument("--fast-reid-config", type=Path)
    parser.add_argument("--fast-reid-weights", type=Path)
    parser.add_argument("--fast-reid-batch-size", type=int, default=16)
    parser.add_argument("--ml", choices=("gbm", "xgb", "sklearn"))
    parser.add_argument("--ml-weights", type=Path)
    parser.add_argument("--dma-weights", type=Path)
    parser.add_argument("--dma-device", default="cpu")
    parser.add_argument("--summary-json", type=Path)
    return parser


def add_model_repo_to_path(model_repo: Path) -> None:
    model_repo = model_repo.resolve()
    sys.path.insert(0, str(model_repo))
    fast_reid_root = model_repo / "fast-reid"
    if fast_reid_root.is_dir():
        sys.path.insert(0, str(fast_reid_root))
    os.environ.setdefault("PYTHONPATH", str(model_repo))


def build_tracker_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        min_box_area=args.min_box_area,
        mot20=args.mot20,
        ltc_motion_ckpt=str(args.ltc_motion_ckpt.resolve()) if args.ltc_motion_ckpt else None,
        ltc_device=args.ltc_device,
        ltc_history_len=args.ltc_history_len,
        ltc_min_history=args.ltc_min_history,
        ltc_covariance_scale=args.ltc_covariance_scale,
        ltc_max_abs_residual=args.ltc_max_abs_residual,
        with_reid=args.with_reid,
        reid_backend=args.reid_backend,
        reid_device=args.reid_device,
        reid_weight=args.reid_weight,
        reid_thresh=args.reid_thresh,
        reid_alpha=args.reid_alpha,
        reid_model=args.reid_model,
        reid_model_path=args.reid_model_path,
        fast_reid_config=str(args.fast_reid_config.resolve()) if args.fast_reid_config else "",
        fast_reid_weights=str(args.fast_reid_weights.resolve()) if args.fast_reid_weights else "",
        fast_reid_batch_size=args.fast_reid_batch_size,
        ml=args.ml,
        ml_weights=str(args.ml_weights.resolve()) if args.ml_weights else None,
        dma_weights=str(args.dma_weights.resolve()) if args.dma_weights else None,
        dma_device=args.dma_device,
    )


def install_dma_gbm_adapter(model_repo: Path) -> None:
    """Install a lightweight yolox.DMA module for GBM inference.

    The model repo's yolox.DMA package imports every backend from __init__,
    including XGBoost. ExhibitFlow's default config uses only the GBM weight
    from the Drive folder, so this adapter exposes the DMAFusion symbol that
    BYTETracker expects without requiring unrelated XGBoost packages.
    """

    if "yolox.DMA" in sys.modules:
        return

    features_path = model_repo / "yolox" / "DMA" / "features.py"
    spec = importlib.util.spec_from_file_location("exhibitflow_dma_features", features_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load DMA features from {features_path}")
    features_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(features_module)

    class PureLightGBMTree:
        def __init__(self, split_feature, threshold, left_child, right_child, leaf_value):
            self.split_feature = split_feature
            self.threshold = threshold
            self.left_child = left_child
            self.right_child = right_child
            self.leaf_value = leaf_value

        def predict_one(self, row) -> float:
            node = 0
            while node >= 0:
                feature_index = self.split_feature[node]
                threshold = self.threshold[node]
                if row[feature_index] <= threshold:
                    node = self.left_child[node]
                else:
                    node = self.right_child[node]
            return self.leaf_value[-node - 1]

    class DynamicWeightGBM:
        def __init__(self, trees):
            self.trees = trees

        def to(self, device):
            return self

        def eval(self):
            return self

        def predict_numpy(self, x_np):
            x_np = np.asarray(x_np, dtype=np.float32)
            raw = np.zeros((x_np.shape[0],), dtype=np.float64)
            for tree in self.trees:
                raw += np.asarray([tree.predict_one(row) for row in x_np], dtype=np.float64)
            w_motion = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
            w_reid = 1.0 - w_motion
            return np.stack([w_motion, w_reid], axis=1).astype(np.float32)

        @staticmethod
        def _parse_numbers(value: str, cast):
            if not value:
                return []
            return [cast(item) for item in value.split()]

        @classmethod
        def _parse_tree(cls, block: str):
            values = {}
            for line in block.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                values[key] = value
            return PureLightGBMTree(
                split_feature=cls._parse_numbers(values.get("split_feature", ""), int),
                threshold=cls._parse_numbers(values.get("threshold", ""), float),
                left_child=cls._parse_numbers(values.get("left_child", ""), int),
                right_child=cls._parse_numbers(values.get("right_child", ""), int),
                leaf_value=cls._parse_numbers(values.get("leaf_value", ""), float),
            )

        @classmethod
        def load(cls, path: str):
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            model_str = payload["model_str"]
            trees = [
                cls._parse_tree(block)
                for block in model_str.split("\n\n")
                if block.startswith("Tree=")
            ]
            if not trees:
                raise ValueError(f"no LightGBM trees found in {path}")
            return cls(trees), payload.get("stats", None)

    class DMAFusion:
        def __init__(self, model, stats: dict, device: str = "cpu"):
            if stats is None:
                raise ValueError("DMA checkpoint has no normalisation stats")
            self.model = model.to(device)
            self.model.eval()
            self.device = device
            self.mean = np.asarray(stats["mean"], dtype=np.float32)
            self.std = np.asarray(stats["std"], dtype=np.float32)
            self.feature_indices = stats.get("feature_indices", None)

        @classmethod
        def from_checkpoint(cls, ckpt_path: str, device: str = "cpu"):
            if Path(ckpt_path).suffix != ".gbm":
                raise ValueError("ExhibitFlow DMA adapter currently supports .gbm checkpoints")
            model, stats = DynamicWeightGBM.load(ckpt_path)
            return cls(model, stats, device=device)

        def fuse(self, tracks, detections, motion_cost, appearance_cost, kf, frame_id: int):
            if len(tracks) == 0 or len(detections) == 0:
                return motion_cost
            has_any_app = any(t.smooth_feat is not None for t in tracks) and any(
                d.curr_feat is not None for d in detections
            )
            if not has_any_app:
                return motion_cost
            feat_matrix = features_module.extract_batch_features(tracks, detections, kf, frame_id)
            flat = feat_matrix.reshape(-1, feat_matrix.shape[-1])
            if self.feature_indices is not None:
                flat = flat[:, self.feature_indices]
            flat = (flat - self.mean) / self.std
            flat = np.clip(flat, -10.0, 10.0)
            weights = self.model.predict_numpy(flat)
            w_motion = weights[:, 0].reshape(len(tracks), len(detections))
            w_reid = weights[:, 1].reshape(len(tracks), len(detections))

            for i, track in enumerate(tracks):
                if track.smooth_feat is None:
                    w_motion[i, :] = 1.0
                    w_reid[i, :] = 0.0
            for j, detection in enumerate(detections):
                if detection.curr_feat is None:
                    w_motion[:, j] = 1.0
                    w_reid[:, j] = 0.0
            return w_motion * motion_cost + w_reid * appearance_cost

    module = types.ModuleType("yolox.DMA")
    module.DMAFusion = DMAFusion
    module.FEAT_DIM = getattr(features_module, "FEAT_DIM", None)
    module.FEAT_NAMES = getattr(features_module, "FEAT_NAMES", None)
    module.extract_pair_features = features_module.extract_pair_features
    sys.modules["yolox.DMA"] = module


class ExhibitFlowFastReIDExtractor:
    """FastReID inference-only adapter for the vendored repo.

    The upstream fast_reid_interfece.py imports training/evaluation modules that
    are not needed for video inference and are incomplete in this model repo
    checkout. This adapter keeps the same crop/preprocess/checkpoint behavior
    while importing only the inference pieces.
    """

    def __init__(self, config_file: str, weights_path: str, device: str = "cpu", batch_size: int = 16):
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device("cuda" if device != "cpu" else "cpu")
        self.batch_size = max(1, int(batch_size))

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHTS", weights_path, "MODEL.DEVICE", self.device.type])
        cfg.MODEL.BACKBONE.PRETRAIN = False
        cfg.freeze()
        self.cfg = cfg

        self.model = build_model(cfg)
        Checkpointer(self.model).load(weights_path)
        self.model.eval().to(self.device)
        if self.device.type == "cuda":
            self.model.half()

        self.input_hw = tuple(cfg.INPUT.SIZE_TEST)

    def extract(self, frame, tlbrs):
        if frame is None or len(tlbrs) == 0:
            return [None] * len(tlbrs)
        height, width = frame.shape[:2]
        patches = []
        valid_indices = []
        input_h, input_w = self.input_hw
        for index, tlbr in enumerate(np.asarray(tlbrs, dtype=np.float32)):
            x1, y1, x2, y2 = tlbr[:4]
            x1 = max(0, min(width - 1, int(np.floor(x1))))
            y1 = max(0, min(height - 1, int(np.floor(y1))))
            x2 = max(0, min(width, int(np.ceil(x2))))
            y2 = max(0, min(height, int(np.ceil(y2))))
            if x2 <= x1 or y2 <= y1:
                continue
            patch = frame[y1:y2, x1:x2, :][:, :, ::-1]
            patch = cv2.resize(patch, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            tensor = torch.as_tensor(patch.astype("float32").transpose(2, 0, 1))
            if self.device.type == "cuda":
                tensor = tensor.half()
            patches.append(tensor)
            valid_indices.append(index)

        features = [None] * len(tlbrs)
        if not patches:
            return features

        with torch.no_grad():
            for start in range(0, len(patches), self.batch_size):
                batch = torch.stack(patches[start : start + self.batch_size], dim=0).to(self.device)
                output = self.model(batch)
                output[torch.isinf(output)] = 1.0
                output = F.normalize(output, dim=1).detach().cpu().numpy()
                for offset, feat in enumerate(output):
                    features[valid_indices[start + offset]] = np.asarray(feat, dtype=np.float32)
        return features


def build_exhibitflow_reid_extractor(args):
    backend = getattr(args, "reid_backend", "fast")
    if backend != "fast":
        from yolox.tracker.reid import build_reid_extractor as upstream_build_reid_extractor

        return upstream_build_reid_extractor(args)
    weights = getattr(args, "fast_reid_weights", "") or getattr(args, "reid_model_path", "")
    extractor = ExhibitFlowFastReIDExtractor(
        config_file=getattr(args, "fast_reid_config", ""),
        weights_path=weights,
        device=getattr(args, "reid_device", "cpu"),
        batch_size=getattr(args, "fast_reid_batch_size", 16),
    )
    print(
        f"[ReID] Backend: ExhibitFlow FastReID | config: {getattr(args, 'fast_reid_config', '')} "
        f"| weights: {weights} | device: {getattr(args, 'reid_device', 'cpu')}"
    )
    return extractor


class Predictor:
    def __init__(self, model, exp, device: torch.device):
        from yolox.data.data_augment import preproc
        from yolox.utils import postprocess

        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self._preproc = preproc
        self._postprocess = postprocess

    def inference(self, frame):
        height, width = frame.shape[:2]
        image, ratio = self._preproc(frame, self.test_size, self.rgb_means, self.std)
        image = torch.from_numpy(image).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
            outputs = self._postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs, {"height": height, "width": width, "ratio": ratio, "raw_img": frame}


def scaled_for_display(frame, scale: float):
    if scale == 1.0:
        return frame
    height, width = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))


def main() -> None:
    args = make_parser().parse_args()
    args.model_repo = args.model_repo.resolve()
    if not torch.cuda.is_available():
        for attr in ("device", "ltc_device", "reid_device"):
            if getattr(args, attr, None) == "cuda":
                print(f"[Device] CUDA unavailable; falling back {attr}=cpu")
                setattr(args, attr, "cpu")
    add_model_repo_to_path(args.model_repo)

    from yolox.exp import get_exp
    install_dma_gbm_adapter(args.model_repo)
    import yolox.tracker.byte_tracker as byte_tracker_module
    from yolox.tracking_utils.timer import Timer
    from yolox.utils import get_model_info
    from yolox.utils.visualize import plot_tracking

    byte_tracker_module.build_reid_extractor = build_exhibitflow_reid_extractor
    BYTETracker = byte_tracker_module.BYTETracker

    exp = get_exp(str(args.exp_file.resolve()), None)
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = exp.get_model().to(device)
    model.eval()
    checkpoint = torch.load(str(args.checkpoint.resolve()), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print(f"[ByteTrack] {get_model_info(model, exp.test_size)}")
    print(f"[ByteTrack] checkpoint: {args.checkpoint}")

    cap = cv2.VideoCapture(str(args.video.resolve()))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    frame_rate = int(round(video_fps or args.fps or 30))

    args.result_txt.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.save_video or args.annotated_video:
        if args.annotated_video is None:
            args.annotated_video = args.result_txt.with_suffix(".mp4")
        args.annotated_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.annotated_video), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height)
        )

    predictor = Predictor(model, exp, device)
    tracker = BYTETracker(build_tracker_args(args), frame_rate=frame_rate)
    timer = Timer()
    frame_id = 0
    result_lines: list[str] = []
    processed_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timer.tic()
        outputs, img_info = predictor.inference(frame)
        online_tlwhs = []
        online_ids = []
        online_scores = []
        if outputs[0] is not None:
            online_targets = tracker.update(
                outputs[0], [img_info["height"], img_info["width"]], exp.test_size, frame=frame
            )
            for target in online_targets:
                tlwh = target.tlwh
                tid = target.track_id
                vertical = tlwh[2] / max(1e-6, tlwh[3]) > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(target.score)
                    result_lines.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{target.score:.2f},-1,-1,-1\n"
                    )
        timer.toc()
        fps_text = 1.0 / max(1e-5, timer.average_time)
        annotated = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1, fps=fps_text)
        if writer is not None:
            writer.write(annotated)
        if args.show:
            cv2.imshow(args.window_name, scaled_for_display(annotated, args.show_scale))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
        if frame_id % 20 == 0:
            print(f"[ByteTrack] frame={frame_id} fps={fps_text:.2f}")
        frame_id += 1
        processed_frames += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyWindow(args.window_name)

    args.result_txt.write_text("".join(result_lines), encoding="utf-8")
    summary = {
        "result_txt": str(args.result_txt),
        "annotated_video": str(args.annotated_video) if args.annotated_video else None,
        "processed_frames": processed_frames,
        "detections": len(result_lines),
    }
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
