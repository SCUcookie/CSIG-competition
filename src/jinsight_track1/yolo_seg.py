"""Ultralytics YOLO-seg training and submission utilities.

The raw challenge layout is ``sequence/{img,mask}``.  This module builds a
symlinked YOLO dataset (images are never copied), trains an optional
Ultralytics model, and converts predicted masks to the competition centroid
TXT format.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .data import natural_key
from .evaluation import point_metrics
from .postprocess import centroids
from .submission import package, write_txt
from .types import SequencePrediction, TrackPoint

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal installs
    cv2 = None

WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"


def _sequences(root: Path) -> list[Path]:
    return sorted((p for p in root.iterdir() if p.is_dir() and (p / "img").is_dir()), key=natural_key)


def _files(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in d.iterdir() if p.is_file()}


def _polygon_labels(mask_path: Path) -> list[str]:
    if cv2 is None:
        raise RuntimeError("opencv-python is required for mask conversion")
    mask = np.asarray(Image.open(mask_path).convert("L"))
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = binary.shape
    result = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            # Single-pixel/line targets are common in infrared masks.  A
            # one-pixel component has no non-zero-area contour, so encode its
            # bounding cell as a valid four-vertex polygon instead of silently
            # dropping the object.
            x, y, cw, ch = cv2.boundingRect(contour)
            points = np.asarray([(x, y), (x + max(cw, 1), y),
                                 (x + max(cw, 1), y + max(ch, 1)),
                                 (x, y + max(ch, 1))])
        else:
            points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        coords = []
        for x, y in points:
            coords.extend((float(x) / w, float(y) / h))
        result.append("0 " + " ".join(f"{v:.8f}" for v in coords))
    return result


def prepare_yolo_dataset(raw_root: str | Path, output_root: str | Path,
                         val_root: str | Path | None = None,
                         max_sequences: int | None = None,
                         max_frames: int | None = None) -> dict:
    """Create a symlinked YOLO dataset and return conversion statistics."""
    raw = Path(raw_root); out = Path(output_root)
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    roots = [(raw, "train")]
    if val_root is not None:
        roots.append((Path(val_root), "val"))
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"raw_root": str(raw), "output_root": str(out), "sequences": 0,
             "frames": 0, "empty_labels": 0, "objects": 0, "missing_masks": 0}
    for source_root, source_split in roots:
        if not source_root.is_dir(): raise FileNotFoundError(source_root)
        seqs = _sequences(source_root)
        if max_sequences: seqs = seqs[:max_sequences]
        for seq in seqs:
            imgs, masks = _files(seq / "img"), _files(seq / "mask")
            stems = sorted(imgs, key=lambda x: natural_key(Path(x)))
            if max_frames:
                stems = stems[:max_frames]
            # Sequence-level validation split is intentional: no frame leakage.
            split = "val" if source_root.name.lower() in {"val", "validation"} else source_split
            for stem in stems:
                image = imgs[stem]; target = out / "images" / split / f"{seq.name}__{stem}{image.suffix.lower()}"
                label = out / "labels" / split / f"{seq.name}__{stem}.txt"
                if not target.exists(): target.symlink_to(image.resolve())
                lines = _polygon_labels(masks[stem]) if stem in masks else []
                if stem not in masks: stats["missing_masks"] += 1
                label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="ascii")
                stats["frames"] += 1; stats["objects"] += len(lines)
                stats["empty_labels"] += not lines
            stats["sequences"] += 1
    yaml = out / "dataset.yaml"
    yaml.write_text("path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [target]\n" % out.resolve(), encoding="utf-8")
    (out / "conversion_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def ensure_weights(path: str | Path, allow_download: bool = False) -> dict:
    path = Path(path)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(f"weights not found: {path}; pass --download-weights")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(WEIGHT_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "sha256": digest, "source": WEIGHT_URL if allow_download else "local"}


def train_yolo(raw_root: str | Path, work_dir: str | Path, weights: str | Path = "yolov8n-seg.pt",
               val_root: str | Path | None = None,
               device: str = "0,1,2,3", imgsz: int = 640, epochs: int = 50,
               max_sequences: int | None = None, max_frames: int | None = None,
               download_weights: bool = False, workers: int = 4) -> dict:
    """Train YOLOv8n-seg. Ultralytics is imported only when this is called."""
    from ultralytics import YOLO
    work = Path(work_dir); work.mkdir(parents=True, exist_ok=True)
    data_dir = work / "yolo_dataset"
    if device != "cpu":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this environment; use --device cpu for a smoke run")
    stats = prepare_yolo_dataset(raw_root, data_dir, val_root, max_sequences, max_frames)
    weight_info = ensure_weights(weights, download_weights)
    config = {"raw_root": str(raw_root), "val_root": str(val_root) if val_root else None,
              "work_dir": str(work), "weights": weight_info,
              "device": device, "imgsz": imgsz, "epochs": epochs, "amp": True,
              "cache": False, "max_sequences": max_sequences, "max_frames": max_frames}
    (work / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    model = YOLO(str(weights))
    result = model.train(data=str(data_dir / "dataset.yaml"), imgsz=imgsz, epochs=epochs,
                         device=device, amp=True, cache=False, workers=workers,
                         project=str(work), name="ultralytics", exist_ok=True)
    best = work / "ultralytics" / "weights" / "best.pt"
    report = {"config": config, "conversion": stats, "best": str(best),
              "results_csv": str(work / "ultralytics" / "results.csv"),
              "status": "complete" if best.exists() else "finished_without_best"}
    (work / "train_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def _as_numpy(value) -> np.ndarray:
    """Convert an Ultralytics tensor-like value without requiring torch here."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _polygon_centroid(points: np.ndarray, fallback: tuple[float, float]) -> tuple[float, float]:
    """Return the area centroid of an original-image polygon.

    Ultralytics ``Masks.xy`` polygons are already scaled from the inference
    canvas back to the original image. Degenerate tiny masks fall back to the
    corresponding original-image bounding-box centre.
    """
    polygon = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(polygon) < 3:
        return fallback
    x, y = polygon[:, 0], polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(cross.sum())
    if abs(area2) < 1e-9:
        return fallback
    cx = float(((x + np.roll(x, -1)) * cross).sum() / (3.0 * area2))
    cy = float(((y + np.roll(y, -1)) * cross).sum() / (3.0 * area2))
    return cx, cy


def _prediction_points(result, min_conf: float) -> list[tuple[float, float, float]]:
    if result.masks is None or result.boxes is None:
        return []
    # Do not calculate centroids directly on result.masks.data: that tensor can
    # use the imgsz/letterboxed inference canvas rather than the original image
    # shape. Masks.xy and Boxes.xywh are explicitly mapped to original pixels.
    polygons = result.masks.xy
    confs = _as_numpy(result.boxes.conf).reshape(-1)
    centres = _as_numpy(result.boxes.xywh).reshape(-1, 4)[:, :2]
    if len(polygons) != len(confs) or len(centres) != len(confs):
        raise RuntimeError("Ultralytics returned inconsistent masks, boxes, and confidences")
    h, w = (int(v) for v in result.orig_shape)
    found = []
    for polygon, centre, score in zip(polygons, centres, confs):
        score = float(score)
        if score < min_conf:
            continue
        x, y = _polygon_centroid(polygon, (float(centre[0]), float(centre[1])))
        if not np.isfinite((x, y)).all():
            raise RuntimeError("prediction contains a non-finite centroid")
        if not (0.0 <= x < w and 0.0 <= y < h):
            raise RuntimeError(
                f"original-image centroid ({x:.3f}, {y:.3f}) is outside {w}x{h}"
            )
        found.append((x, y, score))
    return found


def infer_yolo(model_path: str | Path, data_root: str | Path, output_dir: str | Path,
               device: str = "0", conf: float = .25, coordinate_order: str = "xy",
               max_sequences: int | None = None, max_frames: int | None = None,
               make_zip: bool = True, batch: int = 16, imgsz: int = 640) -> dict:
    from ultralytics import YOLO
    model = YOLO(str(model_path)); root = Path(data_root); out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seqs = _sequences(root)[:max_sequences] if max_sequences else _sequences(root)
    total = 0; frames_seen = 0
    for seq in seqs:
        imgs = _files(seq / "img"); stems = sorted(imgs, key=lambda x: natural_key(Path(x)))
        if max_frames: stems = stems[:max_frames]
        pred_frames = {}
        sources = [str(imgs[stem]) for stem in stems]
        results = model.predict(
            source=sources, device=device, conf=conf, verbose=False,
            stream=True, batch=batch, imgsz=imgsz,
        )
        for frame_no, (stem, result) in enumerate(zip(stems, results), 1):
            pts = _prediction_points(result, conf)
            pred_frames[frame_no] = [TrackPoint(frame_no, 0, x, y) for x, y, _ in pts]
            total += len(pts); frames_seen += 1
        if len(pred_frames) != len(stems):
            raise RuntimeError(f"inference returned {len(pred_frames)} of {len(stems)} frames for {seq.name}")
        write_txt(SequencePrediction(seq.name, pred_frames), out / f"{seq.name}.txt", coordinate_order, overwrite=True)
    zip_path = package(out, out.with_suffix(".zip"), expected=len(seqs), overwrite=True, coordinate_order=coordinate_order) if make_zip else None
    return {"sequences": len(seqs), "frames": frames_seen, "detections": total,
            "output_dir": str(out), "zip": str(zip_path) if zip_path else None,
            "coordinate_order": coordinate_order, "centroid_coordinates": "original-image pixels",
            "batch": batch, "imgsz": imgsz, "proxy_metric": "not computed without labels"}


def _metric_summary(totals: dict[str, float], confidence: float) -> dict:
    tp, fp, fn = int(totals["tp"]), int(totals["fp"]), int(totals["fn"])
    return {
        "confidence": confidence,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def evaluate_yolo(model_path: str | Path, val_root: str | Path, output_dir: str | Path,
                  device: str = "0", conf: float = .25, radius: float = 2.,
                  batch: int = 16, conf_grid: list[float] | None = None,
                  max_sequences: int | None = None,
                  max_frames: int | None = None, imgsz: int = 640) -> dict:
    """Run validation and report local point metrics (not official scoring)."""
    from ultralytics import YOLO
    thresholds = sorted(set(float(v) for v in (conf_grid or [conf])))
    if not thresholds or any(v <= 0.0 or v >= 1.0 for v in thresholds):
        raise ValueError("confidence thresholds must be between 0 and 1")
    model = YOLO(str(model_path)); root = Path(val_root)
    totals = {value: defaultdict(float) for value in thresholds}
    resolution_totals = {value: {} for value in thresholds}
    sequences = 0; frames_seen = 0; missing_masks = 0
    seqs = _sequences(root)[:max_sequences] if max_sequences else _sequences(root)
    for seq in seqs:
        imgs, masks = _files(seq / "img"), _files(seq / "mask")
        stems = sorted(imgs, key=lambda x: natural_key(Path(x)))
        if max_frames:
            stems = stems[:max_frames]
        sources = [str(imgs[stem]) for stem in stems]
        results = model.predict(
            source=sources, device=device, conf=min(thresholds), verbose=False,
            stream=True, batch=batch, imgsz=imgsz,
        )
        returned = 0
        for stem, result in zip(stems, results):
            returned += 1; frames_seen += 1
            scored = _prediction_points(result, min(thresholds))
            if stem in masks:
                truth_mask = np.asarray(Image.open(masks[stem]).convert("L"))
                truth = [(d.x, d.y) for d in centroids(truth_mask, .5, 1)]
            else:
                missing_masks += 1
                truth_mask = np.zeros(result.orig_shape, dtype=np.uint8)
                truth = []
            height, width = truth_mask.shape
            resolution = f"{width}x{height}"
            for threshold in thresholds:
                pred = [(x, y) for x, y, score in scored if score >= threshold]
                metrics = point_metrics(pred, truth, radius)
                bucket = resolution_totals[threshold].setdefault(resolution, defaultdict(float))
                for key in ("tp", "fp", "fn"):
                    totals[threshold][key] += metrics[key]
                    bucket[key] += metrics[key]
        if returned != len(stems):
            raise RuntimeError(f"evaluation returned {returned} of {len(stems)} frames for {seq.name}")
        sequences += 1
    sweep = [_metric_summary(totals[value], value) for value in thresholds]
    best = max(sweep, key=lambda item: (item["f1"], item["recall"], item["precision"]))
    best_resolution = {
        name: _metric_summary(values, best["confidence"])
        for name, values in sorted(resolution_totals[best["confidence"]].items())
    }
    report = {
        "sequences": sequences,
        "frames": frames_seen,
        "missing_masks": missing_masks,
        **best,
        "best_confidence": best["confidence"],
        "sweep": sweep,
        "best_resolution_breakdown": best_resolution,
        "inference_confidence": min(thresholds),
        "radius_pixels": radius,
        "batch": batch,
        "imgsz": imgsz,
        "centroid_coordinates": "original-image pixels",
        "metric_status": "local proxy; not official scorer",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "val_proxy_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
