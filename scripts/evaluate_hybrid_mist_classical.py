"""Reproduce the Hybrid-MIST registered-background traditional branch."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from jinsight_track1.deeppro_adapter import _files, _load_sequence_images, _sequences
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.postprocess import centroids
from jinsight_track1.deeppro_adapter import component_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--sequences")
    parser.add_argument(
        "--registration", choices=("phase", "sift", "ecc"), default="phase"
    )
    parser.add_argument("--background-samples", type=int, default=61)
    parser.add_argument("--thresholds", default=".5,1,1.5,2,2.5,3,4,5,6,8,10")
    parser.add_argument("--methods", default="residual,dog,fused")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--max-area", type=int, default=20)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def robust_standardize(score: np.ndarray) -> np.ndarray:
    sample = score[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    return (score - centre) / max(0.25, 1.4826 * mad)


def phase_transforms(frames: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:]
    size = (width, height)
    reference = cv2.GaussianBlur(frames[0].astype(np.float32), (0, 0), 1.0)
    window = cv2.createHanningWindow(size, cv2.CV_32F)
    transforms = [np.eye(3, dtype=np.float64)]
    previous = np.zeros(2, dtype=float)
    for frame in frames[1:]:
        current = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.0)
        shift, response = cv2.phaseCorrelate(reference, current, window)
        if response > 0.01 and np.isfinite(shift).all():
            previous = np.asarray(shift)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, 2] = previous
        transforms.append(matrix)
    return np.stack(transforms)


def sift_transforms(frames: np.ndarray) -> np.ndarray:
    sift = cv2.SIFT_create(nfeatures=2500, contrastThreshold=0.02)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    reference = cv2.equalizeHist(frames[0])
    keypoints_ref, descriptors_ref = sift.detectAndCompute(reference, None)
    transforms = [np.eye(3, dtype=np.float64)]
    previous = transforms[0]
    for frame in frames[1:]:
        current = cv2.equalizeHist(frame)
        keypoints, descriptors = sift.detectAndCompute(current, None)
        matrix = None
        if (
            descriptors_ref is not None
            and descriptors is not None
            and len(descriptors_ref) >= 8
            and len(descriptors) >= 8
        ):
            pairs = matcher.knnMatch(descriptors_ref, descriptors, k=2)
            good = [first for first, second in pairs if first.distance < 0.75 * second.distance]
            if len(good) >= 8:
                source = np.float32([keypoints_ref[item.queryIdx].pt for item in good])
                target = np.float32([keypoints[item.trainIdx].pt for item in good])
                affine, _ = cv2.estimateAffinePartial2D(
                    source,
                    target,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=1.5,
                    maxIters=3000,
                    confidence=0.995,
                    refineIters=20,
                )
                if affine is not None and np.isfinite(affine).all():
                    candidate = np.eye(3, dtype=np.float64)
                    candidate[:2] = affine
                    # Reject catastrophic matches while retaining real camera motion.
                    if np.linalg.norm(candidate[:2, 2]) < 0.3 * np.hypot(*frame.shape):
                        matrix = candidate
        if matrix is None:
            matrix = previous.copy()
        transforms.append(matrix)
        previous = matrix
    return np.stack(transforms)


def ecc_transforms(frames: np.ndarray) -> np.ndarray:
    height, width = frames.shape[1:]
    reference = cv2.GaussianBlur(frames[0].astype(np.float32) / 255, (0, 0), 1.0)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    transforms = [np.eye(3, dtype=np.float64)]
    previous = transforms[0]
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        60,
        1e-5,
    )
    for frame in frames[1:]:
        current = cv2.GaussianBlur(frame.astype(np.float32) / 255, (0, 0), 1.0)
        shift, response = cv2.phaseCorrelate(reference, current, window)
        initial = previous[:2].astype(np.float32)
        if response > 0.01 and np.isfinite(shift).all():
            initial[:, :2] = np.eye(2, dtype=np.float32)
            initial[:, 2] = shift
        try:
            _, affine = cv2.findTransformECC(
                reference,
                current,
                initial,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                5,
            )
            candidate = np.eye(3, dtype=np.float64)
            candidate[:2] = affine
            if (
                np.isfinite(candidate).all()
                and np.linalg.norm(candidate[:2, 2]) < 0.3 * np.hypot(height, width)
            ):
                previous = candidate
        except cv2.error:
            pass
        transforms.append(previous.copy())
    return np.stack(transforms)


def canonical_background(
    frames: np.ndarray, transforms: np.ndarray, sample_count: int
) -> np.ndarray:
    height, width = frames.shape[1:]
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames))).astype(int)
    registered = []
    for index in indices:
        inverse = np.linalg.inv(transforms[index])
        registered.append(
            cv2.warpAffine(
                frames[index],
                inverse[:2],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    return np.median(np.stack(registered), axis=0).astype(np.float32)


def response_maps(frame, background, transform):
    height, width = frame.shape
    current_background = cv2.warpAffine(
        background,
        transform[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    residual = robust_standardize(current_background - frame.astype(np.float32))
    fine = cv2.GaussianBlur(residual, (0, 0), 0.65)
    coarse = cv2.GaussianBlur(residual, (0, 0), 2.4)
    dog = robust_standardize(fine - coarse)
    fused = np.maximum(residual, 0) + np.maximum(dog, 0)
    return {"residual": residual, "dog": dog, "fused": fused}


def main():
    args = parse_args()
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    selected_sequences = set(args.sequences.split(",")) if args.sequences else None
    thresholds = [float(value) for value in args.thresholds.split(",")]
    methods = args.methods.split(",")
    totals = {
        method: {threshold: defaultdict(int) for threshold in thresholds}
        for method in methods
    }
    sequence_reports = {}
    frames_seen = 0
    for sequence in _sequences(Path(args.root)):
        if selected_sequences and sequence.name not in selected_sequences:
            continue
        stems, frames = _load_sequence_images(sequence, args.max_frames)
        if not len(frames):
            continue
        height, width = frames.shape[1:]
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        if args.registration == "phase":
            transforms = phase_transforms(frames)
        elif args.registration == "sift":
            transforms = sift_transforms(frames)
        else:
            transforms = ecc_transforms(frames)
        background = canonical_background(
            frames, transforms, args.background_samples
        )
        masks = _files(sequence / "mask")
        per_sequence = {
            method: {threshold: defaultdict(int) for threshold in thresholds}
            for method in methods
        }
        oracle_values = defaultdict(list)
        for index, stem in enumerate(stems):
            mask = np.asarray(Image.open(masks[stem]).convert("L"))
            truth = [(point.x, point.y) for point in centroids(mask, 0.5, 1)]
            maps = response_maps(frames[index], background, transforms[index])
            for method in methods:
                score = maps[method]
                for x, y in truth:
                    oracle_values[method].append(float(score[round(y), round(x)]))
                for threshold in thresholds:
                    predicted = component_points(
                        score,
                        threshold,
                        min_area=args.min_area,
                        max_area=args.max_area,
                    )
                    metrics = point_metrics(predicted, truth, args.radius)
                    for key in ("tp", "fp", "fn"):
                        totals[method][threshold][key] += metrics[key]
                        per_sequence[method][threshold][key] += metrics[key]
        rows = {}
        for method in methods:
            sweep = []
            for threshold in thresholds:
                values = per_sequence[method][threshold]
                tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
                sweep.append(
                    {
                        "threshold": threshold,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "precision": tp / max(1, tp + fp),
                        "recall": tp / max(1, tp + fn),
                        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                    }
                )
            rows[method] = {
                "best": max(sweep, key=lambda row: row["f1"]),
                "truth_response": {
                    str(level): float(np.quantile(oracle_values[method], level))
                    if oracle_values[method]
                    else None
                    for level in (0, 0.1, 0.5, 0.9, 1)
                },
                "sweep": sweep,
            }
        sequence_reports[sequence.name] = rows
        frames_seen += len(stems)
        print(
            f"hybrid-mist {sequence.name} frames={frames_seen} "
            + " ".join(f"{method}={rows[method]['best']['f1']:.4f}" for method in methods),
            flush=True,
        )

    aggregate = {}
    for method in methods:
        sweep = []
        for threshold in thresholds:
            values = totals[method][threshold]
            tp, fp, fn = (int(values[key]) for key in ("tp", "fp", "fn"))
            sweep.append(
                {
                    "threshold": threshold,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "precision": tp / max(1, tp + fp),
                    "recall": tp / max(1, tp + fn),
                    "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                }
            )
        aggregate[method] = {
            "best": max(sweep, key=lambda row: row["f1"]),
            "sweep": sweep,
        }
    report = {
        "settings": vars(args),
        "frames": frames_seen,
        "aggregate": aggregate,
        "sequences": sequence_reports,
    }
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
