"""Dense constant-velocity shift-and-stack evaluation for weak point targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from jinsight_track1.deeppro_adapter import DeepProDetector, _files, _sequences
from jinsight_track1.postprocess import centroids
from jinsight_track1.tracking import assign_track_ids


def parse_range(text: str) -> np.ndarray:
    start, stop, step = map(float, text.split(":"))
    return np.arange(start, stop + step * 0.25, step)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--resolutions", default="640x512")
    parser.add_argument("--sequences")
    parser.add_argument(
        "--concatenate",
        action="store_true",
        help="Search selected sequences as one continuous video while estimating each background separately.",
    )
    parser.add_argument("--roi", default="380,220,510,410", help="x0,y0,x1,y1")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--score-source",
        choices=(
            "spatial",
            "hybrid_phase_dog",
            "hybrid_phase_fused",
            "hybrid_phase_sparse",
            "deeppro_log",
        ),
        default="spatial",
    )
    parser.add_argument("--background-samples", type=int, default=61)
    parser.add_argument(
        "--hybrid-registration", choices=("phase", "ecc"), default="phase"
    )
    parser.add_argument("--candidate-threshold", type=float, default=2.0)
    parser.add_argument("--candidate-min-area", type=int, default=1)
    parser.add_argument("--candidate-max-area", type=int, default=20)
    parser.add_argument("--deeppro-source")
    parser.add_argument("--deeppro-weights")
    parser.add_argument("--deeppro-device", default="1")
    parser.add_argument("--deeppro-tile-size", type=int, default=1024)
    parser.add_argument("--deeppro-tile-halo", type=int, default=12)
    parser.add_argument("--deeppro-bidirectional", action="store_true")
    parser.add_argument(
        "--deeppro-dog-weight",
        type=float,
        default=0.5,
        help="Add this multiple of phase-background DoG to standardized log probability.",
    )
    parser.add_argument("--vx", default="-0.03:0.02:0.002")
    parser.add_argument("--vy", default="-0.02:0.05:0.002")
    parser.add_argument("--min-speed", type=float, default=0.003)
    parser.add_argument("--peaks-per-velocity", type=int, default=2)
    parser.add_argument("--peak-separation", type=int, default=3)
    parser.add_argument("--top-candidates", type=int, default=50)
    parser.add_argument("--path-separation", type=float, default=3.0)
    parser.add_argument(
        "--temporal-reducer",
        choices=("mean", "median", "q25", "mean_std", "fraction_gt1"),
        default="mean",
    )
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--output")
    return parser.parse_args()


def robust_standardize(score: np.ndarray) -> np.ndarray:
    sample = score[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    return (score - centre) / max(0.25, 1.4826 * mad)


def spatial_score(frame: np.ndarray) -> np.ndarray:
    image = frame.astype(np.float32)
    centre = cv2.GaussianBlur(image, (0, 0), 0.65)
    surround = cv2.GaussianBlur(image, (0, 0), 2.4)
    return robust_standardize(surround - centre).astype(np.float32)


def truth_points(mask_path: Path):
    mask = np.asarray(Image.open(mask_path).convert("L"))
    return [(point.x, point.y) for point in centroids(mask, 0.5, 1)]


def phase_registered_background(frames: np.ndarray, sample_count: int):
    height, width = frames.shape[1:]
    reference = cv2.GaussianBlur(frames[0].astype(np.float32), (0, 0), 1.0)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    transforms = []
    previous = np.zeros(2, dtype=float)
    for index, frame in enumerate(frames):
        if index:
            current = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 1.0)
            shift, response = cv2.phaseCorrelate(reference, current, window)
            if response > 0.01 and np.isfinite(shift).all():
                previous = np.asarray(shift)
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, 2] = previous
        transforms.append(matrix)
    transforms = np.stack(transforms)
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames))).astype(int)
    registered = []
    for index in indices:
        registered.append(
            cv2.warpAffine(
                frames[index],
                np.linalg.inv(transforms[index])[:2],
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
        )
    background = np.median(np.stack(registered), axis=0).astype(np.float32)
    return background, transforms


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


def registered_background(
    frames: np.ndarray, sample_count: int, registration: str
):
    if registration == "phase":
        return phase_registered_background(frames, sample_count)
    transforms = ecc_transforms(frames)
    height, width = frames.shape[1:]
    indices = np.linspace(0, len(frames) - 1, min(sample_count, len(frames))).astype(
        int
    )
    registered = [
        cv2.warpAffine(
            frames[index],
            np.linalg.inv(transforms[index])[:2],
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        for index in indices
    ]
    return np.median(np.stack(registered), axis=0).astype(np.float32), transforms


def hybrid_score(
    frame: np.ndarray, background: np.ndarray, transform: np.ndarray, fused: bool
):
    height, width = frame.shape
    current_background = cv2.warpAffine(
        background,
        transform[:2],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    raw = current_background - frame.astype(np.float32)
    sample = raw[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    residual = (raw - centre) / max(0.25, 1.4826 * mad)
    dog_raw = cv2.GaussianBlur(residual, (0, 0), 0.65) - cv2.GaussianBlur(
        residual, (0, 0), 2.4
    )
    sample = dog_raw[::4, ::4]
    centre = float(np.median(sample))
    mad = float(np.median(np.abs(sample - centre)))
    dog = (dog_raw - centre) / max(0.25, 1.4826 * mad)
    return (
        np.maximum(residual, 0) + np.maximum(dog, 0)
        if fused
        else dog.astype(np.float32)
    )


def sparse_candidate_score(
    score: np.ndarray, threshold: float, min_area: int, max_area: int
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (score >= threshold).astype(np.uint8), connectivity=8
    )
    valid = np.zeros(count, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    valid[1:] = (areas[1:] >= min_area) & (areas[1:] <= max_area)
    return np.where(valid[labels], score, 0).astype(np.float32)


def block_scores(
    sequence: Path,
    block_size: int,
    score_source: str,
    background_samples: int,
    detector: DeepProDetector | None = None,
    deeppro_bidirectional: bool = False,
    deeppro_dog_weight: float = 0.0,
    candidate_threshold: float = 2.0,
    candidate_min_area: int = 1,
    candidate_max_area: int = 20,
    hybrid_registration: str = "phase",
):
    images = _files(sequence / "img")
    masks = _files(sequence / "mask")
    stems = sorted(set(images) & set(masks))
    frame_arrays = [
        np.asarray(Image.open(images[stem]).convert("L"), dtype=np.uint8)
        for stem in stems
    ]
    needs_hybrid = score_source.startswith("hybrid_phase") or (
        score_source == "deeppro_log" and deeppro_dog_weight
    )
    if needs_hybrid:
        background, transforms = registered_background(
            np.stack(frame_arrays), background_samples, hybrid_registration
        )
    else:
        background = transforms = None
    if score_source == "deeppro_log":
        if detector is None:
            raise ValueError("deeppro_log requires a DeepPro detector")
        probabilities = detector.predict(
            np.stack(frame_arrays),
            bidirectional=deeppro_bidirectional,
            motion_compensation=False,
        )
    else:
        probabilities = None
    maps = []
    times = []
    accumulated = None
    count = 0
    block_time = []
    truths = []
    for time, (stem, frame) in enumerate(zip(stems, frame_arrays)):
        if accumulated is None:
            accumulated = np.zeros(frame.shape, dtype=np.float32)
        if score_source == "spatial":
            score = spatial_score(frame)
        elif score_source == "deeppro_log":
            score = robust_standardize(
                np.log10(np.maximum(probabilities[time], 1e-9))
            ).astype(np.float32)
            if deeppro_dog_weight:
                score += deeppro_dog_weight * hybrid_score(
                    frame, background, transforms[time], fused=False
                )
        else:
            score = hybrid_score(
                frame,
                background,
                transforms[time],
                fused=score_source.endswith("_fused"),
            )
            if score_source == "hybrid_phase_sparse":
                score = sparse_candidate_score(
                    score,
                    candidate_threshold,
                    candidate_min_area,
                    candidate_max_area,
                )
        accumulated += score
        count += 1
        block_time.append(time)
        truths.append(truth_points(masks[stem]))
        if count == block_size or time == len(stems) - 1:
            maps.append(accumulated / count)
            times.append(float(np.mean(block_time)))
            accumulated.fill(0)
            count = 0
            block_time = []
    return stems, truths, maps, np.asarray(times)


def local_peaks(score: np.ndarray, count: int, separation: int):
    work = score.copy()
    result = []
    for _ in range(count):
        index = int(np.argmax(work))
        value = float(work.flat[index])
        y, x = np.unravel_index(index, work.shape)
        result.append((x, y, value))
        cv2.circle(work, (int(x), int(y)), separation, -np.inf, -1)
    return result


def search(
    maps,
    times: np.ndarray,
    roi,
    velocities_x: np.ndarray,
    velocities_y: np.ndarray,
    min_speed: float,
    peaks_per_velocity: int,
    peak_separation: int,
    temporal_reducer: str,
):
    x0, y0, x1, y1 = roi
    grid_x, grid_y = np.meshgrid(
        np.arange(x0, x1, dtype=np.float32),
        np.arange(y0, y1, dtype=np.float32),
    )
    reference_time = float(np.mean(times))
    candidates = []
    for velocity_y in velocities_y:
        for velocity_x in velocities_x:
            if np.hypot(velocity_x, velocity_y) < min_speed:
                continue
            sampled_maps = []
            for score, time in zip(maps, times):
                delta = time - reference_time
                query_x = (grid_x + velocity_x * delta).astype(np.float32)
                query_y = (grid_y + velocity_y * delta).astype(np.float32)
                sampled = cv2.remap(
                    score,
                    query_x,
                    query_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=-2,
                )
                valid = (
                    (query_x >= 0)
                    & (query_x < score.shape[1] - 1)
                    & (query_y >= 0)
                    & (query_y < score.shape[0] - 1)
                )
                sampled_maps.append(np.where(valid, sampled, np.nan))
            sampled_stack = np.stack(sampled_maps)
            if temporal_reducer == "mean":
                averaged = np.nanmean(sampled_stack, axis=0)
            elif temporal_reducer == "median":
                averaged = np.nanmedian(sampled_stack, axis=0)
            elif temporal_reducer == "q25":
                averaged = np.nanquantile(sampled_stack, 0.25, axis=0)
            elif temporal_reducer == "mean_std":
                averaged = np.nanmean(sampled_stack, axis=0) - 0.5 * np.nanstd(
                    sampled_stack, axis=0
                )
            elif temporal_reducer == "fraction_gt1":
                averaged = np.nanmean(sampled_stack > 1.0, axis=0)
            else:
                raise ValueError(f"unknown temporal reducer: {temporal_reducer}")
            for local_x, local_y, value in local_peaks(
                averaged, peaks_per_velocity, peak_separation
            ):
                candidates.append(
                    {
                        "midpoint": np.asarray([x0 + local_x, y0 + local_y], dtype=float),
                        "velocity": np.asarray([velocity_x, velocity_y], dtype=float),
                        "score": value,
                        "reference_time": reference_time,
                        "features": {
                            "mean": float(np.nanmean(sampled_stack[:, local_y, local_x])),
                            "std": float(np.nanstd(sampled_stack[:, local_y, local_x])),
                            "min": float(np.nanmin(sampled_stack[:, local_y, local_x])),
                            "q10": float(
                                np.nanquantile(
                                    sampled_stack[:, local_y, local_x], 0.10
                                )
                            ),
                            "q25": float(
                                np.nanquantile(
                                    sampled_stack[:, local_y, local_x], 0.25
                                )
                            ),
                            "median": float(
                                np.nanmedian(sampled_stack[:, local_y, local_x])
                            ),
                            "q75": float(
                                np.nanquantile(
                                    sampled_stack[:, local_y, local_x], 0.75
                                )
                            ),
                            "q90": float(
                                np.nanquantile(
                                    sampled_stack[:, local_y, local_x], 0.90
                                )
                            ),
                            "max": float(np.nanmax(sampled_stack[:, local_y, local_x])),
                            "fraction_gt0": float(
                                np.nanmean(sampled_stack[:, local_y, local_x] > 0)
                            ),
                            "fraction_gt1": float(
                                np.nanmean(sampled_stack[:, local_y, local_x] > 1)
                            ),
                            "fraction_gt2": float(
                                np.nanmean(sampled_stack[:, local_y, local_x] > 2)
                            ),
                        },
                    }
                )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def distance(first, second, frame_count: int) -> float:
    times = np.asarray([0, (frame_count - 1) / 2, frame_count - 1])
    first_xy = first["midpoint"] + (times - first["reference_time"])[:, None] * first[
        "velocity"
    ]
    second_xy = second["midpoint"] + (
        times - second["reference_time"]
    )[:, None] * second["velocity"]
    return float(np.mean(np.linalg.norm(first_xy - second_xy, axis=1)))


def select_candidates(candidates, count: int, separation: float, frame_count: int):
    kept = []
    for candidate in candidates:
        if any(distance(candidate, other, frame_count) < separation for other in kept):
            continue
        kept.append(candidate)
        if len(kept) >= count:
            break
    return kept


def predict(candidate, time: int):
    return candidate["midpoint"] + (
        time - candidate["reference_time"]
    ) * candidate["velocity"]


def evaluate(candidates, truths, radius: float):
    rows = []
    for limit in range(1, len(candidates) + 1):
        tp = fp = fn = 0
        for time, truth_frame in enumerate(truths):
            predicted = np.asarray(
                [predict(candidate, time) for candidate in candidates[:limit]]
            ).reshape(-1, 2)
            truth = np.asarray(truth_frame, dtype=float).reshape(-1, 2)
            matched = 0
            if len(predicted) and len(truth):
                distances = np.linalg.norm(predicted[:, None] - truth[None, :], axis=2)
                row_ids, column_ids = linear_sum_assignment(distances)
                matched = sum(
                    distances[row_id, column_id] <= radius
                    for row_id, column_id in zip(row_ids, column_ids)
                )
            tp += matched
            fp += len(predicted) - matched
            fn += len(truth) - matched
        rows.append(
            {
                "candidates": limit,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": tp / max(1, tp + fp),
                "recall": tp / max(1, tp + fn),
                "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            }
        )
    return rows


def oracle(truths, maps, block_times):
    height, width = maps[0].shape
    tracked = assign_track_ids(truths, (height, width), gate_fraction=0.02, max_age=5)
    tracks = {}
    for frame_id, points in tracked.items():
        for point in points:
            tracks.setdefault(point.track_id, []).append(
                (frame_id - 1, point.x, point.y)
            )
    result = []
    for track_id, entries in tracks.items():
        if len(entries) < 8:
            continue
        values = np.asarray(entries)
        design = np.column_stack((np.ones(len(values)), values[:, 0]))
        coefficients = np.linalg.lstsq(design, values[:, 1:], rcond=None)[0]
        scores = []
        for score, time in zip(maps, block_times):
            x, y = np.rint(coefficients[0] + coefficients[1] * time).astype(int)
            if 0 <= x < width and 0 <= y < height:
                scores.append(float(score[y, x]))
        fitted = design @ coefficients
        result.append(
            {
                "track_id": int(track_id),
                "length": len(entries),
                "intercept": coefficients[0].tolist(),
                "velocity": coefficients[1].tolist(),
                "rmse": float(
                    np.sqrt(np.mean(np.sum((values[:, 1:] - fitted) ** 2, axis=1)))
                ),
                "score": float(np.mean(scores)),
            }
        )
    return result


def evaluate_sequence(
    name: str,
    resolution: str,
    stems,
    truths,
    maps,
    times,
    args: argparse.Namespace,
    roi,
):
    raw = search(
        maps,
        times,
        roi,
        parse_range(args.vx),
        parse_range(args.vy),
        args.min_speed,
        args.peaks_per_velocity,
        args.peak_separation,
        args.temporal_reducer,
    )
    candidates = select_candidates(
        raw, args.top_candidates, args.path_separation, len(stems)
    )
    metrics = evaluate(candidates, truths, args.radius)
    best = max(metrics, key=lambda row: row["f1"])
    report = {
        "sequence": name,
        "resolution": resolution,
        "frames": len(stems),
        "truth_points": sum(map(len, truths)),
        "oracle": oracle(truths, maps, times),
        "candidates": [
            {
                "rank": rank,
                "midpoint": candidate["midpoint"].tolist(),
                "velocity": candidate["velocity"].tolist(),
                        "score": candidate["score"],
                        "reference_time": candidate["reference_time"],
                        "features": candidate["features"],
                    }
            for rank, candidate in enumerate(candidates, 1)
        ],
        "evaluation": metrics,
        "best": best,
    }
    print(f"shift-stack {name}: {best}", flush=True)
    return report


def main() -> None:
    args = parse_args()
    resolutions = set(args.resolutions.split(",")) if args.resolutions else None
    roi = tuple(map(int, args.roi.split(",")))
    reports = []
    selected_sequences = set(args.sequences.split(",")) if args.sequences else None
    detector = None
    if args.score_source == "deeppro_log":
        if not args.deeppro_source or not args.deeppro_weights:
            raise ValueError(
                "--deeppro-source and --deeppro-weights are required for deeppro_log"
            )
        detector = DeepProDetector(
            args.deeppro_source,
            args.deeppro_weights,
            device=args.deeppro_device,
            tile_size=args.deeppro_tile_size,
            tile_halo=args.deeppro_tile_halo,
        )
    eligible = []
    for sequence in _sequences(Path(args.root)):
        if selected_sequences and sequence.name not in selected_sequences:
            continue
        images = _files(sequence / "img")
        if not images:
            continue
        width, height = Image.open(next(iter(images.values()))).size
        resolution = f"{width}x{height}"
        if resolutions and resolution not in resolutions:
            continue
        eligible.append((sequence, resolution))

    if args.concatenate and eligible:
        all_stems = []
        all_truths = []
        all_maps = []
        all_times = []
        frame_offset = 0
        resolutions = {resolution for _, resolution in eligible}
        if len(resolutions) != 1:
            raise ValueError(f"Cannot concatenate mixed resolutions: {resolutions}")
        for sequence, _ in eligible:
            stems, truths, maps, times = block_scores(
                sequence,
                args.block_size,
                args.score_source,
                args.background_samples,
                detector,
                args.deeppro_bidirectional,
                args.deeppro_dog_weight,
                args.candidate_threshold,
                args.candidate_min_area,
                args.candidate_max_area,
                args.hybrid_registration,
            )
            all_stems.extend(f"{sequence.name}/{stem}" for stem in stems)
            all_truths.extend(truths)
            all_maps.extend(maps)
            all_times.extend(times + frame_offset)
            frame_offset += len(stems)
        reports.append(
            evaluate_sequence(
                "+".join(sequence.name for sequence, _ in eligible),
                next(iter(resolutions)),
                all_stems,
                all_truths,
                all_maps,
                np.asarray(all_times),
                args,
                roi,
            )
        )
    else:
        for sequence, resolution in eligible:
            stems, truths, maps, times = block_scores(
                sequence,
                args.block_size,
                args.score_source,
                args.background_samples,
                detector,
                args.deeppro_bidirectional,
                args.deeppro_dog_weight,
                args.candidate_threshold,
                args.candidate_min_area,
                args.candidate_max_area,
                args.hybrid_registration,
            )
            reports.append(
                evaluate_sequence(
                    sequence.name,
                    resolution,
                    stems,
                    truths,
                    maps,
                    times,
                    args,
                    roi,
                )
            )
    report = {"settings": vars(args), "sequences": reports}
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
