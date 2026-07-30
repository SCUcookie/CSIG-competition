import numpy as np
from types import SimpleNamespace
from jinsight_track1.postprocess import centroids
from jinsight_track1.windowing import infer_sequence
from jinsight_track1.detector import FakeDetector
from jinsight_track1.types import SequencePrediction,TrackPoint
from jinsight_track1.submission import render,parse
from jinsight_track1.evaluation import point_metrics
from jinsight_track1.yolo_seg import _prediction_points
from jinsight_track1.deeppro_adapter import component_points, temporal_windows
from jinsight_track1.tracking import assign_track_ids
from jinsight_track1.tbd_patch import extract_patch_channels, spatial_score
def test_centroid_xy_and_eight_connectivity():
    a=np.zeros((10,12)); a[2:4,7:9]=1; a[4,9]=1
    d=centroids(a,min_area=2)[0]; assert abs(d.x-7.8)<.3 and abs(d.y-3.0)<.3
def test_windows_cover_once():
    frames=[np.full((3,4),i) for i in range(10)]; assert len(infer_sequence(frames,FakeDetector(),4,2))==10
def test_submission_xy_yx_roundtrip():
    p=SequencePrediction("s",{1:[TrackPoint(1,4,2.5,9.5)]}); text=render(p,"yx"); assert "4 9.500000 2.500000" in text; assert parse(text,coordinate_order="yx").frames[1][0].x==2.5
def test_metrics_zero_and_exact():
    assert point_metrics([],[])['f1']==0; assert point_metrics([[1,2]],[[1,2]])['f1']==1


def _fake_seg_result(shape, centre, polygon, confidence=.9):
    return SimpleNamespace(
        orig_shape=shape,
        masks=SimpleNamespace(
            # This deliberately unrelated 640 canvas catches regressions that
            # accidentally calculate centroids on masks.data again.
            data=np.pad(np.ones((1, 2, 2)), ((0, 0), (100, 538), (300, 338))),
            xy=[np.asarray(polygon, dtype=float)],
        ),
        boxes=SimpleNamespace(
            conf=np.asarray([confidence]),
            xywh=np.asarray([[centre[0], centre[1], 10.0, 10.0]]),
        ),
    )


def test_yolo_centroid_uses_original_image_polygon_for_mixed_resolutions():
    cases = [
        ((256, 256), (80.0, 120.0)),
        ((512, 640), (500.0, 300.0)),
        ((733, 742), (600.0, 200.0)),
        ((1024, 1024), (800.0, 700.0)),
    ]
    for shape, (x, y) in cases:
        polygon = [(x-2, y-3), (x+2, y-3), (x+2, y+3), (x-2, y+3)]
        point = _prediction_points(_fake_seg_result(shape, (x, y), polygon), .25)[0]
        assert np.allclose(point[:2], (x, y))


def test_yolo_centroid_falls_back_to_original_box_and_filters_confidence():
    result = _fake_seg_result((256, 256), (90.0, 40.0), [(90.0, 40.0)], .2)
    assert _prediction_points(result, .25) == []
    point = _prediction_points(result, .1)[0]
    assert np.allclose(point[:2], (90.0, 40.0))


def test_yolo_centroid_falls_back_when_polygon_mapping_is_outside_image():
    result = _fake_seg_result(
        (512, 640),
        (320.0, 250.0),
        [(975.0, 500.0), (985.0, 500.0), (985.0, 508.0), (975.0, 508.0)],
    )
    assert np.allclose(_prediction_points(result, .25)[0][:2], (320.0, 250.0))


def test_yolo_detection_model_uses_original_box_centres():
    result = SimpleNamespace(
        orig_shape=(512, 640),
        masks=None,
        boxes=SimpleNamespace(
            conf=np.asarray([.9, .1]),
            xywh=np.asarray([[320.0, 250.0, 12.0, 12.0],
                             [100.0, 100.0, 12.0, 12.0]]),
        ),
    )
    assert _prediction_points(result, .25) == [(320.0, 250.0, .9)]


def test_deeppro_temporal_windows_cover_and_end_align():
    windows = temporal_windows(100, 40, 4)
    assert windows == [(0, 40), (36, 76), (60, 100)]
    assert set().union(*(set(range(a, b)) for a, b in windows)) == set(range(100))
    assert temporal_windows(12, 40, 4) == [(0, 12)]


def test_deeppro_component_points_are_xy_and_area_filtered():
    probability = np.zeros((10, 12), dtype=float)
    probability[2:4, 7:9] = .9
    probability[8, 1] = .8
    points = component_points(probability, .5, min_area=2)
    assert len(points) == 1
    assert np.allclose(points[0], (7.5, 2.5))


def test_deeppro_component_point_modes_use_probability_shape():
    probability = np.zeros((5, 6), dtype=float)
    probability[2, 2] = .6
    probability[2, 3] = .9
    binary = component_points(probability, .5, centroid_mode="binary")[0]
    weighted = component_points(probability, .5, centroid_mode="weighted")[0]
    peak = component_points(probability, .5, centroid_mode="peak")[0]
    assert np.allclose(binary, (2.5, 2.0))
    assert 2.5 < weighted[0] < 3.0 and weighted[1] == 2.0
    assert peak == (3.0, 2.0)


def test_tracking_ids_are_stable_without_dropping_initial_detections():
    frames = [[(2, 3)], [(3, 3)], [], [(5, 3)]]
    tracked = assign_track_ids(frames, (100, 100), max_age=3)
    assert tracked[1][0].track_id == tracked[2][0].track_id == tracked[4][0].track_id
    assert len(tracked[1]) == 1 and tracked[3] == []


def test_polarity_invariant_patch_response_and_vectorized_features():
    image = np.full((25, 27), 100, dtype=np.uint8)
    image[12, 13] = 70
    dark = spatial_score(image, "dark")
    absolute = spatial_score(image, "absolute")
    assert absolute.shape == image.shape
    assert np.all(absolute >= 0)
    assert np.allclose(absolute, np.abs(dark))
    patches = extract_patch_channels(
        image, absolute, np.asarray([[13, 12], [0, 0]]), size=9
    )
    assert patches.shape == (2, 2, 9, 9)
    assert np.isfinite(patches).all()
