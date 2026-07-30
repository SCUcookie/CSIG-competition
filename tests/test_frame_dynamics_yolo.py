import numpy as np

from jinsight_track1.frame_dynamics_yolo import frame_dynamics, point_box_labels


def test_frame_dynamics_channels_and_gain():
    previous2 = np.array([[10, 20]], dtype=np.uint8)
    previous = np.array([[12, 18]], dtype=np.uint8)
    current = np.array([[15, 10]], dtype=np.uint8)
    result = frame_dynamics(current, previous, previous2, diff_gain=4)
    assert result.tolist() == [[[15, 12, 20], [10, 32, 40]]]


def test_point_box_labels_preserve_component_centroid():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[4:6, 10:12] = 255
    label = point_box_labels(mask, box_size=8)[0].split()
    assert label[0] == "0"
    assert np.allclose([float(x) for x in label[1:]], [10.5 / 40, 4.5 / 20, 8 / 40, 8 / 20])
