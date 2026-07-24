import numpy as np
from jinsight_track1.postprocess import centroids
from jinsight_track1.windowing import infer_sequence
from jinsight_track1.detector import FakeDetector
from jinsight_track1.types import SequencePrediction,TrackPoint
from jinsight_track1.submission import render,parse
from jinsight_track1.evaluation import point_metrics
def test_centroid_xy_and_eight_connectivity():
    a=np.zeros((10,12)); a[2:4,7:9]=1; a[4,9]=1
    d=centroids(a,min_area=2)[0]; assert abs(d.x-7.8)<.3 and abs(d.y-3.0)<.3
def test_windows_cover_once():
    frames=[np.full((3,4),i) for i in range(10)]; assert len(infer_sequence(frames,FakeDetector(),4,2))==10
def test_submission_xy_yx_roundtrip():
    p=SequencePrediction("s",{1:[TrackPoint(1,4,2.5,9.5)]}); text=render(p,"yx"); assert "4 9.500000 2.500000" in text; assert parse(text,coordinate_order="yx").frames[1][0].x==2.5
def test_metrics_zero_and_exact():
    assert point_metrics([],[])['f1']==0; assert point_metrics([[1,2]],[[1,2]])['f1']==1
