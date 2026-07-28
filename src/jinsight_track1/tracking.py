import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import TrackPoint
def track_detections(frames, image_shape, gate_fraction=.02, max_age=5, min_hits=2):
    gate=float(np.hypot(*image_shape))*gate_fraction; tracks={}; next_id=1; result={}
    for frame_id,dets in enumerate(frames,1):
        candidates=[]
        for tid,t in tracks.items():
            pred=t[:2]+t[2:4]
            for i,d in enumerate(dets):
                dist=float(np.linalg.norm(pred-[d.x,d.y]))
                if dist<=gate: candidates.append((dist,tid,i))
        used_t=set(); used_d=set()
        for _,tid,i in sorted(candidates):
            if tid in used_t or i in used_d: continue
            d=dets[i]; t=tracks[tid]; pos=np.array([d.x,d.y]); t[2:4]=pos-t[:2]; t[:2]=pos; t[4]=0; t[5]+=1
            used_t.add(tid); used_d.add(i)
            if t[5]>=min_hits: result.setdefault(frame_id,[]).append(TrackPoint(frame_id,tid,d.x,d.y))
        for tid,t in list(tracks.items()):
            if tid not in used_t:
                t[:2]+=t[2:4]; t[4]+=1
                if t[4]>max_age: del tracks[tid]
        for i,d in enumerate(dets):
            if i not in used_d: tracks[next_id]=np.array([d.x,d.y,0.,0.,0.,1.]); next_id+=1
    return result


def assign_track_ids(frames, image_shape, gate_fraction=.02, max_age=3):
    """Assign stable IDs while retaining every detection for the main F1 task."""
    gate = max(3.0, float(np.hypot(*image_shape)) * gate_fraction)
    tracks = {}
    next_id = 1
    result = {}
    for frame_id, detections in enumerate(frames, 1):
        points = np.asarray(detections, dtype=float).reshape(-1, 2)
        track_ids = sorted(tracks)
        assignments = {}
        if track_ids and len(points):
            predicted = []
            for track_id in track_ids:
                track = tracks[track_id]
                elapsed = frame_id - track["frame"]
                predicted.append(track["position"] + track["velocity"] * elapsed)
            cost = np.linalg.norm(np.asarray(predicted)[:, None] - points[None, :], axis=2)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] <= gate:
                    assignments[int(column)] = track_ids[int(row)]

        output = []
        matched_tracks = set()
        for detection_index, point in enumerate(points):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = next_id
                next_id += 1
                velocity = np.zeros(2, dtype=float)
            else:
                previous = tracks[track_id]
                elapsed = max(1, frame_id - previous["frame"])
                observed = (point - previous["position"]) / elapsed
                velocity = .5 * previous["velocity"] + .5 * observed
            tracks[track_id] = {
                "position": point.copy(),
                "velocity": velocity,
                "frame": frame_id,
            }
            matched_tracks.add(track_id)
            output.append(TrackPoint(frame_id, track_id, float(point[0]), float(point[1])))
        for track_id in list(tracks):
            if track_id not in matched_tracks and frame_id - tracks[track_id]["frame"] > max_age:
                del tracks[track_id]
        result[frame_id] = sorted(output, key=lambda point: point.track_id)
    return result
