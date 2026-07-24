import numpy as np
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
