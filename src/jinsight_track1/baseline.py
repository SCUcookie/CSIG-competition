from pathlib import Path
import json
import numpy as np
from PIL import Image
from .data import discover_sequences
from .postprocess import centroids
from .types import Detection, SequencePrediction, TrackPoint
from .submission import write_txt, package
from scipy import ndimage

def _image_mask_pairs(sequence):
    image_dir=sequence.frames[0].parent
    if image_dir.name.lower() not in {"img","images","frames"}: return []
    mask_dir=image_dir.parent/"mask"
    by_stem={p.stem:p for p in mask_dir.glob("*")} if mask_dir.is_dir() else {}
    return [(p,by_stem[p.stem]) for p in sequence.frames if p.stem in by_stem]

def train_threshold(root, output, max_frames=2000, seed=7):
    rng=np.random.default_rng(seed); values=[]; labels=[]; seen=0
    for seq in discover_sequences(root):
        for image_path,mask_path in _image_mask_pairs(seq):
            image=np.asarray(Image.open(image_path),dtype=np.float32)
            mask=np.asarray(Image.open(mask_path))>0
            if image.shape!=mask.shape: continue
            flat=image.ravel(); target=mask.ravel()
            take=min(2048,flat.size)
            idx=rng.choice(flat.size,take,replace=False)
            values.append(flat[idx]); labels.append(target[idx]); seen+=1
            if seen>=max_frames: break
        if seen>=max_frames: break
    if not values: raise RuntimeError("no paired image/mask frames found")
    x=np.concatenate(values); y=np.concatenate(labels); candidates=np.unique(np.quantile(x,np.linspace(.70,.999,64)))
    best=None
    for threshold in candidates:
        pred=x>=threshold; tp=np.count_nonzero(pred&y); fp=np.count_nonzero(pred&~y); fn=np.count_nonzero(~pred&y)
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        item=(float(f1),float(threshold),int(tp),int(fp),int(fn))
        if best is None or item>best: best=item
    model={"model":"global_intensity_threshold","threshold":best[1],"sampled_frames":seen,"sampled_pixels":int(x.size),"proxy_f1":best[0],"tp":best[2],"fp":best[3],"fn":best[4],"seed":seed}
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(model,indent=2)+"\n",encoding="ascii")
    return model

def infer_threshold(root, model_path, output_dir, max_sequences=None, max_frames=None, track=False, coordinate_order="xy", start=0, end=None, make_package=True):
    model=json.loads(Path(model_path).read_text(encoding="ascii")); threshold=float(model["threshold"])
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); sequences=discover_sequences(root,max_sequences,max_frames)[start:end]
    for seq in sequences:
        frames={}
        for frame_id,path in enumerate(seq.frames,1):
            image=np.asarray(Image.open(path),dtype=np.float32); mask=(image>=threshold).astype(np.float32)
            # Bound noisy threshold masks to a few local maxima for a usable CPU submission pass.
            maxima=(image==ndimage.maximum_filter(image,size=5)) & (image>=threshold)
            ys,xs=np.where(maxima)
            order=np.argsort(image[ys,xs])[::-1][:10]
            detections=[Detection(float(xs[i]),float(ys[i]),float(image[ys[i],xs[i]])) for i in order]
            frames[frame_id]=[TrackPoint(frame_id,0,d.x,d.y) for d in detections]
        write_txt(SequencePrediction(seq.name,frames),output_dir/(seq.name+".txt"),coordinate_order,overwrite=True)
    zip_path=output_dir.with_suffix(".zip")
    if make_package: package(output_dir,zip_path,expected=len(sequences),overwrite=True,coordinate_order=coordinate_order)
    return {"sequences":len(sequences),"zip":str(zip_path) if make_package else None,"threshold":threshold}
