import argparse, json, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from .data import inspect, discover_sequences, read_frame
from .detector import FakeDetector
from .windowing import infer_sequence
from .postprocess import centroids
from .types import SequencePrediction, TrackPoint
from .tracking import track_detections
from .submission import write_txt, package
from .baseline import train_threshold, infer_threshold
from .yolo_seg import prepare_yolo_dataset, train_yolo, infer_yolo, evaluate_yolo
def smoke():
    with tempfile.TemporaryDirectory(prefix="jinsight-smoke-") as td:
        root=Path(td)/"data"; out=Path(td)/"pred"; root.mkdir(); out.mkdir()
        for name,shape in (("seq_2",(18,24)),("seq_10",(20,30))):
            d=root/name; d.mkdir()
            for i in range(5):
                a=np.zeros(shape,dtype=np.uint8); a[(i+3)%shape[0],(i+5)%shape[1]]=255
                Image.fromarray(a).save(d/f"frame_{i+1:05d}.png")
        seqs=discover_sequences(root); total=0
        for s in seqs:
            masks=infer_sequence([read_frame(p) for p in s.frames],FakeDetector(),3,1)
            detections=[centroids(m,.5,1) for m in masks]; tracked=track_detections(detections,read_frame(s.frames[0]).shape)
            frames={i+1:tracked.get(i+1,[]) for i in range(len(masks))}
            write_txt(SequencePrediction(s.name,frames),out/f"{s.name}.txt"); total+=len(frames)
        z=package(out,Path(td)/"submission.zip",expected=len(seqs)); print(json.dumps({"sequences":len(seqs),"frames":total,"zip":str(z),"root_entries":len(__import__('zipfile').ZipFile(z).namelist())}))
def main(argv=None):
    p=argparse.ArgumentParser(prog="jinsight-track1"); sub=p.add_subparsers(dest="command",required=True)
    q=sub.add_parser("inspect-data"); q.add_argument("root"); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int); q.add_argument("--json",action="store_true")
    sub.add_parser("smoke")
    q=sub.add_parser("package"); q.add_argument("directory"); q.add_argument("output"); q.add_argument("--expected",type=int); q.add_argument("--overwrite",action="store_true"); q.add_argument("--coordinate-order",choices=["xy","yx"],default="xy")
    q=sub.add_parser("infer"); q.add_argument("root"); q.add_argument("output"); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int); q.add_argument("--window",type=int,default=8); q.add_argument("--overlap",type=int,default=2); q.add_argument("--track",action="store_true"); q.add_argument("--coordinate-order",choices=["xy","yx"],default="xy")
    sub.add_parser("estimate-hardware")
    q=sub.add_parser("train"); q.add_argument("root"); q.add_argument("output"); q.add_argument("--max-frames",type=int,default=2000); q.add_argument("--seed",type=int,default=7)
    q=sub.add_parser("baseline-infer"); q.add_argument("root"); q.add_argument("model"); q.add_argument("output"); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int); q.add_argument("--start",type=int,default=0); q.add_argument("--end",type=int); q.add_argument("--no-package",action="store_true"); q.add_argument("--coordinate-order",choices=["xy","yx"],default="xy")
    q=sub.add_parser("yolo-prepare", help="convert sequence img/mask data to a symlinked YOLO-seg dataset")
    q.add_argument("train_root"); q.add_argument("output_root"); q.add_argument("--val-root"); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int)
    q=sub.add_parser("yolo-train", help="train YOLOv8n-seg on the challenge data")
    q.add_argument("train_root"); q.add_argument("work_dir"); q.add_argument("--val-root"); q.add_argument("--weights",default="yolov8n-seg.pt"); q.add_argument("--device",default="0,1,2,3"); q.add_argument("--imgsz",type=int,default=640); q.add_argument("--epochs",type=int,default=50); q.add_argument("--workers",type=int,default=4); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int); q.add_argument("--download-weights",action="store_true")
    q=sub.add_parser("yolo-infer", help="infer a test/validation root and build submission ZIP")
    q.add_argument("model"); q.add_argument("data_root"); q.add_argument("output"); q.add_argument("--device",default="0"); q.add_argument("--conf",type=float,default=.25); q.add_argument("--max-sequences",type=int); q.add_argument("--max-frames",type=int); q.add_argument("--coordinate-order",choices=["xy","yx"],default="xy"); q.add_argument("--no-package",action="store_true")
    q=sub.add_parser("yolo-eval", help="local proxy point metrics against validation masks")
    q.add_argument("model"); q.add_argument("val_root"); q.add_argument("output"); q.add_argument("--device",default="0"); q.add_argument("--conf",type=float,default=.25); q.add_argument("--radius",type=float,default=2.)
    a=p.parse_args(argv)
    if a.command=="smoke": return smoke()
    if a.command=="inspect-data":
        value=inspect(a.root,a.max_sequences,a.max_frames); print(json.dumps(value,indent=2)); return
    if a.command=="package": return print(package(a.directory,a.output,a.expected,a.overwrite,a.coordinate_order))
    if a.command=="estimate-hardware": print("Planning baseline: 8 CPU cores, 32 GB RAM, 250 GB free NVMe; this is not a training run."); return
    if a.command=="train": print(json.dumps(train_threshold(a.root,a.output,a.max_frames,a.seed),indent=2)); return
    if a.command=="baseline-infer": print(json.dumps(infer_threshold(a.root,a.model,a.output,a.max_sequences,a.max_frames,False,a.coordinate_order,a.start,a.end,not a.no_package),indent=2)); return
    if a.command=="yolo-prepare": print(json.dumps(prepare_yolo_dataset(a.train_root,a.output_root,a.val_root,a.max_sequences,a.max_frames),indent=2)); return
    if a.command=="yolo-train": print(json.dumps(train_yolo(a.train_root,a.work_dir,a.weights,a.val_root,a.device,a.imgsz,a.epochs,a.max_sequences,a.max_frames,a.download_weights,a.workers),indent=2)); return
    if a.command=="yolo-infer": print(json.dumps(infer_yolo(a.model,a.data_root,a.output,a.device,a.conf,a.coordinate_order,a.max_sequences,a.max_frames,not a.no_package),indent=2)); return
    if a.command=="yolo-eval": print(json.dumps(evaluate_yolo(a.model,a.val_root,a.output,a.device,a.conf,a.radius),indent=2)); return
    # infer intentionally uses only the deterministic fake detector in this baseline.
    seqs=discover_sequences(a.root,a.max_sequences,a.max_frames); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for s in seqs:
        frames=[read_frame(x) for x in s.frames]; masks=infer_sequence(frames,FakeDetector(),a.window,a.overlap); ds=[centroids(x) for x in masks]
        tracked=track_detections(ds,frames[0].shape) if a.track else {}
        pred=SequencePrediction(s.name,{i+1:(tracked.get(i+1,[]) if a.track else [TrackPoint(i+1,0,d.x,d.y) for d in ds[i]]) for i in range(len(ds))})
        write_txt(pred,out/f"{s.name}.txt",a.coordinate_order)
    package(out,Path(out).with_suffix(".zip"),len(seqs),True,a.coordinate_order)
if __name__=="__main__": main()
