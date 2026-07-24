from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import io, re
from .types import SequencePrediction, TrackPoint
def _xy(p, order): return (p.x,p.y) if order=="xy" else (p.y,p.x)
def render(pred, coordinate_order="xy"):
    if coordinate_order not in ("xy","yx"): raise ValueError("coordinate order must be xy or yx")
    lines=[]
    for frame in sorted(pred.frames):
        pts=pred.frames[frame]; row=[f"{frame:05d}",str(len(pts))]
        for p in pts:
            x,y=_xy(p,coordinate_order); row += [str(p.track_id),f"{x:.6f}",f"{y:.6f}"]
        lines.append(" ".join(row))
    return "\n".join(lines)+"\n"
def parse(text, sequence_name="sequence", coordinate_order="xy"):
    if coordinate_order not in ("xy","yx"): raise ValueError("coordinate order must be xy or yx")
    frames={}
    for line_no,line in enumerate(text.splitlines(),1):
        parts=line.split()
        if len(parts)<2 or not parts[0].isdigit() or len(parts[0])<5: raise ValueError(f"invalid line {line_no}")
        n=int(parts[1]); rest=parts[2:]
        if len(rest)!=3*n: raise ValueError(f"target count mismatch on line {line_no}")
        pts=[]
        for i in range(n):
            try: tid=int(rest[3*i]); a=float(rest[3*i+1]); b=float(rest[3*i+2])
            except ValueError as e: raise ValueError(f"invalid point on line {line_no}") from e
            x,y=(a,b) if coordinate_order=="xy" else (b,a); pts.append(TrackPoint(int(parts[0]),tid,x,y))
        frames[int(parts[0])]=pts
    return SequencePrediction(sequence_name,frames)
def write_txt(pred, path, coordinate_order="xy", overwrite=False):
    path=Path(path)
    if path.exists() and not overwrite: raise FileExistsError(path)
    path.write_text(render(pred,coordinate_order),encoding="ascii")
def package(directory, output, expected=None, overwrite=False, coordinate_order="xy"):
    directory=Path(directory); output=Path(output)
    if output.exists() and not overwrite: raise FileExistsError(output)
    files=sorted(directory.glob("*.txt"))
    if expected is not None and len(files)!=expected: raise ValueError("txt count does not match expected sequences")
    names=set()
    for p in files:
        if p.name in names or p.name in ("","."): raise ValueError("duplicate file")
        names.add(p.name); parse(p.read_text(encoding="ascii"),p.stem,coordinate_order)
    with ZipFile(output,"w",ZIP_DEFLATED) as z:
        for p in files: z.writestr(p.name,p.read_bytes())
    with ZipFile(output) as z:
        if any("/" in n or "\\" in n for n in z.namelist()): raise ValueError("zip contains a subdirectory")
        if len(z.namelist())!=len(files): raise ValueError("unexpected zip entry count")
        for n in z.namelist(): parse(z.read(n).decode("ascii"),Path(n).stem,coordinate_order)
    return output
