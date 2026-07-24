import numpy as np
from scipy import ndimage
from .types import Detection
def centroids(mask, threshold=.5, min_area=1):
    a=np.asarray(mask, dtype=float)
    if a.ndim != 2: raise ValueError("mask must be 2-D")
    on=a>=threshold; structure=np.ones((3,3),dtype=np.uint8); labels,count=ndimage.label(on,structure)
    ids=np.arange(1,count+1)
    sizes=ndimage.sum(on,labels,index=ids); intensity=ndimage.sum(a,labels,index=ids)
    centers=ndimage.center_of_mass(a,labels,index=ids)
    found=[]
    for label,size,total,(y,x) in zip(ids,sizes,intensity,centers):
        if size<min_area: continue
        found.append(Detection(float(x),float(y),float(total/size)))
    return sorted(found,key=lambda d:(d.y,d.x))
