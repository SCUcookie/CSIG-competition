import numpy as np
def infer_sequence(frames, detector, window=8, overlap=2):
    if window < 1 or overlap < 0 or overlap >= window: raise ValueError("require window>=1 and 0<=overlap<window")
    out=[None]*len(frames); step=window-overlap; start=0
    while start<len(frames):
        end=min(len(frames), start+window); pred=list(detector.predict(frames[start:end]))
        if len(pred)!=end-start: raise ValueError("detector output length mismatch")
        for i,x in enumerate(pred):
            if out[start+i] is None: out[start+i]=np.asarray(x)
        if end==len(frames): break
        start += step
    return out
