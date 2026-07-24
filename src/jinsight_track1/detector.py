import numpy as np
class FakeDetector:
    def predict(self, frames):
        out=[]
        for frame in frames:
            a=np.asarray(frame, dtype=float); a=(a-a.min())/(a.max()-a.min() or 1)
            out.append(a)
        return out
