import numpy as np
from scipy.optimize import linear_sum_assignment
def point_metrics(pred, truth, radius=2.):
    p=np.asarray(pred,float).reshape(-1,2); t=np.asarray(truth,float).reshape(-1,2); tp=0
    if len(p) and len(t):
        cost=np.linalg.norm(p[:,None]-t[None,:],axis=2); rows,cols=linear_sum_assignment(cost)
        tp=sum(cost[i,j]<=radius for i,j in zip(rows,cols))
    fp=len(p)-tp; fn=len(t)-tp
    return {"tp":int(tp),"fp":int(fp),"fn":int(fn),"precision":tp/(tp+fp) if tp+fp else 0.,"recall":tp/(tp+fn) if tp+fn else 0.,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
