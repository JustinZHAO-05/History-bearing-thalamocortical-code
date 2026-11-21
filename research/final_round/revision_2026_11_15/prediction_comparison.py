"""Compare local forecasts with the independent reference mean response."""
import os
import experiments as ex
from pathlib import Path
import numpy as np,json
ROOT=Path(__file__).resolve().parent;OUT=Path(os.environ.get('NOV15_RESULTS',str(ROOT/'results')))
rows=[]
for i,p in enumerate(sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db'))):
 net=ex.v1.eng.IndependentNetwork(p);tr=ex.receiver_run(net,ex.v1.eng.contexts(12,97001,(2,10,18)),False,97200+i)
 yy=tr['y'][310:400,:,0];ref=yy[...,0]-yy[...,1];template=np.stack([ref[:,j::3].mean(1) for j in range(3)],axis=1)
 z=np.load(OUT/f'receiver_{net.name}.npz');actual=z['actual'];pred=z['prediction'];qidx=np.arange(actual.shape[1])%3
 tp=template[:,qidx];scores={}
 for name,pr in [('gain',pred[:,:,0]),('template',tp)]:
  scores[name]=dict(rmse=float(np.sqrt(np.mean((pr-actual)**2))),r2=float(1-np.sum((pr-actual)**2)/np.sum((actual-actual.mean())**2)),within_evidence={})
  for j,q in enumerate((2,10,18)):
   a=actual[:,j::3];b=pr[:,j::3];scores[name]['within_evidence'][str(q)]=dict(rmse=float(np.sqrt(np.mean((b-a)**2))),r2=float(1-np.sum((b-a)**2)/np.sum((a-a.mean())**2)))
 row=dict(model=net.name,scores=scores);rows.append(row);np.savez_compressed(OUT/f'reference_template_{net.name}.npz',template=template)
 print(json.dumps(row),flush=True)
ex.dump(OUT/'prediction_comparison.json',rows)
