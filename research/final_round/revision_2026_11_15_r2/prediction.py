"""Matched gain/recurrence comparisons and reference-template skill."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,csv
import numpy as np
import experiments as ex
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'revision_2026_11_15/results';OUT=ex.OUT
def main():
 rows=[];flat=[]
 for path in sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db')):
  net=ex.v1.eng.IndependentNetwork(path);z=np.load(OLD/f'receiver_{net.name}.npz');actual=z['actual'];base=z['base'];q=np.arange(len(base))%3
  gains=z['reference_gains'];pre=z['presensory_gains'];initial=z['initial_difference'];delta=np.repeat(initial[:,None],4,axis=1);pred=np.empty((90,len(base),4))
  for t in range(90):
   for j in range(4):
    D=gains[t,q] if j in (0,2) else pre[q];dr=delta[:,j]*D;pred[t,:,j]=dr@net.output
    recurrence=dr@net.W.T if j in (0,1) else 0.
    delta[:,j]+=net.alpha*(recurrence-delta[:,j])
  assert np.max(abs(pred[:,:,:2]-z['prediction'][:,:,:2]))<1e-12
  assert np.max(abs(pred[:,:,3]-z['prediction'][:,:,2]))<1e-12
  template=np.load(OLD/f'reference_template_{net.name}.npz')['template'][:,q]
  predictions=np.concatenate([pred,template[:,:,None]],axis=-1);names=['evidence_recurrence','pre_recurrence','evidence_leak','pre_leak','reference_template'];scores={}
  for j,name in enumerate(names):
   p=predictions[:,:,j];error=(p-actual)**2;score=dict(rmse=float(np.sqrt(error.mean())),r2=float(1-error.sum()/np.sum((actual-actual.mean())**2)),template_relative_skill=float(1-error.sum()/np.sum((actual-template)**2)),within_evidence={})
   for qi,number in enumerate((2,10,18)):
    a=actual[:,qi::3];b=p[:,qi::3];ss=float(np.sum((b-a)**2));den=float(np.sum((a-template[:,qi::3])**2));score['within_evidence'][str(number)]=dict(rmse=float(np.sqrt(np.mean((a-b)**2))),r2=float(1-ss/np.sum((a-a.mean())**2)),template_relative_skill=float(1-ss/den))
    flat.append(dict(model=net.name,predictor=name,q=number,**score['within_evidence'][str(number)]))
   scores[name]=score
  compare={}
  for name,j,k in [('template_minus_gain',4,0),('same_gain_leak_minus_recurrence',2,0),('pre_gain_minus_evidence_gain',1,0)]:
   diff=(((predictions[:,:,j]-actual)**2)-((predictions[:,:,k]-actual)**2)).mean(0);compare[name]=ex.interval(diff,base)
  row=dict(model=net.name,n_reference=12,n_test=24,scores=scores,paired_mse_difference=compare);rows.append(row)
  np.savez_compressed(OUT/f'prediction_{net.name}.npz',prediction=predictions,actual=actual,base=base,initial_difference=initial,reference_gains=gains)
 ex.dump(OUT/'prediction_summary.json',rows)
 with (OUT/'prediction_by_strength.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print(json.dumps([dict(model=r['model'],rmse={k:v['rmse'] for k,v in r['scores'].items()},comparisons=r['paired_mse_difference']) for r in rows],indent=2))
if __name__=='__main__':main()
