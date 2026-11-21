"""Separate geometric and trajectory checks for the focused revision."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.optimize import brentq
import experiments as ex
ROOT=Path(__file__).resolve().parent;OUT=ex.OUT
def geometry():
 rows=[];rng=np.random.default_rng(99515)
 for path in sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db')):
  model=ex.previous.Model(path);z=np.load(OUT/f'axis_{model.name}.npz');B=model.W[:400,400:470];cap=2*np.minimum(z['source_mean'],20-z['source_mean']).reshape(-1,70);d=z['natural_source_difference'].reshape(-1,70);target=np.sum((d@B.T)**2,axis=1);largest=0.;maxroot=0.
  for j,q in enumerate(z['random_axes']):
   c=z['source_candidate_difference'][j].reshape(-1,70);actual=np.sum((c@B.T)**2,axis=1)
   for idx in rng.choice(len(cap),64,replace=False):
    knots=np.unique(np.r_[0.,np.divide(cap[idx],abs(q),out=np.full(70,np.inf),where=abs(q)>1e-18)]);knots=knots[np.isfinite(knots)]
    energy=np.sum((np.clip(knots[:,None]*q,-cap[idx],cap[idx])@B.T)**2,axis=1);hit=np.flatnonzero(energy>=target[idx])
    if len(hit):
     h=hit[0];l=max(0,h-1);root=brentq(lambda k:np.sum((B@np.clip(k*q,-cap[idx],cap[idx]))**2)-target[idx],knots[l],knots[h],xtol=1e-13)
     expected=np.sum((B@np.clip(root*q,-cap[idx],cap[idx]))**2);maxroot=max(maxroot,abs(expected-target[idx])/max(target[idx],1e-20))
    else:expected=energy.max()
    largest=max(largest,abs(actual[idx]-expected)/max(target[idx],1e-20))
  assert largest<1e-8 and maxroot<1e-8
  rows.append(dict(model=model.name,direct_points=256,max_relative_energy_difference=largest,max_root_relative_error=maxroot,source_bounds=float((abs(z['source_candidate_difference'])-2*np.minimum(z['source_mean'],20-z['source_mean'])).max())))
 ex.dump(OUT/'independent_geometry.json',rows);return rows
def trajectory():
 path=sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db'))[0];m=ex.previous.Model(path);ctx=ex.v1.eng.contexts(48,99102,(2,10,18));u=ctx['u'][:9];x=np.repeat(ctx['x0'][:9,None],2,axis=1)
 for t in range(1400):x=m.step(x,u[:,:,t])
 natural=x.copy();donor=[]
 for t in range(1400,1600):donor.append(m.rate(x)[:,:,400:470].copy());x=m.step(x,u[:,:,t])
 x=np.repeat(natural.mean(1)[:,None],2,axis=1);B=m.W[:400,400:470]
 for t in range(1400,1710):
  r=m.rate(x);drive=r@m.W.T
  if t<1600:drive[...,:400]+=donor[t-1400]@B.T-r[...,400:470]@B.T
  x=(1-m.alpha)*x+m.alpha*(drive+u[:,:,t,None]*m.Win)
 meta=json.loads((OUT/f'lowdose_{m.name}.json').read_text());saved=np.load(OUT/f'lowdose_{m.name}.npz');indices=[0,1,2,10];x=np.repeat(x[:,None],4,axis=1)
 common=x[:,0,:,:400].mean(1);shift=.5*(np.clip(common,2,18)-common);x[:,1,:,:400]+=meta['selected_displacement']*shift[:,None]
 x[:,2,:,:400]+=meta['selected_displacement']*np.linalg.norm(shift,axis=1)[:,None,None]*saved['random_directions'][0][None,None]
 trace=[]
 for t in range(1710,2200):
  r=m.rate(x);trace.append(r@m.Wout+m.bias);drive=r@m.W.T
  if t<1730:drive[:,3]-=meta['selected_attenuation']*(drive[:,3]-drive[:,3].mean(1,keepdims=True))
  x=(1-m.alpha)*x+m.alpha*(drive+u[:,None,:,t,None]*m.Win)
 error=float(np.max(abs(np.array(trace)-saved['output_trace'][:,:9][:,:,indices])))
 assert error<1e-9,error
 result=dict(model=m.name,bases=3,conditions=indices,max_output_error=error);ex.dump(OUT/'independent_receiver_trajectory.json',result);return result
if __name__=='__main__':print(json.dumps(dict(geometry=geometry(),receiver=trajectory()),indent=2))
