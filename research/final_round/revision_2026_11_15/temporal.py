import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import os
import experiments as ex
from pathlib import Path
import numpy as np,json,hashlib
from concurrent.futures import ProcessPoolExecutor
ROOT=Path(__file__).resolve().parent;OUT=Path(os.environ.get('NOV15_RESULTS',str(ROOT/'results')))
def run(job):
 i,p=job;net=ex.v1.eng.IndependentNetwork(p);ctx=ex.v1.eng.contexts(48,96003,(0,2,18,20));init,rate=ex.v1.donor(net,ctx)
 mu=rate.mean(2);d=rate[:,:,0]-rate[:,:,1];cap=2*np.minimum(mu,20-mu);e,inh=ex.classes(p);B=net.B;M=B.T@B;A=np.array([B[e].sum(0),B[inh].sum(0)])
 cur=d.copy();h=cur@M;h0=h.copy();rough=np.sum(np.diff(cur,axis=0)**2,axis=(0,2));naturalrough=rough.copy();rng=np.random.default_rng(98500+i)
 for k in range(256):
  ids=rng.choice(70,3,replace=False);z=np.cross(A[0,ids],A[1,ids]);nrm=np.linalg.norm(z)
  if nrm<1e-15:continue
  z/=nrm;den=z@M[np.ix_(ids,ids)]@z
  if den<1e-20:continue
  alpha=-2*(h[:,:,ids]@z)/den;delta=alpha[:,:,None]*z;cand=cur[:,:,ids]+delta
  change=2*np.sum(delta*(h[:,:,ids]-h0[:,:,ids]),axis=(0,2))+np.sum(alpha**2,axis=0)*den
  newrough=rough-np.sum(np.diff(cur[:,:,ids],axis=0)**2,axis=(0,2))+np.sum(np.diff(cand,axis=0)**2,axis=(0,2))
  good=(np.max(abs(cand)-cap[:,:,ids],axis=(0,2))<1e-10)&(change>1e-16)&(newrough<=1.25*naturalrough+1e-15)
  for idx in ids:cur[:,good,idx]=cand[:,good,list(ids).index(idx)]
  h[:,good]+=alpha[:,good,None]*(z@M[ids]);rough[good]=newrough[good]
 bd=d@B.T;bc=cur@B.T;energy=np.sum(bd**2,axis=-1);error=np.max(np.divide(abs(np.sum(bc**2,axis=-1)-energy),energy,out=np.zeros_like(energy),where=energy>1e-15));ei=float(abs((cur-d)@A.T).max())
 assert error<1e-8 and ei<1e-7 and np.max(abs(cur)-cap)<1e-8
 n=len(init);x=np.repeat(init.mean(axis=1)[:,None,None],2,axis=1);x=np.repeat(x,2,axis=2);y=np.empty((800,n,2,2))
 for k in range(800):
  r=net.rates(x);y[k]=r@net.output+net.bias;drive=(r.reshape(-1,500)@net.W.T).reshape(n,2,2,500)
  if k<200:
   drive[:,0,:,:400]+=rate[k]@B.T-r[:,0,:,400:470]@B.T
   sr=mu[k,:,None]+np.array([1.,-1.])[None,:,None]*cur[k,:,None]/2
   drive[:,1,:,:400]+=sr@B.T-r[:,1,:,400:470]@B.T
  drive+=ctx['u'][:,None,:,1400+k,None]*net.input;x+=net.alpha*(drive-x)
 ep=ex.endpoints(y,ctx);base=ctx['base_context'];ratio=np.sqrt(np.sum((bc-bd)**2,axis=-1))/np.maximum(np.sqrt(energy),1e-10)
 summary=dict(model=net.name,nbase=48,relative_change=float(ratio.mean()),roughness_ratio_max=float(np.max(np.divide(rough,naturalrough,out=np.ones_like(rough),where=naturalrough>1e-15))),changed_fraction=float(np.mean(ratio>1e-8)),energy_error=float(error),ei_error=ei,history_loss=ex.boot(ep['H'][:,0]-ep['H'][:,1],base),sensory_change=ex.boot(ep['sensory'][:,1]-ep['sensory'][:,0],base),current=ex.boot(ep['current'],base))
 np.savez_compressed(OUT/f'temporal_{net.name}.npz',output_trace=y,source_difference=d,source_map=cur,source_mean=mu,base=base,**ep);ex.dump(OUT/f'temporal_{net.name}.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 paths=sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db'))
 ex.dump(ROOT/'temporal_execution.json',dict(protocol_sha256=hashlib.sha256((ROOT/'temporal_protocol.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 with ProcessPoolExecutor(2) as pool:list(pool.map(run,enumerate(paths)))
