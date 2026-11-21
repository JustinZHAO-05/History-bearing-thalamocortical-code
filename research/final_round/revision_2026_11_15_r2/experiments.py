"""Focused source correction and calibrated receiver experiments."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,hashlib,time,argparse,importlib.util
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parent;FINAL=ROOT.parent
spec=importlib.util.spec_from_file_location('nov15_experiments',FINAL/'revision_2026_11_15/experiments.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
v1=previous.v1
OUT=Path(os.environ.get('FOCUSED_RESULTS',str(ROOT/'results')));OUT.mkdir(parents=True,exist_ok=True)
DOSES=np.array([.001,.003,.01,.03,.1,.3,1.])
def dump(p,a):p.write_text(json.dumps(a,indent=2,ensure_ascii=False),encoding='utf-8')
def interval(a,base,seed=99401):
 b=np.stack([a[base==j].mean(0) for j in np.unique(base)]);ix=np.random.default_rng(seed).integers(0,len(b),(10000,len(b)))
 return dict(mean=b.mean(0).tolist(),ci=np.quantile(b[ix].mean(1),[.025,.975],axis=0).T.tolist())

def clipped_ray(cap,q,B,target):
 """First target crossing, otherwise maximal-norm breakpoint; complete path."""
 M=B.T@B;n=len(cap);row=np.arange(n)
 times=np.divide(cap,abs(q),out=np.full_like(cap,np.inf),where=abs(q)>1e-18)
 order=np.argsort(times,axis=1);active=np.broadcast_to(q@M,(n,70)).copy();fixed=np.zeros_like(active)
 C=np.zeros(n);b=np.zeros(n);a=np.full(n,q@M@q);bestE=np.zeros(n);bestK=np.zeros(n)
 lo=np.zeros(n);hi=np.zeros(n);last=np.zeros(n);found=np.zeros(n,bool)
 for j in range(70):
  idx=order[:,j];k=times[row,idx];valid=np.isfinite(k)
  en=np.zeros(n);en[valid]=C[valid]+2*k[valid]*b[valid]+k[valid]**2*a[valid]
  better=valid&(en>bestE);bestE[better]=en[better];bestK[better]=k[better]
  hit=valid&(~found)&(en>=target);lo[hit]=last[hit];hi[hit]=k[hit];found[hit]=True
  v=np.sign(q[idx])*cap[row,idx];u=q[idx];fm=fixed[row,idx].copy();um=active[row,idx].copy();diag=M[idx,idx]
  C+=2*v*fm+v*v*diag;b+=v*um-u*fm-v*u*diag;a+=-2*u*um+u*u*diag
  fixed+=v[:,None]*M[idx];active-=u[:,None]*M[idx];last[valid]=k[valid]
 # Direct coefficients on the first crossing interval avoid accumulated error.
 mid=(lo+hi)/2;U=np.where(mid[:,None]<times,q,0);F=np.clip(mid[:,None]*q,-cap,cap)-mid[:,None]*U
 aa=np.sum((U@B.T)**2,axis=1);bb=np.sum((F@B.T)*(U@B.T),axis=1);cc=np.sum((F@B.T)**2,axis=1)
 for _ in range(60):
  mid=(lo+hi)/2;en=aa*mid*mid+2*bb*mid+cc;high=en>=target
  hi=np.where(found&high,mid,hi);lo=np.where(found&~high,mid,lo)
 scale=np.where(found,(lo+hi)/2,bestK);scale=np.where(target>1e-20,scale,0)
 candidate=np.clip(scale[:,None]*q,-cap,cap)
 return candidate,found,scale

def source(job):
 i,path=job;t=time.time();net=v1.eng.IndependentNetwork(path)
 old=np.load(FINAL/'revision_2026_11_14/results'/(net.name+'.npz'));mu=old['source_mean'];d=old['natural_source_difference'];axes=old['random_axes'];cap=2*np.minimum(mu,20-mu)
 target=np.sum((d@net.B.T)**2,axis=-1).ravel();sign=np.sign(d@axes[0]);sign=np.where(sign==0,1,sign)
 candidates=[];geometry=[];scales=[]
 for j,q in enumerate(axes):
  c,hit,k=clipped_ray(cap.reshape(-1,70),q,net.B,target);c=c.reshape(d.shape)*sign[...,None]
  actual=np.linalg.norm(c@net.B.T,axis=-1);natural=np.sqrt(target).reshape(actual.shape)
  rel=np.divide(abs(actual-natural),natural,out=np.zeros_like(actual),where=natural>1e-12)
  cosine=np.divide(c@axes[0],np.linalg.norm(c,axis=-1),out=np.zeros_like(actual),where=np.linalg.norm(c,axis=-1)>1e-12)
  eigen=np.linalg.eigvalsh(c.reshape(-1,70).T@c.reshape(-1,70));pr=float(eigen.sum()**2/(eigen@eigen))
  geometry.append(dict(axis=j,matching_fraction_1pct=float((rel<=.01+1e-10).mean()),exact_target_fraction=float(hit.mean()),relative_norm_error_mean=float(rel.mean()),relative_norm_error_max=float(rel.max()),matched_norm_error_max=float(rel.ravel()[hit].max()) if hit.any() else 0.,postclip_axis_cosine_abs=float(abs(cosine).mean()),participation_ratio=pr,source_bound_excess=float((abs(c)-cap).max())))
  candidates.append(c);scales.append(k.reshape(d.shape[:-1]))
 candidates=np.array(candidates);initial,rate=v1.donor(net,v1.CTEST);assert np.max(abs(rate.mean(2)-mu))<1e-12
 outputs,checks=v1.sim(net,v1.CTEST,initial,rate,mu,candidates);base=outputs['base_context']
 for j in [0,1,2,7]:assert np.max(abs(outputs['output_trace'][:,:,j]-old['output_trace'][:,:,j]))<1e-11
 metrics={k:interval(outputs[k],base) for k in ('current_success','history_success','history_opposite','history_signed_margin','current_sensory_margin')}
 rec=dict(model=net.name,nbase=12,conditions=list(v1.NAMES),geometry=geometry,metrics=metrics,checks=checks,heldout_axis_fraction=float(np.sum((d@axes[0])**2)/np.sum(d*d)),seconds=time.time()-t)
 np.savez_compressed(OUT/f'axis_{net.name}.npz',**outputs,source_mean=mu,natural_source_difference=d,source_candidate_difference=candidates,random_axes=axes,scale=np.array(scales))
 dump(OUT/f'axis_{net.name}.json',rec);print(json.dumps(dict(stage='axis',model=net.name,seconds=rec['seconds'],match=[x['matching_fraction_1pct'] for x in geometry],history=metrics['history_signed_margin']['mean'])),flush=True)
 return rec

def prefix(net,ctx):
 initial,rate=v1.donor(net,ctx);x=np.repeat(initial.mean(1)[:,None],2,axis=1)
 for k in range(310):
  r=net.rates(x);drive=r@net.W.T
  if k<200:drive[...,:400]+=rate[k]@net.B.T-r[...,400:470]@net.B.T
  drive+=ctx['u'][:,:,1400+k,None]*net.input;x+=net.alpha*(drive-x)
 return x

def receiver_arms(doses,attenuations):
 arms=[dict(name='own',type='own',dose=0.)]
 for dose in doses:
  arms.append(dict(name=f'inward_{dose:g}',type='inward',dose=float(dose)))
  for j in range(8):arms.append(dict(name=f'random{j}_{dose:g}',type='random',direction=j,dose=float(dose)))
 for dose in attenuations:arms.append(dict(name=f'recurrent_{dose:g}',type='recurrent',dose=float(dose)))
 return arms

def run_receiver(net,ctx,x0,arms,random,steps,weighted=False):
 n=len(x0);m=len(arms);x=np.repeat(x0[:,None],m,axis=1)
 common=x0[:,:,:400].mean(1);shift=.5*(np.clip(common,2,18)-common);norm=np.linalg.norm(shift,axis=1)
 for j,a in enumerate(arms):
  if a['type']=='inward':x[:,j,:,:400]+=a['dose']*shift[:,None]
  elif a['type']=='random':x[:,j,:,:400]+=a['dose']*norm[:,None,None]*random[a['direction']][None,None]
 startdx=x[:,:,0]-x[:,:,1];assert np.max(abs(startdx-(x0[:,0]-x0[:,1])[:,None]))<1e-11
 initialdr=net.rates(x)[:,:,0]-net.rates(x)[:,:,1]
 y=np.empty((steps,n,m,2));rn2=np.empty((steps,n,m));align=np.empty_like(rn2)
 weights=np.zeros((3,3,m,6,3));qs=np.arange(n)%3;normw=np.linalg.norm(net.output)
 lower=np.r_[np.zeros(470),np.full(30,9.)];upper=np.r_[np.full(470,20.),np.full(30,11.)];slope=np.r_[np.ones(470),np.full(30,10.)]
 for k in range(steps):
  r=net.rates(x);y[k]=r@net.output+net.bias;dr=r[:,:,0]-r[:,:,1];delta=y[k,:,:,0]-y[k,:,:,1]
  rn2[k]=np.sum(dr*dr,axis=-1);align[k]=np.divide(abs(delta),normw*np.sqrt(rn2[k]),out=np.zeros_like(delta),where=rn2[k]>1e-24)
  if weighted and 10<=k<90:
   mn=np.minimum(x[:,:,0],x[:,:,1]);mx=np.maximum(x[:,:,0],x[:,:,1]);pot=abs(x[:,:,0]-x[:,:,1])*slope*abs(net.output);actual=dr*net.output
   masks=[mx<=lower,(mn>lower)&(mx<upper),mn>=upper,(mn<=lower)&(mx>lower)&(mx<upper),(mn>lower)&(mn<upper)&(mx>=upper),(mn<=lower)&(mx>=upper)]
   assert np.all(np.sum(np.array(masks),axis=0)==1)
   for ci,mask in enumerate(masks):
    for qi in range(3):
     for ri,sl in enumerate((slice(0,400),slice(400,470),slice(470,500))):
      keep=mask[qs==qi,:,sl]
      weights[qi,ri,:,ci,0]+=np.sum(pot[qs==qi,:,sl]*keep,axis=(0,2))
      weights[qi,ri,:,ci,1]+=np.sum(abs(actual[qs==qi,:,sl])*keep,axis=(0,2))
      weights[qi,ri,:,ci,2]+=np.sum(actual[qs==qi,:,sl]*keep,axis=(0,2))
  drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
  if k<20:
   for j,a in enumerate(arms):
    if a['type']=='recurrent':drive[:,j]-=a['dose']*(drive[:,j]-drive[:,j].mean(1,keepdims=True))
  drive+=ctx['u'][:,None,:,1710+k,None]*net.input;x+=net.alpha*(drive-x)
 return dict(output_trace=y,rate_norm_squared=rn2,alignment=align,initial_state_difference=startdx,initial_rate_difference=initialdr,unit_shift_norm=norm,activation_weights=weights)

def current_endpoints(y,ctx):
 cur=y[10:90];mn=cur.min(0);mx=cur.max(0);target=np.sign(ctx['counts'][:,:,4]-10)[:,None,:]
 success=np.where(target>0,(mx>=.75)&(mn>=-.75),(mn<=-.75)&(mx<=.75)).mean(-1)
 return dict(sensory=np.mean(cur.mean(0)*target,axis=-1),current=success,current_H=(cur[:,:,:,0]-cur[:,:,:,1]).mean(0)/2)

def select_dose(ep,arms,ctx,family):
 base=ctx['base_context'];selected=0.;details=[]
 for dose in DOSES:
  ids=[j for j,a in enumerate(arms) if a['dose']==dose and ((a['type'] in ('inward','random')) if family=='displacement' else a['type']=='recurrent')]
  passed=True;checks=[]
  for qi in (0,2):
   keep=np.arange(len(base))%3==qi;change=ep['sensory'][keep][:,ids]-ep['sensory'][keep,0,None];iv=interval(change,base[keep]);ci=np.array(iv['ci']);loss=(ep['current'][keep,0,None]-ep['current'][keep][:,ids]).mean(0)
   good=(ci[:,0]>-.02)&(ci[:,1]<.02)&(loss<=.01+1e-12);passed=passed and bool(good.all());checks.append(dict(q=(2,10,18)[qi],sensory=iv,commitment_loss=loss.tolist(),passed=good.tolist()))
  if passed:selected=float(dose)
  details.append(dict(dose=float(dose),passed=passed,arms=ids,checks=checks))
 return selected,details

def receiver(job):
 i,path=job;t=time.time();net=v1.eng.IndependentNetwork(path);rng=np.random.default_rng(99300+i);random=rng.normal(size=(8,400));random/=np.linalg.norm(random,axis=1)[:,None]
 cal=v1.eng.contexts(12,99101,(2,10,18));allarms=receiver_arms(DOSES,DOSES);caldata=run_receiver(net,cal,prefix(net,cal),allarms,random,90);ep=current_endpoints(caldata['output_trace'],cal)
 dose,detail=select_dose(ep,allarms,cal,'displacement');atten,adetail=select_dose(ep,allarms,cal,'recurrent')
 np.savez_compressed(OUT/f'calibration_{net.name}.npz',**caldata,**ep,base=cal['base_context'],random_directions=random)
 test=v1.eng.contexts(48,99102,(2,10,18));arms=receiver_arms([dose],[atten]);data=run_receiver(net,test,prefix(net,test),arms,random,490,True);ep=current_endpoints(data['output_trace'],test);y=data['output_trace'];H=(y[410:490,:,:,0]-y[410:490,:,:,1]).mean(0)/2;ep['H']=H
 base=test['base_context'];responses={}
 for qi,q in enumerate((2,10,18)):
  keep=np.arange(len(base))%3==qi;b=base[keep];rr={k:interval(v[keep],b) for k,v in ep.items()}
  rr['sensory_change']=interval(ep['sensory'][keep]-ep['sensory'][keep,0,None],b);rr['later_change']=interval(H[keep]-H[keep,0,None],b);rr['current_H_change']=interval(ep['current_H'][keep]-ep['current_H'][keep,0,None],b)
  for name,v in [('rate_rms',np.sqrt(data['rate_norm_squared'][10:90].mean(0))),('alignment_rms',np.sqrt(((y[10:90,:,:,0]-y[10:90,:,:,1])**2).mean(0))/np.maximum(normw:=np.linalg.norm(net.output)*np.sqrt(data['rate_norm_squared'][10:90].mean(0)),1e-20)),('temporal_consistency',abs(ep['current_H']*2)/np.maximum(np.sqrt(((y[10:90,:,:,0]-y[10:90,:,:,1])**2).mean(0)),1e-20))]:rr[name]=interval(v[keep],b)
  random_mean=H[keep,2:10].mean(1);rr['inward_minus_random_H']=interval(H[keep,1]-random_mean,b);rr['random_mean_H_change']=interval(random_mean-H[keep,0],b)
  responses[str(q)]=rr
 rec=dict(model=net.name,n_calibration=12,n_test=48,selected_displacement=dose,selected_attenuation=atten,calibration_arms=allarms,calibration_displacement=detail,calibration_attenuation=adetail,arms=arms,responses=responses,seconds=time.time()-t,mean_actual_shift=float(dose*data['unit_shift_norm'].mean()),activation_categories=['below','linear','upper','lower_cross','upper_cross','both_cross'],activation_weight_names=['potential_readout','absolute_readout','signed_readout'])
 np.savez_compressed(OUT/f'lowdose_{net.name}.npz',**data,**ep,base=base,counts=test['counts'],random_directions=random)
 dump(OUT/f'lowdose_{net.name}.json',rec)
 print(json.dumps(dict(stage='receiver',model=net.name,seconds=rec['seconds'],dose=dose,attenuation=atten,ambiguous_later_change=responses['10']['later_change'],ambiguous_inward_minus_random=responses['10']['inward_minus_random_H'])),flush=True);return rec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['axis','receiver']);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--models',type=int,default=9);a=ap.parse_args()
 paths=sorted((v1.R3/'data/hosford_history_rnn').glob('*/*.db'))[:a.models]
 dump(OUT/f'{a.stage}_design.json',dict(protocol_sha256=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models={str(p.relative_to(v1.R3)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},stage=a.stage))
 fn=source if a.stage=='axis' else receiver
 jobs=[(i,p) for i,p in enumerate(paths) if not (OUT/f'{"axis" if a.stage=="axis" else "lowdose"}_{v1.eng.IndependentNetwork(p).name}.json').exists()]
 if a.workers==1:
  for job in jobs:fn(job)
 else:
  with ProcessPoolExecutor(a.workers) as pool:list(pool.map(fn,jobs))
if __name__=='__main__':main()
