#!/usr/bin/env python3
"""Exactly match PUL-PPC output-difference norm + native E/I aggregate under source-rate feasibility.

Prepared as declared amendment after detecting naive random-axis energy mismatch.
"""
import os
for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[s]='1'
from pathlib import Path
import json,pickle,sqlite3,time,hashlib,sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));import source_space_v1 as v1
NAMES=('natural','allmean_own','allmean_mean','allmean_map0','allmean_map1','allmean_opposite')

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def native_classes(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    row=dict(db.execute('select * from currentRNNParameters order by id desc limit 1').fetchone());db.close()
    arrays={k:pickle.loads(v) for k,v in row.items() if isinstance(v,bytes)}
    e=np.asarray(arrays['PPCexh_idx'],dtype=int);i=np.asarray(arrays['PPCinh_idx'],dtype=int)
    assert len(e)==280 and len(i)==120 and len(np.unique(np.r_[e,i]))==400
    return e,i

def geometry(rate,B,A,seed,attempts=32):
    mu=rate.mean(axis=2);d=rate[:,:,0]-rate[:,:,1]
    cap=2*np.minimum(mu,20-mu)
    slack=np.maximum(0,cap-np.abs(d))
    # Source values on the native extreme vertices are fixed under this
    # exact fixed-background and fixed-energy control.
    D=slack.reshape(-1,70)
    S=d.reshape(-1,70);C=cap.reshape(-1,70)
    # Metric in source and target spaces.
    M=B.T@B
    BD=S@B.T;norm=np.linalg.norm(BD,axis=-1)
    # 2x2 exact weighted projection to remove E/I totals.
    G00=(D*(A[0]**2)).sum(axis=1)
    G01=(D*(A[0]*A[1])).sum(axis=1)
    G11=(D*(A[1]**2)).sum(axis=1)
    det=G00*G11-G01*G01
    valid=det>1e-13*np.maximum(1.,G00*G11)
    best=S.copy();score=np.zeros(len(S));kept=np.zeros(len(S),bool)
    feasible_candidates=np.zeros(len(S),dtype=int)
    rng=np.random.default_rng(seed)
    for j in range(attempts):
        v=rng.normal(size=S.shape)
        init=D*v
        rhs0=init@A[0];rhs1=init@A[1]
        u0=np.divide(G11*rhs0-G01*rhs1,det,out=np.zeros_like(rhs0),where=valid)
        u1=np.divide(G00*rhs1-G01*rhs0,det,out=np.zeros_like(rhs0),where=valid)
        z=D*(v-u0[:,None]*A[0]-u1[:,None]*A[1])
        Bz=z@B.T
        zz=np.sum(Bz*Bz,axis=1)
        dz=np.sum(Bz*BD,axis=1)
        alpha=np.divide(-2*dz,zz,out=np.zeros_like(dz),where=zz>1e-20)
        delta=alpha[:,None]*z
        dprime=S+delta
        feas=(np.max(np.abs(dprime)-C,axis=1)<1e-9)&(norm>1e-10)&valid
        rel=np.divide(np.linalg.norm((delta@B.T),axis=1),norm,out=np.zeros_like(norm),where=norm>1e-10)
        feasible_candidates+=feas.astype(int)
        good=feas & (rel>score)
        best[good]=dprime[good];score[good]=rel[good];kept[good]=True
    shape=d.shape
    out=best.reshape(shape)
    actual=out.reshape(-1,70)@B.T
    nrm0=np.sum(BD*BD,axis=1);nrm1=np.sum(actual*actual,axis=1)
    relerr=np.divide(abs(nrm1-nrm0),nrm0,out=np.zeros_like(nrm0),where=nrm0>1e-18)
    eierr=float(np.max(np.abs((out-d).reshape(-1,70)@A.T)))
    result=dict(feasible_fraction=float(np.mean(kept)),mean_feasible_candidates=float(feasible_candidates.mean()),
                mean_receiver_relative_change=float(score.mean()),median_receiver_relative_change=float(np.median(score)),
                q90_receiver_relative_change=float(np.quantile(score,.9)),
                source_coordinate_drift_rms=float(np.sqrt(np.mean((out-d)**2))),
                pair_mean_maxerr=0.,max_source_rate=float(np.max(mu+out/2)),min_source_rate=float(np.min(mu-out/2)),
                energy_relative_error_max=float(np.max(relerr)),aggregate_ei_maxerr=eierr,
                component_maxviolation=float(np.max(abs(out)-cap)))
    assert relerr.max()<1e-9 and eierr<1e-8 and result['component_maxviolation']<1e-8,result
    return out,result,score.reshape(d.shape[:-1])

def simulate(net,ctx,initial,rate,mu,maps):
    n=len(initial);m=len(NAMES);x=np.repeat(initial[:,None],m,axis=1)
    for j in range(1,m):x[:,j]=initial.mean(axis=1,keepdims=True)
    sign=np.array([1.,-1.])[None,:,None]
    y=np.empty((800,n,m,2));norm=np.empty((800,n,m),dtype=np.float32)
    for k in range(800):
        r=net.rates(x);y[k]=r@net.output+net.bias
        norm[k]=np.linalg.norm(r[:,:,0]-r[:,:,1],axis=-1)
        dr=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j in range(1,m):
                if j==1:src=rate[k]
                elif j==2:src=np.repeat(mu[k,:,None],2,axis=1)
                elif j==5:src=rate[k,:,::-1]
                else:src=mu[k,:,None]+.5*sign*maps[j-3,k,:,None]
                dr[:,j,:,:400]+=src@net.B.T-r[:,j,:,400:470]@net.B.T
        dr+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(dr-x)
    margin=np.stack([y[w].mean(axis=0) for w in v1.WINDOWS],axis=-1)
    ymin=np.stack([y[w].min(axis=0) for w in v1.WINDOWS],axis=-1)
    ymax=np.stack([y[w].max(axis=0) for w in v1.WINDOWS],axis=-1)
    tar=np.sign(ctx['counts'][:,:,4]-10)[:,None,:]
    cs=np.where(tar>0,(ymax[...,0]>=.75)&(ymin[...,0]>=-.75),(ymin[...,0]<=-.75)&(ymax[...,0]<=.75))
    sign=np.array([1.,-1.])[None,None,:]
    hc=np.where(sign>0,(ymax[...,1]>=.75)&(ymin[...,1]>=-.75),(ymin[...,1]<=-.75)&(ymax[...,1]<=.75))
    h=np.mean(margin[...,1]*sign,axis=-1)
    assert np.max(np.abs(y[:,:,2,0]-y[:,:,2,1]))<1e-10
    return dict(output_trace=y,rate_difference_norm=norm,margin=margin,min_out=ymin,max_out=ymax,
       current_success=cs.mean(axis=-1),history_success=hc.mean(axis=-1),history_signed=h,
       current_signed_sensory=np.mean(margin[...,0]*tar,axis=-1),base=ctx['base_context'])

def run_model(task):
    index,p=task;start=time.time();net=v1.eng.IndependentNetwork(p)
    ctx=v1.CTEST
    init,rate=v1.donor(net,ctx)
    E,I=native_classes(p)
    A=np.stack([net.B[E].sum(axis=0),net.B[I].sum(axis=0)])
    mu=rate.mean(axis=2)
    maps=[];checks=[];scores=[]
    for k in range(2):
        mapped,qc,score=geometry(rate,net.B,A,95100+index*20+k)
        maps.append(mapped);checks.append(qc);scores.append(score)
    val=simulate(net,ctx,init,rate,mu,np.array(maps))
    base=ctx['base_context'];rng=np.random.default_rng(95500+index)
    ix=rng.integers(0,12,size=(6000,12))
    outcomes={}
    for name in ['current_success','history_success','history_signed','current_signed_sensory']:
        a=val[name];b=np.stack([a[base==i].mean(axis=0) for i in range(12)])
        outcomes[name]=dict(mean=dict(zip(NAMES,b.mean(axis=0).tolist())),
              ci95=dict(zip(NAMES,np.quantile(b[ix].mean(axis=1),[.025,.975],axis=0).T.tolist())))
    summary=dict(model=net.name,model_sha=sha(p),nbase=12,conditions=NAMES,
          geometry=checks,outcomes=outcomes,time_seconds=time.time()-start)
    np.savez_compressed(ROOT/'results/reflection_v2'/f'{net.name}.npz',**val,maps=np.array(maps),geometry_score=np.array(scores))
    (ROOT/'results/reflection_v2'/f'{net.name}.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(dict(model=net.name,time=round(summary['time_seconds'],1),geo=[(round(x['feasible_fraction'],3),round(x['mean_receiver_relative_change'],3)) for x in checks],
      H={n:round(v,3) for n,v in outcomes['history_signed']['mean'].items()},
      current={n:round(v,3) for n,v in outcomes['current_success']['mean'].items()})),flush=True)
    return summary

def main():
    ROOT.joinpath('results/reflection_v2').mkdir(parents=True,exist_ok=True)
    paths=sorted((v1.R3/'data/hosford_history_rnn').glob('*/*.db'))
    freeze=dict(protocol='source_reflection_v2',status='frozen_before_behavioral_evaluation',
      source_protocol_sha=sha(ROOT/'protocols/source_reflection_v2.freeze.md'),source_code_sha=sha(__file__),
      models=[p.name+'@'+p.parent.name for p in paths],nbase=12,seeds=[95100+i*20 for i in range(len(paths))])
    (ROOT/'results/reflection_v2/protocol.freeze.json').write_text(json.dumps(freeze,indent=2))
    out=[]
    for i,p in enumerate(paths):
        out.append(run_model((i,p)))
        (ROOT/'results/reflection_v2/summary.json').write_text(json.dumps(out,indent=2))
    (ROOT/'results/reflection_v2/completion.json').write_text(json.dumps(dict(status='complete',models=9),indent=2))
if __name__=='__main__':main()
