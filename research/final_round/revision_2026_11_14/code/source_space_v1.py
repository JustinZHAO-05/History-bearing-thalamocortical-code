#!/usr/bin/env python3
"""Source-constrained PUL->PPC replay under learned original connectivity.

All histories/weights frozen; do not select random axes on behavioral performance.
"""
import os
for s in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[s]='1'
from pathlib import Path
import hashlib,time,json,sys,argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
R3=Path(os.environ.get('R3_RESEARCH_ROOT',str(ROOT.parents[1]/'round3_consciousness')))
sys.path.insert(0,str(R3/'reviews/independent_audit'))
import rank_one_control as eng
eng.DATA=R3/'data/hosford_history_rnn'

NBASE=12
CTRAIN=eng.contexts(NBASE,94001,(2,))
CTEST=eng.contexts(NBASE,94002,(2,18))
NAMES=('natural','allmean_own','allmean_mean','allmean_learned','allmean_random0','allmean_random1','allmean_random2','allmean_opposite')
WINDOWS=(slice(320,400),slice(720,800))

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def donor(net,ctx):
    x=np.repeat(ctx['x0'][:,None],2,axis=1)
    for t in range(1400):x=net.step(x,ctx['u'][:,:,t])
    initial=x.copy();rates=np.empty((200,len(x),2,70))
    for k in range(200):
        rates[k]=net.rates(x)[:,:,400:470]
        x=net.step(x,ctx['u'][:,:,1400+k])
    return initial,rates

def src_axis(r):
    d=r[:,:,0]-r[:,:,1]
    arr=d.reshape(-1,70)
    e,V=np.linalg.eigh(arr.T@arr)
    v=V[:,-1]
    if np.mean(arr@v)<0:v=-v
    return v,float(e[-1]/max(e.sum(),1e-20))

def prep_candidates(rate,B,v,seed):
    mu=rate.mean(axis=2); d=rate[:,:,0]-rate[:,:,1]
    cap=2*np.minimum(mu,20-mu)
    nrm=np.linalg.norm((d.reshape(-1,70)@B.T).reshape(*d.shape[:-1],400),axis=-1)
    a=d@v
    rng=np.random.default_rng(seed)
    axes=[v]
    for _ in range(3):
        q=rng.normal(size=70);q-=q@v*v;q/=np.linalg.norm(q)
        axes.append(q)
    candidates=[];checks=[]
    # Independent grid search over amplification of signed axis coefficient.
    # Clipping can make the target norm nonmonotone; search is over the grid
    # and then local bounded refinement, never tuned to any decision outcome.
    for j,q in enumerate(axes):
        signed=a[...,None]*q
        best=np.zeros_like(signed);besterr=np.abs(nrm)
        matchmask=np.zeros_like(nrm,dtype=bool)
        bestk=np.zeros_like(nrm)
        # Also include strong scaling where source-constrained feasibility becomes saturated.
        grid=np.r_[0.,np.exp(np.linspace(-10,19,65))]
        for k in grid:
            c=np.clip(k*signed,-cap,cap)
            y=(c.reshape(-1,70)@B.T).reshape(*c.shape[:-1],400)
            nn=np.linalg.norm(y,axis=-1)
            diff=np.abs(nn-nrm)
            better=diff<besterr
            best=np.where(better[...,None],c,best)
            bestk=np.where(better,k,bestk)
            besterr=np.minimum(besterr,diff)
        # Fine 1D local minimization around best grid seed; preserves feasibility.
        # Search log-factor grid with smaller steps around recorded best scale.
        for factor in (np.exp(np.linspace(-.46,.46,20)),):
            for f in factor:
                k=bestk*f
                c=np.clip(k[...,None]*signed,-cap,cap)
                y=(c.reshape(-1,70)@B.T).reshape(*c.shape[:-1],400)
                nn=np.linalg.norm(y,axis=-1)
                diff=np.abs(nn-nrm)
                better=diff<besterr
                best=np.where(better[...,None],c,best)
                bestk=np.where(better,k,bestk)
                besterr=np.minimum(besterr,diff)
        after=(best.reshape(-1,70)@B.T).reshape(*best.shape[:-1],400)
        actual=np.linalg.norm(after,axis=-1)
        ratio=np.divide(actual,nrm,out=np.ones_like(actual),where=nrm>1e-12)
        err=np.divide(abs(actual-nrm),nrm,out=np.zeros_like(actual),where=nrm>1e-12)
        candidates.append(best)
        checks.append(dict(axis=j,matching_fraction_1pct=float(np.mean(err<=.01)),matching_fraction_5pct=float(np.mean(err<=.05)),
            relative_energy_error_median=float(np.median(err)),relative_energy_error_mean=float(np.mean(err)),
            mean_actual_over_target_norm=float(np.mean(ratio)),
            source_difference_rms=float(np.sqrt(np.mean(best*best))),
            weighted_difference_rms=float(np.sqrt(np.mean(after*after))),
            percentage_on_source_boundary=float(np.mean((abs(best)>=cap-1e-9)&(cap>1e-8))),
            source_direction_dot_learned=float(np.dot(q,v))))
    return mu,d,np.array(candidates),np.array(axes),checks

def sim(net,ctx,initial,rate,mu,cands):
    n=len(initial); m=len(NAMES)
    x=np.repeat(initial[:,None],m,axis=1)
    for j in range(1,m):x[:,j]=initial.mean(axis=1,keepdims=True)
    pair_sign=np.array([1.,-1.])[None,:,None]
    y=np.empty((800,n,m,2),dtype=np.float64)
    histnorm=np.empty((800,n,m),dtype=np.float32)
    maxsham=0;zerocontrol=0;minsource=np.inf;maxsource=-np.inf
    for k in range(800):
        rates=net.rates(x)
        y[k]=rates@net.output+net.bias
        histnorm[k]=np.linalg.norm(rates[:,:,0]-rates[:,:,1],axis=-1)
        drive=(rates.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j in range(1,m):
                if j==1:sr=rate[k]
                elif j==2:sr=np.repeat(mu[k,:,None],2,axis=1)
                elif j==7:sr=rate[k,:,::-1]
                else:sr=mu[k,:,None]+0.5*pair_sign*cands[j-3,k,:,None]
                minsource=min(minsource,float(np.min(sr)));maxsource=max(maxsource,float(np.max(sr)))
                b=sr@net.B.T
                old=rates[:,j,:,400:470]@net.B.T
                drive[:,j,:,:400]+=b-old
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    # native output checks and source-pair symmetry
    assert minsource>=-1e-9 and maxsource<=20+1e-9,(minsource,maxsource)
    assert np.max(np.abs(y[:,:,2,0]-y[:,:,2,1]))<1e-10
    margin=np.stack([y[w].mean(axis=0) for w in WINDOWS],axis=-1)
    mn=np.stack([y[w].min(axis=0) for w in WINDOWS],axis=-1)
    mx=np.stack([y[w].max(axis=0) for w in WINDOWS],axis=-1)
    target=np.sign(ctx['counts'][:,:,4]-10)[:,None,:]
    current=np.where(target>0,(mx[...,0]>=.75)&(mn[...,0]>=-.75),(mn[...,0]<=-.75)&(mx[...,0]<=.75))
    history=np.array([1,-1])[None,None,:]
    correct=np.where(history>0,(mx[...,1]>=.75)&(mn[...,1]>=-.75),(mn[...,1]<=-.75)&(mx[...,1]<=.75))
    opposite=np.where(history<0,(mx[...,1]>=.75)&(mn[...,1]>=-.75),(mn[...,1]<=-.75)&(mx[...,1]<=.75))
    signed=np.mean(margin[...,1]*history,axis=-1)
    sensory=np.mean(margin[...,0]*target,axis=-1)
    out=dict(output_trace=y,history_norm_trace=histnorm,margin=margin,response_min=mn,response_max=mx,
       current_success=current.astype(np.float64).mean(axis=-1),history_success=correct.astype(np.float64).mean(axis=-1),
       history_opposite=opposite.astype(np.float64).mean(axis=-1),history_signed_margin=signed,current_sensory_margin=sensory,
       base_context=ctx['base_context'],counts=ctx['counts'])
    outchecks=dict(min_source=minsource,max_source=maxsource,mean_source_pair_output_max=float(np.max(np.abs(y[:,:,2,0]-y[:,:,2,1]))),
                   own_natural_margin_max_diff=float(np.max(np.abs(margin[:,0]-margin[:,1]))))
    return out,outchecks

def permodel(job):
    index,p=job;t=time.time();net=eng.IndependentNetwork(p)
    _,trainrate=donor(net,CTRAIN)
    axis,frac=src_axis(trainrate)
    initial,test_rates=donor(net,CTEST)
    mu,d,cands,axes,geometry=prep_candidates(test_rates,net.B,axis,94010+index*20)
    outputs,checks=sim(net,CTEST,initial,test_rates,mu,cands)
    base=CTEST['base_context'];nbase=NBASE
    metrics={};rng=np.random.default_rng(94311+index);ix=rng.integers(0,nbase,(8000,nbase))
    for name in ['current_success','history_success','history_opposite','history_signed_margin','current_sensory_margin']:
        ar=outputs[name];g=np.stack([ar[base==b].mean(axis=0) for b in range(nbase)])
        boot=g[ix].mean(axis=1)
        metrics[name]=dict(means=dict(zip(NAMES,g.mean(axis=0).tolist())),ci95=dict(zip(NAMES,np.quantile(boot,[.025,.975],axis=0).T.tolist())))
    signed=outputs['history_signed_margin'];g=np.stack([signed[base==b].mean(axis=0) for b in range(nbase)])
    advantage=g[:,3]-np.mean(np.abs(g[:,4:7]),axis=1)
    advantage_ci=np.quantile(advantage[ix].mean(axis=1),[.025,.975]).tolist()
    np.savez_compressed(ROOT/'results'/f'{net.name}.npz',**outputs,axis=axis,random_axes=axes,source_mean=mu,
                        natural_source_difference=d,source_candidate_difference=cands)
    result=dict(model=net.name,model_hash=sha(p),seconds=time.time()-t,training_source_first_axis_energy=frac,
                geometry=geometry,checks=checks,metrics=metrics,learned_minus_random_abs=dict(mean=float(advantage.mean()),ci95=advantage_ci),
                n_base=NBASE,conditions=NAMES)
    (ROOT/'results'/f'{net.name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(model=net.name,seconds=round(result['seconds'],2),hist={k:round(v,4) for k,v in metrics['history_signed_margin']['means'].items()},
        current={k:round(v,3) for k,v in metrics['current_success']['means'].items()},match=[round(c['matching_fraction_1pct'],3) for c in geometry])),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--models',type=int,default=9);args=ap.parse_args()
    ROOT.joinpath('results').mkdir(parents=True,exist_ok=True)
    files=sorted((R3/'data/hosford_history_rnn').glob('*/*.db'))[:args.models]
    freeze=dict(protocol='source_space_v1',n_base=NBASE,n_models=len(files),train_seed=94001,test_seed=94002,
                q_current=[2,18],axis_random_seed_base=94010,conditions=NAMES,
                protocol_sha256=sha(ROOT/'protocols/source_space_v1.freeze.md'),code_sha256=sha(__file__),
                model_shas={p.name+'@'+p.parent.name:sha(p) for p in files},status='frozen_before_source_behavioral_evaluation')
    (ROOT/'results/protocol.freeze.json').write_text(json.dumps(freeze,indent=2))
    np.savez_compressed(ROOT/'results/context_test.npz',**CTEST)
    np.savez_compressed(ROOT/'results/context_train.npz',**CTRAIN)
    results=[]
    if args.workers==1:
        for i,p in enumerate(files):results.append(permodel((i,p)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fut=[pool.submit(permodel,(i,p)) for i,p in enumerate(files)]
            for f in as_completed(fut):
                results.append(f.result())
                (ROOT/'results/summary.json').write_text(json.dumps(sorted(results,key=lambda x:x['model']),indent=2))
    (ROOT/'results/summary.json').write_text(json.dumps(sorted(results,key=lambda x:x['model']),indent=2))
    (ROOT/'results/completion.json').write_text(json.dumps(dict(status='complete',models=len(results),code_sha256=sha(__file__),nbase=NBASE),indent=2))
if __name__=='__main__':main()
