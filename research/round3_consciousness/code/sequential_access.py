"""Exploratory original-task assay: old history survives an intervening choice.

Four original discrimination trials -> source intervention in ITI -> strong
fifth discrimination trial -> ambiguous sixth history trial. Both behavioral
operations occur in one uninterrupted trajectory under the same trained output.
"""
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from rnn_core import ROOT,Model,models,initialize
from cognitive_causal import source_target

def contexts(n,seed):
    rng=np.random.default_rng(seed);qs=[0,2,18,20];K=len(qs)
    u=np.zeros((n*K,2,2200));counts=np.zeros((n*K,2,6),int);x0=np.empty((n*K,500))
    oldq=np.r_[np.arange(10),np.arange(11,21)]
    for j in range(n):
        while True:
            c=rng.choice(oldq,4);s=np.sum(c-10)
            if abs(s)>10:break
        if s<0:c=20-c
        for t,q in enumerate(c):
            pp=np.r_[np.ones(q),-np.ones(20-q)];rng.shuffle(pp)
            for z in range(K):
                a=t*400+50;u[j*K+z,0,a:a+60]=np.repeat(pp,3);u[j*K+z,1,a:a+60]=np.repeat(-pp,3)
                counts[j*K+z,:,t]=[q,20-q]
        order=rng.permutation(20);amb=-np.ones(20);amb[rng.permutation(20)[:10]]=1
        init=rng.normal(0,1/500,500)
        for z,q in enumerate(qs):
            pp=-np.ones(20);pp[order[:q]]=1
            u[j*K+z,:,1650:1710]=np.repeat(pp,3);u[j*K+z,:,2050:2110]=np.repeat(amb,3)
            counts[j*K+z,:,4]=q;counts[j*K+z,:,5]=10;x0[j*K+z]=init
    assert np.all(counts[:,0,:5].mean(-1)>10) and np.all(counts[:,1,:5].mean(-1)<10)
    return dict(u=u,x0=x0,counts=counts,start=np.array(1400),end=np.array(2200),base_context=np.repeat(np.arange(n),K),seed=np.array(seed))

def conditions():
    cs=[dict(name='natural',state='intact',source='natural',duration=0)]
    for d in [100,200]:
        for state,ss in [('intact',['own','mean','opposite','pattern','gain','zero']),('PPC_mean',['own','mean','opposite']),('all_mean',['own','mean','opposite'])]:
            for s in ss:cs.append(dict(name=f'{state}_{s}_{d}',state=state,source=s,duration=d))
    return cs

def evaluate(model,ctx,cs):
    n=len(ctx['u']);m=len(cs);pair=initialize(model,ctx);mean=pair.mean(1);inp=ctx['u'];donor=np.empty((200,n,2,400))
    native_y=np.empty((n,2,800));snaps=[100,200,320,399,600,720,799];refs={};xx=pair.copy()
    for k in range(800):
        r=model.rate(xx);native_y[:,:,k]=r@model.Wout+model.bias
        if k<200:donor[k]=r[:,:,400:470]@model.W[:400,400:470].T
        if k in snaps:refs[k]=xx.copy()
        xx=model.step(xx.reshape(n*2,500),inp[:,:,1400+k].reshape(-1)).reshape(n,2,500)
    x=np.repeat(pair[:,None],m,axis=1)
    for j,c in enumerate(cs):
        if c['state']=='PPC_mean':x[:,j,:,:400]=mean[:,None,:400]
        elif c['state']=='all_mean':x[:,j]=mean[:,None]
    out=np.zeros((n,m,2,2));mi=np.full_like(out,np.inf);ma=np.full_like(out,-np.inf)
    comp=np.zeros((n,m,2,2,3));norm=np.zeros((len(snaps),n,m,3));proj=np.zeros_like(norm)
    act=np.zeros((len(snaps),n,m,2,3));sham=0
    mosaic=np.ones(400)
    for k in range(800):
        r=model.rate(x);y=r@model.Wout+model.bias
        for j,c in enumerate(cs):
            if c['state']=='intact' and c['source'] in ['natural','own']:
                sham=max(sham,float(np.max(np.abs(y[:,j]-native_y[:,:,k]))))
        response=0 if 320<=k<400 else (1 if k>=720 else None)
        if response is not None:
            out[:,:,:,response]+=y/80;mi[:,:,:,response]=np.minimum(mi[:,:,:,response],y);ma[:,:,:,response]=np.maximum(ma[:,:,:,response],y)
            for z,sl in enumerate([slice(0,400),slice(400,470),slice(470,500)]):comp[:,:,:,response,z]+=r[:,:,:,sl]@model.Wout[sl]/80
        if k in snaps:
            ix=snaps.index(k);d=x[:,:,0]-x[:,:,1];ref=refs[k][:,0]-refs[k][:,1]
            for z,sl in enumerate([slice(0,400),slice(400,470),slice(470,500)]):
                norm[ix,:,:,z]=np.linalg.norm(d[:,:,sl],axis=-1)
                proj[ix,:,:,z]=np.sum(d[:,:,sl]*ref[:,None,sl],axis=-1)/np.maximum(np.sum(ref[:,sl]**2,axis=-1)[:,None],1e-30)
                act[ix,:,:,:,z]=r[:,:,:,sl].mean(-1)
        cur=(r.reshape(n*m*2,500)@model.W.T).reshape(n,m,2,500)
        if k<200:
            b=donor[k];targets={s:source_target(b,s,mosaic) for s in set(c['source'] for c in cs) if s!='natural'}
            for j,c in enumerate(cs):
                if k<c['duration']:
                    cur[:,j,:,:400]+=targets[c['source']] - r[:,j,:,400:470]@model.W[:400,400:470].T
        cur+=inp[:,:,1400+k][:,None,:,None]*model.Win
        x=(1-model.alpha)*x+model.alpha*cur
    checks=dict(sham_max_abs_output=sham,output_component_max_error=float(np.max(np.abs(comp.sum(-1)+model.bias-out))))
    assert sham<1e-7
    return dict(margin=out,response_min=mi,response_max=ma,components=comp,history_norm=norm,history_projection=proj,activity_time=act,snapshot_steps=np.array(snaps),checks=checks)

def run_one(job):
    p,cp,cs,od=job;m=Model(p);ctx=dict(np.load(cp));t=time.time();res=evaluate(m,ctx,cs);checks=res.pop('checks');np.savez_compressed(Path(od)/(m.name+'.npz'),**res)
    rows=[];hs=np.array([1,-1]);s=np.sign(ctx['counts'][:,:,4]-10)
    for j,c in enumerate(cs):
        y=res['margin'][:,j];hi=res['response_max'][:,j];lo=res['response_min'][:,j]
        current_choice=(y[:,:,0]>0)==(s>0);history_choice=(y[:,:,1]>0)==(hs>0)
        current_native=np.where(s>0,(hi[:,:,0]>=.75)&(lo[:,:,0]>=-.75),(lo[:,:,0]<=-.75)&(hi[:,:,0]<=.75))
        history_native=np.where(hs>0,(hi[:,:,1]>=.75)&(lo[:,:,1]>=-.75),(lo[:,:,1]<=-.75)&(hi[:,:,1]<=.75))
        rows.append(dict(**c,current_sign_accuracy=float(current_choice.mean()),history_sign_accuracy=float(history_choice.mean()),current_author_success=float(current_native.mean()),history_author_success=float(history_native.mean()),
                         signed_history_margin_current=float(np.mean(y[:,:,0]*hs)),signed_history_margin_history=float(np.mean(y[:,:,1]*hs))))
    row=dict(model=m.name,sha256=m.sha256,seconds=time.time()-t,checks=checks,conditions=rows)
    (Path(od)/(m.name+'.json')).write_text(json.dumps(row,indent=2));return row

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=1);a=ap.parse_args()
    od=ROOT/'results'/a.out;od.mkdir(parents=True,exist_ok=True);ctx=contexts(a.n,a.seed);cp=od/'contexts.npz';np.savez_compressed(cp,**ctx);cs=conditions();(od/'conditions.json').write_text(json.dumps(cs,indent=2))
    mani=dict(n_base=a.n,seed=a.seed,n_contexts=len(ctx['u']),status='running',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (od/'run_manifest.json').write_text(json.dumps(mani,indent=2));summary=[];t=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run_one,(str(p),str(cp),cs,str(od))) for p in models()]
        for f in as_completed(jobs):
            x=f.result();summary.append(x);print(json.dumps({'model':x['model'],'seconds':x['seconds'],'natural':x['conditions'][0],'checks':x['checks']}),flush=True)
            (od/'summary.json').write_text(json.dumps(sorted(summary,key=lambda z:z['model']),indent=2))
    mani.update(status='complete',seconds=time.time()-t);(od/'run_manifest.json').write_text(json.dumps(mani,indent=2))
if __name__=='__main__':main()
