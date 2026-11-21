"""Fixed-weight causal assays of source content, gain and retained state.

Each run saves all trial/context outcomes, including unsuccessful checkpoints.
No fitted decoder, output retraining, or behavior-based context selection.
"""
from pathlib import Path
import argparse,json,time,hashlib,platform,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from rnn_core import Model,models,paired_histories,initialize,ROOT

def make_conditions(task,compact=False):
    rows=[dict(name='natural',state='intact',source='natural',duration=0)]
    durations=[100,200] if task=='current' or compact else [20,50,100,200]
    sources=['own','mean','opposite','pattern','gain','pattern_l2','gain_l2','mosaic','zero']
    for d in durations:
        for s in sources:
            rows.append(dict(name=f'intact_{s}_{d}',state='intact',source=s,duration=d))
    if task=='history':
        for state in ['PPC_mean','all_mean','outside_mean']:
            for d in ([100] if compact else [100,200]):
                for s in ['own','mean','opposite','pattern','gain']:
                    rows.append(dict(name=f'{state}_{s}_{d}',state=state,source=s,duration=d))
    return rows

def make_context(n,seed,task):
    if task=='history':return paired_histories(n,seed,task='history')
    # Common histories and pip ordering across eight lawful, nonambiguous
    # discrimination mixtures at original trial position five.
    base=paired_histories(n,seed,task='current',probe_counts=(10,))
    qs=np.array([2,6,8,9,11,12,14,18]);k=len(qs)
    ctx={key:(np.repeat(value,k,axis=0) if np.ndim(value)>0 else value.copy()) for key,value in base.items()}
    ctx['base_context']=np.repeat(np.arange(n),k);ctx['probe_index']=np.tile(np.arange(k),n)
    rng=np.random.default_rng(seed+9090)
    for j in range(n):
        order=rng.permutation(20)
        for i,q in enumerate(qs):
            p=-np.ones(20);p[order[:q]]=1
            ctx['u'][j*k+i,:,1650:1710]=np.repeat(p,3)
            ctx['counts'][j*k+i,:,-1]=q
    return ctx

def source_target(b,kind,mosaic):
    other=b[:,::-1];mu=b.mean(1,keepdims=True)
    if kind=='own':return b
    if kind=='opposite':return other
    if kind=='mean':return np.broadcast_to(mu,b.shape)
    if kind=='zero':return np.zeros_like(b)
    if kind=='mosaic':return mu+(b-mu)*mosaic
    if kind in ['pattern','gain']:
        g=b.mean(-1,keepdims=True);go=other.mean(-1,keepdims=True)
    elif kind in ['pattern_l2','gain_l2']:
        g=np.linalg.norm(b,axis=-1,keepdims=True);go=np.linalg.norm(other,axis=-1,keepdims=True)
    else:raise ValueError(kind)
    if kind.startswith('pattern'):
        return other*np.divide(g,go,out=np.zeros_like(g),where=go>1e-30)
    return b*np.divide(go,g,out=np.zeros_like(g),where=g>1e-30)

def evaluate(model,ctx,conds):
    n=len(ctx['u']);m=len(conds);start=int(ctx['start']);end=int(ctx['end'])
    u=ctx['u'];pair=initialize(model,ctx);mu0=pair.mean(1)
    donor=np.empty((200,n,2,400));native_y=np.empty((n,2,400));refs={}
    snapsteps=[20,50,100,200,250,320,399]
    xx=pair.copy()
    for k in range(400):
        r=model.rate(xx);native_y[:,:,k]=r@model.Wout+model.bias
        if k<200:donor[k]=r[:,:,400:470]@model.W[:400,400:470].T
        if k in snapsteps:refs[k]=xx.copy()
        xx=model.step(xx.reshape(n*2,500),u[:,:,start+k].reshape(-1)).reshape(n,2,500)
    x=np.repeat(pair[:,None],m,axis=1)
    for j,c in enumerate(conds):
        if c['state']=='PPC_mean':x[:,j,:,:400]=mu0[:,None,:400]
        elif c['state']=='outside_mean':x[:,j,:,400:]=mu0[:,None,400:]
        elif c['state']=='all_mean':x[:,j]=mu0[:,None]
    rng=np.random.default_rng(314159);mosaic=np.r_[np.ones(200),-np.ones(200)];rng.shuffle(mosaic)
    out=np.zeros((n,m,2));vmax=np.full_like(out,-np.inf);vmin=np.full_like(out,np.inf)
    comp=np.zeros((n,m,2,3));activity=np.zeros_like(comp)
    projections=np.zeros((len(snapsteps),n,m,3));pair_norm=np.zeros_like(projections)
    activity_time=np.zeros((len(snapsteps),n,m,2,3))
    sham=0.;target_min=np.inf;meanmatch=0.;normmatch=0.;mosaic_normerr=0.
    for k in range(400):
        r=model.rate(x);y=r@model.Wout+model.bias
        for j,c in enumerate(conds):
            if c['state']=='intact' and c['source'] in ['natural','own']:
                sham=max(sham,float(np.max(np.abs(y[:,j]-native_y[:,:,k]))))
        if k>=320:
            out+=y/80;vmax=np.maximum(vmax,y);vmin=np.minimum(vmin,y)
            for z,sl in enumerate([slice(0,400),slice(400,470),slice(470,500)]):
                comp[:,:,:,z]+=(r[:,:,:,sl]@model.Wout[sl])/80
                activity[:,:,:,z]+=r[:,:,:,sl].mean(-1)/80
        if k in snapsteps:
            ix=snapsteps.index(k);d=x[:,:,0]-x[:,:,1];dr=refs[k][:,0]-refs[k][:,1]
            for z,sl in enumerate([slice(0,400),slice(400,470),slice(470,500)]):
                den=np.sum(dr[:,sl]**2,axis=-1)
                projections[ix,:,:,z]=np.sum(d[:,:,sl]*dr[:,None,sl],axis=-1)/np.maximum(den[:,None],1e-30)
                pair_norm[ix,:,:,z]=np.linalg.norm(d[:,:,sl],axis=-1)
                activity_time[ix,:,:,:,z]=r[:,:,:,sl].mean(-1)
        current=(r.reshape(n*m*2,500)@model.W.T).reshape(n,m,2,500)
        if k<200:
            b=donor[k];targets={s:source_target(b,s,mosaic) for s in set(c['source'] for c in conds) if s!='natural'}
            target_min=min(target_min,min(float(t.min()) for t in targets.values()))
            if 'pattern' in targets:meanmatch=max(meanmatch,float(np.max(np.abs(targets['pattern'].mean(-1)-b.mean(-1)))))
            if 'pattern_l2' in targets:normmatch=max(normmatch,float(np.max(np.abs(np.linalg.norm(targets['pattern_l2'],axis=-1)-np.linalg.norm(b,axis=-1)))))
            if 'mosaic' in targets:mosaic_normerr=max(mosaic_normerr,float(np.max(np.abs(np.linalg.norm(targets['mosaic'][:,0]-targets['mosaic'][:,1],axis=-1)-np.linalg.norm(b[:,0]-b[:,1],axis=-1)))))
            for j,c in enumerate(conds):
                if k>=c['duration'] or c['source']=='natural':continue
                native=r[:,j,:,400:470]@model.W[:400,400:470].T
                current[:,j,:,:400]+=targets[c['source']]-native
        current+=u[:,:,start+k][:,None,:,None]*model.Win
        x=(1-model.alpha)*x+model.alpha*current
    checks=dict(sham_max_abs_output=sham,pattern_mean_current_max_error=meanmatch,pattern_l2_norm_max_error=normmatch,mosaic_diffnorm_max_error=mosaic_normerr,minimum_replayed_current=target_min,
                output_components_max_error=float(np.max(np.abs(comp.sum(-1)+model.bias-out))),zero_gain_entries=int(np.sum(donor.mean(-1)<=1e-30)))
    assert sham<1e-7,checks
    assert target_min>=-1e-10,checks
    assert meanmatch<1e-9 and normmatch<1e-8 and mosaic_normerr<1e-8,checks
    return dict(margin=out,response_max=vmax,response_min=vmin,components=comp,activity=activity,history_projection=projections,history_norm=pair_norm,activity_time=activity_time,snapshot_steps=np.array(snapsteps),checks=checks)

def run_one(args):
    path,ctxpath,conds,outdir=args;tick=time.time();model=Model(path)
    context=dict(np.load(ctxpath));res=evaluate(model,context,conds);checks=res.pop('checks')
    np.savez_compressed(Path(outdir)/(model.name+'.npz'),**res)
    signs=np.array([1.,-1.]);natural=res['margin'][:,0]
    rows=[]
    for j,c in enumerate(conds):
        y=res['margin'][:,j];hi=res['response_max'][:,j];lo=res['response_min'][:,j]
        old_correct=((y>0)==(signs>0));native_correct=np.where(signs>0,(hi>=.75)&(lo>=-.75),(lo<=-.75)&(hi<=.75))
        current_sign=np.sign(context['counts'][:,:,-1]-10)
        curr=((y>0)==(current_sign>0));valid=current_sign!=0
        row=dict(**c,mean_signed_history_margin=float(np.mean(y*signs)),history_sign_accuracy=float(old_correct.mean()),history_author_success=float(native_correct.mean()),
                 current_sign_accuracy=float(curr[valid].mean()) if np.any(valid) else None,
                 natural_choice_disagreement=float(np.mean((y>0)!=(natural>0))))
        rows.append(row)
    entry=dict(model=model.name,sha256=model.sha256,seconds=time.time()-tick,checks=checks,conditions=rows,
               readout_norms=[float(np.linalg.norm(model.Wout[s])) for s in [slice(0,400),slice(400,470),slice(470,500)]])
    (Path(outdir)/(model.name+'.json')).write_text(json.dumps(entry,indent=2))
    return entry

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=['history','current'],required=True);ap.add_argument('--n',type=int,required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--compact',action='store_true');a=ap.parse_args()
    outdir=ROOT/'results'/a.out;outdir.mkdir(parents=True,exist_ok=True)
    ctx=make_context(a.n,a.seed,a.task);ctxpath=outdir/'contexts.npz';np.savez_compressed(ctxpath,**ctx)
    conds=make_conditions(a.task,a.compact);(outdir/'conditions.json').write_text(json.dumps(conds,indent=2))
    paths=models();assert len(paths)==9
    manifest=dict(task=a.task,n_base=a.n,n_stimulus_contexts=len(ctx['u']),seed=a.seed,conditions=len(conds),status='running',python=sys.version,
                  code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('rnn_core.py')]})
    (outdir/'run_manifest.json').write_text(json.dumps(manifest,indent=2));beg=time.time();summary=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run_one,(str(p),str(ctxpath),conds,str(outdir))) for p in paths]
        for f in as_completed(jobs):
            item=f.result();summary.append(item);print(json.dumps({k:item[k] for k in ['model','seconds','checks']}),flush=True)
            (outdir/'summary.json').write_text(json.dumps(sorted(summary,key=lambda x:x['model']),indent=2))
    manifest.update(status='complete',total_seconds=time.time()-beg)
    (outdir/'run_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
