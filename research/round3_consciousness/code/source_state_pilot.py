"""Exploratory source × retained-state cognitive intervention, no refitting."""
from pathlib import Path
import argparse,json,time,hashlib,sys,platform
import numpy as np
from rnn_core import Model,models,paired_histories,initialize,ROOT

def conditions():
    out=[dict(name='natural',state='intact',source='natural',duration=0)]
    for duration in [100,200]:
        for state in ['intact','PPC_mean','outside_mean','all_mean']:
            for source in ['own','mean','opposite']:
                out.append(dict(name=f'{state}_{source}_{duration}',state=state,source=source,duration=duration))
        for source in ['mosaic','zero']:
            out.append(dict(name=f'intact_{source}_{duration}',state='intact',source=source,duration=duration))
    return out

def evaluate(model,context,conds):
    n=len(context['u']);m=len(conds);start=int(context['start']);end=int(context['end'])
    u=context['u'];xpair=initialize(model,context);mean=xpair.mean(1)
    # Natural donors are measured before interventions; only their PPC arrival
    # currents are replayed, never their direct contribution to the readout.
    donor=np.empty((200,n,2,400));natural_y=np.empty((n,2,end-start))
    xx=xpair.copy();natural_snap=[]
    for t in range(start,end):
        r=model.rate(xx)
        if t<start+200:donor[t-start]=r[:,:,400:470]@model.W[:400,400:470].T
        natural_y[:,:,t-start]=r@model.Wout+model.bias
        if t in [start+200,end-1]:natural_snap.append(xx.copy())
        xx=model.step(xx.reshape(n*2,500),u[:,:,t].reshape(-1)).reshape(n,2,500)
    x=np.repeat(xpair[:,None,:,:],m,axis=1)
    for j,c in enumerate(conds):
        if c['state']=='PPC_mean':x[:,j,:,:400]=mean[:,None,:400]
        if c['state']=='outside_mean':x[:,j,:,400:]=mean[:,None,400:]
        if c['state']=='all_mean':x[:,j]=mean[:,None,:]
    rng=np.random.default_rng(314159);mosaic=np.r_[np.ones(200),-np.ones(200)];rng.shuffle(mosaic)
    out=np.zeros((n,m,2));components=np.zeros((n,m,2,3));act=np.zeros((n,m,2,3))
    snapshots=[];ytrace=np.empty((n,m,2,end-start))
    for t in range(start,end):
        r=model.rate(x);y=r@model.Wout+model.bias;ytrace[:,:,:,t-start]=y
        if t>=end-80:
            out+=y/80
            for z,sl in enumerate([slice(0,400),slice(400,470),slice(470,500)]):
                components[:,:,:,z]+=(r[:,:,:,sl]@model.Wout[sl])/80
                act[:,:,:,z]+=r[:,:,:,sl].mean(-1)/80
        if t in [start+100,start+200,end-1]:snapshots.append(x.copy())
        current=(r.reshape(n*m*2,500)@model.W.T).reshape(n,m,2,500)
        if t<start+200:
            b=donor[t-start];mu=b.mean(1,keepdims=True);delta=b-mu
            for j,c in enumerate(conds):
                if t-start>=c['duration']:continue
                s=c['source']
                if s=='own':target=b
                elif s=='mean':target=np.broadcast_to(mu,b.shape)
                elif s=='opposite':target=mu-delta
                elif s=='mosaic':target=mu+delta*mosaic
                elif s=='zero':target=np.zeros_like(b)
                else:continue
                native=r[:,j,:,400:470]@model.W[:400,400:470].T
                current[:,j,:,:400]+=target-native
        current+=u[:,:,t][:,None,:,None]*model.Win
        x=(1-model.alpha)*x+model.alpha*current
    sham=max(float(np.max(np.abs(ytrace[:,j]-natural_y))) for j,c in enumerate(conds) if c['state']=='intact' and c['source'] in ['natural','own'])
    checks=dict(sham_max_abs_output=sham,output_component_max_error=float(np.max(np.abs(out-components.sum(-1)-model.bias))),
                mosaic_pairmean_max_error=float(np.max(np.abs((donor.mean(2,keepdims=True)+(donor-donor.mean(2,keepdims=True))*mosaic).mean(2)-donor.mean(2)))),
                mosaic_diffnorm_max_error=float(np.max(np.abs(np.linalg.norm((donor[:,:,0]-donor[:,:,1])*mosaic,axis=-1)-np.linalg.norm(donor[:,:,0]-donor[:,:,1],axis=-1)))))
    assert sham<1e-7,checks
    return dict(margin=out,components=components,activity=act,states=np.asarray(snapshots),natural_states=np.asarray(natural_snap),initial_states=xpair,output_trace=ytrace,checks=checks)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=4);ap.add_argument('--seed',type=int,default=83001);ap.add_argument('--out',default='source_state_pilot');ap.add_argument('--models',type=int,default=9);a=ap.parse_args()
    outdir=ROOT/'results'/a.out;outdir.mkdir(parents=True,exist_ok=True)
    ctx=paired_histories(a.n,a.seed);np.savez_compressed(outdir/'contexts.npz',**ctx)
    conds=conditions();(outdir/'conditions.json').write_text(json.dumps(conds,indent=2))
    summary=[];beg=time.time()
    for p in models()[:a.models]:
        m=Model(p);tick=time.time();res=evaluate(m,ctx,conds);checks=res.pop('checks')
        np.savez_compressed(outdir/(m.name+'.npz'),**res)
        rows=[]
        for j,c in enumerate(conds):
            margin=res['margin'][:,j];signs=np.array([1,-1])
            rows.append(dict(**c,mean_signed_margin=float(np.mean(margin*signs)),history_choice_accuracy=float(np.mean((margin>0)==(signs>0))),mean_pair_contrast=float(np.mean(margin[:,0]-margin[:,1]))))
        entry=dict(model=m.name,sha256=m.sha256,seconds=time.time()-tick,checks=checks,conditions=rows)
        summary.append(entry);(outdir/'summary.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps({'model':m.name,'seconds':entry['seconds'],'sham':checks['sham_max_abs_output'],'rows':[{k:r[k] for k in ['name','mean_signed_margin','history_choice_accuracy']} for r in rows]}),flush=True)
    manifest=dict(seed=a.seed,n=a.n,python=sys.version,platform=platform.platform(),total_seconds=time.time()-beg,code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('rnn_core.py')]})
    (outdir/'run_manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
