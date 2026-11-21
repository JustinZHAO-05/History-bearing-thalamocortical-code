"""Exact finite-pair explanatory propagation, not a predictive test.

Uses activation secants (cf. DeepLIFT Rescale, Shrikumar et al., ICML 2017)
and discrete variation of constants. The pair-dependent operator is measured
from the actual counterfactual trajectories. Component shares are conditional
accounts under this operator, not separately identified biological mediators.
"""
from pathlib import Path
import argparse,json,time,hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from rnn_core import ROOT,Model,models,initialize
from cognitive_causal import source_target

CONDS=[dict(name='intact_own_100',state='intact',source='own',duration=100),
       dict(name='intact_opposite_100',state='intact',source='opposite',duration=100),
       dict(name='PPC_mean_own_100',state='PPC_mean',source='own',duration=100)]

def evaluate(model,ctx):
    n=len(ctx['u']);m=len(CONDS);pair=initialize(model,ctx);inp=ctx['u']
    donor=np.empty((200,n,2,400));xx=pair.copy()
    for k in range(200):
        donor[k]=model.rate(xx)[:,:,400:470]@model.W[:400,400:470].T
        xx=model.step(xx.reshape(n*2,500),inp[:,:,1400+k].reshape(-1)).reshape(n,2,500)
    x=np.repeat(pair[:,None],m,axis=1)
    for j,c in enumerate(CONDS):
        if c['state']=='PPC_mean':x[:,j,:,:400]=pair.mean(1)[:,None,:400]
    d0=x[:,:,0]-x[:,:,1]
    z=np.zeros((n,m,4,500));z[:,:,0,:400]=d0[:,:,:400];z[:,:,1,400:]=d0[:,:,400:]
    outputs=np.zeros((n,m,2));shares=np.zeros((n,m,2,4));rate_energy=np.zeros((n,m,2));output_energy=np.zeros((n,m,2));component_energy=np.zeros((n,m,2,4))
    snapsteps=[0,100,200,320,399,600,720,799];norms=np.zeros((len(snapsteps),n,m,4));dnorm=np.zeros((len(snapsteps),n,m))
    errors=dict(state_closure_max=0.,rate_secant_max=0.,source_factorization_max=0.,output_closure_max=0.,secant_min=np.inf,secant_max=-np.inf,zero_difference_coordinates=0)
    for k in range(800):
        r=model.rate(x);d=x[:,:,0]-x[:,:,1];dr=r[:,:,0]-r[:,:,1]
        mid=x.mean(2);fallback=((mid>0)&(mid<20)).astype(float);fallback[:,:,470:]=10*((mid[:,:,470:]>9)&(mid[:,:,470:]<11))
        sec=np.divide(dr,d,out=fallback.copy(),where=d!=0)
        errors['zero_difference_coordinates']+=int(np.sum(d==0))
        errors['state_closure_max']=max(errors['state_closure_max'],float(np.max(np.abs(z.sum(2)-d))))
        errors['rate_secant_max']=max(errors['rate_secant_max'],float(np.max(np.abs(sec*d-dr))))
        errors['secant_min']=min(errors['secant_min'],float(sec.min()));errors['secant_max']=max(errors['secant_max'],float(sec.max()))
        dy=dr@model.Wout
        response=0 if 320<=k<400 else (1 if k>=720 else None)
        if response is not None:
            outputs[:,:,response]+=dy/80
            shares[:,:,response]+=(z*sec[:,:,None])@model.Wout/80
            rate_energy[:,:,response]+=np.sum(dr*dr,axis=-1)/80
            output_energy[:,:,response]+=dy*dy/80
            component_energy[:,:,response]+=np.sum((z*sec[:,:,None])**2,axis=-1)/80
        if k in snapsteps:
            idx=snapsteps.index(k);norms[idx]=np.linalg.norm(z,axis=-1);dnorm[idx]=np.linalg.norm(d,axis=-1)
        rz=z*sec[:,:,None]
        znext=(1-model.alpha)*z+model.alpha*(rz.reshape(n*m*4,500)@model.W.T).reshape(n,m,4,500)
        current=(r.reshape(n*m*2,500)@model.W.T).reshape(n,m,2,500)
        if k<100:
            for j,c in enumerate(CONDS):
                b=source_target(donor[k],c['source'],np.ones(400))
                current[:,j,:,:400]+=b-r[:,j,:,400:470]@model.W[:400,400:470].T
                # Graph surgery removes endogenous PUL→PPC propagation for
                # every component, followed by the recorded external source.
                znext[:,j,:,:400]-=model.alpha[:400]*(rz[:,j,:,400:470]@model.W[:400,400:470].T)
                g=b.mean(-1,keepdims=True);q=np.divide(b,g,out=np.zeros_like(b),where=g>1e-30)
                dg=g[:,0]-g[:,1];dq=q[:,0]-q[:,1]
                amp=dg*q.mean(1);pattern=g.mean(1)*dq
                errors['source_factorization_max']=max(errors['source_factorization_max'],float(np.max(np.abs(amp+pattern-(b[:,0]-b[:,1])))))
                znext[:,j,2,:400]+=model.alpha[:400]*amp
                znext[:,j,3,:400]+=model.alpha[:400]*pattern
        current+=inp[:,:,1400+k][:,None,:,None]*model.Win
        x=(1-model.alpha)*x+model.alpha*current;z=znext
    errors['output_closure_max']=float(np.max(np.abs(shares.sum(-1)-outputs)))
    assert errors['state_closure_max']<1e-7 and errors['output_closure_max']<1e-8,errors
    assert errors['secant_min']>=-1e-10 and errors['secant_max']<=10+1e-10,errors
    rate_norm=np.sqrt(rate_energy);alignment=np.divide(outputs,rate_norm*np.linalg.norm(model.Wout),out=np.zeros_like(outputs),where=rate_norm>1e-30)
    return dict(output_pair_difference=outputs,component_pair_differences=shares,rate_difference_rmsnorm=rate_norm,readout_alignment=alignment,
                instantaneous_output_difference_rms=np.sqrt(output_energy),component_rate_rmsnorm=np.sqrt(component_energy),component_state_norm=norms,state_difference_norm=dnorm,snapshot_steps=np.array(snapsteps),errors=errors)

def run_one(job):
    path,cp,od=job;m=Model(path);ctx=dict(np.load(cp));t=time.time();r=evaluate(m,ctx);errors=r.pop('errors');np.savez_compressed(Path(od)/(m.name+'.npz'),**r)
    summary=dict(model=m.name,seconds=time.time()-t,errors=errors,
                 mean_half_pair_output=(r['output_pair_difference'].mean(0)/2).tolist(),
                 mean_rate_difference_norm=r['rate_difference_rmsnorm'].mean(0).tolist(),
                 mean_alignment=r['readout_alignment'].mean(0).tolist(),
                 mean_half_pair_components=(r['component_pair_differences'].mean(0)/2).tolist())
    (Path(od)/(m.name+'.json')).write_text(json.dumps(summary,indent=2));return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--contexts',required=True);p.add_argument('--base-n',type=int,default=16);p.add_argument('--out',default='finite_history_calculus');p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    ctx=dict(np.load(a.contexts));keep=ctx['base_context']<a.base_n
    ctx={k:(v[keep] if np.ndim(v)>0 else v) for k,v in ctx.items()}
    od=ROOT/'results'/a.out;od.mkdir(parents=True,exist_ok=True);cp=od/'contexts.npz';np.savez_compressed(cp,**ctx)
    (od/'conditions.json').write_text(json.dumps(CONDS,indent=2));summary=[]
    meta=dict(method='actual-pair conditional finite secant account; not an independent prediction',base_n=a.base_n,context_source=a.contexts,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),status='running')
    (od/'run_manifest.json').write_text(json.dumps(meta,indent=2));tick=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(run_one,(str(p),str(cp),str(od))) for p in models()]
        for f in as_completed(jobs):
            x=f.result();summary.append(x);print(json.dumps(x),flush=True);(od/'summary.json').write_text(json.dumps(sorted(summary,key=lambda z:z['model']),indent=2))
    meta.update(status='complete',seconds=time.time()-tick);(od/'run_manifest.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
