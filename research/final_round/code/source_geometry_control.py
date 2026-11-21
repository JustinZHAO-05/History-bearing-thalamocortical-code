"""Fresh, source-energy-matched direction controls; protocol ../protocols/geometry_control_v1.md.

Uses the frozen independent Round3 evaluator for checkpoint loading, original
task contexts and current-only amplitude matching. No learned weight is changed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
R3=Path(os.environ.get('FINAL_R3_ROOT',str(ROOT.parent/'round3_consciousness')))
AUDIT=R3/'reviews/independent_audit'
sys.path.insert(0,str(AUDIT))
import rank_one_control as engine
import residual_addback_confirmation as amplitude
if os.environ.get('R3_HOSFORD_DATA'):
    engine.DATA=Path(os.environ['R3_HOSFORD_DATA'])
OUT=ROOT/'results/source_geometry_v1'
PRIOR=AUDIT/'rank_one_heldout'
CONDITIONS=[
 {'name':'natural','state':'intact','source':'natural'},
 {'name':'intact_full_own','state':'intact','source':'full_own'},
 {'name':'intact_full_opposite','state':'intact','source':'full_opposite'},
 {'name':'allmean_full_own','state':'mean','source':'full_own'},
 {'name':'allmean_mean','state':'mean','source':'mean'},
 {'name':'intact_axis_opposite','state':'intact','source':'axis_opposite'},
 {'name':'allmean_axis_own','state':'mean','source':'axis_own'},
]+[{'name':f'allmean_random_{i:02d}','state':'mean','source':'random','direction':i} for i in range(16)]

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def evaluate(net,ctx,v,seed):
    init,donor=engine.initial_and_donors(net,ctx)
    mu=donor.mean(axis=2);d=donor[:,:,0]-donor[:,:,1]
    coeff=d@v
    learned,scale,check=amplitude.scalar_energy_match(coeff[...,None]*v,mu,d)
    rng=np.random.default_rng(seed)
    axes=[];msgs=[];geometry=[];checks=[check]
    support=np.any(net.B!=0,axis=1)
    normd=np.linalg.norm(d,axis=-1)
    for i in range(16):
        a=rng.normal(size=400);a[~support]=0
        a-=np.dot(a,v)*v;a/=np.linalg.norm(a)
        projected=coeff[...,None]*a
        msg,sc,chk=amplitude.scalar_energy_match(projected,mu,d)
        cosine_v=np.divide(msg@v,normd,out=np.zeros_like(normd),where=normd>1e-20)
        cosine_d=np.divide(np.sum(msg*d,axis=-1),normd**2,out=np.zeros_like(normd),where=normd>1e-20)
        geometry.append(dict(direction=i,preclip_axis_dot=float(a@v),
            postclip_v_cosine_mean=float(cosine_v.mean()),postclip_v_absolute_cosine_mean=float(np.abs(cosine_v).mean()),
            postclip_d_cosine_mean=float(cosine_d.mean()),postclip_d_absolute_cosine_mean=float(np.abs(cosine_d).mean())))
        axes.append(a);msgs.append(msg);checks.append(chk)
    # Keeping the complete declared set in memory avoids repeated nonlinear matching.
    n=len(init);m=len(CONDITIONS)
    x=np.repeat(init[:,None],m,axis=1)
    for j,c in enumerate(CONDITIONS):
        if c['state']=='mean':x[:,j]=init.mean(axis=1,keepdims=True)
    y=np.empty((800,n,m,2));pmerr=0.;minimum=np.inf
    pair_sign=np.array([1.,-1.])[None,:,None]
    for k in range(800):
        r=net.rates(x);y[k]=r@net.output+net.bias
        drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j,c in enumerate(CONDITIONS):
                source=c['source']
                if source=='natural':continue
                if source=='full_own':b=donor[k]
                elif source=='full_opposite':b=donor[k,:,::-1]
                elif source=='mean':b=np.repeat(mu[k,:,None],2,axis=1)
                elif source=='axis_own':b=mu[k,:,None]+.5*pair_sign*learned[k,:,None]
                elif source=='axis_opposite':b=mu[k,:,None]-.5*pair_sign*learned[k,:,None]
                elif source=='random':b=mu[k,:,None]+.5*pair_sign*msgs[c['direction']][k,:,None]
                else:raise ValueError(source)
                pmerr=max(pmerr,float(np.abs(b.mean(axis=1)-mu[k]).max()))
                minimum=min(minimum,float(b.min()))
                drive[:,j,:,:400]+=b-r[:,j,:,400:470]@net.B.T
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    windows=[slice(320,400),slice(720,800)]
    arr=dict(margin=np.stack([y[w].mean(axis=0) for w in windows],axis=-1),
      response_min=np.stack([y[w].min(axis=0) for w in windows],axis=-1),
      response_max=np.stack([y[w].max(axis=0) for w in windows],axis=-1),
      output_trace=y,learned_axis=v,random_axes=np.array(axes),base_context=ctx['base_context'],counts=ctx['counts'])
    qc=dict(own_sham_max_abs_output=float(np.abs(y[:,:,0]-y[:,:,1]).max()),
      mean_source_zero_history_max_abs=float(np.abs(y[:,:,4,0]-y[:,:,4,1]).max()),
      pair_mean_current_max_abs_error=pmerr,minimum_current=minimum,
      norm_matching_relative_energy_error_max=max(z['norm_energy_relative_error_max'] for z in checks),
      infeasible_targets=sum(z['infeasible_norm_targets'] for z in checks),
      checkpoint_message_checks=checks)
    assert qc['own_sham_max_abs_output']<1e-9 and qc['mean_source_zero_history_max_abs']<1e-9,qc
    assert pmerr<1e-9 and minimum>=-1e-10 and qc['infeasible_targets']==0,qc
    assert qc['norm_matching_relative_energy_error_max']<1e-9,qc
    return arr,qc,geometry

def summarize(arr,ctx):
    metrics=amplitude.endpoint_arrays(arr,ctx)
    base=ctx['base_context'];nb=len(np.unique(base))
    grouped={k:np.stack([v[base==b].mean(axis=(0,2)) for b in range(nb)]) for k,v in metrics.items()}
    ix=np.random.default_rng(84199).integers(0,nb,size=(10000,nb))
    rows=[]
    for j,c in enumerate(CONDITIONS):
        rows.append(dict(**c,outcomes={k:dict(mean=float(v[:,j].mean()),ci95=np.quantile(v[ix,j].mean(axis=1),[.025,.975]).tolist()) for k,v in grouped.items()}))
    H=grouped['history_signed_margin']
    boot=H[ix].mean(axis=1)
    delta=H[:,6].mean()-np.abs(H[:,7:].mean(axis=0)).mean()
    bd=boot[:,6]-np.abs(boot[:,7:]).mean(axis=1)
    primary=dict(learned_minus_mean_random_absolute_history=float(delta),ci95=np.quantile(bd,[.025,.975]).tolist(),
      learned_history=float(H[:,6].mean()),random_signed_history=H[:,7:].mean(axis=0).tolist(),
      random_absolute_history_mean=float(np.abs(H[:,7:].mean(axis=0)).mean()),
      random_absolute_history_max=float(np.abs(H[:,7:].mean(axis=0)).max()),
      n_random_absolute_at_least_learned=int(np.sum(np.abs(H[:,7:].mean(axis=0))>=H[:,6].mean())),
      interpretation='Bootstrap over base histories, sampled directions held fixed; axes are within-model controls.')
    arr.update({'base_'+k:v for k,v in grouped.items()})
    return rows,primary

def run(index,path):
    t=time.time();digest=sha(path);net=engine.IndependentNetwork(path)
    ctx=dict(np.load(OUT/'contexts.npz'))
    with np.load(PRIOR/(net.name+'.npz')) as z:v=z['source_axis'].copy()
    arr,qc,geom=evaluate(net,ctx,v,84110+index)
    rows,primary=summarize(arr,ctx)
    np.savez_compressed(OUT/(net.name+'.npz'),**arr)
    assert sha(path)==digest
    result=dict(model=net.name,checkpoint_sha256=digest,checkpoint_unchanged=True,
      seconds=time.time()-t,checks=qc,geometry=geom,conditions=rows,primary=primary)
    (OUT/(net.name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(model=net.name,seconds=result['seconds'],primary=primary,
      native_current=[c['outcomes']['current_author_success']['mean'] for c in rows])),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=OUT/'protocol.freeze.json'
    if freeze.exists():raise RuntimeError('Preserve completed/frozen version; use a fresh directory for reproduction.')
    paths=sorted(engine.DATA.glob('*/*.db'))
    assert len(paths)==9,len(paths)
    record=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_fresh_context_generation',
      code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocols/geometry_control_v1.md'),
      engine_sha256=sha(engine.__file__),amplitude_sha256=sha(amplitude.__file__),
      context_seed=84101,n_base=24,n_random_axes=16,axis_seed_base=84110,conditions=CONDITIONS,
      checkpoint_sha256={str(p):sha(p) for p in paths},
      learned_axes_files={p.name:sha(p) for p in sorted(PRIOR.glob('*.npz')) if p.name!='contexts.npz'})
    freeze.write_text(json.dumps(record,indent=2))
    ctx=engine.contexts(24,84101)
    np.savez_compressed(OUT/'contexts.npz',**ctx)
    start=time.time();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,i,p) for i,p in enumerate(paths)]
        for f in as_completed(futures):
            results.append(f.result())
            (OUT/'summary.json').write_text(json.dumps(sorted(results,key=lambda z:z['model']),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',models=len(results),seconds=time.time()-start,code_sha256=sha(__file__),contexts_sha256=sha(OUT/'contexts.npz')),indent=2))

if __name__=='__main__':main()
