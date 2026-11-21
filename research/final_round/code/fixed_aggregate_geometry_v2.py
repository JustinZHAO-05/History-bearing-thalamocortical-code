"""Corrected native-cell-class source control; preserves the frozen v1 archive."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse, hashlib, json, os, sys, time, pickle, sqlite3
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
R3=Path(os.environ.get('FINAL_R3_ROOT',str(ROOT.parent/'round3_consciousness')))
sys.path.insert(0,str(R3/'reviews/independent_audit'))
import rank_one_control as engine
import residual_addback_confirmation as prior
if os.environ.get('R3_HOSFORD_DATA'):engine.DATA=Path(os.environ['R3_HOSFORD_DATA'])
OUT=ROOT/'results/fixed_aggregate_geometry_v2'
CONDITIONS=[
 dict(name='natural',state='intact',source='natural'),
 dict(name='intact_own',state='intact',source='own'),
 dict(name='allmean_own',state='mean',source='own'),
 dict(name='allmean_mean',state='mean',source='mean'),
]+[dict(name=f'allmean_map_{i}',state='mean',source='mapped',map=i) for i in range(4)]

def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()

def native_classes(path):
    db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    row=dict(db.execute('SELECT * FROM currentRNNParameters ORDER BY id DESC LIMIT 1').fetchone());db.close()
    arrays={k:pickle.loads(v) for k,v in row.items() if isinstance(v,bytes)}
    classes={label:np.sort(np.asarray(arrays[key],dtype=int).ravel()) for label,key in [('E','PPCexh_idx'),('I','PPCinh_idx')]}
    assert len(classes['E'])==280 and len(classes['I'])==120
    assert np.array_equal(np.sort(np.r_[classes['E'],classes['I']]),np.arange(400))
    mask=np.asarray(arrays['SynapseMask'])
    assert np.all(mask[:,classes['E']]>=0) and np.all(mask[:,classes['I']]<=0)
    return classes

def transform(d,mu,support,classes,seed):
    rng=np.random.default_rng(seed);shape=d.shape
    z=d.reshape(-1,400).copy();cap=(2*mu).reshape(-1,400)
    groups=[indices[support[indices]] for indices in classes.values()]
    accepted=0;possible=0
    for sweep in range(64):
        for group in groups:
            perm=rng.permutation(group);triplets=perm[:3*(len(perm)//3)].reshape(-1,3)
            old=z[:,triplets];m=old.mean(axis=-1,keepdims=True);r=old-m
            theta=rng.uniform(-np.pi,np.pi,size=(1,len(triplets),1))
            cross=np.stack([r[:,:,2]-r[:,:,1],r[:,:,0]-r[:,:,2],r[:,:,1]-r[:,:,0]],axis=-1)/np.sqrt(3.)
            new=m+np.cos(theta)*r+np.sin(theta)*cross
            valid=np.all(np.abs(new)<=cap[:,triplets],axis=-1)
            z[:,triplets]=np.where(valid[:,:,None],new,old)
            accepted+=int(valid.sum());possible+=valid.size
    out=z.reshape(shape);qc={'accepted_triplet_fraction':accepted/possible,'seed':seed}
    residual=d.copy();newres=out.copy()
    for label,sl in classes.items():
        original=d[...,sl];changed=out[...,sl]
        sums=original.sum(axis=-1);ener=(original**2).sum(axis=-1)
        qc[label+'_sum_error_max']=float(np.abs(changed.sum(axis=-1)-sums).max())
        delta=np.abs((changed**2).sum(axis=-1)-ener)
        qc[label+'_norm_energy_relative_error_max']=float(np.divide(delta,ener,out=np.zeros_like(delta),where=ener>1e-24).max())
        residual[...,sl]-=original.mean(axis=-1,keepdims=True)
        newres[...,sl]-=changed.mean(axis=-1,keepdims=True)
    def cos(a,b):
        denom=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
        return np.divide((a*b).sum(axis=-1),denom,out=np.zeros_like(denom),where=denom>1e-20)
    qc['original_difference_cosine_mean']=float(cos(d,out).mean())
    qc['within_class_residual_cosine_mean']=float(cos(residual,newres).mean())
    qc['relative_difference_change_rms']=float(np.linalg.norm(out-d)/np.linalg.norm(d))
    qc['minimum_pair_current']=float(min((mu+out/2).min(),(mu-out/2).min()))
    assert max(qc['E_norm_energy_relative_error_max'],qc['I_norm_energy_relative_error_max'])<1e-10,qc
    assert max(qc['E_sum_error_max'],qc['I_sum_error_max'])<1e-8 and qc['minimum_pair_current']>=-1e-10,qc
    return out,qc

def evaluate(net,ctx,idx,classes):
    init,donor=engine.initial_and_donors(net,ctx)
    n=len(init);m=len(CONDITIONS)
    # One identical source per base before the subsequently crossed fifth input.
    shared=donor.reshape(200,24,4,2,400)
    prefixerror=float(np.abs(shared-shared[:,:,:1]).max());assert prefixerror<1e-12
    d=shared[:,:,0,0]-shared[:,:,0,1];mu=shared[:,:,0].mean(axis=2)
    maps=[];qcs=[];support=np.any(net.B!=0,axis=1)
    for j in range(4):
        changed,qc=transform(d,mu,support,classes,84210+idx*100+j)
        maps.append(np.repeat(changed,4,axis=1));qcs.append(qc)
    common=donor.mean(axis=2)
    x=np.repeat(init[:,None],m,axis=1)
    for j,c in enumerate(CONDITIONS):
        if c['state']=='mean':x[:,j]=init.mean(axis=1,keepdims=True)
    y=np.empty((800,n,m,2));sign=np.array([1.,-1.])[None,:,None];minb=np.inf;pmerr=0.
    for k in range(800):
        r=net.rates(x);y[k]=r@net.output+net.bias
        drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j,c in enumerate(CONDITIONS):
                s=c['source']
                if s=='natural':continue
                if s=='own':b=donor[k]
                elif s=='mean':b=np.repeat(common[k,:,None],2,axis=1)
                else:b=common[k,:,None]+.5*sign*maps[c['map']][k,:,None]
                minb=min(minb,float(b.min()));pmerr=max(pmerr,float(np.abs(b.mean(axis=1)-common[k]).max()))
                drive[:,j,:,:400]+=b-r[:,j,:,400:470]@net.B.T
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    windows=[slice(320,400),slice(720,800)]
    arrays=dict(margin=np.stack([y[w].mean(axis=0) for w in windows],axis=-1),
        response_min=np.stack([y[w].min(axis=0) for w in windows],axis=-1),response_max=np.stack([y[w].max(axis=0) for w in windows],axis=-1),
        output_trace=y,base_context=ctx['base_context'],counts=ctx['counts'])
    qc=dict(prefix_source_max_abs_error=prefixerror,own_sham_max_abs_output=float(np.abs(y[:,:,0]-y[:,:,1]).max()),
        mean_source_pair_output_max_abs=float(np.abs(y[:,:,3,0]-y[:,:,3,1]).max()),pair_mean_current_error=pmerr,minimum_current=minb,maps=qcs)
    assert qc['own_sham_max_abs_output']<1e-9 and qc['mean_source_pair_output_max_abs']<1e-9,qc
    return arrays,qc

def summarize(a,ctx):
    metrics=prior.endpoint_arrays(a,ctx);base=ctx['base_context']
    grouped={k:np.stack([v[base==b].mean(axis=(0,2)) for b in range(24)]) for k,v in metrics.items()}
    ix=np.random.default_rng(84299).integers(0,24,size=(10000,24))
    rows=[]
    for j,c in enumerate(CONDITIONS):
        rows.append(dict(**c,outcomes={k:dict(mean=float(v[:,j].mean()),ci95=np.quantile(v[ix,j].mean(axis=1),[.025,.975]).tolist()) for k,v in grouped.items()}))
    H=grouped['history_signed_margin'];delta=H[:,2]-H[:,4:].mean(axis=1)
    boot=H[ix].mean(axis=1)
    absdelta=abs(H[:,2].mean())-np.abs(H[:,4:].mean(axis=0)).mean()
    abboot=np.abs(boot[:,2])-np.abs(boot[:,4:]).mean(axis=1)
    primary=dict(full_history=float(H[:,2].mean()),mapped_history=H[:,4:].mean(axis=0).tolist(),
        full_minus_mean_mapped=float(delta.mean()),ci95=np.quantile(delta[ix].mean(axis=1),[.025,.975]).tolist(),
        absolute_full_minus_mean_absolute_mapped=float(absdelta),absolute_ci95=np.quantile(abboot,[.025,.975]).tolist())
    a.update({'base_'+k:v for k,v in grouped.items()})
    return rows,primary

def run(idx,path):
    t=time.time();digest=sha(path);net=engine.IndependentNetwork(path);ctx=dict(np.load(OUT/'contexts.npz'))
    classes=native_classes(path)
    arrays,checks=evaluate(net,ctx,idx,classes);rows,primary=summarize(arrays,ctx)
    checks['native_cell_classes']={label:indices.tolist() for label,indices in classes.items()}
    checks['dale_signs_verified']=True
    assert sha(path)==digest
    np.savez_compressed(OUT/(net.name+'.npz'),**arrays)
    result=dict(model=net.name,checkpoint_sha256=digest,checkpoint_unchanged=True,seconds=time.time()-t,checks=checks,conditions=rows,primary=primary)
    (OUT/(net.name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(model=net.name,seconds=result['seconds'],primary=primary,
      cosines=[z['within_class_residual_cosine_mean'] for z in checks['maps']],
      native_current=[z['outcomes']['current_author_success']['mean'] for z in rows])),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);freeze=OUT/'protocol.freeze.json'
    if freeze.exists():raise RuntimeError('Preserve frozen outputs; reproduce in a fresh output directory.')
    paths=sorted(engine.DATA.glob('*/*.db'));assert len(paths)==9
    freeze.write_text(json.dumps(dict(frozen_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_new_context_generation',
      code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocols/fixed_aggregate_geometry_v2.md'),
      engine_sha256=sha(engine.__file__),context_seed=84201,n_base=24,n_maps=4,sweeps=64,conditions=CONDITIONS,
      checkpoint_sha256={str(p):sha(p) for p in paths}),indent=2))
    np.savez_compressed(OUT/'contexts.npz',**engine.contexts(24,84201))
    t=time.time();results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,i,p) for i,p in enumerate(paths)]
        for f in as_completed(futures):
            results.append(f.result());(OUT/'summary.json').write_text(json.dumps(sorted(results,key=lambda z:z['model']),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',models=len(results),seconds=time.time()-t,code_sha256=sha(__file__),contexts_sha256=sha(OUT/'contexts.npz')),indent=2))

if __name__=='__main__':main()
