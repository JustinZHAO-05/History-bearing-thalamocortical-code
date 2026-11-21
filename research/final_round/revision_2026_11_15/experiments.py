"""Fixed-weight source and receiver experiments for the November revision."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json,hashlib,time,pickle,sqlite3,argparse
import numpy as np
from scipy.ndimage import uniform_filter1d
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'revision_2026_11_14'
sys.path.insert(0,str(OLD/'code'))
import source_space_v1 as v1
sys.path.insert(0,str(ROOT.parents[1]/'round3_consciousness/code'))
from rnn_core import Model
OUT=Path(os.environ.get('NOV15_RESULTS',str(ROOT/'results')));OUT.mkdir(exist_ok=True)

def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def boot(a,base,seed=96099):
    b=np.stack([a[base==k].mean(axis=0) for k in np.unique(base)])
    ix=np.random.default_rng(seed).integers(0,len(b),(6000,len(b)))
    return dict(mean=b.mean(axis=0).tolist(),ci=np.quantile(b[ix].mean(axis=1),[.025,.975],axis=0).T.tolist())
def classes(path):
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row;r=dict(c.execute('select * from currentRNNParameters order by id desc limit 1').fetchone())
    return [np.asarray(pickle.loads(r[k]),int) for k in ('PPCexh_idx','PPCinh_idx')]

def sparse_maps(rate,B,A,seed):
    mu=rate.mean(axis=2);d=rate[:,:,0]-rate[:,:,1];shape=d.shape
    S=d.reshape(-1,70);cap=(2*np.minimum(mu,20-mu)).reshape(-1,70)
    M=B.T@B;H=S@M;cur=S.copy();h=H.copy();distance=np.zeros(len(S))
    rng=np.random.default_rng(seed);maps=[]
    for j in range(256):
        ids=rng.choice(70,3,replace=False);z=np.cross(A[0,ids],A[1,ids])
        zn=np.linalg.norm(z)
        if zn>1e-15:
            z/=zn;den=z@M[np.ix_(ids,ids)]@z
            if den>1e-20:
                alpha=-2*(h[:,ids]@z)/den
                delta=alpha[:,None]*z
                cand=cur[:,ids]+delta
                # Squared receiving distance from the original difference.
                change=2*np.sum(delta*(h[:,ids]-H[:,ids]),axis=1)+alpha*alpha*den
                good=(np.max(np.abs(cand)-cap[:,ids],axis=1)<1e-10)&(change>1e-16)
                cur[np.ix_(good,ids)]=cand[good]
                h[good]+=alpha[good,None]*(z@M[ids])
                distance[good]+=change[good]
        if j==63:maps.append(cur.reshape(shape).copy())
    maps.append(cur.reshape(shape).copy())
    sm=uniform_filter1d(maps[-1]-d,size=5,axis=0,mode='nearest').reshape(-1,70)
    # Smoothing preserves Az=0. Exact line intersection restores energy.
    den=np.einsum('ij,ij->i',sm@M,sm);num=np.sum((S@M)*sm,axis=1)
    alpha=np.divide(-2*num,den,out=np.zeros_like(num),where=den>1e-20)
    cand=S+alpha[:,None]*sm
    good=np.max(abs(cand)-cap,axis=1)<1e-10
    smooth=S.copy();smooth[good]=cand[good];maps.append(smooth.reshape(shape))
    checks=[]
    bd=S@B.T;energy=np.sum(bd**2,axis=1)
    for mp in maps:
        flat=mp.reshape(-1,70);new=flat@B.T;diff=new-bd
        norm=np.linalg.norm(bd,axis=1)
        ec=np.divide(abs(np.sum(new**2,axis=1)-energy),energy,out=np.zeros_like(energy),where=energy>1e-15)
        check=dict(relative_change=float(np.mean(np.divide(np.linalg.norm(diff,axis=1),norm,out=np.zeros_like(norm),where=norm>1e-10))),
          changed_fraction=float(np.mean(np.linalg.norm(diff,axis=1)>1e-8)),energy_error=float(ec.max()),ei_error=float(abs((flat-S)@A.T).max()),
          bound_error=float(np.max(abs(flat)-cap)),roughness=float(np.mean(np.diff(mp,axis=0)**2)),natural_roughness=float(np.mean(np.diff(d,axis=0)**2)))
        assert check['energy_error']<1e-8 and check['ei_error']<1e-7 and check['bound_error']<1e-8,check
        checks.append(check)
    return mu,d,np.array(maps),checks

def endpoints(y,ctx):
    margin=np.stack([y[w].mean(axis=0) for w in v1.WINDOWS],axis=-1)
    mn=np.stack([y[w].min(axis=0) for w in v1.WINDOWS],axis=-1)
    mx=np.stack([y[w].max(axis=0) for w in v1.WINDOWS],axis=-1)
    target=np.sign(ctx['counts'][:,:,4]-10)[:,None,:]
    cs=np.where(target>0,(mx[...,0]>=.75)&(mn[...,0]>=-.75),(mn[...,0]<=-.75)&(mx[...,0]<=.75))
    signs=np.array([1.,-1.])[None,None,:]
    hs=np.where(signs>0,(mx[...,1]>=.75)&(mn[...,1]>=-.75),(mn[...,1]<=-.75)&(mx[...,1]<=.75))
    opp=np.where(signs<0,(mx[...,1]>=.75)&(mn[...,1]>=-.75),(mn[...,1]<=-.75)&(mx[...,1]<=.75))
    return dict(H=np.mean(margin[...,1]*signs,axis=-1),sensory=np.mean(margin[...,0]*target,axis=-1),current=cs.mean(axis=-1),history=hs.mean(axis=-1),opposite=opp.mean(axis=-1),margin=margin)

def source(index,path):
    net=v1.eng.IndependentNetwork(path);ctx=v1.eng.contexts(48,96002,(0,2,18,20))
    initial,rate=v1.donor(net,ctx);e,i=classes(path)
    A=np.stack([net.B[e].sum(axis=0),net.B[i].sum(axis=0)])
    mu,d,maps,checks=sparse_maps(rate,net.B,A,96100+index)
    names=['natural','own','mean','sparse64','sparse256','smooth','opposite']
    n=len(initial);m=len(names);x=np.repeat(initial[:,None],m,axis=1)
    x[:,1:]=initial.mean(axis=1)[:,None,None]
    y=np.empty((800,n,m,2));norm=np.empty((800,n,m),np.float32)
    for k in range(800):
        r=net.rates(x);y[k]=r@net.output+net.bias;norm[k]=np.linalg.norm(r[:,:,0]-r[:,:,1],axis=-1)
        drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j in range(1,m):
                sr=rate[k] if j==1 else (np.repeat(mu[k,:,None],2,axis=1) if j==2 else (rate[k,:,::-1] if j==6 else mu[k,:,None]+np.array([1.,-1.])[None,:,None]*maps[j-3,k,:,None]/2))
                drive[:,j,:,:400]+=sr@net.B.T-r[:,j,:,400:470]@net.B.T
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    ep=endpoints(y,ctx);base=ctx['base_context']
    assert np.max(abs(y[:,:,2,0]-y[:,:,2,1]))<1e-10
    results={k:boot(v,base) for k,v in ep.items() if k!='margin'}
    losses=np.stack([ep['H'][:,1]-ep['H'][:,j] for j in (3,4,5)],axis=1)
    sensory=np.stack([ep['sensory'][:,j]-ep['sensory'][:,1] for j in (3,4,5)],axis=1)
    results['history_loss']=boot(losses,base);results['sensory_change']=boot(sensory,base)
    # Saved discovery arrays: actual held-out variance and post-clipping geometry.
    old=np.load(OLD/'results'/f'{net.name}.npz');od=old['natural_source_difference'];axis=old['axis'];cd=old['source_candidate_difference'][0]
    frac=float(np.sum((od@axis)**2)/np.sum(od**2));cos=np.divide(cd@axis,np.linalg.norm(cd,axis=-1),out=np.zeros(cd.shape[:-1]),where=np.linalg.norm(cd,axis=-1)>1e-12)
    eig=np.linalg.eigvalsh(cd.reshape(-1,70).T@cd.reshape(-1,70));eff=float(eig.sum()**2/(eig@eig))
    rec=dict(model=net.name,nbase=48,names=names,geometry=checks,outcomes=results,heldout_axis_fraction=frac,postclip_axis_cosine_abs=float(np.mean(abs(cos))),postclip_participation_ratio=eff)
    np.savez_compressed(OUT/f'source_{net.name}.npz',output_trace=y,rate_norm=norm,base=base,counts=ctx['counts'],maps=maps,natural_difference=d,source_mean=mu,**ep)
    dump(OUT/f'source_{net.name}.json',rec);return rec

def secant(net,x):
    r=net.rates(x);dx=x[:,0]-x[:,1];dr=r[:,0]-r[:,1]
    local=((x.mean(axis=1)>0)&(x.mean(axis=1)<20)).astype(float)
    local[:,470:]=10*((x.mean(axis=1)[:,470:]>9)&(x.mean(axis=1)[:,470:]<11))
    return np.divide(dr,dx,out=local,where=abs(dx)>1e-12)

def receiver_run(net,ctx,intervene,seed):
    initial,donor=v1.donor(net,ctx);n=len(initial);m=4 if intervene else 1
    x=np.repeat(initial.mean(axis=1)[:,None,None],m,axis=1);x=np.repeat(x,2,axis=2)
    y=np.empty((800,n,m,2));rn=np.empty((800,n,m));align=np.empty_like(rn)
    regimes=np.empty((800,m,3));Ds=[];preD=None;startdelta=None;shift_norm=None
    rand=np.random.default_rng(seed).normal(size=(n,400));rand/=np.linalg.norm(rand,axis=1)[:,None]
    for k in range(800):
        if k==240:preD=secant(net,x[:,0])
        if k==310:
            startdelta=x[:,0,0]-x[:,0,1]
            if intervene:
                common=x[:,0,:,:400].mean(axis=1);shift=.5*(np.clip(common,2,18)-common)
                shift_norm=np.linalg.norm(shift,axis=1)
                before=x[:,:,0]-x[:,:,1]
                x[:,1,:,:400]+=shift[:,None];x[:,2,:,:400]+=rand[:,None]*shift_norm[:,None,None]
                assert np.max(abs((x[:,:,0]-x[:,:,1])-before))<1e-12
        if 310<=k<400:Ds.append(secant(net,x[:,0]))
        r=net.rates(x);y[k]=r@net.output+net.bias
        dr=r[:,:,0]-r[:,:,1];rn[k]=np.linalg.norm(dr,axis=-1)
        align[k]=np.divide(abs(dr@net.output),np.linalg.norm(net.output)*rn[k],out=np.zeros_like(rn[k]),where=rn[k]>1e-12)
        regimes[k,:,0]=(r[...,:400]<=1e-12).mean(axis=(0,2,3));regimes[k,:,1]=((r[...,:400]>1e-12)&(r[...,:400]<20)).mean(axis=(0,2,3));regimes[k,:,2]=(r[...,:400]>=20).mean(axis=(0,2,3))
        drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:drive[...,:400]+=donor[k,:,None]@net.B.T-r[...,400:470]@net.B.T
        if intervene and 310<=k<330:drive[:,3]=drive[:,3].mean(axis=1,keepdims=True)
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    return dict(y=y,rate_norm=rn,alignment=align,regimes=regimes,D=np.array(Ds),preD=preD,delta=startdelta,shift_norm=shift_norm)

def receiver(index,path):
    net=v1.eng.IndependentNetwork(path);qs=(2,10,18)
    train=v1.eng.contexts(12,97001,qs);test=v1.eng.contexts(24,97002,qs)
    tr=receiver_run(net,train,False,97200+index);te=receiver_run(net,test,True,97200+index)
    dref=np.stack([tr['D'][:,j::3].mean(axis=1) for j in range(3)],axis=1)
    dpre=np.stack([tr['preD'][j::3].mean(axis=0) for j in range(3)])
    delta=np.repeat(te['delta'][:,None],3,axis=1);pred=np.empty((90,len(delta),3));qidx=np.arange(len(delta))%3
    for k in range(90):
        ds=[dref[k,qidx],dpre[qidx],dpre[qidx]]
        for j in range(3):
            dr=delta[:,j]*ds[j];pred[k,:,j]=dr@net.output
            rec=dr@net.W.T if j<2 else np.zeros_like(dr)
            delta[:,j]+=net.alpha*(rec-delta[:,j])
    actual=te['y'][310:400,:,0,0]-te['y'][310:400,:,0,1]
    score=[]
    for j in range(3):
        score.append(dict(rmse=float(np.sqrt(np.mean((pred[:,:,j]-actual)**2))),r2=float(1-np.sum((pred[:,:,j]-actual)**2)/np.sum((actual-actual.mean())**2)),
         early20ms_rmse=float(np.sqrt(np.mean((pred[:4,:,j]-actual[:4])**2)))))
    ep=endpoints(te['y'],test);base=test['base_context'];byq={}
    for j,q in enumerate(qs):
        keep=np.arange(len(base))%3==j
        byq[str(q)]={k:boot(v[keep],base[keep],97499) for k,v in ep.items() if k!='margin'}
        byq[str(q)]['H_change']=boot(ep['H'][keep]-ep['H'][keep,0,None],base[keep],97499)
        byq[str(q)]['sensory_change']=boot(ep['sensory'][keep]-ep['sensory'][keep,0,None],base[keep],97499)
    names=['own_reconstructed','common_inward','common_random','recurrent_difference_pause']
    rec=dict(model=net.name,nbase=24,names=names,by_strength=byq,predictions=score,prediction_names=['evidence_specific_gain','presensory_gain','leak_only'],mean_shift_norm=float(te['shift_norm'].mean()))
    np.savez_compressed(OUT/f'receiver_{net.name}.npz',output_trace=te['y'],rate_norm=te['rate_norm'],alignment=te['alignment'],regimes=te['regimes'],prediction=pred,actual=actual,reference_gains=dref,presensory_gains=dpre,initial_difference=te['delta'],base=base,counts=test['counts'],**ep)
    dump(OUT/f'receiver_{net.name}.json',rec);return rec

def verify_native(path):
    a=v1.eng.IndependentNetwork(path);b=Model(path);rng=np.random.default_rng(97601)
    x=rng.normal(5,6,(8,500));u=rng.normal(size=8)
    for _ in range(20):
        xa=a.step(x,u);xb=b.step(x,u);assert np.max(abs(xa-xb))<1e-11;x=xa
    return a.name

def run(job):
    i,p=job;start=time.time();verify_native(p)
    for kind,fun in [('source',source),('receiver',receiver)]:
        dest=OUT/f'{kind}_{v1.eng.IndependentNetwork(p).name}.json'
        if not dest.exists():fun(i,p)
        print(kind,p.name,p.parent.name,round(time.time()-start,1),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    paths=sorted((v1.R3/'data/hosford_history_rnn').glob('*/*.db'))
    dump(ROOT/'execution.json',dict(protocol_sha256=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest(),code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),models={str(p.relative_to(v1.R3)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(args.workers) as pool:list(pool.map(run,enumerate(paths)))
    dump(OUT/'completion.json',dict(source=len(list(OUT.glob('source_*.json'))),receiver=len(list(OUT.glob('receiver_*.json')))))
