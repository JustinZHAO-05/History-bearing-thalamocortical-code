#!/usr/bin/env python3
"""Cross future stimulus mixtures with trial position, using the root evaluator.

This exploratory stress test includes explicitly out-of-training conditions.
It reports native outputs and state contrasts, never accuracy on those cells.
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
from pathlib import Path
import argparse,csv,hashlib,itertools,json,sys,time
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np

HERE=Path(__file__).resolve().parent
R3=HERE.parents[1]
sys.path.insert(0,str(R3/"code"))
from rnn_core import Model,models
from sequential_access import evaluate,contexts as original_contexts

QS=(2,10,18)
CONDITIONS=[
    dict(name="natural",state="intact",source="natural",duration=0),
    dict(name="own100",state="intact",source="own",duration=100),
    dict(name="opposite100",state="intact",source="opposite",duration=100),
]


def make_contexts(n=12,seed=83031):
    rng=np.random.default_rng(seed)
    u=np.zeros((n*9,2,2200))
    counts=np.zeros((n*9,2,6),int)
    x0=np.empty((n*9,500))
    oldq=np.r_[np.arange(10),np.arange(11,21)]
    fifth=[];sixth=[]
    for j in range(n):
        while True:
            c=rng.choice(oldq,4)
            total=np.sum(c-10)
            if abs(total)>10:break
        if total<0:c=20-c
        for t,q in enumerate(c):
            pp=np.r_[np.ones(q),-np.ones(20-q)]
            rng.shuffle(pp)
            a=t*400+50
            for z in range(9):
                u[j*9+z,0,a:a+60]=np.repeat(pp,3)
                u[j*9+z,1,a:a+60]=np.repeat(-pp,3)
                counts[j*9+z,:,t]=[q,20-q]
        # Exactly the same two permutations and x0 draw as original_contexts;
        # its second permutation selects the ten positive sixth-trial pips.
        order5=rng.permutation(20)
        order6=rng.permutation(20)
        init=rng.normal(0,1/500,500)
        for z,(q5,q6) in enumerate(itertools.product(QS,QS)):
            pp5=-np.ones(20);pp5[order5[:q5]]=1
            pp6=-np.ones(20);pp6[order6[:q6]]=1
            u[j*9+z,:,1650:1710]=np.repeat(pp5,3)
            u[j*9+z,:,2050:2110]=np.repeat(pp6,3)
            counts[j*9+z,:,4]=q5;counts[j*9+z,:,5]=q6
            x0[j*9+z]=init;fifth.append(q5);sixth.append(q6)
    assert np.all(np.sum(counts[:,0,:4]-10,axis=1)>10)
    assert np.all(np.sum(counts[:,1,:4]-10,axis=1)<-10)
    assert np.all(counts[:,0,:5].mean(-1)>10)
    assert np.all(counts[:,1,:5].mean(-1)<10)
    for j in range(n):
        assert np.array_equal(u[j*9:(j+1)*9,:,:1650],np.broadcast_to(u[j*9,:,:1650],(9,2,1650)))
        assert np.array_equal(x0[j*9:(j+1)*9],np.broadcast_to(x0[j*9],(9,500)))
        for i5 in range(3):
            block=u[j*9+i5*3:j*9+i5*3+3,:,:2050]
            assert np.array_equal(block,np.broadcast_to(block[0],block.shape))
    ctx=dict(u=u,x0=x0,counts=counts,start=np.array(1400),end=np.array(2200),
             base_context=np.repeat(np.arange(n),9),seed=np.array(seed),
             q5=np.array(fifth),q6=np.array(sixth),
             legal_original_training_sequence=np.array([(q5!=10 and q6==10) for q5,q6 in zip(fifth,sixth)]))
    # Independent generation identity on the two overlapping lawful cells.
    ref=original_contexts(n,seed)
    for q5,ref_offset in ((2,1),(18,2)):
        ix=np.flatnonzero((ctx["q5"]==q5)&(ctx["q6"]==10))
        ri=np.arange(n)*4+ref_offset
        assert np.array_equal(ctx["u"][ix],ref["u"][ri])
        assert np.array_equal(ctx["x0"][ix],ref["x0"][ri])
        assert np.array_equal(ctx["counts"][ix],ref["counts"][ri])
    return ctx


def one_model(job):
    path,context_path,outdir=job
    start=time.time();model=Model(path)
    ctx=dict(np.load(context_path));nbase=len(ctx["u"])//9
    result=evaluate(model,ctx,CONDITIONS)
    checks=result.pop("checks")
    margin=result["margin"].reshape(nbase,3,3,3,2,2)
    norms=result["history_norm"].reshape(7,nbase,3,3,3,3)
    checks["first_response_future_q6_invariance_max_error"]=float(np.max(np.ptp(margin[...,0],axis=2)))
    checks["pre_fifth_state_norm_future_invariance_max_error"]=float(max(np.max(np.ptp(norms[k].reshape(nbase,9,3,3),axis=1)) for k in (0,1)))
    checks["pre_sixth_state_norm_q6_invariance_max_error"]=float(np.max(np.ptp(norms[4],axis=2)))
    for key in ("first_response_future_q6_invariance_max_error","pre_fifth_state_norm_future_invariance_max_error","pre_sixth_state_norm_q6_invariance_max_error"):
        assert checks[key]<1e-9,(key,checks[key])
    np.savez_compressed(Path(outdir)/(model.name+".npz"),**result)
    rows=[]
    for q5,q6 in itertools.product(QS,QS):
        ix=(ctx["q5"]==q5)&(ctx["q6"]==q6)
        for ci,condition in enumerate(CONDITIONS):
            for response,position in enumerate((5,6)):
                y=result["margin"][ix,ci,:,response]
                lo=result["response_min"][ix,ci,:,response]
                hi=result["response_max"][ix,ci,:,response]
                H=(y[:,0]-y[:,1])/2
                snapshot=3 if response==0 else 6
                regional_norm=result["history_norm"][snapshot,ix,ci]
                total_norm=np.linalg.norm(regional_norm,axis=1)
                row=dict(model=model.name,condition=condition["name"],q5=q5,q6=q6,response_position=position,
                         original_training_lawful_sequence=bool(q5!=10 and q6==10),
                         stimulus_at_position_lawful=bool((position==5 and q5!=10) or (position==6 and q6==10)),
                         n_base=nbase,mean_signed_history_contrast=float(H.mean()),mean_absolute_history_contrast=float(np.abs(H).mean()),
                         sd_signed_history_contrast=float(H.std(ddof=1)),minimum_history_contrast=float(H.min()),maximum_history_contrast=float(H.max()),
                         mean_output_positive_history=float(y[:,0].mean()),mean_output_negative_history=float(y[:,1].mean()),
                         mean_response_min_positive_history=float(lo[:,0].mean()),mean_response_max_positive_history=float(hi[:,0].mean()),
                         mean_response_min_negative_history=float(lo[:,1].mean()),mean_response_max_negative_history=float(hi[:,1].mean()),
                         mean_end_response_history_norm=float(total_norm.mean()),
                         mean_end_response_ppc_history_norm=float(regional_norm[:,0].mean()),
                         mean_end_response_pul_history_norm=float(regional_norm[:,1].mean()),
                         mean_end_response_trn_history_norm=float(regional_norm[:,2].mean()))
                rows.append(row)
    # Checkpoint-level summaries; no p-values across released model files.
    H=(margin[:,:,:,:,0,:]-margin[:,:,:,:,1,:])/2
    contrasts=[]
    for ci,condition in enumerate(CONDITIONS):
        # Sixth future cannot affect fifth output, so use q6=10 once.
        h5=H[:,:,1,ci,0]
        d5=h5[:,1]-(h5[:,0]+h5[:,2])/2
        a5=np.abs(h5[:,1])-(np.abs(h5[:,0])+np.abs(h5[:,2]))/2
        entry=dict(condition=condition["name"],position5_signed_ambiguous_minus_strong=float(d5.mean()),
                   position5_absolute_ambiguous_minus_strong=float(a5.mean()),
                   position5_mean_signed_by_q5=h5.mean(axis=0).tolist(),
                   position5_mean_absolute_by_q5=np.abs(h5).mean(axis=0).tolist(),
                   position6_for_fixed_q5=[])
        for i5,q5 in enumerate(QS):
            h6=H[:,i5,:,ci,1]
            d6=h6[:,1]-(h6[:,0]+h6[:,2])/2
            a6=np.abs(h6[:,1])-(np.abs(h6[:,0])+np.abs(h6[:,2]))/2
            entry["position6_for_fixed_q5"].append(dict(q5=q5,signed_ambiguous_minus_strong=float(d6.mean()),
                    absolute_ambiguous_minus_strong=float(a6.mean()),mean_signed_by_q6=h6.mean(axis=0).tolist(),
                    mean_absolute_by_q6=np.abs(h6).mean(axis=0).tolist()))
        contrasts.append(entry)
    entry=dict(model=model.name,model_sha256=model.sha256,seconds=time.time()-start,checks=checks,cell_rows=rows,contrasts=contrasts)
    (Path(outdir)/(model.name+".json")).write_text(json.dumps(entry,indent=2)+"\n")
    return entry


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--workers",type=int,default=2);a=ap.parse_args()
    assert 1<=a.workers<=2
    out=R3/"results/position_mixture_stress";out.mkdir(parents=True,exist_ok=True)
    context=make_contexts();cp=out/"contexts.npz";np.savez_compressed(cp,**context)
    (out/"conditions.json").write_text(json.dumps(CONDITIONS,indent=2)+"\n")
    paths=models();assert len(paths)==9
    code_files=[Path(__file__),R3/"code/sequential_access.py",R3/"code/rnn_core.py",R3/"code/cognitive_causal.py"]
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),status="running",n_base=12,seed=83031,
                  n_crossed_contexts=108,n_pair_members=2,n_conditions=3,n_models=9,workers=a.workers,blas_threads_per_worker=1,
                  intervention_steps=[1400,1500],response_windows=[[1720,1800],[2120,2200]],
                  q5=list(QS),q6=list(QS),
                  code_sha256={str(p.relative_to(R3)):hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files},
                  contexts_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                  semantic_scope="Exploratory crossed position/mixture stress test; fifth ambiguous or sixth nonambiguous cells are OOD. No accuracy claims for these cells.")
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    tic=time.time();summary=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one_model,(str(p),str(cp),str(out))) for p in paths]
        for future in as_completed(futures):
            entry=future.result();summary.append(entry)
            ordered=sorted(summary,key=lambda x:x["model"])
            (out/"summary.json").write_text(json.dumps(ordered,indent=2)+"\n")
            cell_rows=[r for e in ordered for r in e["cell_rows"]]
            with (out/"cell_summary.csv").open("w",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(cell_rows[0]));w.writeheader();w.writerows(cell_rows)
            print(json.dumps(dict(model=entry["model"],seconds=entry["seconds"],checks=entry["checks"],contrasts=entry["contrasts"])),flush=True)
    manifest.update(status="complete",seconds=time.time()-tic,completed_utc=datetime.now(timezone.utc).isoformat())
    (out/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":main()
