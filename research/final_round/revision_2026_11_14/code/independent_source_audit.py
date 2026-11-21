#!/usr/bin/env python3
"""Independent numerical review of saved PUL source-space claims.
Recomputes original donor rates using separately implemented published Model.
"""
import os
for t in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[t]='1'
from pathlib import Path
import sys,json,sqlite3,pickle,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
R3=Path(os.environ.get('R3_RESEARCH_ROOT',str(ROOT.parents[1]/'round3_consciousness')))
sys.path.insert(0,str(R3/'code'))
from rnn_core import Model

def main():
    ctx=dict(np.load(ROOT/'results/context_test.npz'))
    entries=[]
    for p in sorted((R3/'data/hosford_history_rnn').glob('*/*.db')):
        net=Model(p);name=net.name;assert name==p.parent.name.replace('TBPTT_bio_','').replace('_valid','')+'_'+p.stem
        x=np.repeat(ctx['x0'][:,None,:],2,axis=1)
        for t in range(1400):x=net.step(x,ctx['u'][:,:,t])
        donor=np.empty((200,len(x),2,70))
        for t in range(200):
            donor[t]=net.rate(x)[...,400:470]
            x=net.step(x,ctx['u'][:,:,1400+t])
        mu=donor.mean(axis=2);d=donor[:,:,0]-donor[:,:,1]
        B=net.W[:400,400:470]
        db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        row=dict(db.execute('select * from currentRNNParameters order by id desc limit 1').fetchone());db.close()
        e=np.asarray(pickle.loads(row['PPCexh_idx']),int);i=np.asarray(pickle.loads(row['PPCinh_idx']),int)
        A=np.stack([B[e].sum(axis=0),B[i].sum(axis=0)])
        full=np.sum((d@B.T)**2,axis=-1)
        p1=ROOT/'results'/f'{name}.npz'
        p2=ROOT/'results/reflection_v2'/f'{name}.npz'
        with np.load(p1) as a, np.load(p2) as z:
            assert np.allclose(a['natural_source_difference'],d,atol=1e-10,rtol=1e-10)
            assert np.allclose(a['source_mean'],mu,atol=1e-10,rtol=1e-10)
            assert np.allclose(a['margin'][:,1],z['margin'][:,1],atol=1e-11,rtol=1e-11)
            c=a['source_candidate_difference']
            assert c.shape==(4,200,len(x),70)
            cap=2*np.minimum(mu,20-mu)
            vmax=float(np.max(np.abs(c)-cap))
            assert vmax<=1e-9
            exact=z['maps']
            ge=[]
            for j in range(2):
                w=exact[j]
                max_bound_violation=float(np.max(np.abs(w)-cap))
                max_ei=float(np.max(np.abs((w-d)@A.T)))
                en=np.sum((w@B.T)**2,axis=-1)
                relerr=float(np.max(np.divide(np.abs(en-full),full,out=np.zeros_like(full),where=full>1e-16)))
                abs_err=float(np.max(np.abs(en-full)))
                rel_geo=float(np.mean(np.divide(np.linalg.norm((w-d)@B.T,axis=-1),np.sqrt(full),out=np.zeros_like(full),where=full>1e-12)))
                assert max_bound_violation<1e-8 and max_ei<1e-8 and relerr<1e-9
                ge.append(dict(map=j,EI_error=max_ei,energy_relative_error=relerr,energy_absolute_error=abs_err,
                               bounds_violation=max_bound_violation,receiver_change=rel_geo))
            assert np.max(np.abs(z['output_trace'][:,:,2,0]-z['output_trace'][:,:,2,1]))<1e-9
            checks=dict(model=name,geometry=ge,natural_rate_max=float(np.max(donor)),
                matched_v1_fullown_vs_v2_max=float(np.max(np.abs(a['margin'][:,1]-z['margin'][:,1]))),
                v1_source_bounds_max_violation=vmax,
                v1_mean_source_pair_max=float(np.max(np.abs(a['output_trace'][:,:,2,0]-a['output_trace'][:,:,2,1]))))
            entries.append(checks);print(name,ge,flush=True)
    out=ROOT/'results/source_independent_audit.json'
    out.write_text(json.dumps(dict(status='PASS',n_models=len(entries),results=entries),indent=2))
if __name__=='__main__':main()
