#!/usr/bin/env python3
"""Second, predeclared carryover sensitivity: temporal adjustment and separation."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from scipy import optimize,special
import biology_carryover_audit as audit
from analyze_trials import fit_logit

HERE=Path(__file__).resolve().parent

def firth_objective(b,X,y):
    eta=X@b;p=special.expit(eta);w=np.maximum(p*(1-p),1e-20);I=X.T@(w[:,None]*X)
    sign,ld=np.linalg.slogdet(I)
    if sign<=0:return 1e100,np.full_like(b,1e10)
    inv=np.linalg.solve(I,np.eye(I.shape[0]));leverage=w*np.sum((X@inv)*X,axis=1)
    value=np.logaddexp(0,eta).sum()-y@eta-.5*ld
    grad=X.T@(p-y-leverage*(.5-p))
    return float(value),grad

def firth(X,y):
    rank=int(np.linalg.matrix_rank(X));npar=X.shape[1]
    if rank<npar:return {'stable':False,'rank':rank,'n_parameters':npar,'reason':'rank deficient'}
    result=optimize.minimize(lambda b:firth_objective(b,X,y),np.zeros(npar),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':2000})
    val,grad=firth_objective(result.x,X,y);g=float(np.max(abs(grad)))
    return dict(coefficients=result.x.tolist(),objective=val,max_abs_gradient=g,converged=bool(result.success),message=str(result.message),rank=rank,n_parameters=npar,n_trials=len(y),stable=bool(g<1e-4 and np.isfinite(result.x).all() and np.max(abs(result.x))<50),n_iterations=result.nit)


def gradient_check():
    rng=np.random.default_rng(872901);X=np.column_stack([np.ones(80),rng.normal(size=(80,6))]);y=rng.integers(2,size=80);b=rng.normal(0,.2,7)
    val,grad=firth_objective(b,X,y);eps=1e-6;approx=[]
    for j in range(len(b)):
        a=b.copy();c=b.copy();a[j]+=eps;c[j]-=eps;approx.append((firth_objective(a,X,y)[0]-firth_objective(c,X,y)[0])/(2*eps))
    return dict(seed=872901,finite_difference_step=eps,max_abs_gradient_difference=float(np.max(abs(grad-approx))))


def time_design(rr):
    X,y,ni,names,centers,rows=audit.design(rr,'lag_adjusted');sessions=sorted(set(r['session_index'] for r in rows));extra=[];en=[]
    for ss in sessions:
        ix=np.array([r['session_index']==ss for r in rows]);t=np.array([r['trial'] for r in rows],float)
        t=t/ max(r['trial'] for r in rr if r['session_index']==ss)*2-1
        for power in (1,2):
            p=t**power;p=p-p[ix].mean()
            for key in ('bias','s','h'):
                fac=np.ones(len(rows)) if key=='bias' else np.array([r[key] for r in rows])
                extra.append(p*ix*fac);en.append(f'session{ss}_time{power}_{key}')
    return np.column_stack([X,np.column_stack(extra)]),y,ni,names+en,centers,rows


def assignment(sessions):
    sr=[]
    for ss in sessions:
        a=ss['z'][4];changes=np.r_[0,np.flatnonzero(np.diff(a)!=0)+1,len(a)];length=np.diff(changes);levels=a[changes[:-1]]
        ac=[float(np.corrcoef(a[:-k],a[k:])[0,1]) for k in range(1,6)]
        transitions=[]
        for prev in (0,1):
            q=a[:-1]==prev;transitions.append(float(np.mean(a[1:][q])))
        sr.append(dict(mouse=ss['mouse'],session=ss['name'],n_trials=len(a),laser_rate=float(np.mean(a)),lag_autocorrelation_1_to_5=ac,prob_laser_given_previous_0_1=transitions,laser_run_max=int(max(length[levels==1])),nonlaser_run_max=int(max(length[levels==0])),first120=''.join(str(int(x)) for x in a[:120])))
    return sr


def main():
    check=gradient_check();assert check['max_abs_gradient_difference']<1e-6
    out=dict(protocol_sha256=hashlib.sha256((HERE/'biology_carryover_protocol_v2.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),gradient_check=check,groups={},seed=20261123,bootstrap_resamples=100000)
    rng=np.random.default_rng(20261123)
    for label,src in audit.SOURCES.items():
        sessions,*_=audit.read_source(label,src);rr=audit.make_rows(sessions);mice=sorted(set(r['mouse'] for r in rr))
        dd=[(m,time_design([r for r in rr if r['mouse']==m])) for m in mice]
        fits=[fit_logit(X,y,ni) for m,(X,y,ni,nn,cc,rows) in dd]
        pen=any(not f['stable'] for f in fits)
        if pen:fits=[fit_logit(X,y,ni,1e-4) for m,(X,y,ni,nn,cc,rows) in dd]
        rec=[]
        for (m,(X,y,ni,names,centers,rows)),f in zip(dd,fits):
            rec.append(dict(mouse=m,n_trials=len(y),n_sessions=ni,coefficients=dict(zip(names,f['coefficients'][ni:])),session_intercepts=f['coefficients'][:ni],fit={k:v for k,v in f.items() if k!='coefficients'}))
        adjusted=dict(all_mice_penalized=pen,all_stable=all(f['stable'] for f in fits),mice=rec,summary={key:audit.summary([r['coefficients'][key] for r in rec],rng) for key in ('history','laser_history','stimulus','laser_stimulus')})
        print(label,'lag_time',adjusted['summary']['laser_history'],'stable',adjusted['all_stable'],flush=True)
        subset=[]
        for m in mice:
            X,y,ni,names,centers,rows=audit.design([r for r in rr if r['mouse']==m],'no_prior_laser');f=firth(X,y)
            subset.append(dict(mouse=m,n_trials=len(y),n_sessions=ni,n_laser=sum(r['L']==1 for r in rows),n_no_laser=sum(r['L']==0 for r in rows),coefficients=dict(zip(names,f['coefficients'][ni:])) if 'coefficients' in f else None,session_intercepts=f.get('coefficients',[])[:ni],fit={k:v for k,v in f.items() if k!='coefficients'}))
        sf=dict(mice=subset,all_stable=all(f['fit']['stable'] for f in subset))
        if all(r['coefficients'] is not None for r in subset):sf['summary']={key:audit.summary([r['coefficients'][key] for r in subset],rng) for key in ('history','laser_history','stimulus','laser_stimulus')}
        print(label,'subset_firth',sf.get('summary',{}).get('laser_history'), 'stable',sf['all_stable'],flush=True)
        out['groups'][label]=dict(lag_time_adjusted=adjusted,no_prior_laser_firth=sf,assignment_sessions=assignment(sessions))
        (HERE/'biology_carryover_v2_results.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
