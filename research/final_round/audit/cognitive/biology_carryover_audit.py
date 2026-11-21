#!/usr/bin/env python3
"""Frozen supplementary retrospective analysis; read biology_carryover_protocol.md."""
from pathlib import Path
import sys, json, hashlib, csv
from datetime import datetime, timezone
import numpy as np
from scipy import stats

HERE=Path(__file__).resolve().parent
R3=HERE.parents[2]/'round3_consciousness'
OLD=R3/'scouts/r3_data_and_function'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(OLD/'pydeps'))
import h5py
from analyze_trials import fit_logit
from analyze_biology import read_source, SOURCES


def lag(a,k):
    return np.r_[np.full(k,np.nan),a[:-k]]


def make_rows(sessions):
    rr=[]
    for j,ss in enumerate(sessions):
        z=ss['z'];s=2*z[3]-1;n=len(s)
        hs=np.stack([lag(s,k) for k in range(1,6)])
        ls=np.stack([lag(z[4],k) for k in range(1,6)])
        pc=lag(2*z[2]-1,1);pf=lag(2*(z[0]==1)-1,1)
        for t in range(n):
            h=float(np.mean(hs[:,t]))
            r=dict(mouse=int(ss['mouse']),session=ss['name'],session_index=j,trial=t,s=float(s[t]),h=h,L=float(z[4,t]),target=float(z[1,t]),choice=float(z[2,t]),feedback=float(z[0,t]),history_trial=bool(z[5,t]==1 and np.isclose(z[3,t],.5)),previous_choice=float(pc[t]),previous_feedback=float(pf[t]),p1=float(ls[0,t]),p25=float(np.mean(ls[1:,t])),prior5=float(np.mean(ls[:,t])),no_prior_laser=bool(np.all(ls[:,t]==0)))
            r['regular_eligible']=bool(not r['history_trial'] and np.isfinite(h) and r['target'] in (0,1) and r['choice'] in (0,1) and r['L'] in (0,1) and r['previous_choice'] in (-1,1) and np.all(np.isin(ls[:,t],(0,1))))
            rr.append(r)
    return rr


def design(rr,mode):
    rr=[r for r in rr if r['regular_eligible'] and (mode!='no_prior_laser' or r['no_prior_laser'])]
    sessions=sorted(set(r['session_index'] for r in rr));sm={s:j for j,s in enumerate(sessions)}
    a=lambda k:np.array([r[k] for r in rr],float)
    s,h,L=a('s'),a('h'),a('L');X=[s,h,L,s*L,h*L]
    names=['stimulus','history','laser_bias','laser_stimulus','laser_history']
    centers={}
    if mode=='lag_adjusted':
        for key in ('p1','p25'):
            p=a(key);centers[key]=float(p.mean());p=p-p.mean()
            for suffix,fac in [('',np.ones_like(s)),('_stimulus',s),('_history',h),('_laser',L),('_laser_stimulus',s*L),('_laser_history',h*L)]:
                X.append(p*fac);names.append(key+suffix)
        for key in ('previous_choice','previous_feedback'):
            p=a(key);centers[key]=float(p.mean());p=p-p.mean()
            X.extend([p,p*L]);names.extend([key,key+'_laser'])
    I=np.zeros((len(rr),len(sessions)))
    I[np.arange(len(rr)),[sm[r['session_index']] for r in rr]]=1
    return np.column_stack([I,np.column_stack(X)]),a('choice'),len(sessions),names,centers,rr


def summary(vals,rng):
    vals=np.asarray(vals);n=len(vals);boot=rng.integers(n,size=(100000,n))
    return dict(n_mice=n,mean=float(vals.mean()),median=float(np.median(vals)),bootstrap95_mean=np.quantile(vals[boot].mean(1),[.025,.975]).tolist(),values=vals.tolist(),wilcoxon_exact_two_sided_p=float(stats.wilcoxon(vals,method='exact').pvalue),n_negative=int((vals<0).sum()),n_positive=int((vals>0).sum()))


def audit_cache(sessions,perf,pmouse):
    out=[];errors=[]
    for ss,ref in zip(sessions,perf):
        z=ss['z'];catch=(z[5]==1)&np.isclose(z[3],.5);L=z[4]==1;ok=z[0]==1
        masks=[~catch&~L,~catch&L,catch&~L,catch&L]
        vals=np.array([ok[m].mean() if m.any() else np.nan for m in masks])
        assert np.array_equal(np.isnan(vals),np.isnan(ref))
        errors.append(np.nanmax(abs(vals-ref)));out.append(vals)
    out=np.stack(out);ms=np.array([s['mouse'] for s in sessions])
    mv=np.stack([np.nanmean(out[ms==m],axis=0) for m in sorted(set(ms))])
    return dict(n_sessions=len(sessions),n_mice=len(mv),n_trials=sum(s['z'].shape[1] for s in sessions),max_session_error=float(max(errors)),max_mouse_error=float(np.max(abs(mv-pmouse))))


def conflict(rr,rng):
    vals=[];records=[]
    for m in sorted(set(r['mouse'] for r in rr)):
        selected=[r for r in rr if r['mouse']==m and not r['history_trial'] and np.isfinite(r['h']) and abs(r['h'])>1e-12 and abs(r['s'])>1e-12 and r['choice'] in (0,1) and r['target']==(r['s']>0)]
        counts=[];rates=[]
        for c,L in [(True,0),(True,1),(False,0),(False,1)]:
            rows=[r for r in selected if (r['s']*r['h']>0)==c and r['L']==L]
            counts.append(len(rows));rates.append(np.mean([r['feedback']==1 for r in rows]))
        d=(rates[3]-rates[2])-(rates[1]-rates[0]);vals.append(d)
        records.append(dict(mouse=m,counts=counts,rates=rates,interaction=float(d)))
    return dict(mice=records,interaction=summary(vals,rng),total_eligible_trials=sum(sum(r['counts']) for r in records))


def main():
    rng=np.random.default_rng(20261122)
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=hashlib.sha256((HERE/'biology_carryover_protocol.md').read_bytes()).hexdigest(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),seed=20261122,bootstrap_resamples=100000,groups={})
    all_rows=[]
    for label,src in SOURCES.items():
        sessions,perf,pmouse,*_=read_source(label,src)
        raw=OLD/'source_data'/src
        rr=make_rows(sessions);mice=sorted(set(r['mouse'] for r in rr));group=dict(source=str(raw),source_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),cache_audit=audit_cache(sessions,perf,pmouse),conflict_recalculation=conflict(rr,rng),models={},assignment_balance=[])
        assert group['cache_audit']['max_session_error']<1e-12 and group['cache_audit']['max_mouse_error']<1e-12
        for r in rr:all_rows.append(dict(intervention=label,**r))
        for m in mice:
            rows=[r for r in rr if r['mouse']==m and r['regular_eligible']]
            bal=dict(mouse=m,n=len(rows),n_laser=sum(r['L']==1 for r in rows),laser_rate=float(np.mean([r['L'] for r in rows])),contrasts={})
            for key in ('s','h','p1','p25','prior5','previous_choice','previous_feedback'):
                x=np.array([r[key] for r in rows]);L=np.array([r['L'] for r in rows]);d=x[L==1].mean()-x[L==0].mean()
                bal['contrasts'][key]=dict(laser_minus_control=float(d),standardized_mean_difference=float(d/x.std(ddof=1)))
            x=np.array([abs(r['s']) for r in rows]);d=x[L==1].mean()-x[L==0].mean();bal['contrasts']['abs_stimulus']=dict(laser_minus_control=float(d),standardized_mean_difference=float(d/x.std(ddof=1)))
            group['assignment_balance'].append(bal)
        for mode in ('baseline','lag_adjusted','no_prior_laser'):
            dd=[(m,design([r for r in rr if r['mouse']==m],mode)) for m in mice]
            fits=[fit_logit(X,y,ni) for m,(X,y,ni,nn,cc,rows) in dd]
            penalty=any(not f['stable'] for f in fits)
            if penalty:fits=[fit_logit(X,y,ni,1e-4) for m,(X,y,ni,nn,cc,rows) in dd]
            rec=[]
            for (m,(X,y,ni,names,centers,rows)),f in zip(dd,fits):
                rec.append(dict(mouse=m,n_trials=len(y),n_sessions=ni,n_laser=sum(r['L']==1 for r in rows),n_no_laser=sum(r['L']==0 for r in rows),centers=centers,coefficients=dict(zip(names,f['coefficients'][ni:])),session_intercepts=f['coefficients'][:ni],fit={k:v for k,v in f.items() if k!='coefficients'}))
            sm={key:summary([r['coefficients'][key] for r in rec],rng) for key in ('history','laser_history','stimulus','laser_stimulus')}
            group['models'][mode]=dict(all_mice_penalized=penalty,all_stable=all(f['stable'] for f in fits),mice=rec,summary=sm)
            print(label,mode,sm['laser_history'], 'stable',all(f['stable'] for f in fits),flush=True)
        d=np.array(group['models']['lag_adjusted']['summary']['laser_history']['values'])-np.array(group['models']['baseline']['summary']['laser_history']['values'])
        group['adjusted_minus_baseline_history_interaction']=summary(d,rng)
        out['groups'][label]=group
        (HERE/'biology_carryover_results.json').write_text(json.dumps(out,indent=2)+'\n')
    with (HERE/'biology_derived_trials.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    print('COMPLETED',len(all_rows),'raw rows',flush=True)

if __name__=='__main__':main()
