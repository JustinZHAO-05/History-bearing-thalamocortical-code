#!/usr/bin/env python3
"""V2 frozen explanatory model comparison and condition-dependent history use."""
from __future__ import annotations

import itertools
import json
from datetime import datetime, timezone

import numpy as np
from scipy import special

from analyze_biology import BASE, SOURCES, read_source, paired_stats
from analyze_trials import features, model_design, fit_logit


def summarize(values, rng):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if not finite.any():
        return {"n_mice":0,"excluded_nonfinite":int(len(values))}
    v = values[finite]
    n = len(v)
    return {**paired_stats(v, rng.integers(n, size=(100000,n))),
            "excluded_nonfinite":int((~finite).sum())}


def fit_group(designs):
    fitted = [fit_logit(X,y,ni) for m,X,y,ni in designs]
    penalty = any(not f["stable"] for f in fitted)
    if penalty:
        fitted = [fit_logit(X,y,ni,1e-4) for m,X,y,ni in designs]
    for (m,X,y,ni),fit in zip(designs,fitted):
        eta = X@np.array(fit["coefficients"])
        nll = float(np.logaddexp(0,eta).sum()-y@eta)
        fit["unpenalized_nll"] = nll
        fit["AIC_descriptive"] = 2*nll + 2*X.shape[1]
        fit["mouse"] = m
        fit["n_sessions"] = ni
    return fitted


def current_metadata(sessions):
    combined = {k:[] for k in ("s","h","L","target","correct")}
    for session in sessions:
        f=features(session["z"])
        v=(~f["history_trial"])&np.isfinite(f["h"])&f["valid_target"]&f["valid_choice"]&f["coherent_feedback"]
        for key in combined:
            combined[key].append(f[key][v])
    return {k:np.concatenate(v) for k,v in combined.items()}


def shapley_standardized(X, ni, coefficients, metadata):
    b=np.asarray(coefficients)
    f=metadata
    s,h,target=f["s"],f["h"],f["target"]
    valid=(abs(s)>1e-12)&(abs(h)>1e-12)&(target==(s>0))
    congruent=s*h>0
    baseline=X[:,:ni]@b[:ni]+b[ni]*s+b[ni+1]*h
    increments=np.column_stack([np.full(len(s),b[ni+2]),b[ni+3]*s,b[ni+4]*h])
    def pcorrect(eta):
        p=special.expit(eta)
        return np.where(target==1,p,1-p)
    contributions=np.zeros_like(increments)
    for order in itertools.permutations(range(3)):
        eta=baseline.copy();p0=pcorrect(eta)
        for k in order:
            eta=eta+increments[:,k];p1=pcorrect(eta)
            contributions[:,k]+=(p1-p0)/6
            p0=p1
    net=pcorrect(baseline+increments.sum(axis=1))-pcorrect(baseline)
    assert np.max(abs(contributions.sum(axis=1)-net))<1e-12
    out={"n_standardized_trials":int(valid.sum()),"components":{}}
    for k,name in enumerate(("side_bias","sensory_gain","history_bias")):
        d0=float(contributions[valid&congruent,k].mean())
        d1=float(contributions[valid&(~congruent),k].mean())
        out["components"][name]={"congruent":d0,"conflict":d1,"interaction":d1-d0}
    out["full"]={key:sum(v[key] for v in out["components"].values())
                 for key in ("congruent","conflict","interaction")}
    # Explicitly change only the sensory coefficient from its no-laser value.
    only_s=pcorrect(baseline+increments[:,1])-pcorrect(baseline)
    out["sensory_change_alone"]={"congruent":float(only_s[valid&congruent].mean()),
                                 "conflict":float(only_s[valid&(~congruent)].mean())}
    out["sensory_change_alone"]["interaction"]=out["sensory_change_alone"]["conflict"]-out["sensory_change_alone"]["congruent"]
    return out


def conditional_design(sessions, categorical=False):
    blocks=[];choices=[];sid=[]
    for i,session in enumerate(sessions):
        f=features(session["z"])
        v=np.isfinite(f["h"])&f["valid_choice"]&f["valid_target"]&np.isfinite(f["s"])
        s,h,L=f["s"][v],f["h"][v],f["L"][v]
        a=(abs(s)<1e-12).astype(float) if categorical else abs(s)
        blocks.append(np.column_stack([s,h,a,a*h,L,s*L,h*L,a*L,a*h*L]))
        choices.append(f["choice"][v]);sid.extend([i]*int(v.sum()))
    ni=len(sessions)
    return np.column_stack([np.eye(ni)[np.asarray(sid)],np.vstack(blocks)]),np.concatenate(choices),ni


def magnitude_matched(sessions):
    all_results=[]
    for m in sorted(set(s["mouse"] for s in sessions)):
        counts=np.zeros((5,3,4),int);successes=np.zeros_like(counts)
        for session in [s for s in sessions if s["mouse"]==m]:
            f=features(session["z"]);s,h,L=f["s"],f["h"],f["L"]
            valid=(~f["history_trial"])&(abs(s)>1e-12)&(abs(h)>1e-12)&np.isfinite(h)&f["valid_choice"]&(f["target"]==(s>0))
            sb=np.minimum(np.searchsorted([0,.2,.4,.6,.8,1],abs(s),side="right")-1,4)
            hb=np.minimum(np.searchsorted([0,.1,.25,1],abs(h),side="right")-1,2)
            congruent=s*h>0
            cell=2*(~congruent).astype(int)+L.astype(int)
            for t in np.flatnonzero(valid):
                counts[sb[t],hb[t],cell[t]]+=1
                successes[sb[t],hb[t],cell[t]]+=int(f["correct"][t])
        keep=counts.min(axis=2)>=5
        if keep.any():
            n=counts[keep];k=successes[keep]
            weights=n.min(axis=1).astype(float);weights/=weights.sum()
            rates=(weights[:,None]*(k/n)).sum(axis=0)
            out={"mouse":m,"qualified_bins":int(keep.sum()),"all_eligible_trials":int(counts.sum()),
                 "trials_in_qualified_bins":int(counts[keep].sum()),"minimum_cell_count":int(counts[keep].min()),
                 "rates_congruent_NL_Laser_conflict_NL_Laser":rates.tolist(),
                 "delta_congruent":float(rates[1]-rates[0]),"delta_conflict":float(rates[3]-rates[2]),
                 "interaction":float((rates[3]-rates[2])-(rates[1]-rates[0]))}
        else:
            out={"mouse":m,"qualified_bins":0,"all_eligible_trials":int(counts.sum()),"trials_in_qualified_bins":0,
                 "delta_congruent":float("nan"),"delta_conflict":float("nan"),"interaction":float("nan")}
        all_results.append(out)
    return all_results


def main():
    rng=np.random.default_rng(20261113)
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"analysis":"frozen_v2_explanatory_iteration","groups":{}}
    for label,source in SOURCES.items():
        sessions,*_=read_source(label,source)
        grouped=[(m,[s for s in sessions if s["mouse"]==m]) for m in sorted(set(s["mouse"] for s in sessions))]
        base_designs=[(m,*model_design(ss)) for m,ss in grouped]
        nested={}
        for name,drop in (("full",None),("sensory_only",4),("history_only",3)):
            designs=[]
            for m,X,y,ni in base_designs:
                XX=X if drop is None else np.delete(X,ni+drop,axis=1)
                designs.append((m,XX,y,ni))
            nested[name]=fit_group(designs)
        comparisons=[]
        for i,(m,ss) in enumerate(grouped):
            full=nested["full"][i]
            row={"mouse":m,"n_trials":full["n_trials"],"n_sessions":full["n_sessions"]}
            for name in ("full","sensory_only","history_only"):
                f=nested[name][i]
                row[name+"_nll"]=f["unpenalized_nll"]
                row[name+"_AIC"]=f["AIC_descriptive"]
                if name!="full":
                    row[name+"_deviance_loss"]=2*(f["unpenalized_nll"]-full["unpenalized_nll"])
                    row[name+"_delta_AIC_vs_full"]=f["AIC_descriptive"]-full["AIC_descriptive"]
            comparisons.append(row)
        shapley=[]
        for (m,ss),(m2,X,y,ni),fit in zip(grouped,base_designs,nested["full"]):
            assert m==m2
            shapley.append({"mouse":m,**shapley_standardized(X,ni,fit["coefficients"],current_metadata(ss))})
        shapley_summary={}
        for component in ("side_bias","sensory_gain","history_bias","full","sensory_change_alone"):
            def get(row,key):
                return row[component][key] if component in ("full","sensory_change_alone") else row["components"][component][key]
            shapley_summary[component]={k:summarize([get(r,k) for r in shapley],rng) for k in ("congruent","conflict","interaction")}
        conditional={}
        for categorical in (False,True):
            name="zero_evidence_category" if categorical else "continuous_evidence_magnitude"
            designs=[(m,*conditional_design(ss,categorical)) for m,ss in grouped]
            fits=fit_group(designs)
            values={}
            for a in ([0,1] if categorical else [0,.5,1]):
                values[str(a)]=[]
                for fit in fits:
                    ni=fit["n_sessions"];b=fit["coefficients"]
                    values[str(a)].append(b[ni+6]+a*b[ni+8])
            conditional[name]={"fits":fits,"history_laser_change":{a:summarize(v,rng) for a,v in values.items()},
                                "context_laser_history_coefficient":summarize([f["coefficients"][f["n_sessions"]+8] for f in fits],rng)}
        mm=magnitude_matched(sessions)
        # Audit the temporal interpretation using an independent target identity.
        audit_rows=[]
        ambiguous=[]
        for m,ss in grouped:
            target_n=target_match=ties=0
            for session in ss:
                f=features(session["z"])
                valid=f["history_trial"]&np.isfinite(f["h"])
                ties+=int(np.sum(valid&(abs(f["h"])<1e-12)))
                valid&=abs(f["h"])>=1e-12
                target_n+=int(valid.sum());target_match+=int(np.sum(f["target"][valid]==(f["h"][valid]>0)))
            audit_rows.append({"mouse":m,"non_tied_history_targets":target_n,"target_matches_preceding_five_stimuli":target_match,"history_ties":ties,
                               "source_first_session":ss[0]["name"],"source_last_session":ss[-1]["name"]})
            for flag in (False,True):
                for laser in (0,1):
                    n=k=0
                    for session in ss:
                        f=features(session["z"])
                        v=(abs(f["s"])<1e-12)&(f["history_trial"]==flag)&(f["L"]==laser)&np.isfinite(f["h"])&(abs(f["h"])>1e-12)&f["valid_choice"]
                        n+=int(v.sum());k+=int(np.sum(f["choice"][v]==(f["h"][v]>0)))
                    ambiguous.append({"mouse":m,"flagged_history":flag,"laser":laser,"n":n,"history_congruent_choices":k,"rate":k/n if n else None})
        result["groups"][label]={"nested_current_choice_models":nested,"nested_comparisons":comparisons,
            "nested_summary":{name:{"summed_deviance_loss":float(sum(r[name+"_deviance_loss"] for r in comparisons)),
                                     "mice_AIC_favors_full":int(sum(r[name+"_delta_AIC_vs_full"]>0 for r in comparisons)),
                                     "delta_AIC_vs_full_by_mouse":[r[name+"_delta_AIC_vs_full"] for r in comparisons]}
                              for name in ("sensory_only","history_only")},
            "shapley_mice":shapley,"shapley_summary":shapley_summary,
            "conditional_history_use":conditional,"magnitude_matched_mice":mm,
            "magnitude_matched_summary":{k:summarize([r[k] for r in mm],rng) for k in ("delta_congruent","delta_conflict","interaction")},
            "history_target_chronology_audit":audit_rows,"ambiguous_choices_descriptive":ambiguous}
        print(label,"nested",result["groups"][label]["nested_summary"])
        for c in ("sensory_gain","history_bias","full"):
            q=shapley_summary[c]["interaction"]
            print("Shapley interaction",c,q["mean"],q["bootstrap95_mean"])
        print("matched",result["groups"][label]["magnitude_matched_summary"]["interaction"])
        print("conditional continuous",{k:(v["mean"],v["bootstrap95_mean"]) for k,v in conditional["continuous_evidence_magnitude"]["history_laser_change"].items()})
        print("conditional category",{k:(v["mean"],v["bootstrap95_mean"]) for k,v in conditional["zero_evidence_category"]["history_laser_change"].items()})
    (BASE/"results/biology_v2_mechanism_results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
