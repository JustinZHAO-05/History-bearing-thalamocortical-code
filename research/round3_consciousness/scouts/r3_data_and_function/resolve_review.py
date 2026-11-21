#!/usr/bin/env python3
"""Fixed sensitivity analyses addressing two independent review concerns."""
import json
from datetime import datetime, timezone
import numpy as np
from scipy.interpolate import BSpline
from analyze_biology import BASE, SOURCES, read_source
from analyze_trials import features
from analyze_mechanism_v2 import fit_group, summarize

KNOTS=np.r_[np.repeat(-1.,4),[-.5,0.,.5],np.repeat(1.,4)]

def design(sessions,spline=False):
    blocks=[];ys=[];sid=[]
    for i,session in enumerate(sessions):
        f=features(session["z"])
        v=(~f["history_trial"])&np.isfinite(f["h"])&f["valid_target"]&f["valid_choice"]
        s,h,L=f["s"][v],f["h"][v],f["L"][v]
        if spline:
            B=BSpline.design_matrix(s,KNOTS,3).toarray()[:,:-1]
            B0=BSpline.design_matrix(np.array([0.]),KNOTS,3).toarray()[:,:-1]
            B=B-B0
        else:
            B=s[:,None]
        # History-by-laser is the final coefficient in both variants.
        blocks.append(np.column_stack([B,h,L,B*L[:,None],h*L]))
        ys.append(f["choice"][v]);sid.extend([i]*int(v.sum()))
    ni=len(sessions)
    return np.column_stack([np.eye(ni)[np.array(sid)],np.vstack(blocks)]),np.concatenate(ys),ni


def main():
    rng=np.random.default_rng(20261114)
    out={"created_utc":datetime.now(timezone.utc).isoformat(),"analysis":"frozen_v2_1_independent_review_response",
         "spline_degree":3,"spline_internal_knots":[-.5,0,.5],"spline_boundaries":[-1,1],"groups":{}}
    for label,source in SOURCES.items():
        sessions,*_=read_source(label,source)
        audit=[]
        for L in (0,1):
            row={"laser":L,"all_trials":0,"nonbinary_or_nan_choices":0,"feedback_identity_disagreements":0,
                 "all_regular_after_five":0,"regular_binary_target_choice":0,"regular_coherent_current_model":0}
            for session in sessions:
                f=features(session["z"]);v=f["L"]==L
                row["all_trials"]+=int(v.sum())
                row["nonbinary_or_nan_choices"]+=int(np.sum(v&(~f["valid_choice"])))
                row["feedback_identity_disagreements"]+=int(np.sum(v&(~f["coherent_feedback"])))
                regular=v&(~f["history_trial"])&np.isfinite(f["h"])
                row["all_regular_after_five"]+=int(regular.sum())
                eligible=regular&f["valid_choice"]&f["valid_target"]
                row["regular_binary_target_choice"]+=int(eligible.sum())
                row["regular_coherent_current_model"]+=int(np.sum(eligible&f["coherent_feedback"]))
            audit.append(row)
        variants={}
        for spline in (False,True):
            designs=[(m,*design([s for s in sessions if s["mouse"]==m],spline))
                     for m in sorted(set(s["mouse"] for s in sessions))]
            fits=fit_group(designs)
            summary=summarize([f["coefficients"][-1] for f in fits],rng)
            name="fixed_spline_all_binary" if spline else "linear_all_binary"
            variants[name]={"fits":fits,"history_by_laser":summary}
            print(label,name,"history interaction",summary["mean"],summary["bootstrap95_mean"],summary["wilcoxon_exact_two_sided_p"],"stable",all(f["stable"] for f in fits))
        out["groups"][label]={"selection_by_laser":audit,"models":variants}
        print(label,"selection",audit)
    (BASE/"results/biology_v2_1_review_results.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":main()
