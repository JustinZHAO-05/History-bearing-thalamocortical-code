#!/usr/bin/env python3
"""Summarize the frozen crossed stress test without fitting any parameters."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R3=Path(__file__).resolve().parents[2]
OUT=R3/"results/position_mixture_stress"
QS=[2,10,18]
COND=["natural","own100","opposite100"]


def describe(x):
    x=np.asarray(x,float)
    return dict(n_checkpoints=len(x),mean=float(x.mean()),median=float(np.median(x)),
                minimum=float(x.min()),maximum=float(x.max()),n_positive=int(np.sum(x>0)),
                n_negative=int(np.sum(x<0)),values=x.tolist())


def main():
    entries=json.loads((OUT/"summary.json").read_text());assert len(entries)==9
    context=dict(np.load(OUT/"contexts.npz"))
    nbase=len(context["u"])//9
    within=[];effects=[];ratios=[];curves={}
    max_checks={}
    for entry in entries:
        name=entry["model"]
        for k,v in entry["checks"].items():max_checks[k]=max(max_checks.get(k,0),v)
        data=np.load(OUT/(name+".npz"))
        H=((data["margin"][:,:,0]-data["margin"][:,:,1])/2).reshape(nbase,3,3,3,2)
        D=np.linalg.norm(data["history_norm"],axis=-1).reshape(7,nbase,3,3,3)
        curves[name]={}
        for ci,condition in enumerate(COND):
            for pos,past in [(5,None),(6,2),(6,10),(6,18)]:
                i5=QS.index(past) if past is not None else None
                h=H[:,:,1,ci,0] if pos==5 else H[:,i5,:,ci,1]
                d=D[3,:,:,1,ci] if pos==5 else D[6,:,i5,:,ci]
                curves[name][(condition,pos,past)]={"H":h.mean(0),"D":d.mean(0)}
                signed=h[:,1]-(h[:,0]+h[:,2])/2
                absolute=np.abs(h[:,1])-(np.abs(h[:,0])+np.abs(h[:,2]))/2
                row=dict(model=name,condition=condition,position=pos,fixed_previous_q5=past,
                         signed_ambiguous_minus_strong=float(signed.mean()),absolute_ambiguous_minus_strong=float(absolute.mean()),
                         mean_H_weak_count=float(h[:,0].mean()),mean_H_ambiguous=float(h[:,1].mean()),mean_H_high_count=float(h[:,2].mean()),
                         mean_abs_H_strong=float(np.abs(h[:,[0,2]]).mean()),mean_abs_H_ambiguous=float(np.abs(h[:,1]).mean()),
                         mean_norm_strong=float(d[:,[0,2]].mean()),mean_norm_ambiguous=float(d[:,1].mean()))
                within.append(row)
                if ci==0:
                    ratios.append(dict(model=name,position=pos,fixed_previous_q5=past,
                                       absolute_output_contrast_strong_over_ambiguous=row["mean_abs_H_strong"]/row["mean_abs_H_ambiguous"],
                                       neural_norm_strong_over_ambiguous=row["mean_norm_strong"]/row["mean_norm_ambiguous"]))
        for pos,past in [(5,None),(6,2),(6,10),(6,18)]:
            nat=curves[name][("natural",pos,past)]["H"]
            opp=curves[name][("opposite100",pos,past)]["H"]
            delta=opp-nat
            effects.append(dict(model=name,position=pos,fixed_previous_q5=past,natural_H_ambiguous=float(nat[1]),
                                opposite_H_ambiguous=float(opp[1]),opposite_reverses_mean_ambiguous_H=bool(nat[1]*opp[1]<0),
                                source_effect_low_count=float(delta[0]),source_effect_ambiguous=float(delta[1]),source_effect_high_count=float(delta[2]),
                                source_effect_ambiguous_minus_strong=float(delta[1]-(delta[0]+delta[2])/2)))
    summaries=[]
    for pos,past in [(5,None),(6,2),(6,10),(6,18)]:
        rr=[r for r in ratios if r["position"]==pos and r["fixed_previous_q5"]==past]
        ee=[r for r in effects if r["position"]==pos and r["fixed_previous_q5"]==past]
        row=dict(position=pos,fixed_previous_q5=past,conditions={})
        for condition in COND:
            cc=[r for r in within if r["condition"]==condition and r["position"]==pos and r["fixed_previous_q5"]==past]
            row["conditions"][condition]={k:describe([r[k] for r in cc]) for k in (
                "signed_ambiguous_minus_strong","absolute_ambiguous_minus_strong","mean_H_weak_count","mean_H_ambiguous","mean_H_high_count",
                "mean_abs_H_strong","mean_abs_H_ambiguous","mean_norm_strong","mean_norm_ambiguous")}
        row["natural_ratios"]={k:describe([r[k] for r in rr]) for k in ("absolute_output_contrast_strong_over_ambiguous","neural_norm_strong_over_ambiguous")}
        row["source_effect"]={k:describe([r[k] for r in ee]) for k in ("source_effect_low_count","source_effect_ambiguous","source_effect_high_count","source_effect_ambiguous_minus_strong")}
        row["opposite_reverses_mean_ambiguous_H_count"]=sum(r["opposite_reverses_mean_ambiguous_H"] for r in ee)
        summaries.append(row)
    summary=dict(n_base=nbase,n_models=9,checkpoint_order=[e["model"] for e in entries],checks=max_checks,
                 contrasts=summaries,
                 inference="Descriptive checkpoint and paired-context contrasts; OOD stress test, no OOD accuracy inference or checkpoint-population p-values.")
    (OUT/"aggregate_analysis.json").write_text(json.dumps(summary,indent=2)+"\n")
    for filename,rows in [("within_position_contrasts.csv",within),("source_intervention_contrasts.csv",effects),("expression_state_ratios.csv",ratios)]:
        with (OUT/filename).open("w",newline="") as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"svg.fonttype":"none","savefig.facecolor":"white"})
    color={"natural":"#254F7A","opposite100":"#B64C41"}
    panels=[(5,None),(6,2),(6,10),(6,18)]
    fig,axs=plt.subplots(2,4,figsize=(13,6.7),sharey="row")
    for j,(pos,past) in enumerate(panels):
        for k,metric in enumerate(("H","D")):
            ax=axs[k,j]
            if metric=="H":ax.axhline(0,color="#acb3bc",lw=.8)
            for condition in ("natural","opposite100"):
                a=np.stack([curves[e["model"]][(condition,pos,past)][metric] for e in entries])
                for yy in a:ax.plot(QS,yy,color=color[condition],alpha=.24,lw=.7)
                ax.plot(QS,a.mean(0),"o-",color=color[condition],lw=2,ms=4,label=condition)
            ax.set(xticks=QS,xticklabels=["2\nstrong −","10\nambiguous","18\nstrong +"],xlabel=f"Trial {pos} positive-pip count")
            ax.text(-.12,1.08,"ABCDEFGH"[k*4+j],transform=ax.transAxes,weight="bold",fontsize=12)
            if k==0:
                note="Ambiguous probe is OOD" if pos==5 else ("Strong probes are OOD" if past!=10 else "All sequences are OOD")
                title="Trial 5: fixed pre-trial state" if pos==5 else f"Trial 6: preceding q₅ = {past}"
                ax.set_title(title+"\n"+note,fontsize=9)
            if j==0:ax.set_ylabel("Signed paired-history output contrast H" if metric=="H" else "End-response state-difference norm")
    axs[0,0].legend(frameon=False,fontsize=8,loc="lower left")
    fig.subplots_adjust(hspace=.5,wspace=.25,bottom=.12,top=.9)
    for fmt in ("pdf","svg","png"):fig.savefig(OUT/("position_mixture_expression."+fmt),dpi=200,bbox_inches="tight")
    plt.close(fig)
    # Direct per-checkpoint comparison of expression loss and remaining state
    # difference, avoiding an average that would hide model-specific scales.
    fig,ax=plt.subplots(figsize=(6,4))
    rr=[r for r in ratios if r["position"]==5]
    for i,r in enumerate(rr):
        x=r["neural_norm_strong_over_ambiguous"];y=r["absolute_output_contrast_strong_over_ambiguous"]
        ax.scatter(x,y,color="#254F7A",s=32)
        ax.annotate(str(i+1),(x,y),xytext=(5,2),textcoords="offset points",fontsize=8)
    ax.plot([0,1.2],[0,1.2],ls="--",color="#a4aab2",lw=1)
    ax.set(xlim=(0,1.2),ylim=(0,.15),xlabel="Strong / ambiguous: state-difference norm",ylabel="Strong / ambiguous: absolute history output",
           title="Same trial position, same pre-trial state")
    ax.text(.03,.95,"Natural condition; numbers identify checkpoints",transform=ax.transAxes,va="top",fontsize=8)
    for fmt in ("pdf","svg","png"):fig.savefig(OUT/("position5_state_vs_expression."+fmt),dpi=200,bbox_inches="tight")
    plt.close(fig)
    for s in summaries:
        n=s["conditions"]["natural"];print(json.dumps(dict(position=s["position"],q5=s["fixed_previous_q5"],
              natural_H=[n[k]["mean"] for k in ("mean_H_weak_count","mean_H_ambiguous","mean_H_high_count")],
              absolute_ambiguity_advantage_range=[n["absolute_ambiguous_minus_strong"]["minimum"],n["absolute_ambiguous_minus_strong"]["maximum"]],
              source_ambiguity_contrast=s["source_effect"]["source_effect_ambiguous_minus_strong"]["mean"])))


if __name__=="__main__":main()
