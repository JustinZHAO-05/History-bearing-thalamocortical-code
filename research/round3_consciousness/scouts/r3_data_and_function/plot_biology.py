#!/usr/bin/env python3
"""Publication-sized plots of executed biological reanalyses (no synthetic data)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
R=BASE/"results"
v0=json.loads((R/"biology_v0_results.json").read_text())["groups"]
v1=json.loads((R/"biology_v1_trial_results.json").read_text())["groups"]
v2=json.loads((R/"biology_v2_mechanism_results.json").read_text())["groups"]
labels={"PPC_suppression":"PPC suppression", "PUL_suppression":"PUL suppression", "TRN_PUL_activation":"TRN→PUL activation"}
colors={"PPC_suppression":"#254F7A", "PUL_suppression":"#238478", "TRN_PUL_activation":"#7F5A9D"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "axes.labelsize":10,"axes.titlesize":11,"pdf.fonttype":42,"svg.fonttype":"none",
                     "savefig.facecolor":"white"})

def style(ax,letter):
    ax.axhline(0,color="#aab1ba",lw=.8,zorder=0)
    ax.text(-.11,1.08,letter,transform=ax.transAxes,fontweight="bold",fontsize=13)
    ax.tick_params(axis="both",labelsize=9)

def estimates(ax, stats, xs, color, scale=1.):
    for x,st in zip(xs,stats):
        y=scale*st["mean"];lo,hi=np.array(st["bootstrap95_mean"])*scale
        ax.errorbar(x,y,yerr=np.array([[y-lo],[hi-y]]),fmt="D",ms=5,color=color,lw=2,capsize=4,zorder=5)

def save(fig,name):
    for fmt in ("pdf","svg","png"):
        fig.savefig(R/(name+"."+fmt),dpi=220,bbox_inches="tight")
    plt.close(fig)

fig,axs=plt.subplots(2,3,figsize=(10.8,6.6),sharey="row")
for j,(key,title) in enumerate(labels.items()):
    color=colors[key]
    ax=axs[0,j];style(ax,"ABC"[j])
    g=v0[key];arr=np.array([g["delta_current"]["values"],g["delta_history"]["values"]]).T*100
    for row in arr:ax.plot([0,1],row,"o-",color=color,alpha=.43,lw=.8,ms=3.5)
    estimates(ax,[g["delta_current"],g["delta_history"]],[0,1],"#15191e",100)
    ax.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=["Current trials","History trials"],title=f"{title}\n{g['n_mice']} source-labeled mice")
    if j==0:ax.set_ylabel("Laser change in accuracy\n(percentage points)")
    ax=axs[1,j];style(ax,"DEF"[j])
    g=v1[key]["conflict"]["primary"]
    for r in g["mice"]:ax.plot([0,1],100*np.array([r["delta_congruent"],r["delta_conflict"]]),"o-",color=color,alpha=.43,lw=.8,ms=3.5)
    estimates(ax,[g["delta_congruent"],g["delta_conflict"]],[0,1],"#15191e",100)
    ax.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=["History agrees","History conflicts"])
    if j==0:ax.set_ylabel("Laser change in current-trial accuracy\n(percentage points)")
    q=g["interaction"]
    ax.text(.5,1.025,f"Conflict advantage: {100*q['mean']:.1f} pp",ha="center",va="bottom",transform=ax.transAxes,fontsize=9)
fig.subplots_adjust(hspace=.45,wspace=.23,bottom=.1,top=.9)
save(fig,"biology_task_tradeoff_and_conflict")

fig,axs=plt.subplots(2,3,figsize=(10.8,6.6),sharey="row")
for j,(key,title) in enumerate(labels.items()):
    color=colors[key];ax=axs[0,j];style(ax,"ABC"[j])
    g=v1[key]["choice"]["primary"]
    for r in g["mice"]:
        b=r["coefficients"];ax.plot([0,1],[b["laser_stimulus"],b["laser_history"]],"o-",color=color,alpha=.43,lw=.8,ms=3.5)
    estimates(ax,[g["coefficient_summaries"][k] for k in ("laser_stimulus","laser_history")],[0,1],"#15191e")
    ax.set(xlim=(-.35,1.35),xticks=[0,1],xticklabels=["Current evidence","Past evidence"],title=title)
    if j==0:ax.set_ylabel("Laser change in native-choice\nlogit coefficient")
    ax=axs[1,j];style(ax,"DEF"[j]);g=v2[key]
    components=("side_bias","sensory_gain","history_bias")
    for r in g["shapley_mice"]:
        ax.plot([0,1,2],[100*r["components"][c]["interaction"] for c in components],"o-",color=color,alpha=.35,lw=.6,ms=3)
    estimates(ax,[g["shapley_summary"][c]["interaction"] for c in components],[0,1,2],"#15191e",100)
    ax.set(xlim=(-.4,2.4),xticks=[0,1,2],xticklabels=["Side bias","Sensory gain","History bias"])
    if j==0:ax.set_ylabel("Shapley contribution to conflict advantage\n(percentage points)")
fig.subplots_adjust(hspace=.38,wspace=.23,bottom=.1,top=.91)
save(fig,"biology_choice_mechanism")

fig,axs=plt.subplots(1,3,figsize=(10.8,3.2),sharey=True)
for j,(key,title) in enumerate(labels.items()):
    ax=axs[j];style(ax,"ABC"[j]);color=colors[key]
    g=v2[key]["conditional_history_use"]["continuous_evidence_magnitude"]
    x=np.array([0,.5,1])
    for fit in g["fits"]:
        b=np.array(fit["coefficients"]);ni=fit["n_sessions"]
        ax.plot(x,b[ni+6]+b[ni+8]*x,color=color,alpha=.32,lw=.8)
    ss=[g["history_laser_change"][str(a)] for a in (0,.5,1)]
    ax.plot(x,[s["mean"] for s in ss],color="#15191e",lw=1.5)
    estimates(ax,ss,x,"#15191e")
    ax.set(xlim=(-.08,1.08),xticks=x,xlabel="Current evidence magnitude |s|",title=title)
    if j==0:ax.set_ylabel("Laser change in history influence\n(log-odds coefficient)")
fig.subplots_adjust(wspace=.22,bottom=.21,top=.82)
save(fig,"biology_conditional_history_use")
print("Saved three scientific figures as PDF, SVG, and PNG.")
