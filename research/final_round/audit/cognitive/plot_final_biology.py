#!/usr/bin/env python3
"""Final biological main figure; all observed values come from audited result JSON."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
HERE=Path(__file__).resolve().parent;FINAL=HERE.parents[1];R3=HERE.parents[2]/'round3_consciousness';OUT=FINAL/'figures';OUT.mkdir(exist_ok=True)
v0=json.loads((R3/'scouts/r3_data_and_function/results/biology_v0_results.json').read_text())
v1=json.loads((HERE/'biology_carryover_results.json').read_text());v2=json.loads((HERE/'biology_carryover_v2_results.json').read_text())
groups=list(v1['groups']);short=['PPC','PUL','TRN→PUL'];colors=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.default':'regular'})
fig,axs=plt.subplots(2,2,figsize=(12.5,9.4),layout='constrained',gridspec_kw={'height_ratios':[1,1.03]})
ax=axs[0,0];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_title('A  History has a signed effect on current choice',loc='left',fontweight='bold')
def box(x,y,w,h,txt,color):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.015',facecolor=color,edgecolor='none',alpha=.13);ax.add_patch(p);ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=10.5)
def arrow(a,b,color='.35'):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=color))
box(.04,.70,.37,.16,'Past five stimuli\nHistory $h$',colors[0]);box(.60,.70,.35,.16,'Current stimulus\nEvidence $s$',colors[1])
box(.17,.42,.66,.17,'Choice model\n$\\mathrm{logit}\\,P(Y=1)=a+\\beta_s s+\\beta_h h$', '#444444')
arrow((.225,.69),(.37,.60),colors[0]);arrow((.775,.69),(.66,.60),colors[1]);
ax.text(.04,.92,'Intertrial laser: 0.5 s',fontsize=10,ha='left',color='#444444')
box(.03,.09,.43,.19,'History agrees with evidence\nHistory supports current choice',colors[0]);box(.54,.09,.43,.19,'History opposes evidence\nHistory biases the other choice',colors[1]);arrow((.36,.41),(.245,.29));arrow((.64,.41),(.755,.29))
ax.text(.5,.005,'Schematic: reducing history influence preferentially helps conflicting trials',fontsize=9.3,ha='center')
# Actual accuracy effects.
ax=axs[0,1];table=[]
for j,g in enumerate(groups):
 for k,(name,offset,marker) in enumerate([('delta_current',-.12,'o'),('delta_history',.12,'s')]):
  q=v0['groups'][g][name];m=100*q['mean'];lo,hi=100*np.array(q['bootstrap95_mean']);ax.errorbar(j+offset,m,yerr=[[m-lo],[hi-m]],fmt=marker,color=colors[j],capsize=4,markersize=7,label=['Current stimulus','History probe'][k] if j==0 else None)
  table.append(dict(panel='B',intervention=g,quantity=name,mean=m,ci_low=lo,ci_high=hi,units='percentage points'))
ax.axhline(0,color='.55',lw=.9);ax.set_xticks(range(3),[f'{s}\n(n={v0["groups"][g]["n_mice"]})' for s,g in zip(short,groups)]);ax.set_ylabel('Laser − control accuracy (percentage points)');ax.set_title('B  Current and history judgments respond differently',loc='left',fontweight='bold',fontsize=11.3);ax.legend(frameon=False,fontsize=9,loc='lower left')
# Interaction in measured accuracy.
ax=axs[1,0]
for j,g in enumerate(groups):
 q=v1['groups'][g]['conflict_recalculation']['interaction'];vals=100*np.array(q['values']);m=100*q['mean'];lo,hi=100*np.array(q['bootstrap95_mean']);jitter=np.linspace(-.13,.13,len(vals));ax.scatter(j+jitter,vals,color=colors[j],s=24,alpha=.6);ax.errorbar(j+.22,m,yerr=[[m-lo],[hi-m]],fmt='D',color=colors[j],capsize=4,markersize=6)
 table.append(dict(panel='C',intervention=g,quantity='conflict_advantage',mean=m,ci_low=lo,ci_high=hi,units='percentage points'))
ax.axhline(0,color='.55',lw=.9);ax.set_xticks(range(3),short);ax.set_ylabel('Extra laser benefit on conflicting trials\n(percentage points)');ax.set_title('C  Laser benefit is greatest under conflicting evidence',loc='left',fontweight='bold',fontsize=11.3);ax.text(.02,.98,'Dots: animals; diamonds: mean ± 95% interval',transform=ax.transAxes,va='top',fontsize=9)
# Full-sample sensitivity.
ax=axs[1,1];variants=['All binary choices','+ previous laser / choice / outcome','+ within-session drift'];modes=['baseline','lag_adjusted','lag_time_adjusted'];offsets=[-.22,0,.22];markers=['o','s','^']
for j,g in enumerate(groups):
 sm=[v1['groups'][g]['models']['baseline']['summary']['laser_history'],v1['groups'][g]['models']['lag_adjusted']['summary']['laser_history'],v2['groups'][g]['lag_time_adjusted']['summary']['laser_history']]
 for k,q in enumerate(sm):
  m=q['mean'];lo,hi=q['bootstrap95_mean'];ax.errorbar(j+offsets[k],m,yerr=[[m-lo],[hi-m]],fmt=markers[k],color=colors[j],capsize=3,markersize=6,label=variants[k] if j==0 else None)
  table.append(dict(panel='D',intervention=g,quantity=modes[k],mean=m,ci_low=lo,ci_high=hi,units='log odds per unit history'))
ax.axhline(0,color='.55',lw=.9);ax.set_xticks(range(3),short);ax.set_ylabel('History × current-laser coefficient');ax.set_title('D  Temporal adjustment retains the PPC effect',loc='left',fontweight='bold',fontsize=11.3);ax.legend(frameon=False,fontsize=8.5,loc='upper center',bbox_to_anchor=(.5,-.14))
for ext in ['pdf','svg','png']:fig.savefig(OUT/('Figure_1_Biological_Function.'+ext),dpi=240)
with (OUT/'Figure_1_Biological_Function_source.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
print(OUT/'Figure_1_Biological_Function.pdf')
