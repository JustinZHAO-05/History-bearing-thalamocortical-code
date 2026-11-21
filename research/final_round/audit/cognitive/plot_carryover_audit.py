#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
HERE=Path(__file__).resolve().parent
v1=json.loads((HERE/'biology_carryover_results.json').read_text());v2=json.loads((HERE/'biology_carryover_v2_results.json').read_text());chron=json.loads((HERE/'source_chronology_audit.json').read_text())
groups=list(v1['groups']);short=['PPC suppression','PUL suppression','TRN→PUL activation'];colors=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axs=plt.subplots(2,2,figsize=(13.2,10),layout='constrained')
# First-source-session laser series, source order retained.
ax=axs[0,0];mat=[];labels=[]
for g,lab in zip(groups,['PPC','PUL','TRN']):
 seen=set()
 for r in v2['groups'][g]['assignment_sessions']:
  if r['mouse'] in seen:continue
  seen.add(r['mouse']);a=np.full(120,np.nan);ss=np.array(list(r['first120']),float);a[:len(ss)]=ss;mat.append(a);labels.append(lab+' '+str(r['mouse']))
ax.imshow(np.array(mat),aspect='auto',interpolation='nearest',cmap=ListedColormap(['#e5e5e5','#404040']),vmin=0,vmax=1)
ax.set_yticks(range(25),labels,fontsize=7.7);ax.set_xlabel('Trial index in first source session');ax.set_title('A  Observed intervention schedules',loc='left',fontweight='bold')

# First-order transition dependence in each animal's observed schedules.
ax=axs[0,1]
for j,g in enumerate(groups):
 vals=np.array([r['transition_difference'] for r in chron[g]['mouse_summary']]);xs=j+np.linspace(-.18,.18,len(vals))
 ax.scatter(xs,vals,color=colors[j],s=40)
ax.axhline(0,color='.5',lw=.8);ax.set_xticks(range(3),['PPC','PUL','TRN→PUL']);ax.set_ylabel('P(laser | previous laser) − P(laser | previous control)');ax.set_title('B  Temporal dependence differs across animals',loc='left',fontweight='bold')
ax.text(.03,.97,'Each point represents one within-cohort animal',transform=ax.transAxes,va='top',fontsize=9)
# Laser rates vs prior-state difference.
ax=axs[1,0]
for j,g in enumerate(groups):
 bal=v1['groups'][g]['assignment_balance'];x=[r['laser_rate'] for r in bal];y=[r['contrasts']['prior5']['standardized_mean_difference'] for r in bal]
 ax.scatter(x,y,color=colors[j],label=short[j],s=38,alpha=.85)
ax.axhline(0,color='.5',lw=.8);ax.axvline(.33,color='.6',ls='--',lw=.8);ax.set_xlabel('Observed laser fraction on eligible current trials');ax.set_ylabel('Prior-five-laser imbalance (standardized)');ax.set_title('C  Recent exposure varies with current condition',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='lower right')
# No prior laser overlap, counts all 25 mice.
ax=axs[1,1];x=0;ticks=[];names=[]
for j,g in enumerate(groups):
 rec=v2['groups'][g]['no_prior_laser_firth']['mice'];start=x
 for r in rec:
  ax.scatter(x,r['n_laser'],marker='o',color=colors[j],s=25)
  ax.scatter(x,r['n_no_laser'],marker='x',color=colors[j],s=32)
  if r['n_no_laser']==0:ax.scatter(x,0,marker='s',facecolors='none',edgecolors='black',s=75)
  x+=1
 ticks.append((start+x-1)/2);names.append(['PPC','PUL','TRN→PUL'][j]);x+=1
ax.set_yscale('symlog',linthresh=2);ax.set_xticks(ticks,names);ax.set_ylabel('Trials after five consecutive control trials');ax.set_title('D  Strict exposure stratum has limited overlap',loc='left',fontweight='bold')
ax.scatter([],[],marker='o',color='.3',label='Current laser');ax.scatter([],[],marker='x',color='.3',label='Current control');ax.scatter([],[],marker='s',facecolors='none',edgecolors='black',label='No controls (4 mice)');ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,.12))
for ext in ['pdf','svg','png']:fig.savefig(HERE/('Figure_S_Carryover_Audit.'+ext),dpi=220)
print(HERE/'Figure_S_Carryover_Audit.pdf')
