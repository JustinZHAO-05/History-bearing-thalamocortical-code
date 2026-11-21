import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,shutil,textwrap
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;F=Path(os.environ.get('NOV15_FIGURES',str(ROOT.parent/'figures')));F.mkdir(parents=True,exist_ok=True);R=Path(os.environ.get('NOV15_RESULTS',str(ROOT/'results')))
S=json.loads((R/'scientific_summary.json').read_text());src=S['source'];rec=S['receiver']
names=[x['model'] for x in src];labels=[('G' if n.startswith('Gtune') else 'S')+n.split('NUM')[-1] for n in names]
colors=plt.cm.viridis(np.linspace(.08,.9,9));blue='#286582';red='#b84b42';green='#218779';gray='#8a959f';purple='#7860a0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ('pdf','svg','png'):fig.savefig(F/(name+'.'+ext),dpi=260,bbox_inches='tight')
 plt.close(fig)
def title(ax,s):ax.set_title(textwrap.fill(s,46),loc='left',fontweight='bold',pad=10,fontsize=9)
def err(ax,x,m,ci,color,label=None,offset=0):
 ci=np.asarray(ci);m=np.asarray(m);ax.errorbar(np.array(x)+offset,m,yerr=[m-ci[:,0],ci[:,1]-m],fmt='o',ms=4,capsize=2,color=color,label=label)
def preserve(stem,new):
 for ext in ('pdf','svg','png'):
  dest=F/(new+'.'+ext)
  if not dest.exists():shutil.copy2(F/(stem+'.'+ext),dest)

def fig2():
 old=ROOT.parent/'revision_2026_11_14/results';zs=[np.load(old/(n+'.npz')) for n in names]
 fig=plt.figure(figsize=(10,8.4),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[.45,1.4,1])
 ax=fig.add_subplot(gs[0,:]);ax.axis('off');title(ax,'A  A source message rebuilds a historical distinction')
 ax.text(.01,.43,'Opposite pasts',transform=ax.transAxes,fontsize=12);ax.annotate('',xy=(.33,.5),xytext=(.2,.5),arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
 ax.text(.34,.43,r'Full-state mean: $x^+=x^-$',transform=ax.transAxes,fontsize=12);ax.annotate('',xy=(.76,.5),xytext=(.66,.5),arrowprops=dict(arrowstyle='->'),xycoords='axes fraction')
 ax.text(.77,.43,'Own / mean / opposite\nPUL source replay',transform=ax.transAxes,color=blue)
 ax=fig.add_subplot(gs[1,:]);title(ax,'B  Different messages establish different later outputs')
 z=zs[0];t=np.arange(800)*5;base=z['base_context'];ix=np.random.default_rng(98009).integers(0,12,(3000,12))
 for j,c,l,ls in [(0,'#9aa4ae','Natural reference',':'),(1,blue,'Own history','-'),(2,gray,'Pair mean','--'),(7,red,'Opposite history','-')]:
  h=(z['output_trace'][:,:,j,0]-z['output_trace'][:,:,j,1])/2;b=np.stack([h[:,base==i].mean(1) for i in range(12)],axis=1)
  ci=np.quantile(b[:,ix].mean(2),[.025,.975],axis=1)
  ax.fill_between(t,ci[0],ci[1],color=c,alpha=.15,lw=0);ax.plot(t,h.mean(1),color=c,label=l,ls=ls,lw=1.7)
 for lo,hi,c,lab in [(0,1000,'#e4eff5','Source replay'),(1600,2000,'#fff1df','Current judgment'),(3600,4000,'#e7f1e7','Historical judgment')]:
  ax.axvspan(lo,hi,color=c,zorder=-5);ax.text((lo+hi)/2,.94,lab,ha='center',transform=ax.get_xaxis_transform(),fontsize=9)
 ax.axhline(0,color=gray,lw=.6);ax.set(xlim=(0,4000),xlabel='Time from source replay onset (ms)',ylabel='Half-pair native output');ax.legend(ncol=4,frameon=False,loc='lower center')
 ax=fig.add_subplot(gs[2,0]);title(ax,'C  Population differences reappear and persist')
 for j,c,l in [(1,blue,'Own / opposite (overlap)'),(2,gray,'Pair mean')]:ax.plot(t,z['history_norm_trace'][:,:,j].mean(1),color=c,label=l)
 ax.axvspan(0,1000,color='#e4eff5');ax.set(xlabel='Time from source replay onset (ms)',ylabel='Population rate separation');ax.legend(frameon=False)
 ax=fig.add_subplot(gs[2,1]);title(ax,'D  Reconstruction across nine circuits')
 for j,c,l,o in [(1,blue,'Own',-.18),(2,gray,'Mean',0),(7,red,'Opposite',.18)]:
  means=[];cis=[]
  for z in zs:
   h=z['history_signed_margin'][:,j];b=np.array([h[z['base_context']==i].mean() for i in range(12)]);means.append(b.mean());cis.append(np.quantile(b[ix].mean(1),[.025,.975]))
  err(ax,range(9),means,cis,c,l,offset=o)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Later historical contrast');ax.legend(ncol=3,frameon=False,fontsize=8,loc='center')
 save(fig,'Figure_2_Reconstruction_and_Temporal_Access')

def fig4():
 fig,axs=plt.subplots(2,3,figsize=(12,7.6),layout='constrained');ax=axs[0,0];ax.axis('off');title(ax,'A  From a bounded source to a message')
 ax.text(.03,.78,r'$r_T^\pm=\mu_T\pm d_T/2$',fontsize=16,transform=ax.transAxes);ax.text(.03,.6,r'$0\leq r_{T,i}^{\pm}\leq20$',fontsize=16,transform=ax.transAxes)
 ax.text(.03,.39,r'$b^\pm=B r_T^\pm$',fontsize=16,color=blue,transform=ax.transAxes);ax.text(.03,.1,'70 PUL activities\nOriginal 400-target projection',transform=ax.transAxes)
 ax=axs[0,1];title(ax,'B  History concentrates along a source axis')
 train=json.loads((ROOT.parent/'revision_2026_11_14/new_scientific_summary.json').read_text())['rows']
 ax.plot(range(9),[r['source_learned_variance'] for r in train],'o-',color=gray,label='Fitting histories');ax.plot(range(9),[r['heldout_axis_fraction'] for r in src],'s-',color=blue,label='Held-out histories');ax.set_xticks(range(9),labels);ax.set(ylim=(0,1.03),ylabel='Explained source-difference energy');ax.legend(frameon=False)
 ax=axs[0,2];title(ax,'C  Available source directions constrain norm matching')
 for i,n in enumerate(names):
  g=json.loads((ROOT.parent/'revision_2026_11_14/results'/(n+'.json')).read_text())['geometry'];ax.scatter(i-.1,g[0]['matching_fraction_1pct'],color=green,marker='D');ax.scatter([i+.06,i+.13,i+.2],[a['matching_fraction_1pct'] for a in g[1:]],color=gray,s=15)
 ax.set_xticks(range(9),labels);ax.set(ylabel='Fraction within 1% of natural norm',ylim=(-.03,1.03));ax.text(.02,.08,'Green: learned axis; gray: alternatives',transform=ax.transAxes,fontsize=8)
 ax=axs[1,0];title(ax,'D  Boundary-aware maps reach new directions')
 for j,c,l in [(0,gray,'64 proposals'),(1,blue,'256 proposals'),(2,green,'Smoothed')]:ax.plot(range(9),[s['geometry'][j]['relative_change'] for s in src],'o-',color=c,label=l,ms=4)
 ax.set_xticks(range(9),labels);ax.set_ylabel(r'Mean $\|B(d^\prime-d)\|/\|Bd\|$');ax.legend(frameon=False,fontsize=8)
 ax=axs[1,1];title(ax,'E  Source reallocation changes historical influence')
 for i,s in enumerate(src):
  xx=[g['relative_change'] for g in s['geometry']];yy=s['outcomes']['history_loss']['mean'];ax.plot(xx,yy,'o-',color=colors[i],label=labels[i],ms=4)
 ax.axhline(0,color=gray,lw=.6);ax.set(xlabel='Actual receiving-difference change',ylabel='Own minus transformed historical contrast');ax.legend(ncol=3,frameon=False,fontsize=7)
 ax=axs[1,2];title(ax,'F  Temporal variation accompanies each map')
 for j,c,l in [(0,gray,'64'),(1,blue,'256'),(2,green,'Smoothed')]:ax.plot(range(9),[s['geometry'][j]['roughness']/max(s['geometry'][j]['natural_roughness'],1e-20) for s in src],'o-',color=c,label=l,ms=4)
 ax.axhline(1,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set(ylabel='Source temporal roughness / natural',yscale='log');ax.legend(frameon=False,ncol=3,fontsize=8)
 save(fig,'Figure_4_Source_Reachability')

def fig5():
 preserve('Figure_5_Fixed_Aggregate','Figure_5-2_Original_Target_Allocation')
 fig,axs=plt.subplots(2,2,figsize=(10,7.8),layout='constrained');old=[]
 for n in names:old.append(json.loads((ROOT.parent/'results/fixed_aggregate_geometry_v2'/(n+'.json')).read_text()))
 ax=axs[0,0];title(ax,'A  Target-current allocation reveals receiver sensitivity')
 means=[d['primary']['full_minus_mean_mapped'] for d in old]
 ci=[d['primary']['ci95'] for d in old]
 err(ax,range(9),means,ci,red);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Historical contrast loss')
 ax=axs[0,1];title(ax,'B  Source-constrained allocation has circuit-specific effects')
 for j,c,l,o in [(0,gray,'64',-.18),(1,blue,'256',0),(2,green,'Smoothed',.18)]:err(ax,range(9),[s['outcomes']['history_loss']['mean'][j] for s in src],[s['outcomes']['history_loss']['ci'][j] for s in src],c,l,o)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Historical contrast loss');ax.legend(frameon=False,ncol=3)
 ax=axs[1,0];title(ax,'C  Current sensory output remains closely matched')
 for j,c,l,o in [(0,gray,'64',-.18),(1,blue,'256',0),(2,green,'Smoothed',.18)]:err(ax,range(9),[s['outcomes']['sensory_change']['mean'][j] for s in src],[s['outcomes']['sensory_change']['ci'][j] for s in src],c,l,o)
 ax.axhspan(-.02,.02,color='#e6f0ed');ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Transformed minus own sensory output')
 ax=axs[1,1];title(ax,'D  Historical loss and current commitment')
 for i,s in enumerate(src):
  ax.scatter(s['outcomes']['current']['mean'][3:6],s['outcomes']['history_loss']['mean'],color=colors[i],s=25,label=labels[i])
 ax.set(xlim=(.97,1.002),xlabel='Current native commitment fraction',ylabel='Historical contrast loss');ax.legend(ncol=3,frameon=False,fontsize=8)
 save(fig,'Figure_5_Fixed_Aggregate')

def fig6():
 preserve('Figure_6_Retention_and_Expression','Figure_6-2_Original_Trajectory_Components')
 fig,axs=plt.subplots(2,3,figsize=(12,7.8),layout='constrained')
 ax=axs[0,0];title(ax,'A  Equal initial differences, altered receiver states');ax.axis('off')
 ax.text(.02,.8,'Same reconstructed history\nSame sensory pip train',transform=ax.transAxes,fontsize=12)
 ax.text(.02,.53,r'$x^\pm\mapsto x^\pm+s$',transform=ax.transAxes,fontsize=17,color=blue)
 ax.text(.02,.27,'Common inward shift\nMatched-norm random shift\n100-ms recurrent-difference pause',transform=ax.transAxes,fontsize=10)
 for q,ax,letter in [(10,axs[0,1],'B'),(2,axs[0,2],'C')]:
  title(ax,letter+('  Ambiguous current response' if q==10 else '  Strong current response'))
  for i,r in enumerate(rec):ax.plot(range(4),r['responses'][str(q)]['current_H']['mean'],'o-',color=colors[i],ms=3,lw=.9)
  ax.set_xticks(range(4),['Own','Inward','Random','Pause'],rotation=20);ax.set_ylabel('Current historical contrast');ax.set_yscale('symlog',linthresh=.15)
 ax=axs[1,0];title(ax,'D  Receiver perturbations affect later history use')
 for j,c,l,o in [(1,blue,'Inward',-.18),(2,gray,'Random',0),(3,red,'Pause',.18)]:err(ax,range(9),[r['responses']['10']['later_change']['mean'][j] for r in rec],[r['responses']['10']['later_change']['ci'][j] for r in rec],c,l,o)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Later H change after ambiguous trial 5');ax.set_yscale('symlog',linthresh=.1);ax.legend(frameon=False,ncol=3,fontsize=8)
 ax=axs[1,1];title(ax,'E  Activity separation after receiver perturbation')
 for i,r in enumerate(rec):ax.plot(range(4),r['responses']['10']['rate_norm']['mean'],'o-',color=colors[i],ms=3,lw=.9)
 ax.set_xticks(range(4),['Own','Inward','Random','Pause'],rotation=20);ax.set_ylabel('Current rate-difference norm')
 ax=axs[1,2];title(ax,'F  The fixed readout sees different activity directions')
 for i,r in enumerate(rec):ax.plot(range(4),r['responses']['10']['alignment']['mean'],'o-',color=colors[i],ms=3,lw=.9,label=labels[i])
 ax.set_xticks(range(4),['Own','Inward','Random','Pause'],rotation=20);ax.set_ylabel('Mean absolute readout alignment');ax.legend(ncol=3,frameon=False,fontsize=7)
 save(fig,'Figure_6_Retention_and_Expression')

def fig7():
 preserve('Figure_7_Position_and_Evidence','Figure_7-2_Original_Evidence_Crossing')
 fig,axs=plt.subplots(2,3,figsize=(12,7.7),layout='constrained')
 table=pd.read_csv(ROOT.parents[1]/'round3_consciousness/results/position_mixture_stress/within_position_contrasts.csv')
 for cond,c,label in [('natural',blue,'Natural'),('opposite100',red,'Opposite source')]:
  for pos,ax,letter in [(5,axs[0,0],'A'),(6,axs[0,1],'B')]:
   rows=table[(table.condition==cond)&(table.position==pos)]
   if pos==6:rows=rows[rows.fixed_previous_q5==2]
   yy=rows[['mean_H_weak_count','mean_H_ambiguous','mean_H_high_count']].to_numpy()
   if len(yy):ax.plot([2,10,18],yy.T,color=c,alpha=.18,lw=.7);ax.plot([2,10,18],yy.mean(0),'o-',color=c,label=label)
   title(ax,letter+f'  Evidence shapes expression at position {pos}');ax.set_xticks([2,10,18],['Strong -','Ambiguous','Strong +']);ax.set_ylabel('Historical contrast');ax.legend(frameon=False,fontsize=8)
 ax=axs[0,2];title(ax,'C  Local gains predict held-out response differences')
 for j,c,l in [(0,blue,'Evidence-specific gain'),(1,gray,'Pre-sensory gain'),(2,red,'Leak only')]:ax.plot(range(9),[r['predictions'][j]['rmse'] for r in rec],'o-',color=c,label=l,ms=4)
 ax.set_xticks(range(9),labels);ax.set_ylabel('Held-out trajectory RMSE');ax.legend(frameon=False,fontsize=8)
 z=np.load(R/f'receiver_{names[0]}.npz');t=np.arange(90)*5
 ax=axs[1,0];title(ax,'D  Predicted and observed G20 trajectories')
 for qi,c,l in [(0,blue,'Strong -'),(1,green,'Ambiguous'),(2,purple,'Strong +')]:
  ax.plot(t,z['actual'][:,qi::3].mean(1),color=c,label=l);ax.plot(t,z['prediction'][:,qi::3,0].mean(1),color=c,ls='--')
 ax.set(xlabel='Time after pip train (ms)',ylabel='Paired native output difference');ax.legend(frameon=False,fontsize=8);ax.text(.02,.03,'Solid: observed; dashed: predicted',transform=ax.transAxes,fontsize=8)
 ax=axs[1,1];title(ax,'E  Evidence-specific gain transfers across histories')
 ax.bar(range(9),[r['predictions'][0]['r2'] for r in rec],color=blue);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Held-out explained variation')
 ax=axs[1,2];title(ax,'F  Mechanism connecting past and present');ax.axis('off')
 ax.text(.02,.82,'Source content\nreconstructs a past',transform=ax.transAxes,color=blue,fontsize=13)
 ax.text(.02,.5,'Recurrent propagation\ncontinues its influence',transform=ax.transAxes,color=green,fontsize=13)
 ax.text(.02,.18,'Evidence-dependent activity gains\nand readout alignment shape choice',transform=ax.transAxes,color=purple,fontsize=12)
 save(fig,'Figure_7_Position_and_Evidence')

if __name__=='__main__':fig2();fig4();fig5();fig6();fig7()
