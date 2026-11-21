"""Focused main figures and complete scientific comparisons."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,importlib.util,shutil,textwrap
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;FINAL=ROOT.parent;F=Path(os.environ.get('FOCUSED_FIGURES',str(FINAL/'figures')));F.mkdir(exist_ok=True)
R=Path(os.environ.get('FOCUSED_RESULTS',str(ROOT/'results')));S=json.loads((R/'scientific_summary.json').read_text());axis=S['axis'];low=S['lowdose'];pred=S['prediction']
names=[x['model'] for x in axis];labels=[('G' if n.startswith('Gtune') else 'S')+n.split('NUM')[-1] for n in names];colors=plt.cm.viridis(np.linspace(.08,.9,9))
blue='#286582';red='#b84b42';green='#218779';gray='#8a959f';purple='#7860a0'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
def title(ax,s):ax.set_title(textwrap.fill(s,34),loc='left',fontsize=9.5,fontweight='bold',pad=10)
def save(fig,stem):
 for ax in fig.axes:
  ticktext=[t.get_text() for t in ax.get_xticklabels()]
  if ticktext==labels:
   for tick in ax.get_xticklabels():tick.set_rotation(40);tick.set_ha('right');tick.set_fontsize(9.5)
  elif ticktext==['Strong -','Ambiguous','Strong +']:ax.tick_params(axis='x',labelsize=9)
 for ext in ('pdf','svg','png'):fig.savefig(F/f'{stem}.{ext}',dpi=250,bbox_inches='tight')
 plt.close(fig)
def err(ax,x,iv,color,label=None,offset=0):
 m=np.array([v['mean'] for v in iv]);ci=np.array([v['ci'] for v in iv]);ax.errorbar(np.array(x)+offset,m,yerr=np.maximum(np.array([m-ci[:,0],ci[:,1]-m]),0),fmt='o',ms=4,capsize=2,color=color,label=label)
def subset(rec,key,j):return dict(mean=rec[key]['mean'][j],ci=rec[key]['ci'][j])
def legend(ax):ax.legend(ncol=3,frameon=False,fontsize=9)

def prepare():
 for stem in ('Figure_2_Reconstruction_and_Temporal_Access','Figure_4_Source_Reachability','Figure_5_Fixed_Aggregate','Figure_6_Retention_and_Expression','Figure_7_Position_and_Evidence'):
  for ext in ('pdf','svg','png'):
   dest=ROOT/'inherited/figures'/f'{stem}.{ext}'
   if not dest.exists():shutil.copy2(FINAL/'figures'/f'{stem}.{ext}',dest)
 for ext in ('pdf','svg','png'):shutil.copy2(ROOT/'inherited/figures'/f'Figure_6_Retention_and_Expression.{ext}',F/f'Figure_6-4_Larger_Receiver_Shifts.{ext}')

def fig4():
 src=json.loads((FINAL/'revision_2026_11_15/results/scientific_summary.json').read_text())['source'];train=json.loads((FINAL/'revision_2026_11_14/new_scientific_summary.json').read_text())['rows']
 fig,axs=plt.subplots(2,3,figsize=(9.8,7.6),layout='constrained');ax=axs[0,0];ax.axis('off');title(ax,'A  Bounded activities define arriving messages')
 for y,text in [(.78,r'$r_T^\pm=\mu_T\pm d_T/2$'),(.57,r'$0\leq r_{T,i}^{\pm}\leq20$'),(.32,r'$b^\pm=B r_T^\pm$')]:ax.text(.03,y,text,fontsize=16,transform=ax.transAxes,color=blue if y==.32 else 'black')
 ax.text(.03,.08,'70 PUL activities; original projection',transform=ax.transAxes)
 ax=axs[0,1];title(ax,'B  A leading axis generalizes across histories');ax.plot(range(9),[r['source_learned_variance'] for r in train],'o-',color=gray,label='12 fitting bases');ax.plot(range(9),[r['heldout_axis_fraction'] for r in axis],'s-',color=blue,label='12 separate test bases');ax.set(ylim=(0,1.03),ylabel='Source-difference energy fraction');ax.set_xticks(range(9),labels);ax.legend(frameon=False,fontsize=8)
 ax=axs[0,2];title(ax,'C  Complete clipped paths resolve matching')
 for i,a in enumerate(axis):ax.scatter(i-.1,a['geometry'][0]['matching_fraction_1pct'],color=green,marker='D',s=27);ax.scatter(i+np.array([.03,.13,.23]),[g['matching_fraction_1pct'] for g in a['geometry'][1:]],color=gray,s=17)
 ax.set_xticks(range(9),labels);ax.set(ylim=(-.03,1.03),ylabel='Fraction within 1% of natural norm');ax.text(.02,.09,'Green: learned; gray: three fixed alternatives',transform=ax.transAxes,fontsize=8)
 ax=axs[1,0];title(ax,'D  Source maps produce measured displacements')
 for j,c,l in [(0,gray,'64 proposals'),(1,blue,'256 proposals'),(2,green,'Smoothed')]:ax.plot(range(9),[s['geometry'][j]['relative_change'] for s in src],'o-',color=c,label=l,ms=4)
 ax.set_xticks(range(9),labels);ax.set_ylabel(r'Mean $\|B(d^\prime-d)\|/\|Bd\|$');ax.legend(frameon=False,fontsize=8)
 ax=axs[1,1];title(ax,'E  Allocation and historical function vary together')
 for i,s in enumerate(src):ax.plot([g['relative_change'] for g in s['geometry']],s['outcomes']['history_loss']['mean'],'o-',color=colors[i],label=labels[i],ms=4)
 ax.axhline(0,color=gray,lw=.6);ax.set(xlabel='Actual receiving-difference change',ylabel='Own minus transformed historical contrast');legend(ax)
 ax=axs[1,2];title(ax,'F  Temporal variation accompanies each map')
 for j,c,l in [(0,gray,'64'),(1,blue,'256'),(2,green,'Smoothed')]:ax.plot(range(9),[s['geometry'][j]['roughness']/s['geometry'][j]['natural_roughness'] for s in src],'o-',color=c,label=l,ms=4)
 ax.axhline(1,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set(ylabel='Source roughness / natural',yscale='log');ax.legend(frameon=False,ncol=3,fontsize=8);save(fig,'Figure_4_Source_Reachability')

def fig6():
 old=json.loads((FINAL.parents[0]/'round3_consciousness/results/analysis/rnn_inference.json').read_text())['finite_calculus']
 fig,axs=plt.subplots(2,3,figsize=(9.8,7.8),layout='constrained')
 for ax,key,letter,text,ylabel in [(axs[0,0],'rate_difference_norm_mean','A','Historical differences remain in activity','Activity-difference RMS norm'),(axs[0,1],'rms_readout_alignment_mean','B','The fixed readout becomes more aligned','Readout RMS alignment')]:
  title(ax,letter+'  '+text)
  for i,r in enumerate(old):ax.plot([0,1],r['conditions'][0][key],'o-',color=colors[i],label=labels[i],ms=4,lw=1)
  ax.set_xticks([0,1],['Current','History']);ax.set_ylabel(ylabel)
  if letter=='B':ax.set_yscale('log');legend(ax)
 ax=axs[0,2];title(ax,'C  Small state shifts alter later history')
 err(ax,range(9),[subset(r['native'],'later_change',1) for r in low],blue,'Inward',-.1);err(ax,range(9),[r['native']['random_change'] for r in low],gray,'Eight-direction mean',.1);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Later historical contrast change');ax.legend(frameon=False,fontsize=8)
 ax=axs[1,0];title(ax,'D  Attenuating recurrence changes later use');err(ax,range(9),[r['native']['recurrent_change'] for r in low],red);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Later historical contrast change')
 ax=axs[1,1];title(ax,'E  Current sensory responses remain close');ax.axhspan(-.02,.02,color='#e6f0ed')
 for q,marker,off in [('2','o',-.12),('18','s',.12)]:
  for j,c,offset in [(1,blue,-.035),(10,red,.035)]:
   iv=[subset(r['responses'][q],'sensory_change',j) for r in low];err(ax,np.arange(9)+off,iv,c,offset=offset)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Signed sensory response change');ax.text(.02,.96,'Blue: inward; red: recurrence\nLeft/right: strong - / +',va='top',transform=ax.transAxes,fontsize=9)
 ax=axs[1,2];title(ax,'F  Threshold crossings carry output difference');df=pd.read_csv(R/'activation_weights.csv');df=df[(df.region=='PPC')&(df.arm==0)]
 for cat,c,l in [('linear',gray,'Both linear'),('lower_cross',blue,'Lower crossing'),('upper_cross',red,'Upper crossing')]:
  arr=np.array([df[(df.q==q)&(df.category==cat)].absolute_readout_fraction.to_numpy() for q in [2,10,18]])
  ax.plot([0,1,2],arr.mean(1),'o-',color=c,label=l)
  ax.scatter(np.repeat([0,1,2],9)+np.tile(np.linspace(-.07,.07,9),3),arr.ravel(),color=c,s=9,alpha=.35)
 ax.set_xticks([0,1,2],['Strong -','Ambiguous','Strong +']);ax.set(ylabel='Absolute PPC readout contribution fraction',ylim=(-.02,1.02));ax.legend(frameon=False,fontsize=8);save(fig,'Figure_6_Retention_and_Expression')

def fig7():
 fig,axs=plt.subplots(2,3,figsize=(9.8,7.8),layout='constrained');table=pd.read_csv(FINAL.parents[0]/'round3_consciousness/results/position_mixture_stress/within_position_contrasts.csv')
 for pos,ax,letter in [(5,axs[0,0],'A'),(6,axs[0,1],'B')]:
  for condition,c,label in [('natural',blue,'Natural'),('opposite100',red,'Opposite source')]:
   rows=table[(table.condition==condition)&(table.position==pos)]
   if pos==6:rows=rows[rows.fixed_previous_q5==2]
   yy=rows[['mean_H_weak_count','mean_H_ambiguous','mean_H_high_count']].to_numpy();ax.plot([2,10,18],yy.T,color=c,alpha=.18,lw=.7);ax.plot([2,10,18],yy.mean(0),'o-',color=c,label=label)
  title(ax,letter+f'  Evidence shapes expression at position {pos}');ax.set_xticks([2,10,18],['Strong -','Ambiguous','Strong +']);ax.set_ylabel('Historical contrast');ax.legend(frameon=False,fontsize=8)
 ax=axs[0,2];title(ax,'C  Matched propagation and template comparisons')
 for key,c,l,ls in [('evidence_recurrence',blue,'Evidence + recurrence','-'),('pre_recurrence',gray,'Pre-sensory + recurrence',':'),('evidence_leak',red,'Same evidence + leak','-'),('reference_template',green,'Reference template','-')]:ax.plot(range(9),[p['scores'][key]['rmse'] for p in pred],'o',ls=ls,color=c,label=l,ms=4)
 ax.set_xticks(range(9),labels);ax.set_ylabel('Held-out trajectory RMSE');ax.legend(frameon=False,fontsize=7)
 ax=axs[1,0];title(ax,'D  Conditional propagation of G20 histories');z=np.load(R/f'prediction_{names[0]}.npz');t=np.arange(90)*5
 for qi,c,l in [(0,blue,'Strong -'),(1,green,'Ambiguous'),(2,purple,'Strong +')]:ax.plot(t,z['actual'][:,qi::3].mean(1),color=c,label=l);ax.plot(t,z['prediction'][:,qi::3,0].mean(1),color=c,ls='--')
 ax.set(xlabel='Time after pip train (ms)',ylabel='Paired native output difference');ax.legend(frameon=False,fontsize=8);ax.text(.02,.03,'Solid: observed; dashed: predicted',transform=ax.transAxes,fontsize=8)
 ax=axs[1,1];title(ax,'E  Gain propagation versus the reference template');err(ax,range(9),[p['paired_mse_difference']['template_minus_gain'] for p in pred],blue);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Template MSE minus gain MSE');ax.text(.02,.96,'Positive: gain advantage',transform=ax.transAxes,va='top',fontsize=8)
 ax=axs[1,2];title(ax,'F  Prediction quality within each evidence');arr=np.array([[p['scores']['evidence_recurrence']['within_evidence'][str(q)]['r2'] for q in (2,10,18)] for p in pred]);im=ax.imshow(arr,cmap='RdBu',vmin=-2.4,vmax=2.4,aspect='auto');ax.set_xticks(range(3),['Strong -','Ambiguous','Strong +']);ax.set_yticks(range(9),labels)
 for i in range(9):
  for j in range(3):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(arr[i,j])>1.5 else 'black')
 fig.colorbar(im,ax=ax,fraction=.05,pad=.03,label=r'Within-evidence $R^2$');save(fig,'Figure_7_Position_and_Evidence')

def supporting():
 fig,axs=plt.subplots(2,2,figsize=(10,7.5),layout='constrained');ax=axs[0,0];title(ax,'A  Realized clipped messages remain low dimensional');ax.bar(range(9),[a['geometry'][0]['participation_ratio'] for a in axis],color=blue);ax.set_xticks(range(9),labels);ax.set_ylabel('Participation-ratio dimension')
 ax=axs[0,1];title(ax,'B  Clipping changes alignment with the fitted axis');ax.plot(range(9),[a['geometry'][0]['postclip_axis_cosine_abs'] for a in axis],'o-',color=green);ax.set_xticks(range(9),labels);ax.set(ylabel='Mean absolute cosine',ylim=(0,1.03))
 ax=axs[1,0];title(ax,'C  Full-path matching preserves learned replay')
 for j,c,l,o in [(1,gray,'Full own',-.12),(3,blue,'Learned clipped axis',.12)]:err(ax,range(9),[subset(a['metrics'],'history_signed_margin',j) for a in axis],c,l,o)
 ax.set_xticks(range(9),labels);ax.set_ylabel('Later historical contrast');ax.legend(frameon=False,fontsize=8)
 ax=axs[1,1];title(ax,'D  All sampled source alternatives are retained')
 for i,a in enumerate(axis):ax.scatter([i-.15,i,i+.15],a['metrics']['history_signed_margin']['mean'][4:7],c=[gray]*3,s=25)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Random-axis historical contrast');save(fig,'Figure_4-6_Corrected_Source_Axes')
 df=pd.read_csv(R/'source_sensory_by_strength.csv');fig,axs=plt.subplots(1,3,figsize=(9.8,4),layout='constrained')
 for j,(ax,mp,l) in enumerate(zip(axs,[3,4,5],['64 proposals','256 proposals','Smoothed'])):
  title(ax,chr(65+j)+'  '+l);ax.axhspan(-.02,.02,color='#e6f0ed');ax.axhline(0,color=gray,lw=.6)
  for qi,(q,c) in enumerate(zip([0,2,18,20],[blue,green,purple,red])):
   rows=df[(df['map']==mp)&(df.q==q)];iv=[dict(mean=r['mean'],ci=[r['low'],r['high']]) for _,r in rows.iterrows()];err(ax,range(9),iv,c,f'{q} positive pips',(qi-1.5)*.15)
  ax.set_xticks(range(9),labels);ax.set(ylabel='Signed sensory response change',ylim=(-.022,.022));ax.legend(frameon=False,fontsize=7,ncol=2)
 save(fig,'Figure_5-3_Source_Sensory_Strengths')
 # Existing larger intervention results, now with explicit signs and colors.
 original=json.loads((FINAL/'revision_2026_11_15/results/scientific_summary.json').read_text())['receiver'];fig,axs=plt.subplots(2,2,figsize=(10,7.4),layout='constrained')
 for ax,q,letter in [(axs[0,0],'2','A'),(axs[0,1],'18','B')]:
  title(ax,f'{letter}  Strong {"negative" if q=="2" else "positive"} current commitment')
  for i,r in enumerate(original):ax.plot(range(4),r['responses'][q]['current']['mean'],'o-',color=colors[i],label=labels[i],ms=3)
  ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set(ylabel='Native commitment fraction',ylim=(-.03,1.04));legend(ax)
 ax=axs[1,0];title(ax,'C  Continuous sensory responses')
 for i,r in enumerate(original):ax.plot(range(4),np.mean([r['responses'][q]['sensory']['mean'] for q in ['2','18']],axis=0),'o-',color=colors[i],ms=3)
 ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set_ylabel('Signed sensory output')
 ax=axs[1,1];title(ax,'D  PPC activity regimes')
 for j,c,l in [(0,gray,'Below threshold'),(1,blue,'Linear'),(2,red,'Upper bound')]:ax.plot(range(4),np.mean([np.load(FINAL/f'revision_2026_11_15/results/receiver_{n}.npz')['regimes'][320:400,:,j].mean(0) for n in names],axis=0),'o-',color=c,label=l)
 ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set_ylabel('PPC unit fraction');ax.legend(frameon=False,fontsize=8);save(fig,'Figure_6-3_Receiver_Function')
 # Dose selection and every fixed random direction on the test sample.
 fig,axs=plt.subplots(2,3,figsize=(9.8,7.8),layout='constrained');ax=axs[0,0];title(ax,'A  Selected doses from independent calibration');ax.plot(range(9),[r['selected_displacement'] for r in low],'o-',color=blue,label='Common displacement');ax.plot(range(9),[r['selected_attenuation'] for r in low],'s-',color=red,label='Recurrent attenuation');ax.set(yscale='log',ylabel='Fraction of original perturbation');ax.set_xticks(range(9),labels);ax.legend(frameon=False,fontsize=8)
 for ax,key,letter,text in [(axs[0,1],'current_H_change','B','Current history under ambiguity'),(axs[0,2],'later_H_change','C','Later history after ambiguity')]:
  title(ax,letter+'  '+text)
  for i,r in enumerate(low):
   yy=r['responses']['10'][key]['mean'];ax.scatter(np.full(8,i)+np.linspace(-.17,.17,8),yy[2:10],color=gray,s=12);ax.scatter(i,yy[1],color=blue,s=27);ax.scatter(i,yy[-1],color=red,s=27,marker='s')
  ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Historical contrast change')
 ax=axs[1,0];title(ax,'D  Test sensory effects of all directions')
 for i,r in enumerate(low):
  for qi,q in enumerate(('2','18')):ax.scatter(np.full(10,i)+(qi-.5)*.17,r['responses'][q]['sensory_change']['mean'][1:],s=12,color=colors[i])
 ax.axhspan(-.02,.02,color='#e6f0ed');ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Signed sensory response change')
 ax=axs[1,1];title(ax,'E  Current and later readout alignment')
 for i,r in enumerate(low):
  a=r['responses']['10']['current_A']['mean'];b=r['responses']['10']['later_A']['mean'];ax.plot([0,1],[a[0],b[0]],'-',color=colors[i]);ax.plot([0,1],[a[1],b[1]],'--',color=colors[i])
 ax.set_xticks([0,1],['Current','Later']);ax.set_ylabel('RMS readout alignment');ax.text(.02,.95,'Solid: own; dashed: small inward shift',va='top',transform=ax.transAxes,fontsize=8)
 ax=axs[1,2];title(ax,'F  Immediate activity differences after the shift')
 for i,n in enumerate(names):
  z=np.load(R/f'lowdose_{n}.npz');dr=z['initial_rate_difference'];ratio=np.linalg.norm(dr,axis=-1)/np.maximum(np.linalg.norm(dr[:,0],axis=-1)[:,None],1e-20);ax.scatter(np.full(8,i)+np.linspace(-.17,.17,8),ratio[:,2:10].mean(0),s=12,color=gray);ax.scatter(i,ratio[:,1].mean(),color=blue,s=25)
 ax.axhline(1,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Rate-difference norm / own');save(fig,'Figure_6-5_Low_Dose_Receiver')
 fig,axs=plt.subplots(2,2,figsize=(10,7.5),layout='constrained');ax=axs[0,0];title(ax,'A  Full gain-by-recurrence comparison')
 for key,c,l,ls in [('evidence_recurrence',blue,'Evidence + recurrence','-'),('pre_recurrence',gray,'Pre + recurrence','-'),('evidence_leak',red,'Evidence + leak','-'),('pre_leak',purple,'Pre + leak',':'),('reference_template',green,'Reference template','-')]:ax.plot(range(9),[p['scores'][key]['rmse'] for p in pred],'o',ls=ls,color=c,label=l,ms=3)
 ax.set_xticks(range(9),labels);ax.set_ylabel('Held-out trajectory RMSE');ax.legend(frameon=False,fontsize=7)
 for ax,q,letter in [(axs[0,1],'2','B'),(axs[1,0],'10','C'),(axs[1,1],'18','D')]:
  title(ax,letter+f'  {q} positive pips: template-relative skill')
  for key,c,l in [('evidence_recurrence',blue,'Gain propagation'),('reference_template',green,'Reference template')]:ax.plot(range(9),[p['scores'][key]['within_evidence'][q]['template_relative_skill'] for p in pred],'o-',color=c,label=l,ms=3)
  ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Skill relative to reference template');ax.legend(frameon=False,fontsize=8)
 save(fig,'Figure_7-3_Prediction_Benchmarks')

def temporal():
 ts=[json.loads((FINAL/f'revision_2026_11_15/results/temporal_{n}.json').read_text()) for n in names];fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for ax,key,letter,text in [(axs[0,0],'relative_change','A','Achieved receiving displacement'),(axs[0,1],'roughness_ratio_max','B','Whole-trajectory roughness constraint'),(axs[1,1],'changed_fraction','D','Coverage of changed time/context points')]:
  title(ax,letter+'  '+text)
  for i,r in enumerate(ts):ax.scatter(i,r[key],s=36,facecolors=colors[i] if r['changed_fraction']>0 else 'none',edgecolors=colors[i],linewidths=1.4)
  ax.set_xticks(range(9),labels);ax.set_ylabel({'relative_change':'Mean relative receiving displacement','roughness_ratio_max':'Maximum roughness / natural','changed_fraction':'Fraction with a changed message'}[key])
  if key=='roughness_ratio_max':ax.axhline(1.25,color=red,ls='--',lw=1)
 ax=axs[1,0];title(ax,'C  History under coherent alternative messages');err(ax,range(9),[r['history_loss'] for r in ts],blue)
 for i,r in enumerate(ts):
  if r['changed_fraction']==0:ax.scatter(i,r['history_loss']['mean'],s=45,facecolors='white',edgecolors=blue,zorder=6)
 ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Own minus transformed historical contrast');ax.text(.02,.96,'Open markers: natural-message fallback',transform=ax.transAxes,va='top',fontsize=8);save(fig,'Figure_4-5_Temporal_Source_Constraints')

if __name__=='__main__':prepare();fig4();fig6();fig7();supporting();temporal()
