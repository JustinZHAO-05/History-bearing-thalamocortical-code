from figures import *
def main():
 ts=[json.loads((R/f'temporal_{n}.json').read_text()) for n in names]
 pc=json.loads((R/'prediction_comparison.json').read_text())
 fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 ax=axs[0,0];title(ax,'A  Whole-trajectory source displacement');ax.bar(range(9),[x['relative_change'] for x in ts],color=blue);ax.set_xticks(range(9),labels);ax.set_ylabel('Relative receiving-difference change')
 ax=axs[0,1];title(ax,'B  Temporal variation remains near natural');ax.bar(range(9),[x['roughness_ratio_max'] for x in ts],color=green);ax.axhline(1.25,color=red,ls='--');ax.set_xticks(range(9),labels);ax.set_ylabel('Maximum per-context roughness ratio')
 ax=axs[1,0];title(ax,'C  Historical effects of temporally matched maps');err(ax,range(9),[x['history_loss']['mean'] for x in ts],[x['history_loss']['ci'] for x in ts],blue);ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Own minus mapped historical contrast')
 ax=axs[1,1];title(ax,'D  Feasible changes across the full interval');ax.bar(range(9),[x['changed_fraction'] for x in ts],color=gray);ax.set_xticks(range(9),labels);ax.set_ylabel('Fraction of changed time/context points')
 save(fig,'Figure_4-5_Temporal_Source_Constraints')
 fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 for qi,q in enumerate(('2','18')):
  ax=axs[0,qi];title(ax,('A' if qi==0 else 'B')+'  Strong-stimulus current commitment')
  for i,r in enumerate(rec):ax.plot(range(4),r['responses'][q]['current']['mean'],'o-',color=colors[i],ms=3,label=labels[i])
  ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set_ylabel('Native commitment fraction');ax.set_ylim(-.03,1.03)
 ax=axs[1,0];title(ax,'C  Strong-stimulus sensory response')
 for i,r in enumerate(rec):ax.plot(range(4),np.mean([r['responses'][q]['sensory']['mean'] for q in ['2','18']],axis=0),'o-',color=colors[i],ms=3)
 ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set_ylabel('Signed continuous sensory output')
 ax=axs[1,1];title(ax,'D  PPC activation regimes during current response')
 for j,c,l in [(0,gray,'Below threshold'),(1,blue,'Linear'),(2,red,'Upper bound')]:
  vals=np.array([np.load(R/f'receiver_{n}.npz')['regimes'][320:400,:,j].mean(0) for n in names]);ax.plot(range(4),vals.mean(0),'o-',color=c,label=l)
 ax.set_xticks(range(4),['Own','Inward','Random','Pause']);ax.set_ylabel('Mean fraction of PPC units');ax.legend(frameon=False,fontsize=8)
 save(fig,'Figure_6-3_Receiver_Function')
 fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
 ax=axs[0,0];title(ax,'A  Independent response template provides a benchmark')
 for key,c,l in [('gain',blue,'Gain propagation'),('template',green,'Reference mean template')]:ax.plot(range(9),[x['scores'][key]['rmse'] for x in pc],'o-',color=c,label=l)
 ax.set_xticks(range(9),labels);ax.set_ylabel('Held-out trajectory RMSE');ax.legend(frameon=False)
 for ax,q,letter in zip([axs[0,1],axs[1,0],axs[1,1]],['2','10','18'],['B','C','D']):
  title(ax,letter+'  Within-evidence prediction: '+q+' positive pips')
  for key,c,l in [('gain',blue,'Gain'),('template',green,'Template')]:ax.plot(range(9),[x['scores'][key]['within_evidence'][q]['r2'] for x in pc],'o-',color=c,label=l)
  ax.axhline(0,color=gray,lw=.6);ax.set_xticks(range(9),labels);ax.set_ylabel('Within-evidence explained variation');ax.legend(frameon=False)
 save(fig,'Figure_7-3_Prediction_Benchmarks')
if __name__=='__main__':main()
