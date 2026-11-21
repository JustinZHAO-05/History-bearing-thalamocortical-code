"""Figure 4 and Figure 4-2 from the frozen final-round source-direction arrays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/source_geometry_v1'
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
COL='#a23839';BLUE='#27748e';GREY='#92999f'

def main():
    records=json.loads((SRC/'summary.json').read_text())
    assert len(records)==9 and (SRC/'completion.json').exists()
    names=[x['model'].replace('Gtune_RateRNNstructNUM','G').replace('sussano_RateRNNstructNUM','S') for x in records]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(11.5,7.7),gridspec_kw={'height_ratios':[1,1]},layout='constrained')
    ax=axs[0,0];ax.axis('off')
    ax.text(0,1.02,'A  Same background and magnitude; changed direction',fontweight='bold',transform=ax.transAxes)
    ax.text(.03,.81,r'Natural pair: $\mu=(b^++b^-)/2$, $d=b^+-b^-$',fontsize=11)
    ax.text(.03,.58,r'Learned axis $v$   or   random $q\perp v$',fontsize=12,color=BLUE)
    ax.annotate('',xy=(.5,.43),xytext=(.5,.53),arrowprops={'arrowstyle':'->','color':BLUE})
    ax.text(.03,.33,r'$\widetilde d=\mathrm{clip}(k\,\mathrm{sign}(d^\top v)q,-2\mu,2\mu)$',fontsize=12)
    ax.text(.03,.13,r'$\|\widetilde d\|_2=\|d\|_2,\qquad \widetilde b^{\pm}=\mu\pm\widetilde d/2$',fontsize=12)
    ax.text(.03,-.02,'All network states start from their pair mean.\nOne-second source replay ends before the shared stimulus.',fontsize=9,va='top')
    ax=axs[0,1]
    for i,r in enumerate(records):
        z=r['primary'];vals=np.abs(z['random_signed_history'])
        jitter=np.linspace(-.17,.17,len(vals))
        ax.scatter(i+jitter,vals,s=17,color=GREY,alpha=.7,label='Random directions' if i==0 else None)
        ax.scatter(i,z['learned_history'],s=47,marker='D',color=COL,label='Learned axis' if i==0 else None,zorder=4)
        h=r['conditions'][3]['outcomes']['history_signed_margin']['mean']
        ax.scatter(i,h,s=44,facecolors='none',edgecolors=BLUE,marker='o',label='Full source' if i==0 else None,zorder=3)
    ax.set(xticks=range(9),xticklabels=names,ylabel='Absolute mean history contrast',ylim=(-.015,.78))
    ax.set_title('B  Learned geometry supports stronger reconstruction',loc='left',fontweight='bold',fontsize=10)
    ax.legend(frameon=False,fontsize=8,loc='upper left',ncol=3)
    ax=axs[1,0]
    for i,r in enumerate(records):
        vals=[c['outcomes']['current_author_success']['mean'] for c in r['conditions'][7:]]
        ax.scatter(i+np.linspace(-.17,.17,16),vals,s=17,color=GREY,alpha=.7)
        ax.scatter(i,r['conditions'][6]['outcomes']['current_author_success']['mean'],s=47,marker='D',color=COL,zorder=4)
    ax.set(xticks=range(9),xticklabels=names,ylim=(0,1.055),ylabel='Native current commitment success')
    ax.set_title('C  Current judgments under direction controls',loc='left',fontweight='bold',fontsize=10)
    ax.axhline(.5,color='#bbbbbb',lw=.6,ls=':')
    ax.text(.02,.11,'24 fresh base histories × 4 strong stimuli × 2 signs\n16 fixed random directions per checkpoint',transform=ax.transAxes,fontsize=9)
    ax=axs[1,1]
    for i,r in enumerate(records):
        z=r['primary'];y=z['learned_minus_mean_random_absolute_history'];lo,hi=z['ci95']
        ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='D',color=COL,capsize=3,ms=5)
    ax.axhline(0,color='#333333',lw=.8)
    ax.set(xticks=range(9),xticklabels=names,ylabel='Learned − mean |random| contrast',ylim=(-.02,.72))
    ax.set_title('D  Advantage persists across fresh histories',loc='left',fontweight='bold',fontsize=10)
    ax.text(.02,.05,'95% intervals: base-history bootstrap;\nsampled axes held fixed',transform=ax.transAxes,fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/('Figure_4_Source_Geometry.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12.2,3.7),layout='constrained')
    table=[]
    for i,r in enumerate(records):
        for j,g in enumerate(r['geometry']):
            h=r['primary']['random_signed_history'][j]
            table.append(dict(checkpoint=r['model'],history_contrast=h,**g))
            axs[0].scatter(i+(j-7.5)/50,g['postclip_v_absolute_cosine_mean'],s=16,color=GREY)
            axs[1].scatter(g['postclip_v_absolute_cosine_mean'],abs(h),s=16,color=BLUE,alpha=.6)
        z=r['primary'];axs[2].scatter(i,z['n_random_absolute_at_least_learned'],color=COL,s=35)
    axs[0].set(xticks=range(9),xticklabels=names,ylabel='Mean |post-clip cosine with learned axis|',title='A  Clipping changes source geometry')
    axs[1].set(xlabel='Mean |post-clip cosine with learned axis|',ylabel='|Mean historical contrast|',title='B  All sampled directions retained')
    axs[2].set(xticks=range(9),xticklabels=names,yticks=range(0,17,2),ylim=(-.5,16.5),ylabel='Random axes at least as strong (of 16)',title='C  Effective alternatives remain')
    for ext in ['pdf','svg','png']:fig.savefig(OUT/('Figure_4-2_Geometry_Controls.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(table).to_csv(OUT/'Figure_4_source_data.csv',index=False)
    rows=[]
    for r in records:
        p=r['primary'];row={k:v for k,v in p.items() if isinstance(v,(float,int))};row['checkpoint']=r['model'];row['ci_low'],row['ci_high']=p['ci95'];rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'Figure_4_summary.csv',index=False)
    print('Saved Figure 4, Figure 4-2, and exact source tables.')

if __name__=='__main__':main()
