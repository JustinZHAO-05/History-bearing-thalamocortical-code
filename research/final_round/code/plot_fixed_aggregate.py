"""Final Figure 5 from the corrected native-cell-class experiment (v2)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/fixed_aggregate_geometry_v2'
OUT=ROOT/'figures'
RED='#a23839';BLUE='#27748e';GREY='#92999f'

def main():
    records=json.loads((SRC/'summary.json').read_text())
    assert len(records)==9 and (SRC/'completion.json').exists()
    names=[r['model'].replace('Gtune_RateRNNstructNUM','G').replace('sussano_RateRNNstructNUM','S') for r in records]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
    ax=axs[0,0];ax.axis('off')
    ax.text(0,1.01,'A  Reallocate inputs within native recipient classes',fontweight='bold',fontsize=10)
    ax.text(.02,.81,r'$b^{\pm}=\mu\pm d/2\quad\longrightarrow\quad\mu\pm d^{\prime}/2$',fontsize=14,color=BLUE)
    ax.text(.02,.64,'Each time, each history, each native E or I target set:',fontsize=10)
    ax.text(.03,.45,r'$\mathbf{1}^{\top}d_G^{\prime}=\mathbf{1}^{\top}d_G,\quad\|d_G^{\prime}\|^2=\|d_G\|^2$',fontsize=14)
    ax.text(.03,.28,r'$\mu^{\prime}=\mu,\qquad |d_i^{\prime}|\leq2\mu_i$',fontsize=14)
    ax.text(.02,.01,'Bounded triplet rotations: 64 sweeps, four fixed maps.\nE/I denotes recipient cell type; PUL arrivals remain positive.\nEach branch keeps its total E-input and total I-input.',fontsize=9,va='bottom')
    ax=axs[0,1]
    for i,r in enumerate(records):
        p=r['primary'];v=np.array(p['mapped_history'])
        ax.scatter(i+np.linspace(-.16,.16,4),v,s=28,c=GREY,label='Redistributed' if i==0 else None,zorder=3)
        ax.scatter(i,p['full_history'],marker='D',s=46,c=RED,label='Full own source' if i==0 else None,zorder=4)
    ax.axhline(0,color='#555555',lw=.7)
    ax.set(xticks=range(9),xticklabels=names,ylabel='Signed later history contrast')
    ax.set_title('B  Fine allocation has checkpoint-dependent effects',loc='left',fontweight='bold',fontsize=10)
    ax.legend(frameon=False,ncol=2,fontsize=8,loc='upper left')
    ymin=min(0,min(min(r['primary']['mapped_history']) for r in records))
    ymax=max(r['primary']['full_history'] for r in records)
    ax.set_ylim(ymin-.025,ymax+.12)
    ax=axs[1,0]
    for i,r in enumerate(records):
        vals=[c['outcomes']['current_author_success']['mean'] for c in r['conditions'][4:]]
        ax.scatter(i+np.linspace(-.16,.16,4),vals,s=28,color=GREY)
        ax.scatter(i,r['conditions'][2]['outcomes']['current_author_success']['mean'],marker='D',s=46,color=RED)
    ax.set(xticks=range(9),xticklabels=names,ylabel='Native current commitment success',ylim=(0,1.055))
    ax.set_title('C  Intervening judgment retains its native criterion',loc='left',fontweight='bold',fontsize=10)
    ax.text(.02,.10,'24 base histories × 4 strong inputs × 2 signs\nSame contexts as the archived index-block control',transform=ax.transAxes,fontsize=9)
    ax=axs[1,1]
    for i,r in enumerate(records):
        p=r['primary'];y=p['full_minus_mean_mapped'];lo,hi=p['ci95']
        ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='D',ms=5,color=RED,capsize=3)
    ax.axhline(0,color='#333333',lw=.8)
    ax.set(xticks=range(9),xticklabels=names,ylabel='Full − mean redistributed history contrast')
    ax.set_title('D  Paired history loss under fixed aggregate input',loc='left',fontweight='bold',fontsize=10)
    ax.text(.02,.94,'95% base-history bootstrap intervals',transform=ax.transAxes,fontsize=8,va='top')
    vals=[x for r in records for x in r['primary']['ci95']]
    ax.set_ylim(min(-.025,min(vals)-.03),max(vals)+.12)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/('Figure_5_Fixed_Aggregate.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12.2,3.8),layout='constrained')
    rows=[]
    for i,r in enumerate(records):
        p=r['primary']
        for j,c in enumerate(r['checks']['maps']):
            row=dict(checkpoint=r['model'],map=j,full_history=p['full_history'],mapped_history=p['mapped_history'][j],
                     native_current=r['conditions'][j+4]['outcomes']['current_author_success']['mean'],**c)
            rows.append(row)
            axs[0].scatter(i+(j-1.5)/12,c['within_class_residual_cosine_mean'],s=27,color=BLUE)
            axs[1].scatter(i+(j-1.5)/12,c['relative_difference_change_rms'],s=27,color=BLUE)
            axs[2].scatter(c['relative_difference_change_rms'],p['full_history']-p['mapped_history'][j],s=27,color=BLUE)
        axs[2].annotate(names[i],(np.mean([c['relative_difference_change_rms'] for c in r['checks']['maps']]),p['full_minus_mean_mapped']),fontsize=8,xytext=(3,3),textcoords='offset points')
    axs[0].set(xticks=range(9),xticklabels=names,ylim=(-.1,1.05),ylabel='Mean within-class residual cosine',title='A  Achieved geometry after bounded rotations')
    axs[1].set(xticks=range(9),xticklabels=names,ylabel='RMS relative source-difference change',title='B  Feasibility constrains redistribution')
    axs[2].set(xlabel='RMS relative source-difference change',ylabel='Full − redistributed history contrast',title='C  All four maps and all checkpoints retained')
    axs[2].axhline(0,color='#777777',lw=.7)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/('Figure_5-1_Achieved_Redistribution.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'Figure_5_source_data.csv',index=False)
    summary=[]
    for r in records:
        p=r['primary'];summary.append(dict(checkpoint=r['model'],full_history=p['full_history'],mapped_mean=np.mean(p['mapped_history']),
            loss=p['full_minus_mean_mapped'],ci_low=p['ci95'][0],ci_high=p['ci95'][1],
            mean_residual_cosine=np.mean([c['within_class_residual_cosine_mean'] for c in r['checks']['maps']]),
            min_native_current=min(c['outcomes']['current_author_success']['mean'] for c in r['conditions'][4:])))
    pd.DataFrame(summary).to_csv(OUT/'Figure_5_summary.csv',index=False)
    print('Saved corrected native-cell-class Figure 5, Figure 5-1 and source tables.')

if __name__=='__main__':main()
