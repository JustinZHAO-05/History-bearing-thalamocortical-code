"""Extended-data receiver competitor: analytic curve and executed discrete cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
R3=ROOT.parent/'round3_consciousness'
OUT=ROOT/'figures'

def main():
    out=OUT
    data=json.loads((R3/'reviews/independent_audit/saturation_counterexample.json').read_text())
    rows=data['rows'];a=.85;k=g=1.;z=np.linspace(-3,3,301)
    H=np.sinh(2*k*a)/(np.cosh(2*g*z)+np.cosh(2*k*a))
    direct=(np.tanh(g*z+k*a)-np.tanh(g*z-k*a))/2
    assert np.max(np.abs(H-direct))<1e-14
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,2,figsize=(10.8,4),layout='constrained')
    ax[0].plot(z,H,color='#a23839',lw=2,label='Historical output contrast')
    ax[0].plot(z,np.full_like(z,a),color='#27748e',ls='--',lw=1.5,label='Half stored-state difference')
    ax[0].set(xlabel='Current evidence z',ylabel='Paired half contrast',ylim=(-.03,.94),title='A  Saturation suppresses expression at fixed memory')
    ax[0].legend(frameon=False,fontsize=8)
    colors=['#27748e' if r['source_multiplier']==1 else '#a23839' for r in rows]
    for i,r in enumerate(rows):
        ax[1].plot([0,1],[r['current_output'],r['history_output']],'-o',color=colors[i],alpha=.65,lw=1,ms=5)
    for y in [-.75,.75]:ax[1].axhline(y,color='#777777',lw=.8,ls=':')
    ax[1].set(xticks=[0,1],xticklabels=['Strong current input','Later ambiguous input'],ylabel='Native output',ylim=(-1.1,1.1),title='B  One memory scalar, two instantaneous rate units')
    ax[1].text(.02,.44,'Blue: own history   Red: replaced history\nAll eight cases satisfy the 0.75 threshold',fontsize=8,transform=ax[1].transAxes)
    for ext in ('pdf','svg','png'):fig.savefig(out/('Figure_6-1_Saturation_Competitor.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(dict(current_evidence=z,history_halfcontrast=H,half_memory_difference=a)).to_csv(out/'Figure_6-1_analytic_source.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'Figure_6-1_discrete_source.csv',index=False)
    print('Saved analytic receiver competitor and the eight executed discrete cases.')
if __name__=='__main__':main()
