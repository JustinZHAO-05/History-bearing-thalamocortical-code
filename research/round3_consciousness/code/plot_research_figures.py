"""Standalone publication-format plots made only from executed R3 results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from rnn_core import ROOT

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':10,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':250})
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
blue='#2379a5';red='#c64343';green='#22866b';purple='#7753a5';gray='#85929b'
payload=json.loads((ROOT/'results/analysis/rnn_inference.json').read_text())
names=[x['model'] for x in payload['history_confirmation']]
labels=[x.replace('Gtune_RateRNNstructNUM','G').replace('sussano_RateRNNstructNUM','S') for x in names]
colors=plt.cm.viridis(np.linspace(.08,.9,9))

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)

def carriers():
    h=payload['history_confirmation'];fig=plt.figure(figsize=(7.3,4.8),layout='constrained');gs=fig.add_gridspec(1,2,width_ratios=[1.35,1])
    ax=fig.add_subplot(gs[0,0]);cs=['natural','intact_mean_100','intact_gain_100','intact_pattern_100','intact_opposite_100','intact_opposite_200']
    a=np.array([[x['condition_margins'][c]['mean'] for c in cs] for x in h])
    im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-.7,vmax=.7)
    ax.set_yticks(range(9),labels);ax.set_xticks(range(6),['Natural','Mean\n500 ms','Amplitude\n500 ms','Pattern\n500 ms','Opposite\n500 ms','Opposite\n1000 ms'],rotation=38,ha='right')
    ax.set_title('A  Which supplied history governs the decision?',loc='left',fontweight='bold',fontsize=9)
    for i in range(9):
        for j in range(6):ax.text(j,i,f'{a[i,j]:.2f}',ha='center',va='center',fontsize=6.2,color='white' if abs(a[i,j])>.43 else '#17202a')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.09);cb.set_label('Native half-pair history contrast')
    ax=fig.add_subplot(gs[0,1]);ys=np.arange(9)
    for i,x in enumerate(h):
        v=x['amplitude_minus_pattern_swap_effect'];l=x['l2_amplitude_minus_pattern_swap_effect']
        ax.errorbar(v['mean'],i,xerr=[[v['mean']-v['ci95'][0]],[v['ci95'][1]-v['mean']]],fmt='o',color=colors[i],markersize=5,capsize=2)
        ax.plot(l['mean'],i+.13,'x',color=colors[i],markersize=4)
    ax.axvline(0,c='#6c757d',lw=.8);ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlabel('Amplitude-swap effect minus pattern-swap effect')
    ax.set_title('B  Both carrier coordinates can encode history',loc='left',fontweight='bold',fontsize=9)
    ax.text(.02,-.23,'Left: pattern dominates   Right: amplitude dominates\nCircles: mean-current normalization (95% context CI)\nCrosses: L2 normalization; all 9 checkpoints retained',transform=ax.transAxes,fontsize=6.8,va='top')
    save(fig,'Figure_2_History_Carriers')

def sequential():
    s=payload['sequential_confirmation'];p=ROOT/'results/sequential_confirmation_v1';cs=json.loads((p/'conditions.json').read_text());ix={c['name']:i for i,c in enumerate(cs)}
    fig=plt.figure(figsize=(7.3,6.8),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[.8,1.4,1.4])
    ax=fig.add_subplot(gs[0,:]);ax.set_xlim(-150,4200);ax.set_ylim(-.1,1.1)
    for x,w,y,c,label in [(0,1000,.72,red,'Source replacement'),(1250,300,.25,blue,'Strong\nstimulus'),(1600,400,.25,blue,'Current\nresponse'),(3250,300,.72,green,'50:50\nstimulus'),(3600,400,.72,green,'History\nresponse')]:
        ax.add_patch(Rectangle((x,y),w,.15,color=c,alpha=.85));ax.text(x+w/2,y+.19,label,ha='center',va='bottom',fontsize=6.5,linespacing=1)
    ax.axvline(500,color=red,ls=':',lw=1);ax.text(500,.58,'500 ms arm',ha='center',fontsize=6.8,color=red)
    ax.set_yticks([]);ax.set_xticks([0,1000,2000,3000,4000]);ax.set_xlabel('Time from end of trial 4 (ms)');ax.set_title('A  Rewrite earlier history, then make two successive decisions',loc='left',fontweight='bold')
    ax.text(-.02,1.2,'48 new base histories × 4 matched strong stimuli × 2 historical signs',transform=ax.transAxes,fontsize=7)
    ax=fig.add_subplot(gs[1,0]);conds=['natural','intact_mean_200','intact_opposite_200'];xx=np.arange(3)
    cur_target=np.load(p/'contexts.npz')['counts'][:,:,4]>10;curves=[]
    for i,n in enumerate(names):
        d=np.load(p/(n+'.npz'));y=d['margin']
        hist=[np.mean((y[:,ix[c],:,1]>0)==np.array([True,False])) for c in conds]
        curves.append([np.mean((y[:,ix[c],:,0]>0)==cur_target) for c in conds])
        ax.plot(xx,hist,'o-',color=colors[i],alpha=.85,ms=3,lw=.9)
    curves=np.asarray(curves)
    ax.plot(xx,curves.mean(0),'s--',color=blue,ms=4,lw=1.3,label='Current sign accuracy: model mean')
    ax.fill_between(xx,curves.min(0),curves.max(0),color=blue,alpha=.15)
    ax.axhline(.5,c='gray',ls=':',lw=.7);ax.set_xticks(xx,['Natural','Mean source','Opposite source']);ax.set_ylim(-.03,1.08);ax.set_ylabel('Accuracy relative to original target');ax.set_title('B  Later historical choice changes',loc='left',fontweight='bold');ax.legend(loc='lower left',fontsize=6.5)
    ax=fig.add_subplot(gs[1,1])
    for i,x in enumerate(s):
        e=x['source_effect']['200'];v=[e['current_mean_abs_effect']['mean'],e['history_mean_abs_effect']['mean']]
        ax.plot([0,1],v,'o-',color=colors[i],lw=.9,ms=4)
    ax.set_yscale('log');ax.set_xlim(-.15,1.3);ax.set_xticks([0,1],['Current response','History response']);ax.set_ylabel('Mean absolute native-output change');ax.set_title('C  Continuous output shows graded selectivity',loc='left',fontweight='bold')
    ax=fig.add_subplot(gs[2,0])
    for i,n in enumerate(names):
        d=np.load(p/(n+'.npz'));y=d['margin']
        v=[np.mean((y[:,ix[c],:,1]>0)==np.array([True,False])) for c in ['PPC_mean_mean_100','PPC_mean_own_100']]
        ax.plot([0,1],v,'o-',color=colors[i],ms=4,lw=.9)
    ax.axhline(.5,c='gray',ls=':',lw=.7);ax.set_xticks([0,1],['Mean source','Own-history source']);ax.set_ylim(.43,1.04);ax.set_ylabel('Later history sign accuracy');ax.set_title('D  Source restores use after cortical averaging',loc='left',fontweight='bold')
    ax=fig.add_subplot(gs[2,1]);xv=np.arange(9)
    for i,n in enumerate(names):
        d=np.load(p/(n+'.npz'));j=ix['intact_opposite_200'];lo=d['response_min'][:,j,:,1];hi=d['response_max'][:,j,:,1]
        pos=(hi>=.75)&(lo>=-.75);neg=(lo<=-.75)&(hi<=.75)
        right=np.c_[pos[:,0],neg[:,1]].mean();wrong=np.c_[neg[:,0],pos[:,1]].mean();none=(~pos&~neg).mean()
        ax.bar(i,right,color=green,width=.7);ax.bar(i,wrong,bottom=right,color=red,width=.7);ax.bar(i,none,bottom=right+wrong,color='#d9dfe2',width=.7)
    ax.set_xticks(xv,labels);ax.set_ylim(0,1.03);ax.set_ylabel('Native commitment fraction');ax.set_title('E  Reversal versus absent commitment',loc='left',fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=green,label='Original'),Patch(color=red,label='Opposite'),Patch(color='#d9dfe2',label='Neither')],loc='upper center',ncol=3,bbox_to_anchor=(.5,-.15),fontsize=6.5)
    save(fig,'Figure_3_Sequential_Use')

def expression():
    cc=payload['finite_calculus'];s=payload['sequential_confirmation'];fig=plt.figure(figsize=(7.3,6.3),layout='constrained');gs=fig.add_gridspec(2,2)
    ax=fig.add_subplot(gs[0,0])
    for i,x in enumerate(cc):
        v=x['conditions'][0]['rate_difference_norm_mean'];ax.plot([0,1],v,'o-',color=colors[i],ms=4,lw=.9)
    ax.set_xticks([0,1],['Current','History']);ax.set_ylabel('RMS norm of paired activity difference');ax.set_title('A  Historical differences remain in activity',loc='left',fontweight='bold',fontsize=9)
    ax=fig.add_subplot(gs[0,1])
    for i,x in enumerate(cc):
        v=x['conditions'][0]['rms_readout_alignment_mean'];ax.plot([0,1],v,'o-',color=colors[i],ms=4,lw=.9)
    ax.set_xticks([0,1],['Current','History']);ax.set_ylabel('RMS output projection / activity norm');ax.set_yscale('log');ax.set_title('B  The same trained readout becomes sensitive',loc='left',fontweight='bold',fontsize=9)
    ax=fig.add_subplot(gs[1,0]);data=np.array([x['conditions'][1]['component_half_pair_output_mean'][1] for x in cc])
    im=ax.imshow(data,aspect='auto',cmap='RdBu',vmin=-.65,vmax=.65);ax.set_yticks(range(9),labels);ax.set_xticks(range(4),['Initial\nPPC','Initial\nPUL/TRN','Source\namplitude','Source\npattern']);ax.set_title('C  Finite response account after replay',loc='left',fontweight='bold',fontsize=9)
    for i in range(9):
        for j in range(4):ax.text(j,i,f'{data[i,j]:.2f}',ha='center',va='center',fontsize=6.3,color='white' if abs(data[i,j])>.4 else '#17202a')
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.05,pad=.12);cb.set_label('Half-pair output contribution (conditional account)',fontsize=7)
    ax=fig.add_subplot(gs[1,1]);data=np.array([x['regional_output_account']['200']['history'] for x in s]);bottom=np.zeros(9);negative=np.zeros(9)
    for j,(c,name) in enumerate([(blue,'PPC'),(green,'PUL'),(purple,'TRN')]):
        ax.bar(np.arange(9),data[:,j],bottom=np.where(data[:,j]>=0,bottom,negative),color=c,label=name,width=.7)
        bottom+=np.maximum(data[:,j],0);negative+=np.minimum(data[:,j],0)
    ax.set_xticks(range(9),labels);ax.set_ylabel('Signed output change: natural minus opposite');ax.set_title('D  Readout expression is distributed',loc='left',fontweight='bold',fontsize=9);ax.legend(ncol=3,fontsize=6.5,loc='upper center',bbox_to_anchor=(.5,-.16))
    save(fig,'Figure_4_Retention_and_Expression')

def weak_boundary():
    p=ROOT/'results/factorization_pilot_current';ctx=np.load(p/'contexts.npz');coh=(ctx['counts'][:,0,-1]-10)/10
    fig,axs=plt.subplots(3,3,figsize=(7.3,5.8),sharex=True,sharey=True,layout='constrained')
    for i,n in enumerate(names):
        d=np.load(p/(n+'.npz'));y=d['margin'][:,0]
        q=sorted(np.unique(coh));a=[y[coh==v].mean() for v in q]
        axs.flat[i].plot(q,a,'o-',color=colors[i],ms=3);axs.flat[i].axhline(0,c='gray',lw=.6);axs.flat[i].axvline(0,c='gray',lw=.6);axs.flat[i].set_title(labels[i],loc='left')
    fig.supxlabel('Current stimulus evidence (original trial 5)');fig.supylabel('Native response-window mean output');fig.suptitle('Preserved qualification failure: nonmonotonic weak-stimulus choices\nExploratory pilot, 4 common histories per mixture; no model excluded',fontsize=10)
    save(fig,'Supplement_1_Weak_Stimulus_Boundary')

if __name__=='__main__':
    carriers();sequential();expression();weak_boundary()
