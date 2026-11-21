#!/usr/bin/env python3
"""Revised Figure 2, revised Figure 4 and Extended Figure 4-3 from executed source-space experiments."""
from pathlib import Path
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
root=Path(__file__).resolve().parents[1]
out=root/'figures';out.mkdir(exist_ok=True)
RES=root/'results'
summary=json.loads((RES/'source_space_findings.json').read_text())
rows=summary['rows'];names=[z['model'] for z in rows]
full_names=['Gtune_RateRNNstructNUM'+n[1:] if n.startswith('G') else 'sussano_RateRNNstructNUM'+n[1:] for n in names]
raw=[np.load(RES/f'{name}.npz') for name in full_names]
exact=[np.load(RES/'reflection_v2'/f'{name}.npz') for name in full_names]
blue='#256382';rust='#b54439';green='#388474';gray='#818c9a';purple='#795a95';ink='#243140'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,
                     'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for suf in ('pdf','png','svg'):
        fig.savefig(out/(name+'.'+suf),dpi=260,bbox_inches='tight')
    plt.close(fig)

def a(ax,t):ax.set_title(t,loc='left',fontweight='bold',fontsize=10,pad=10)

def figure2():
    fig=plt.figure(figsize=(10.1,9.0),layout='constrained')
    gs=fig.add_gridspec(3,2,height_ratios=[.8,1.65,1.5],width_ratios=[1.12,1])
    ax=fig.add_subplot(gs[0,:]);ax.set_xlim(0,10);ax.set_ylim(0,1);ax.axis('off')
    a(ax,'A   Reconstruct a difference after averaging all 500 recipient states')
    ax.text(.2,.73,'Distinct pasts',color=ink,weight='bold',fontsize=10)
    ax.text(3.35,.73,'Pair-average state',color=ink,weight='bold',fontsize=10)
    ax.text(6.6,.73,'Replay into native circuit',color=ink,weight='bold',fontsize=10)
    ax.text(.3,.47,'History +    History −',fontsize=11)
    ax.annotate('',xy=(3.2,.43),xytext=(1.95,.43),arrowprops=dict(arrowstyle='->',color=gray,lw=1.7))
    ax.add_patch(Rectangle((3.55,.28),2.3,.34,edgecolor=gray,facecolor='#f3f5f7',linewidth=1.2))
    ax.text(4.7,.45,'x⁺(t₀) = x⁻(t₀)',ha='center',va='center',fontsize=12)
    ax.annotate('',xy=(6.52,.45),xytext=(5.95,.45),arrowprops=dict(arrowstyle='->',color=gray,lw=1.7))
    ax.text(6.73,.47,'Own   Mean   Opposite',fontsize=10,color=blue)
    ax.text(6.73,.24,'Same future sensory input and trained readout',fontsize=9,color=gray)
    ax=fig.add_subplot(gs[1,0]);a(ax,'B   A returning message rebuilds historical output from zero')
    x=np.arange(800)*5
    # G20 is chosen as the first checkpoint by fixed sorted order, no outcome selection.
    z=raw[0];base=z['output_trace'];
    for j,label,color in [(1,'Own-history replay',blue),(2,'Pair-mean replay',gray),(7,'Opposite-history replay',rust),(3,'Learned source mode',green)]:
        h=(base[:,:,j,0]-base[:,:,j,1]).mean(axis=1)/2
        ax.plot(x,h,label=label,color=color,linewidth=1.7 if j in [1,7] else 1.15,ls='--' if j==2 else '-')
    for lo,hi,c in [(0,1000,'#dceaf0'),(1600,2000,'#fff1df'),(3600,4000,'#e9f1e7')]:ax.axvspan(lo,hi,color=c,zorder=-2)
    ax.axhline(0,color='#9fa6ad',lw=.8);ax.axvline(1000,color='#657b8b',lw=.8,ls=':')
    ax.set_xlim(0,4000);ax.set_xlabel('Time from intervention onset (ms)');ax.set_ylabel('Signed half-pair native output')
    ax.legend(frameon=False,fontsize=7.5,loc='upper center',ncol=2)
    ax=fig.add_subplot(gs[1,1]);a(ax,'C   Reconstruction and reversal across nine checkpoints')
    for i,z in enumerate(raw):
        h=z['margin'];val=(h[:,:,0,1]-h[:,:,1,1]).mean(axis=0)/2
        ax.plot([i]*3+[i],val[[1,2,3,7]],' ',alpha=.6)
        for j,c,m in [(1,blue,'o'),(2,gray,'s'),(3,green,'^'),(7,rust,'D')]:
            ax.scatter(i,val[j],color=c,s=29 if j in (1,7) else 21,marker=m,zorder=3)
    ax.axhline(0,color='#9fa6ad',lw=.8)
    ax.set_xticks(range(9),names);ax.set_ylabel('Final history-window contrast')
    ax.set_ylim(-.85,.9)
    ax.text(.02,.03,'Own / Mean / Learned / Opposite',transform=ax.transAxes,color=gray,fontsize=7)
    ax=fig.add_subplot(gs[2,0]);a(ax,'D   Population separation re-emerges, then persists')
    zz=raw[0]['history_norm_trace'].mean(axis=1)
    for j,lab,col in [(1,'Own',blue),(2,'Mean',gray),(7,'Opposite',rust)]:
        ax.plot(x,zz[:,j],color=col,lw=1.65,label=lab)
    ax.axvspan(0,1000,color='#dceaf0',zorder=-2);ax.axvline(1000,color='#657b8b',ls=':',lw=.8)
    ax.set_xlim(0,4000);ax.set_ylabel('Mean Euclidean rate separation');ax.set_xlabel('Time from intervention onset (ms)')
    ax.legend(frameon=False,ncol=3)
    ax=fig.add_subplot(gs[2,1]);a(ax,'E   The next strong judgment remains correct')
    # Native current commitment success under reset + own/opposite/learned sources.
    js=[1,3,7];xx=np.arange(9);width=.24
    colors=[blue,green,rust]
    for k,j in enumerate(js):
        vals=[np.mean(z['current_success'][:,j]) for z in raw]
        ax.bar(xx+(k-1)*width,vals,width=width,label=['Own','Learned','Opposite'][k],color=colors[k],alpha=.83)
    ax.set_xticks(xx,names);ax.set_ylim(.78,1.04);ax.set_ylabel('Native current commitment');ax.legend(frameon=False,ncol=3,loc='lower left')
    fig.suptitle('Which past is reconstructed? Causal replay into initially identical recipients',fontsize=14,weight='bold')
    save(fig,'Figure_2_Reconstruction_and_Temporal_Access')

def figure4():
    fig,axs=plt.subplots(2,2,figsize=(10.2,8.0),layout='constrained')
    ax=axs[0,0];ax.axis('off');a(ax,'A   A bounded PUL source connects to PPC')
    ax.text(.03,.73,'PUL source activity',fontsize=11,weight='bold',transform=ax.transAxes)
    ax.text(.03,.62,r'$0\leq r_{T,i}^{\pm}\leq20$',fontsize=13,transform=ax.transAxes,color=blue)
    ax.text(.03,.43,'Source history difference',fontsize=10,transform=ax.transAxes)
    ax.text(.03,.32,r'$d_T=r_T^+-r_T^-$',fontsize=12,transform=ax.transAxes)
    ax.annotate('',xy=(.64,.49),xytext=(.5,.49),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=1.6))
    ax.text(.69,.68,'PPC arrival',weight='bold',fontsize=11,transform=ax.transAxes)
    ax.text(.69,.54,r'$B\,d_T$',color=rust,fontsize=14,transform=ax.transAxes)
    ax.text(.69,.37,'Original trained B',fontsize=9,transform=ax.transAxes)
    ax.text(.02,.07,'The source-reachable input set is a bounded\n70-D box mapped through the frozen matrix B.',fontsize=9,color=gray,transform=ax.transAxes)
    ax=axs[0,1];a(ax,'B   Historical source differences are low-dimensional')
    values=[z['source_learned_variance'] for z in rows]
    ax.bar(range(9),values,color=blue,alpha=.88);ax.set_xticks(range(9),names);ax.set_ylim(0,1.04)
    ax.set_ylabel('First source mode / total difference energy')
    ax=axs[1,0];a(ax,'C   Energy matching is often infeasible')
    for i,z in enumerate(rows):
        jname=full_names[i];r=json.loads((RES/(jname+'.json')).read_text())
        v=[k['matching_fraction_1pct'] for k in r['geometry']]
        ax.scatter([i-.12],v[:1],s=45,marker='D',color=green,zorder=4)
        ax.scatter([i+.04,i+.13,i+.22],v[1:],color=gray,s=16,alpha=.9)
    ax.set_xticks(range(9),names);ax.set_ylim(-.03,1.03)
    ax.set_ylabel('Fraction of time/context points\nwithin 1% of target energy')
    ax.text(.03,.04,'Diamond = learned; dots = three random source axes',color=gray,transform=ax.transAxes,fontsize=8)
    ax=axs[1,1];a(ax,'D   Learned source codes retain reconstruction')
    xx=np.arange(9);hist=[r['full_history'] for r in rows];src=[r['single_source_history'] for r in rows]
    rand=[r['source_random_abs_mean'] for r in rows]
    ax.plot(xx,hist,'o-',color=ink,ms=4,label='Native source')
    ax.plot(xx,src,'s-',color=green,ms=4,label='Learned source mode')
    ax.plot(xx,rand,'^:',color=gray,ms=4,label='Mean |random output|')
    ax.set_xticks(xx,names);ax.set_ylabel('Later historical output contrast')
    ax.set_ylim(0,.82);ax.legend(frameon=False,fontsize=8,loc='lower left')
    fig.suptitle('Historical message organization is constrained by the original PUL source space',weight='bold',fontsize=13)
    save(fig,'Figure_4_Source_Reachability')

def supplement():
    fig,axs=plt.subplots(2,2,figsize=(10.2,7.8),layout='constrained')
    ax=axs[0,0];a(ax,'A   Exact receiver-energy + E/I-aggregate constraints')
    ax.axis('off')
    terms=[r'$d_T^\prime=d_T+\alpha z, \qquad A z=0$',
           r'$\alpha=-2\,d_T^\top B^\top Bz/(z^\top B^\top Bz)$',
           r'$\|Bd_T^\prime\|^2=\|Bd_T\|^2,\quad A d_T^\prime=A d_T$',
           r'$|d_{T,i}^\prime|\leq 2\min(\mu_i,20-\mu_i)$']
    for k,t in enumerate(terms):ax.text(.06,.78-k*.19,t,fontsize=13 if k!=2 else 12,transform=ax.transAxes)
    ax.text(.06,.04,'Two independently generated feasible maps per checkpoint;\nall selection uses source geometry only.',transform=ax.transAxes,color=gray,fontsize=9)
    ax=axs[0,1];a(ax,'B   Available reallocation varies across learned circuits')
    xx=np.arange(9)
    for i,z in enumerate(rows):
        ax.scatter(i,z['exact_receiver_change'],color=blue,s=52)
        ax.plot([i,i],[0,z['exact_receiver_change']],color=blue,lw=1,alpha=.35)
    ax.set_xticks(xx,names);ax.set_ylabel('||B(d\u2032 − d)|| / ||Bd|| (time mean)')
    ax.set_ylim(0,.49)
    ax=axs[1,0];a(ax,'C   Historical function under exactly matched source rotations')
    yy=np.array([z['exact_change_H'] for z in rows]);lo=np.array([z['exact_change_H_CI_low'] for z in rows]);hi=np.array([z['exact_change_H_CI_high'] for z in rows])
    ax.errorbar(xx,yy,yerr=[yy-lo,hi-yy],fmt='o',color=rust,capsize=2)
    ax.axhline(0,lw=.8,color='#8c969f');ax.set_xticks(xx,names)
    ax.set_ylabel('Native own-source H − rotated-source H')
    ax=axs[1,1];a(ax,'D   Matching and current-decision controls')
    feas=[z['exact_map_feasible'] for z in rows];cur=[z['exact_map_current_success'] for z in rows]
    ax.bar(xx-.17,feas,width=.34,color=blue,label='Geometry feasible')
    ax.bar(xx+.17,cur,width=.34,color=green,label='Current commitment')
    ax.set_xticks(xx,names);ax.set_ylim(0,1.05);ax.set_ylabel('Fraction');ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.12),ncol=2)
    fig.suptitle('Source-realizable geometry: local rigidity and selective vulnerability',fontsize=13,weight='bold')
    save(fig,'Figure_4-3_Exact_Source_Constraints')

if __name__=='__main__':figure2();figure4();supplement();print('Saved 3 reviewed figures to',out)
