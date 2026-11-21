"""Reproducible inference and descriptive summaries of the R3 RNN assays."""
from pathlib import Path
import json,csv
import numpy as np
from rnn_core import ROOT

def ci(v,ids):
    v=np.asarray(v);b=v[ids].mean(1)
    return dict(mean=float(v.mean()),ci95=np.quantile(b,[.025,.975]).tolist(),n=len(v))

def author_response(lo,hi):
    return (hi>=.75)&(lo>=-.75),(lo<=-.75)&(hi<=.75)

def history():
    p=ROOT/'results/history_confirmation_v1';conds=json.loads((p/'conditions.json').read_text());idx={c['name']:i for i,c in enumerate(conds)}
    rng=np.random.default_rng(83099);ids=rng.integers(96,size=(10000,96));out=[];tab=[]
    for fn in sorted(p.glob('*.npz')):
        if fn.stem=='contexts':continue
        d=np.load(fn);y=d['margin'];score=(y[:,:,0]-y[:,:,1])/2
        a={name:ci(score[:,j],ids) for name,j in idx.items()}
        carrier=score[:,idx['intact_pattern_100']]-score[:,idx['intact_gain_100']]
        car2=score[:,idx['intact_pattern_l2_100']]-score[:,idx['intact_gain_l2_100']]
        rescue=score[:,idx['PPC_mean_own_100']]-score[:,idx['PPC_mean_mean_100']]
        h={'model':fn.stem,'condition_margins':a,'amplitude_minus_pattern_swap_effect':ci(carrier,ids),
           'l2_amplitude_minus_pattern_swap_effect':ci(car2,ids),'PPC_erasure_source_rescue':ci(rescue,ids)}
        # Finite 2x2 factor: cortical sign c and donor sign p, outside-PPC
        # initial state held to the paired mean.
        h['factorial']={}
        for dur in [100,200]:
            own=y[:,idx[f'outside_mean_own_{dur}']];opp=y[:,idx[f'outside_mean_opposite_{dur}']]
            fpp=own[:,0];fmm=own[:,1];fpm=opp[:,0];fmp=opp[:,1]
            h['factorial'][str(dur)]={'retained_cortex':ci((fpp+fpm-fmp-fmm)/4,ids),'arriving_source':ci((fpp+fmp-fpm-fmm)/4,ids),
                                     'interaction':ci((fpp+fmm-fpm-fmp)/4,ids)}
        out.append(h)
        for name,j in idx.items():
            pos,neg=author_response(d['response_min'][:,j],d['response_max'][:,j])
            correct=np.c_[pos[:,0],neg[:,1]];opposite=np.c_[neg[:,0],pos[:,1]]
            tab.append(dict(model=fn.stem,condition=name,half_pair_margin=float(score[:,j].mean()),ci95_low=a[name]['ci95'][0],ci95_high=a[name]['ci95'][1],
                            sign_accuracy=float(np.mean((y[:,j]>0)==np.array([True,False]))),author_correct=float(correct.mean()),author_opposite=float(opposite.mean()),author_neither=float((~pos&~neg).mean())))
    return out,tab

def sequential():
    p=ROOT/'results/sequential_confirmation_v1';conds=json.loads((p/'conditions.json').read_text());idx={c['name']:i for i,c in enumerate(conds)}
    ctx=np.load(p/'contexts.npz');base=ctx['base_context'];n=len(np.unique(base));rng=np.random.default_rng(83099);ids=rng.integers(n,size=(10000,n));out=[];tab=[]
    def block(v):return np.stack([np.asarray(v)[base==k].mean(0) for k in range(n)])
    for fn in sorted(p.glob('*.npz')):
        if fn.stem=='contexts':continue
        d=np.load(fn);y=d['margin'];score=(y[:,:,0]-y[:,:,1])/2
        record={'model':fn.stem,'condition_scores':{},'source_effect':{},'regional_output_account':{}}
        for name,j in idx.items():
            record['condition_scores'][name]={'current':ci(block(score[:,j,0]),ids),'history':ci(block(score[:,j,1]),ids)}
            lo=d['response_min'][:,j];hi=d['response_max'][:,j];pos,neg=author_response(lo,hi)
            cur_sign=ctx['counts'][:,:,4]>10;hist_sign=np.array([True,False])
            cur_acc=((y[:,j,:,0]>0)==cur_sign).mean(1);hist_acc=((y[:,j,:,1]>0)==hist_sign).mean(1)
            ac=np.where(cur_sign,pos[:,:,0],neg[:,:,0]);ah=np.c_[pos[:,0,1],neg[:,1,1]];oh=np.c_[neg[:,0,1],pos[:,1,1]]
            row=dict(model=fn.stem,condition=name,current_sign_accuracy=float(cur_acc.mean()),history_sign_accuracy=float(hist_acc.mean()),current_author_success=float(ac.mean()),history_author_success=float(ah.mean()),history_author_opposite=float(oh.mean()),history_author_neither=float((~pos[:,:,1]&~neg[:,:,1]).mean()),current_history_halfcontrast=float(score[:,j,0].mean()),history_halfcontrast=float(score[:,j,1].mean()))
            for task,v in [('current',cur_acc),('history',hist_acc)]:
                cc=ci(block(v),ids);row[task+'_sign_ci95_low'],row[task+'_sign_ci95_high']=cc['ci95']
            tab.append(row)
        for dur in [100,200]:
            own=idx[f'intact_own_{dur}'];opp=idx[f'intact_opposite_{dur}']
            delta=y[:,opp]-y[:,own];abs_effect=np.abs(delta).mean(1)
            rec={'current_mean_abs_effect':ci(block(abs_effect[:,0]),ids),'history_mean_abs_effect':ci(block(abs_effect[:,1]),ids),
                 'history_minus_current_abs_effect':ci(block(abs_effect[:,1]-abs_effect[:,0]),ids),
                 'current_signed_effect':ci(block(score[:,opp,0]-score[:,own,0]),ids),'history_signed_effect':ci(block(score[:,opp,1]-score[:,own,1]),ids),
                 'max_single_current_abs_effect':float(np.max(abs(delta[:,:,0])))}
            record['source_effect'][str(dur)]=rec
            dd=d['components'][:,own,:,:, :]-d['components'][:,opp,:,:,:]
            signed=(dd[:,0]-dd[:,1])/2
            reg=signed[:,:, :].mean(0)
            record['regional_output_account'][str(dur)]={'current':reg[0].tolist(),'history':reg[1].tolist(),'PPC_fraction_of_signed_history_effect':float(reg[1,0]/reg[1].sum())}
        out.append(record)
    return out,tab

def calculus():
    p=ROOT/'results/finite_history_calculus_v1';out=[];tab=[]
    for fn in sorted(p.glob('*.npz')):
        if fn.stem=='contexts':continue
        d=np.load(fn);m=np.load(ROOT/'results/sequential_confirmation_v1'/fn.name)
        # Inferred norm from same validated readout loaded explicitly.
        from rnn_core import Model,models
        model=next(Model(p) for p in models() if Model(p).name==fn.stem)
        wnorm=np.linalg.norm(model.Wout)
        rmsalign=d['instantaneous_output_difference_rms']/(d['rate_difference_rmsnorm']*wnorm+1e-30)
        persistence=d['output_pair_difference']/(d['instantaneous_output_difference_rms']+1e-30)
        error=float(np.max(np.abs(d['readout_alignment']-rmsalign*persistence)))
        rows=[]
        for j,name in enumerate(['intact_own_100','intact_opposite_100','PPC_mean_own_100']):
            v=dict(condition=name,rate_difference_norm_mean=d['rate_difference_rmsnorm'][:,j].mean(0).tolist(),
                   rms_readout_alignment_mean=rmsalign[:,j].mean(0).tolist(),signed_readout_alignment_mean=d['readout_alignment'][:,j].mean(0).tolist(),
                   temporal_sign_consistency_mean=persistence[:,j].mean(0).tolist(),half_pair_output_mean=(d['output_pair_difference'][:,j].mean(0)/2).tolist(),
                   component_half_pair_output_mean=(d['component_pair_differences'][:,j].mean(0)/2).tolist())
            rows.append(v)
            tab.append(dict(model=fn.stem,condition=name,current_rate_norm=v['rate_difference_norm_mean'][0],history_rate_norm=v['rate_difference_norm_mean'][1],current_rms_alignment=v['rms_readout_alignment_mean'][0],history_rms_alignment=v['rms_readout_alignment_mean'][1],current_halfcontrast=v['half_pair_output_mean'][0],history_halfcontrast=v['half_pair_output_mean'][1]))
        out.append(dict(model=fn.stem,alignment_factorization_max_error=error,conditions=rows))
    return out,tab

def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    od=ROOT/'results/analysis';od.mkdir(exist_ok=True)
    h,ht=history();s,st=sequential();c,ct=calculus()
    payload={'history_confirmation':h,'sequential_confirmation':s,'finite_calculus':c,'inference':'Context bootstrap within each checkpoint; cross-checkpoint summaries descriptive.'}
    (od/'rnn_inference.json').write_text(json.dumps(payload,indent=2))
    save_csv(od/'history_condition_estimates.csv',ht);save_csv(od/'sequential_condition_estimates.csv',st);save_csv(od/'finite_calculus_estimates.csv',ct)
    aggregate={}
    for name in ['natural','intact_opposite_100','intact_opposite_200','PPC_mean_own_100','PPC_mean_mean_100']:
        rr=[r for r in st if r['condition']==name];aggregate[name]={key:dict(median=float(np.median([r[key] for r in rr])),min=float(min(r[key] for r in rr)),max=float(max(r[key] for r in rr))) for key in ['current_sign_accuracy','history_sign_accuracy','current_author_success','history_author_success','history_author_opposite','history_halfcontrast']}
    aggregate['source200_abs_effect']={task:dict(min=float(min(x['source_effect']['200'][task]['mean'] for x in s)),median=float(np.median([x['source_effect']['200'][task]['mean'] for x in s])),max=float(max(x['source_effect']['200'][task]['mean'] for x in s))) for task in ['current_mean_abs_effect','history_mean_abs_effect']}
    (od/'headline_numbers.json').write_text(json.dumps(aggregate,indent=2));print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
