"""Independent raw-array checks for the final English manuscript."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import rank_one_control as audit

HERE=Path(__file__).resolve().parent
R3=HERE.parents[1]


def criterion(lo,hi,target):
    return np.where(target>0,(hi>=.75)&(lo>=-.75),(lo<=-.75)&(hi<=.75))


def scalar_counts(folder):
    path=HERE/folder
    results=[]
    for summary in json.loads((path/'summary.json').read_text()):
        name=summary['model'];names=[c['condition'] for c in summary['conditions']]
        with np.load(path/(name+'.npz')) as z:
            y=z['margin'];lo=z['response_min'];hi=z['response_max']
            current_target=np.sign(z['counts'][:,:,4]-10)
            history_target=np.broadcast_to([1,-1],current_target.shape)
            conditions={}
            for j,label in enumerate(names):
                conditions[label]=dict(n=int(current_target.size),
                    current_sign_correct=int(np.sum((y[:,j,:,0]>0)==(current_target>0))),
                    current_native_correct=int(np.sum(criterion(lo[:,j,:,0],hi[:,j,:,0],current_target))),
                    history_sign_correct=int(np.sum((y[:,j,:,1]>0)==(history_target>0))),
                    history_sign_opposite=int(np.sum((y[:,j,:,1]>0)!=(history_target>0))),
                    history_native_correct=int(np.sum(criterion(lo[:,j,:,1],hi[:,j,:,1],history_target))),
                    history_native_opposite=int(np.sum(criterion(lo[:,j,:,1],hi[:,j,:,1],-history_target))),
                    history_mean_signed_margin=float(np.mean(y[:,j,:,1]*history_target)))
        results.append(dict(model=name,conditions=conditions))
    return results


def sequential():
    path=R3/'results/sequential_confirmation_v1'
    ctx=dict(np.load(path/'contexts.npz'));names=[r['name'] for r in json.loads((path/'conditions.json').read_text())]
    rows=[]
    for f in sorted(path.glob('*NUM*.npz')):
        with np.load(f) as z:
            y=z['margin'];lo=z['response_min'];hi=z['response_max'];target=np.sign(ctx['counts'][:,:,4]-10);hist=np.broadcast_to([1,-1],target.shape)
            conditions={}
            for name in ['natural','intact_opposite_100','intact_opposite_200']:
                j=names.index(name)
                conditions[name]=dict(current_sign_accuracy=float(np.mean((y[:,j,:,0]>0)==(target>0))),
                    current_native_success=float(np.mean(criterion(lo[:,j,:,0],hi[:,j,:,0],target))),
                    historical_contrast=float(np.mean(y[:,j,:,1]*hist)),
                    history_sign_accuracy=float(np.mean((y[:,j,:,1]>0)==(hist>0))),
                    history_native_opposite=float(np.mean(criterion(lo[:,j,:,1],hi[:,j,:,1],-hist))))
            difference=y[:,names.index('natural')]-y[:,names.index('intact_opposite_200')]
            rows.append(dict(model=f.stem,conditions=conditions,mean_absolute_output_change_current_history=abs(difference).mean(axis=(0,1)).tolist()))
    return rows


def finite():
    path=R3/'results/finite_history_calculus_v1';seq=R3/'results/sequential_confirmation_v1'
    source_names=[c['name'] for c in json.loads((seq/'conditions.json').read_text())]
    finite_names=[c['name'] for c in json.loads((path/'conditions.json').read_text())]
    nets={audit.IndependentNetwork(p).name:audit.IndependentNetwork(p) for p in audit.DATA.glob('*/*.db')}
    results=[]
    for f in sorted(path.glob('*NUM*.npz')):
        with np.load(f) as z, np.load(seq/f.name) as s:
            wnorm=np.linalg.norm(nets[f.stem].output)
            ar=np.divide(z['instantaneous_output_difference_rms'],z['rate_difference_rmsnorm']*wnorm,out=np.zeros_like(z['instantaneous_output_difference_rms']),where=z['rate_difference_rmsnorm']>1e-30)
            reference=np.stack([s['margin'][:64,source_names.index(name),0]-s['margin'][:64,source_names.index(name),1] for name in finite_names],axis=1)
            results.append(dict(model=f.stem,mean_rms_alignment_own_current_history=ar[:,0].mean(axis=0).tolist(),
                                mean_rate_norm_own_current_history=z['rate_difference_rmsnorm'][:,0].mean(axis=0).tolist(),
                                ratio_mean_rms_alignment_history_over_current=float(ar[:,0,1].mean()/ar[:,0,0].mean()),
                                component_output_closure_max=float(np.max(np.abs(z['component_pair_differences'].sum(axis=-1)-z['output_pair_difference']))),
                                independent_sequential_match_max=float(np.max(np.abs(reference-z['output_pair_difference'])))))
    return results


def position():
    path=R3/'results/position_mixture_stress';ctx=dict(np.load(path/'contexts.npz'))
    rows=[]
    for f in sorted(path.glob('*NUM*.npz')):
        with np.load(f) as z:
            h=(z['margin'][:,:,0]-z['margin'][:,:,1])/2
            norms=np.sqrt(np.sum(z['history_norm']**2,axis=-1))
            end5=list(z['snapshot_steps']).index(399)
            strong=ctx['q5']!=10;amb=ctx['q5']==10
            meanh=[float(h[ctx['q5']==q,0,0].mean()) for q in [2,10,18]]
            ratio=float(abs(h[strong,0,0]).mean()/abs(h[amb,0,0]).mean())
            normratio=float(norms[end5,strong,0].mean()/norms[end5,amb,0].mean())
            second=[]
            for q5 in [2,10,18]:
                a=(ctx['q5']==q5)&(ctx['q6']==10);b=(ctx['q5']==q5)&(ctx['q6']!=10)
                second.append(float(abs(h[a,0,1]).mean()-abs(h[b,0,1]).mean()))
            rows.append(dict(model=f.stem,fifth_mean_H_q2_q10_q18=meanh,fifth_absolute_strong_over_ambiguous=ratio,
                             fifth_endstate_norm_strong_over_ambiguous=normratio,
                             fifth_opposite_ambiguous_mean_H=float(h[amb,2,0].mean()),
                             sixth_absolute_ambiguous_minus_strong_at_q5_2_10_18=second,
                             natural_own_margin_max_error=float(np.max(np.abs(z['margin'][:,0]-z['margin'][:,1])))))
    return rows


def main():
    counts=scalar_counts('scalar_clipped');uplift=scalar_counts('rank_one_uplift')
    report=dict(manuscript_sha256=hashlib.sha256((R3/'manuscript/Round3_Manuscript_EN.md').read_bytes()).hexdigest(),
                scalar_clipped_counts=counts,rank_one_uplift_counts=uplift,sequential=sequential(),finite=finite(),position=position())
    (HERE/'manuscript_audit_numerical_checks.json').write_text(json.dumps(report,indent=2))
    header=['| Checkpoint | Natural current: sign / native | Scalar opposite current: sign / native | Scalar opposite history: opposite sign / opposite native | Scalar own after all-state averaging: correct history sign / native |','|---|---:|---:|---:|---:|']
    for row in counts:
        c=row['conditions'];nat=c['natural'];opp=c['intact_scalar_opposite'];own=c['all_mean_scalar_own'];name=row['model'].replace('_RateRNNstructNUM',' ')
        header.append(f"| {name} | {nat['current_sign_correct']}/64 / {nat['current_native_correct']}/64 | {opp['current_sign_correct']}/64 / {opp['current_native_correct']}/64 | {opp['history_sign_opposite']}/64 / {opp['history_native_opposite']}/64 | {own['history_sign_correct']}/64 / {own['history_native_correct']}/64 |")
    (HERE/'scalar_classification_counts.md').write_text('\n'.join(header)+'\n')
    print('\n'.join(header))
    print(json.dumps(dict(
        sequential_current_native_range=[min(r['conditions']['intact_opposite_200']['current_native_success'] for r in report['sequential']),max(r['conditions']['intact_opposite_200']['current_native_success'] for r in report['sequential'])],
        sequential_median_history_accuracy=[float(np.median([r['conditions'][c]['history_sign_accuracy'] for r in report['sequential']])) for c in ['natural','intact_opposite_200']],
        finite_alignment_ratio_range=[min(r['ratio_mean_rms_alignment_history_over_current'] for r in report['finite']),max(r['ratio_mean_rms_alignment_history_over_current'] for r in report['finite'])],
        position_fifth_mean_H=np.mean([r['fifth_mean_H_q2_q10_q18'] for r in report['position']],axis=0).tolist(),
        position_fifth_output_ratio_range=[min(r['fifth_absolute_strong_over_ambiguous'] for r in report['position']),max(r['fifth_absolute_strong_over_ambiguous'] for r in report['position'])],
        position_fifth_norm_ratio_range=[min(r['fifth_endstate_norm_strong_over_ambiguous'] for r in report['position']),max(r['fifth_endstate_norm_strong_over_ambiguous'] for r in report['position'])],
    ),indent=2))


if __name__=='__main__':main()
