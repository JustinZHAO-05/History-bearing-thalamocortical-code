"""Prospective fresh-context confirmation of a declared source-code residual.

Preselected boundary checkpoints: Gtune23 and sussano2. Successful control:
Gtune24. Axes remain those learned from 12 histories at seed 83103. Twenty-four
new base histories at seed 83105 are crossed with q=0,2,18,20 and paired signs.
No source axis, network weight, threshold or behavioral parameter is refitted.

Main contrast: opposite source = mu - sign * [(1-lambda)*dclip+lambda*d]/2.
The five lambdas preserve pair mean and nonnegative currents by convexity.
One prespecified comparator rescales the scalar projection and clips it, to
match the full difference's norm at every time/context whenever feasible.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import rank_one_control as audit

HERE=Path(__file__).resolve().parent
OUT=HERE/'residual_addback_fresh_v1'
PRIOR=HERE/'rank_one_heldout'
LAMBDAS=(0.,.25,.5,.75,1.)
NAMES=('Gtune_RateRNNstructNUM23','Gtune_RateRNNstructNUM24','sussano_RateRNNstructNUM2')
CONDITIONS=[dict(name='natural',kind='natural',residual_fraction=None),
            dict(name='own_sham',kind='own',residual_fraction=None)]
CONDITIONS += [dict(name=f'opposite_residual_{v:.2f}',kind='residual',residual_fraction=v) for v in LAMBDAS]
CONDITIONS += [dict(name='opposite_scalar_full_energy',kind='scalar_energy',residual_fraction=None)]


def scalar_energy_match(projection, mu, target_difference):
    """One amplitude coefficient, clipped at fixed mu; no directional fitting.

The target is per-time/context norm equality with the FULL original difference.
Bounds and bisection inspect currents only. Infeasible or unattained targets are
reported and retained; they never trigger behavioral tuning.
"""
    cap=2*mu
    target2=np.sum(target_difference**2,axis=-1)
    achievable2=np.sum((cap*(projection!=0))**2,axis=-1)
    lo=np.zeros_like(target2)
    hi=np.ones_like(target2)
    for _ in range(40):
        trial=np.clip(hi[...,None]*projection,-cap,cap)
        low=np.sum(trial**2,axis=-1)<target2*(1-1e-13)
        if not np.any(low):break
        hi[low]*=2
    for _ in range(55):
        mid=(lo+hi)/2
        trial=np.clip(mid[...,None]*projection,-cap,cap)
        too_small=np.sum(trial**2,axis=-1)<target2
        lo=np.where(too_small,mid,lo)
        hi=np.where(too_small,hi,mid)
    scale=(lo+hi)/2
    result=np.clip(scale[...,None]*projection,-cap,cap)
    actual2=np.sum(result**2,axis=-1)
    relative=np.divide(np.abs(actual2-target2),target2,out=np.zeros_like(target2),where=target2>1e-24)
    checks=dict(
        infeasible_norm_targets=int(np.sum(target2>achievable2+1e-10)),
        norm_energy_relative_error_max=float(relative.max()),
        norm_energy_absolute_error_max=float(np.max(np.abs(actual2-target2))),
        scale_minimum=float(scale.min()),scale_median=float(np.median(scale)),scale_maximum=float(scale.max()),
        fraction_receiver_entries_clipped=float(np.mean(np.abs(result-scale[...,None]*projection)>1e-12)),
    )
    return result,scale,checks


def evaluate(net,ctx,axis):
    initial,donor=audit.initial_and_donors(net,ctx)
    mu=donor.mean(axis=2)
    difference=donor[:,:,0]-donor[:,:,1]
    coefficient=difference@axis
    projection=coefficient[...,None]*axis
    clipped=np.clip(projection,-2*mu,2*mu)
    amplitude,amplitude_scale,amplitude_checks=scalar_energy_match(projection,mu,difference)
    n=len(initial);m=len(CONDITIONS)
    x=np.repeat(initial[:,None],m,axis=1)
    trace=np.empty((800,n,m,2))
    sign=np.array([1,-1])[None,:,None]
    source_energy=np.zeros((n,m))
    source_deviation_energy=np.zeros_like(source_energy)
    original_energy=np.sum(difference**2,axis=(0,2))
    mean_error=0.;minimum_current=np.inf
    for k in range(800):
        r=net.rates(x)
        trace[k]=r@net.output+net.bias
        drive=(r.reshape(-1,500)@net.W.T).reshape(n,m,2,500)
        if k<200:
            for j,condition in enumerate(CONDITIONS):
                kind=condition['kind']
                if kind=='natural':continue
                if kind=='own':
                    target=donor[k]
                    carried=difference[k]
                elif kind=='residual':
                    weight=condition['residual_fraction']
                    carried=(1-weight)*clipped[k]+weight*difference[k]
                    target=mu[k,:,None]-.5*sign*carried[:,None]
                elif kind=='scalar_energy':
                    carried=amplitude[k]
                    target=mu[k,:,None]-.5*sign*carried[:,None]
                else:raise ValueError(kind)
                source_energy[:,j]+=np.sum(carried**2,axis=-1)
                source_deviation_energy[:,j]+=np.sum((carried-difference[k])**2,axis=-1)
                minimum_current=min(minimum_current,float(target.min()))
                mean_error=max(mean_error,float(np.max(np.abs(target.mean(axis=1)-mu[k]))))
                arrival=r[:,j,:,400:470]@net.B.T
                drive[:,j,:,:400]+=target-arrival
        drive+=ctx['u'][:,None,:,1400+k,None]*net.input
        x+=net.alpha*(drive-x)
    windows=(slice(320,400),slice(720,800))
    margin=np.stack([trace[w].mean(axis=0) for w in windows],axis=-1)
    ymin=np.stack([trace[w].min(axis=0) for w in windows],axis=-1)
    ymax=np.stack([trace[w].max(axis=0) for w in windows],axis=-1)
    source_energy[:,0]=original_energy
    source_energy_fraction=np.divide(source_energy,original_energy[:,None],out=np.zeros_like(source_energy),where=original_energy[:,None]>1e-30)
    source_deviation_fraction=np.divide(source_deviation_energy,original_energy[:,None],out=np.zeros_like(source_energy),where=original_energy[:,None]>1e-30)
    checks=dict(own_replay_max_abs_output_error=float(np.max(np.abs(trace[:,:,0]-trace[:,:,1]))),
                pair_mean_current_max_abs_error=mean_error,minimum_replayed_current=minimum_current,
                full_residual_energy_fraction_max_error=float(np.max(np.abs(source_energy_fraction[:,6]-1))),
                **amplitude_checks)
    assert checks['own_replay_max_abs_output_error']<1e-9,checks
    assert mean_error<1e-9 and minimum_current>=-1e-10,checks
    assert checks['full_residual_energy_fraction_max_error']<1e-12,checks
    reduction=dict(projected_difference_energy_fraction=float(np.sum(projection**2)/np.sum(difference**2)),
                   clipped_difference_energy_fraction=float(np.sum(clipped**2)/np.sum(difference**2)),
                   declared_residual_energy_fraction=float(np.sum((difference-clipped)**2)/np.sum(difference**2)),
                   fraction_scalar_receiver_entries_clipped=float(np.mean(np.abs(projection-clipped)>1e-12)))
    arrays=dict(margin=margin,response_min=ymin,response_max=ymax,output_trace=trace,
                source_axis=axis,source_coefficient=coefficient,amplitude_scale=amplitude_scale,
                source_energy_fraction=source_energy_fraction,source_original_difference_energy=original_energy,
                source_residual_error_fraction=source_deviation_fraction,counts=ctx['counts'],base_context=ctx['base_context'])
    return arrays,checks,reduction


def endpoint_arrays(arrays,ctx):
    y=arrays['margin'];lo=arrays['response_min'];hi=arrays['response_max']
    current_target=np.sign(ctx['counts'][:,:,4]-10)[:,None,:]
    history_target=np.array([1,-1])[None,None,:]
    correct_current=np.where(current_target>0,(hi[...,0]>=.75)&(lo[...,0]>=-.75),(lo[...,0]<=-.75)&(hi[...,0]<=.75))
    correct_history=np.where(history_target>0,(hi[...,1]>=.75)&(lo[...,1]>=-.75),(lo[...,1]<=-.75)&(hi[...,1]<=.75))
    opposite_history=np.where(history_target<0,(hi[...,1]>=.75)&(lo[...,1]>=-.75),(lo[...,1]<=-.75)&(hi[...,1]<=.75))
    return dict(
        current_sign_accuracy=((y[...,0]>0)==(current_target>0)).astype(float),
        current_author_success=correct_current.astype(float),
        current_signed_sensory_margin=y[...,0]*current_target,
        current_signed_history_margin=y[...,0]*history_target,
        history_sign_accuracy=((y[...,1]>0)==(history_target>0)).astype(float),
        history_author_success=correct_history.astype(float),
        history_author_opposite=opposite_history.astype(float),
        history_author_neither=(~(correct_history|opposite_history)).astype(float),
        history_signed_margin=y[...,1]*history_target,
    )


def summarize(arrays,ctx):
    metrics=endpoint_arrays(arrays,ctx)
    base=ctx['base_context'];nbase=len(np.unique(base))
    grouped={name:np.stack([values[base==i].mean(axis=(0,2)) for i in range(nbase)]) for name,values in metrics.items()}
    rng=np.random.default_rng(83115)
    bootstrap=rng.integers(0,nbase,size=(20000,nbase))
    rows=[]
    for j,condition in enumerate(CONDITIONS):
        row=dict(**condition)
        row['outcomes']={name:dict(mean=float(value[:,j].mean()),ci95=np.quantile(value[bootstrap,j].mean(axis=1),[.025,.975]).tolist()) for name,value in grouped.items()}
        row['source_energy_fraction_weighted']=float(np.sum(arrays['source_energy_fraction'][:,j]*arrays['source_original_difference_energy'])/np.sum(arrays['source_original_difference_energy']))
        row['source_energy_fraction_mean_context']=float(arrays['source_energy_fraction'][:,j].mean())
        rows.append(row)
    contrasts=[]
    for j in (3,4,5,6,7):
        comparison=dict(condition=CONDITIONS[j]['name'],reference=CONDITIONS[2]['name'],outcomes={})
        for name in ['current_sign_accuracy','current_author_success','current_signed_sensory_margin','history_signed_margin']:
            difference=grouped[name][:,j]-grouped[name][:,2]
            comparison['outcomes'][name]=dict(mean=float(difference.mean()),ci95=np.quantile(difference[bootstrap].mean(axis=1),[.025,.975]).tolist(),n_base_with_positive_difference=int(np.sum(difference>1e-12)),n_base_with_negative_difference=int(np.sum(difference< -1e-12)))
        contrasts.append(comparison)
    full_vs_amplitude={}
    for name in ['current_sign_accuracy','current_author_success','current_signed_sensory_margin','history_signed_margin']:
        difference=grouped[name][:,6]-grouped[name][:,7]
        full_vs_amplitude[name]=dict(mean=float(difference.mean()),ci95=np.quantile(difference[bootstrap].mean(axis=1),[.025,.975]).tolist())
    return rows,contrasts,full_vs_amplitude,grouped


def run(path):
    tick=time.time();before=audit.sha(path)
    net=audit.IndependentNetwork(path)
    ctx=dict(np.load(OUT/'contexts.npz'))
    with np.load(PRIOR/(net.name+'.npz')) as prior:
        axis=prior['source_axis'].copy()
    arrays,checks,reduction=evaluate(net,ctx,axis)
    rows,contrasts,full_vs_amplitude,base_metrics=summarize(arrays,ctx)
    arrays.update({'base_'+key:value for key,value in base_metrics.items()})
    np.savez_compressed(OUT/(net.name+'.npz'),**arrays)
    after=audit.sha(path);assert before==after
    result=dict(model=net.name,checkpoint_sha256=before,checkpoint_unchanged=True,seconds=time.time()-tick,
                axis_array_sha256=hashlib.sha256(axis.tobytes()).hexdigest(),checks=checks,reduction=reduction,
                conditions=rows,contrasts_from_scalar=contrasts,full_residual_minus_energy_matched_scalar=full_vs_amplitude)
    (OUT/(net.name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(model=net.name,seconds=result['seconds'],checks=checks,conditions=[dict(name=r['name'],current_sign=r['outcomes']['current_sign_accuracy']['mean'],current_native=r['outcomes']['current_author_success']['mean'],current_margin=r['outcomes']['current_signed_sensory_margin']['mean'],history_margin=r['outcomes']['history_signed_margin']['mean'],history_sign=r['outcomes']['history_sign_accuracy']['mean'],energy=r['source_energy_fraction_weighted']) for r in rows])),flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    freeze=OUT/'protocol.freeze.json'
    if freeze.exists():raise RuntimeError('Preserve the previous run; a changed analysis needs a new version.')
    paths={audit.IndependentNetwork(path).name:path for path in sorted(audit.DATA.glob('*/*.db'))}
    selected=[paths[name] for name in NAMES]
    protocol=dict(
        frozen_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_new_context_generation_and_model_evaluation',
        design='Fresh-context prospective confirmation of residual add-back for the declared scalar message code',
        preselected_boundary_checkpoints=['Gtune_RateRNNstructNUM23','sussano_RateRNNstructNUM2'],successful_control='Gtune_RateRNNstructNUM24',
        context_seed=83105,n_base_histories=24,q_values=[0,2,18,20],paired_signs=2,
        axis_basis='Unchanged leading source-difference axis learned from 12 independent contexts at seed 83103; no refit',
        intervention_updates=[1400,1600],current_response=[1720,1800],history_response=[2120,2200],
        conditions=CONDITIONS,
        primary_question='Does adding the declared source-code residual restore strong-current competence while retaining later-history reversal?',
        primary_endpoints=['current mean-sign accuracy','current original threshold commitment','current signed sensory margin','later history sign accuracy and signed margin'],
        inference='Paired bootstrap over 24 base histories with four strengths and two signs nested; 20000 resamples, seed83115. All outcomes retained; no universal monotonicity requirement.',
        addback_formula='Opposite paired arrival mu +/- (-[(1-lambda)*dclip+lambda*d])/2 for lambda=0,.25,.5,.75,1; dclip=clip((d dot v)*v,-2*mu,2*mu)',
        amplitude_comparator='One additional arm scales the original scalar projection and applies the same receiverwise clipping to match ||d|| at each time/context by current-only bisection. Exact pair mean and nonnegativity retained; all clipping, scales and matching failures recorded.',
        interpretation='Residual of this declared message code, conditional on its pair-common background; not intrinsic dimensionality or a generic-gain exclusion claim.',
        code_sha256=audit.sha(__file__),independent_engine_sha256=audit.sha(audit.__file__),
        checkpoint_sha256={name:audit.sha(paths[name]) for name in NAMES},
        prior_axis_file_sha256={name:audit.sha(PRIOR/(name+'.npz')) for name in NAMES},
    )
    freeze.write_text(json.dumps(protocol,indent=2))
    ctx=audit.contexts(24,83105)
    np.savez_compressed(OUT/'contexts.npz',**ctx)
    (OUT/'conditions.json').write_text(json.dumps(CONDITIONS,indent=2))
    start=time.time();summaries=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,path) for path in selected]
        for future in as_completed(futures):
            summaries.append(future.result())
            (OUT/'summary.json').write_text(json.dumps(sorted(summaries,key=lambda r:r['model']),indent=2))
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',models=len(summaries),seconds=time.time()-start,contexts_sha256=audit.sha(OUT/'contexts.npz'),code_sha256=audit.sha(__file__)),indent=2))


if __name__=='__main__':main()
