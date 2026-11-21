"""Report and figure for the prospective residual/amplitude confirmation."""
from pathlib import Path
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent
RESULTS=HERE/'residual_addback_fresh_v1'


def main():
    summaries=json.loads((RESULTS/'summary.json').read_text())
    numeric=[];tables=[];gain_rows=[]
    for row in summaries:
        name=row['model'].replace('_RateRNNstructNUM',' ')
        table=[f'### {name}', '', '| Replay | Current sign accuracy | Current native commitment | Current signed margin | Later history sign accuracy | Later signed history margin | Opposite history commitment | Source energy fraction |', '|---|---:|---:|---:|---:|---:|---:|---:|']
        for condition in row['conditions']:
            out=condition['outcomes']
            values={metric:data['mean'] for metric,data in out.items()}
            record=dict(model=name,condition=condition['name'],residual_fraction=condition['residual_fraction'],source_energy_fraction=condition['source_energy_fraction_weighted'],**values)
            numeric.append(record)
            label={'natural':'Natural','own_sham':'Own sham','opposite_scalar_full_energy':'Scalar, full energy'}.get(condition['name'],f"Residual λ={condition['residual_fraction']}")
            table.append(f"| {label} | {values['current_sign_accuracy']:.2%} | {values['current_author_success']:.2%} | {values['current_signed_sensory_margin']:.3f} | {values['history_sign_accuracy']:.2%} | {values['history_signed_margin']:.3f} | {values['history_author_opposite']:.2%} | {condition['source_energy_fraction_weighted']:.3f} |")
        tables.append('\n'.join(table))
        with np.load(RESULTS/(row['model']+'.npz')) as data:
            scale=data['amplitude_scale']
            gain_rows.append(dict(model=name,median=float(np.median(scale)),p95=float(np.quantile(scale,.95)),p99=float(np.quantile(scale,.99)),maximum=float(scale.max()),fraction_above10=float(np.mean(scale>10)),fraction_above100=float(np.mean(scale>100)),**row['checks']))
    with (RESULTS/'all_endpoints.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(numeric[0]));writer.writeheader();writer.writerows(numeric)
    (RESULTS/'amplitude_comparator_details.json').write_text(json.dumps(gain_rows,indent=2))
    text='''# Fresh residual add-back and scalar-amplitude confirmation

Date: 2026-11-11 UTC. Prospective follow-up to the independent spatial-reduction audit. Protocol: `residual_addback_fresh_v1/protocol.freeze.json`. All code and numerical outcomes are retained.

## Scientific result

**Adding the declared source-code residual restores current sensory competence while preserving reversed later-history choices. A rescaled one-scalar message also produces that recovery.** The fresh test therefore resolves the two exploratory current-loss cases and changes the mechanistic interpretation: message strength within a scalar code explains the principal recovery effect, without requiring restoration of the original residual spatial pattern.

Gtune23 reproduces the scalar-only loss on fresh histories: current mean-sign accuracy is 84.38% and native commitment is 80.73%. At 25% residual add-back both become 96.35%; at 50% they reach 100%. The full-energy scalar comparator also reaches 100% for both endpoints. Every opposite-source condition retains a negative later-history margin.

Sussano2 reproduces a smaller loss: current sign accuracy is 97.92% and native commitment is 94.79%. A 25% add-back restores all current signs; full add-back restores all native commitments. The amplitude comparator yields 100% current sign accuracy and 98.96% native commitment (190/192 responses), with a later-history margin −0.568 versus −0.573 for full residual restoration. The successful control Gtune24 remains at 100% current sign and native accuracy throughout.

These results replace the earlier interpretation that the discarded spatial dimensions specifically protect current judgments. The supported statement is that the original reduced code can be underpowered for current competence, and either residual restoration or a suitable amplitude adjustment can correct it while preserving the substituted historical content. The comparator is a time- and context-dependent scalar code with deterministic receiverwise clipping; the result concerns this declared control family.

## Prospective design

The protocol was written before generating the new contexts or evaluating the new conditions. The preselected boundary checkpoints were Gtune23 and sussano2; Gtune24 was the successful control. All three were evaluated without selection or refitting.

- Twenty-four fresh base histories, seed 83105; four strong fifth-trial mixtures q=0,2,18,20; two mirrored historical signs; 192 recipient trajectories per condition and checkpoint.
- The same spatial axis previously learned from 12 separate histories at seed 83103. No axis, weight, decoder, threshold or behavioral parameter was refitted.
- Same intact recipient states for all arms. Source replay covers ITI updates [1400,1600); current response [1720,1800); later ambiguous history response [2120,2200).
- Natural and own-replay controls, five residual fractions λ=0,.25,.5,.75,1, and one prespecified amplitude comparator. All original checkpoint files were opened read-only and hash-checked unchanged afterward.
- Paired bootstrap over the 24 base histories, keeping their four mixtures and two signs nested: 20,000 resamples, seed 83115. No test of universal monotonicity was prespecified or inferred.

Let μ=(b+ + b−)/2, d=b+−b−, v the unchanged learned unit axis, a=d·v, and dclip=clip(a v,−2μ,2μ). The residual intervention uses

`dλ = (1−λ) dclip + λ d`,

with opposite replay `bλ,+ = μ−dλ/2` and `bλ,− = μ+dλ/2`. The original pair mean is preserved and nonnegative currents follow from convexity. “Residual” denotes d−dclip for this declared message code, conditional on the pair-common background.

For the amplitude comparator, a scalar scale k is found using current-only bisection so that `||clip(k a v,−2μ,2μ)|| = ||d||` at each time/context. The axis and pair mean stay fixed. The comparator carries no separately reconstructed residual direction. It uses the original full difference's norm to specify its amplitude, so its scientific role is a concrete alternative source code rather than a behavioral fit.

## All outcomes

History sign accuracy is scored against the original recipient history; lower values under opposite replay reflect substitution toward the opposite history. Opposite history commitment uses the author's response-window threshold rule. Source energy is the time/context-weighted squared norm of the replayed paired difference divided by that of the original paired difference. Confidence intervals for every endpoint are in the per-checkpoint JSON and `summary.json`.

__TABLES__

## Paired uncertainty and functional comparisons

For Gtune23, full residual restoration improves native current commitment by **19.27 percentage points**, with a base-history bootstrap 95% interval **15.10–22.92 points**. The amplitude comparator yields exactly the same categorical improvement. Full restoration improves current signed margin by 0.267 (95% interval 0.206–0.321); scalar energy matching improves it by 0.271 (0.210–0.324).

For sussano2, full residual restoration improves native current commitment by **5.21 points** (2.60–8.33), and scalar energy matching improves it by **4.17 points** (1.56–7.29). Full residual restoration exceeds scalar energy matching by **1.04 points**, with interval **0–2.60**. Its continuous current margin is 0.025 higher (0.008–0.045). The two reconstructed historical margins differ by −0.006, with interval −0.052 to +0.034.

Every opposite-source historical-margin interval remains below zero at every residual fraction and for the amplitude comparator in all three checkpoints. For the weakest original reduced code, sussano2 at λ=0, the history margin is −0.165 (−0.261 to −0.069). Thus the current recovery occurs while the prior substitution remains behaviorally expressed. The historical margin itself varies nonmonotonically in Gtune23, consistent with the protocol's absence of a universal monotonicity requirement.

## Numerical and comparator checks

The maximum natural-versus-own output discrepancy is 4.9×10⁻15. Pair-mean arrival errors are at most 4.5×10⁻16. Minimum replayed current is zero. All amplitude norm targets are feasible; maximum relative squared-norm matching error is 8.1×10⁻16. The original weight-file hashes are unchanged.

Median amplitude scales are 1.299 for Gtune23, 1.002 for Gtune24 and 1.287 for sussano2. The 95th percentiles are 1.519, 1.165 and 4.176. Larger scales occur when the original projection approaches zero: maxima are 2.96, 172.87 and 4859.46 respectively. These are pre-clipping coefficients; every replayed target remains in its fixed nonnegative pair envelope [0,2μ]. The complete scale arrays, clipping fractions and quantiles are saved. Because clipping is explicit, the amplitude comparator is a one-scalar nonlinear source code rather than an exact spatial-rank-one current.

## Revised mechanistic conclusion

The earlier reduced-code boundary is real and replicates in fresh histories. The additional comparator establishes an effective alternative within the same scalar encoding family. Historical content replacement, source magnitude and the receiver's nonlinear trajectory jointly explain the observed current/history dissociation more economically than a claim that a separate residual spatial channel specifically protects current decisions.

The principal result remains source-specific historical substitution after release, with substantially greater historical influence in the later ambiguous decision. This follow-up identifies a concrete reason for the two projection failures and prevents turning an amplitude-sensitive intervention boundary into a broad claim about intrinsic circuit dimensionality or independent gating.

## Files

- `residual_addback_confirmation.py`: independent simulation, intervention definitions and clustered analysis.
- `residual_addback_fresh_v1/protocol.freeze.json`: frozen design and code/weight provenance.
- `residual_addback_fresh_v1/contexts.npz`: all fresh stimuli and initial states.
- Per-checkpoint NPZ: complete output traces, response extrema, base-history metrics, source energies, fixed axes and amplitude coefficients.
- Per-checkpoint JSON and `summary.json`: all endpoints, confidence intervals and paired contrasts.
- `all_endpoints.csv`: compact complete numeric endpoint table.
- `amplitude_comparator_details.json`: scaling and numerical constraints.
- `residual_addback_figure.svg` and `.png`: exact numerical figure with base-history confidence intervals.
'''
    (HERE/'residual_addback_report.md').write_text(text.replace('__TABLES__','\n\n'.join(tables)))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(12,6.7),sharex=True)
        order=['Gtune_RateRNNstructNUM23','sussano_RateRNNstructNUM2','Gtune_RateRNNstructNUM24']
        lookup={r['model']:r for r in summaries}
        for col,name in enumerate(order):
            row=lookup[name];cs=row['conditions'];xs=np.array([0,.25,.5,.75,1])
            for rr,metric in enumerate(['current_author_success','history_signed_margin']):
                ax=axes[rr,col];entries=[cs[j]['outcomes'][metric] for j in range(2,7)]
                means=np.array([v['mean'] for v in entries]);ci=np.array([v['ci95'] for v in entries])
                ax.errorbar(xs,means,yerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='o-',color='#225ea8',capsize=3,label='Residual add-back')
                amp=cs[7]['outcomes'][metric];m=amp['mean'];lo,hi=amp['ci95']
                ax.errorbar([1.17],[m],yerr=[[max(m-lo,0)],[max(hi-m,0)]],fmt='D',mfc='white',mec='#c75125',ecolor='#c75125',capsize=3,label='Scalar energy match')
                ax.set_xlim(-.06,1.27);ax.set_xticks([0,.25,.5,.75,1,1.17]);ax.set_xticklabels(['0','.25','.5','.75','1','Scalar'])
                ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
                if rr==0:
                    ax.set_ylim(.70,1.025);ax.set_title(name.replace('_RateRNNstructNUM',' '),fontweight='bold')
                else:
                    ax.axhline(0,color='0.5',lw=.8);ax.set_ylim(-.76,.02);ax.set_xlabel('Residual fraction λ / comparator')
            axes[0,0].set_ylabel('Current native commitment')
            axes[1,0].set_ylabel('Later signed history margin')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,1.025))
        fig.suptitle('Current recovery preserves the substituted history; scalar amplitude also recovers performance',y=1.085,fontsize=12)
        fig.tight_layout()
        fig.savefig(HERE/'residual_addback_figure.svg',bbox_inches='tight')
        fig.savefig(HERE/'residual_addback_figure.png',dpi=200,bbox_inches='tight')
        plt.close(fig)
    except ImportError as error:
        (HERE/'residual_figure_unavailable.txt').write_text(str(error))
    print(str(HERE/'residual_addback_report.md'))


if __name__=='__main__':main()
