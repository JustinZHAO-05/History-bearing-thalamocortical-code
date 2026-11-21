"""One-scalar nonlinear source code with original pair mean exactly retained.

The previously fixed spatial axis v produces d1=(d dot v)*v. Each target's
contrast is clipped into [-2*mu,2*mu], preserving nonnegative source currents and
the original pair-common mean. This is one scalar passed through a known
receiverwise clipping function at fixed mu, not exact spatial rank one.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import time
import numpy as np
import rank_one_control as audit

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE / "rank_one_heldout"
OUT = HERE / "scalar_clipped"
CONDITIONS = [
    ("natural", "intact", "natural"),
    ("intact_full_own", "intact", "full_own"),
    ("intact_full_opposite", "intact", "full_opposite"),
    ("intact_scalar_own", "intact", "scalar_own"),
    ("intact_scalar_opposite", "intact", "scalar_opposite"),
    ("all_mean_full_own", "all_mean", "full_own"),
    ("all_mean_full_opposite", "all_mean", "full_opposite"),
    ("all_mean_scalar_own", "all_mean", "scalar_own"),
    ("all_mean_scalar_opposite", "all_mean", "scalar_opposite"),
    ("all_mean_mean", "all_mean", "mean"),
]


def evaluate(net, ctx, axis):
    initial, donor = audit.initial_and_donors(net, ctx)
    n, m = len(initial), len(CONDITIONS)
    mu = donor.mean(axis=2)
    difference = donor[:, :, 0] - donor[:, :, 1]
    coefficient = difference @ axis
    projection = coefficient[..., None] * axis
    clipped = np.clip(projection, -2*mu, 2*mu)
    x = np.repeat(initial[:, None], m, axis=1)
    for j, (_, state, _) in enumerate(CONDITIONS):
        if state == "all_mean":
            x[:, j] = initial.mean(axis=1, keepdims=True)
    trace = np.empty((800, n, m, 2))
    sign = np.array([1, -1])[None, :, None]
    mincurrent = np.inf
    mean_error = 0.0
    for k in range(800):
        r = net.rates(x)
        trace[k] = r @ net.output + net.bias
        drive = (r.reshape(-1, 500) @ net.W.T).reshape(n, m, 2, 500)
        if k < 200:
            targets = {
                "full_own": donor[k],
                "full_opposite": donor[k, :, ::-1],
                "scalar_own": mu[k, :, None] + .5*sign*clipped[k, :, None],
                "scalar_opposite": mu[k, :, None] - .5*sign*clipped[k, :, None],
                "mean": np.repeat(mu[k, :, None], 2, axis=1),
            }
            mincurrent = min(mincurrent, min(float(v.min()) for v in targets.values()))
            mean_error = max(mean_error, max(float(np.max(np.abs(v.mean(axis=1)-mu[k]))) for v in targets.values()))
            for j, (_, _, source) in enumerate(CONDITIONS):
                if source != "natural":
                    drive[:, j, :, :400] += targets[source] - r[:, j, :, 400:470] @ net.B.T
        drive += ctx["u"][:, None, :, 1400+k, None] * net.input
        x += net.alpha * (drive-x)
    margin = np.stack([trace[320:400].mean(axis=0), trace[720:800].mean(axis=0)], axis=-1)
    ymin = np.stack([trace[320:400].min(axis=0), trace[720:800].min(axis=0)], axis=-1)
    ymax = np.stack([trace[320:400].max(axis=0), trace[720:800].max(axis=0)], axis=-1)
    checks = dict(own_replay_max_abs_output_error=float(np.max(np.abs(trace[:, :, 0]-trace[:, :, 1]))),
                  mean_source_exact_pair_output_error=float(np.max(np.abs(trace[:, :, -1, 0]-trace[:, :, -1, 1]))),
                  pair_mean_current_max_abs_error=mean_error, minimum_replayed_current=mincurrent)
    assert mincurrent >= -1e-10 and max(abs(v) for v in checks.values()) < 1e-9, checks
    reduction = dict(
        original_projected_energy_fraction=float(np.sum(projection**2)/np.sum(difference**2)),
        clipped_replay_energy_fraction=float(np.sum(clipped**2)/np.sum(difference**2)),
        normalized_reconstruction_error=float(np.sum((difference-clipped)**2)/np.sum(difference**2)),
        fraction_target_entries_clipped=float(np.mean(np.abs(clipped-projection)>1e-14)),
        clipped_energy_remaining_along_original_axis=float(np.sum((clipped@axis)**2)/np.sum(clipped**2)),
    )
    arrays = dict(margin=margin, response_min=ymin, response_max=ymax, output_trace=trace,
                  source_axis=axis, source_coefficient=coefficient, source_mean=mu,
                  replay_difference=clipped, counts=ctx["counts"], base_context=ctx["base_context"])
    return arrays, checks, reduction


def run(path):
    tick=time.time()
    net=audit.IndependentNetwork(path)
    ctx=dict(np.load(PREVIOUS/"heldout_contexts.npz"))
    with np.load(PREVIOUS/(net.name+".npz")) as previous:
        axis=previous["source_axis"]
    arrays,checks,reduction=evaluate(net,ctx,axis)
    np.savez_compressed(OUT/(net.name+".npz"),**arrays)
    audit.CONDITIONS=CONDITIONS
    result=dict(model=net.name,checkpoint_sha256=audit.sha(path),checks=checks,reduction=reduction,
                conditions=audit.summarize(arrays,ctx),seconds=time.time()-tick)
    (OUT/(net.name+".json")).write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    freeze=OUT/"protocol.freeze.json"
    if freeze.exists():
        raise RuntimeError("Retain existing frozen run and choose a new version")
    protocol=dict(written_utc=datetime.now(timezone.utc).isoformat(),status="frozen_before_scalar_clipping_evaluation",
                  design="Resolve common-baseline uplift confound using exact original pair mean and a one-scalar nonlinear source encoder",
                  context_status="Exploratory amendment on eight previously held-out contexts, after strict-rank-one results",
                  code_sha256=audit.sha(__file__),independent_engine_sha256=audit.sha(audit.__file__),conditions=CONDITIONS,
                  source_transform="b_plus/minus = mu +/- 0.5*clip(a*v,-2*mu,2*mu), a=(b_plus-b_minus) dot v",
                  interpretation="One signed scalar coefficient given fixed pair-common background; clipping can add spatial dimensions and is not exact rank one",
                  training_seed=83103,training_n_base=12,heldout_seed=83104,heldout_n_base=8)
    freeze.write_text(json.dumps(protocol,indent=2))
    summaries=[];start=time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,p) for p in sorted(audit.DATA.glob("*/*.db"))]
        for future in as_completed(futures):
            summaries.append(future.result())
            (OUT/"summary.json").write_text(json.dumps(sorted(summaries,key=lambda r:r["model"]),indent=2))
    (OUT/"completion.json").write_text(json.dumps(dict(status="complete",models=len(summaries),seconds=time.time()-start),indent=2))


if __name__ == "__main__":
    main()
