"""Amendment after positivity attenuation in rank_one_control.py.

Keep rank one at full projection amplitude. Add the smallest pair-common,
targetwise baseline uplift required for nonnegative excitatory arrival, and give
the SAME uplift to matched full-difference and pair-mean controls. This is an
exploratory amendment using the original held-out contexts; its predecessor and
failed attenuated controls are retained unchanged.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import numpy as np
import rank_one_control as audit

HERE = Path(__file__).resolve().parent
OUT = HERE / "rank_one_uplift"
PREVIOUS = HERE / "rank_one_heldout"

CONDITIONS = [
    ("natural", "intact", "natural"),
    ("intact_full_own", "intact", "full_own"),
    ("intact_full_opposite", "intact", "full_opposite"),
    ("intact_uplift_full_own", "intact", "uplift_full_own"),
    ("intact_uplift_full_opposite", "intact", "uplift_full_opposite"),
    ("intact_uplift_rank1_own", "intact", "uplift_rank1_own"),
    ("intact_uplift_rank1_opposite", "intact", "uplift_rank1_opposite"),
    ("all_mean_uplift_full_own", "all_mean", "uplift_full_own"),
    ("all_mean_uplift_full_opposite", "all_mean", "uplift_full_opposite"),
    ("all_mean_uplift_rank1_own", "all_mean", "uplift_rank1_own"),
    ("all_mean_uplift_rank1_opposite", "all_mean", "uplift_rank1_opposite"),
    ("all_mean_uplift_mean", "all_mean", "uplift_mean"),
]


def evaluate(net, ctx, axis):
    initial, donor = audit.initial_and_donors(net, ctx)
    n, m = len(initial), len(CONDITIONS)
    mu = donor.mean(axis=2)
    difference = donor[:, :, 0] - donor[:, :, 1]
    coefficient = difference @ axis
    projection = coefficient[..., None] * axis
    uplift = np.maximum(np.abs(projection) / 2 - mu, 0)
    shifted_mu = mu + uplift
    x = np.repeat(initial[:, None], m, axis=1)
    for j, (_, state, _) in enumerate(CONDITIONS):
        if state == "all_mean":
            x[:, j] = initial.mean(axis=1, keepdims=True)
    trace = np.empty((800, n, m, 2))
    region = np.zeros((n, m, 2, 2, 3))
    sign = np.array([1, -1])[None, :, None]
    mincurrent = np.inf
    mean_pair_error = 0.0
    rankone_error = 0.0
    for k in range(800):
        r = net.rates(x)
        trace[k] = r @ net.output + net.bias
        response = 0 if 320 <= k < 400 else (1 if 720 <= k < 800 else None)
        if response is not None:
            for rid, sl in enumerate((slice(0, 400), slice(400, 470), slice(470, 500))):
                region[..., response, rid] += (r[..., sl] @ net.output[sl]) / 80
        drive = (r.reshape(-1, 500) @ net.W.T).reshape(n, m, 2, 500)
        if k < 200:
            targets = {
                "full_own": donor[k],
                "full_opposite": donor[k, :, ::-1],
                "uplift_full_own": donor[k] + uplift[k, :, None],
                "uplift_full_opposite": donor[k, :, ::-1] + uplift[k, :, None],
                "uplift_rank1_own": shifted_mu[k, :, None] + 0.5 * sign * projection[k, :, None],
                "uplift_rank1_opposite": shifted_mu[k, :, None] - 0.5 * sign * projection[k, :, None],
                "uplift_mean": np.repeat(shifted_mu[k, :, None], 2, axis=1),
            }
            mincurrent = min(mincurrent, min(float(t.min()) for t in targets.values()))
            for source, target in targets.items():
                if source.startswith("uplift_"):
                    mean_pair_error = max(mean_pair_error, float(np.max(np.abs(target.mean(axis=1) - shifted_mu[k]))))
            rankone_error = max(rankone_error, float(np.max(np.abs(targets["uplift_rank1_own"][:, 0] - targets["uplift_rank1_own"][:, 1] - projection[k]))))
            for j, (_, _, source) in enumerate(CONDITIONS):
                if source != "natural":
                    drive[:, j, :, :400] += targets[source] - r[:, j, :, 400:470] @ net.B.T
        drive += ctx["u"][:, None, :, 1400+k, None] * net.input
        x += net.alpha * (drive - x)
    margin = np.stack([trace[320:400].mean(axis=0), trace[720:800].mean(axis=0)], axis=-1)
    ymin = np.stack([trace[320:400].min(axis=0), trace[720:800].min(axis=0)], axis=-1)
    ymax = np.stack([trace[320:400].max(axis=0), trace[720:800].max(axis=0)], axis=-1)
    checks = dict(
        own_replay_max_abs_output_error=float(np.max(np.abs(trace[:, :, 0] - trace[:, :, 1]))),
        all_mean_mean_pair_output_max_abs_error=float(np.max(np.abs(trace[:, :, -1, 0] - trace[:, :, -1, 1]))),
        component_sum_max_abs_error=float(np.max(np.abs(region.sum(axis=-1) + net.bias - margin))),
        minimum_replayed_current=mincurrent, matched_pair_mean_max_abs_error=mean_pair_error,
        rank_one_difference_max_abs_error=rankone_error,
    )
    assert mincurrent >= -1e-10, checks
    assert max(abs(v) for key, v in checks.items() if key != "minimum_replayed_current") < 1e-9, checks
    reduction = dict(
        heldout_rank1_energy_fraction=float(np.sum(projection**2) / np.sum(difference**2)),
        common_uplift_norm_over_source_mean_norm=float(np.linalg.norm(uplift) / np.linalg.norm(mu)),
        common_uplift_norm_over_original_difference_norm=float(np.linalg.norm(uplift) / np.linalg.norm(difference)),
        common_uplift_max=float(uplift.max()),
        fraction_receiver_entries_uplifted=float(np.mean(uplift > 1e-14)),
    )
    arrays = dict(margin=margin, response_min=ymin, response_max=ymax, components=region, output_trace=trace,
                  source_axis=axis, source_coefficient=coefficient, common_uplift=uplift, source_mean=mu,
                  donor_difference=difference, counts=ctx["counts"], base_context=ctx["base_context"])
    return arrays, checks, reduction


def run(path):
    tick = time.time()
    net = audit.IndependentNetwork(path)
    ctx = dict(np.load(PREVIOUS / "heldout_contexts.npz"))
    with np.load(PREVIOUS / (net.name + ".npz")) as prev:
        axis = prev["source_axis"]
    arrays, checks, reduction = evaluate(net, ctx, axis)
    np.savez_compressed(OUT / (net.name + ".npz"), **arrays)
    # Reuse only endpoint bookkeeping, while substituting this declared condition list.
    audit.CONDITIONS = CONDITIONS
    rows = audit.summarize(arrays, ctx)
    row = dict(model=net.name, checkpoint_sha256=audit.sha(path), checks=checks, reduction=reduction,
               conditions=rows, seconds=time.time()-tick)
    (OUT / (net.name + ".json")).write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


def main():
    OUT.mkdir(exist_ok=True)
    protocol = dict(
        written_utc=datetime.now(timezone.utc).isoformat(), status="frozen_before_uplift_model_evaluation",
        rationale="The exact-mean rank-one projection can reach the nonnegative boundary in a few receiver entries and require extreme global attenuation. Preserve its full amplitude with a minimal matched common uplift.",
        predecessor="rank_one_heldout; results retained, not overwritten",
        context_status="Reuse prior held-out contexts after inspecting only predecessor results; exploratory amendment",
        basis_status="Unchanged axes learned from 12 separate histories at seed 83103",
        n_heldout_base=8, heldout_seed=83104, conditions=CONDITIONS,
        intervention_updates=[1400, 1600], code_sha256=audit.sha(__file__), independent_engine_sha256=audit.sha(audit.__file__),
        transformation="d1=(d dot v)v; uplift=max(abs(d1)/2-mu,0); new pair source=mu+uplift +/- d1/2; full-difference controls use same uplift",
        endpoints="Native continuous margin; mean-sign accuracy; original response-window threshold correct/opposite/neither; all checkpoints",
    )
    frozen = OUT / "protocol.freeze.json"
    if frozen.exists():
        raise RuntimeError("Prior freeze exists; preserve and use a new version for revisions")
    frozen.write_text(json.dumps(protocol, indent=2))
    paths = sorted(audit.DATA.glob("*/*.db"))
    summaries = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, p) for p in paths]
        for future in as_completed(futures):
            summaries.append(future.result())
            (OUT / "summary.json").write_text(json.dumps(sorted(summaries, key=lambda r:r["model"]), indent=2))
    (OUT / "completion.json").write_text(json.dumps(dict(status="complete", models=len(summaries), seconds=time.time()-start), indent=2))


if __name__ == "__main__":
    main()
