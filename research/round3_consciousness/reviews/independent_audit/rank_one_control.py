"""Independent held-out reduction of history-bearing PUL->PPC current.

Audit-only implementation: no imports from the root evaluator. Loads the released
weight database read-only, learns one spatial source-difference axis from 12
separate histories, and replays its signed projection in eight fresh sequential
history contexts. No weights, output decoder, or behavioural criterion is fitted.

The reduction is spatial rank one; its scalar coefficient may vary with time.
It is a sufficiency control for source content, not a claim that the whole
network's dynamics are one dimensional.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import json
import pickle
import sqlite3
import time
import numpy as np

HERE = Path(__file__).resolve().parent
R3 = HERE.parents[1]
DATA = R3.parent / "round2_consciousness/external/hosford_history_rnn"
SOURCE = DATA / "CustomRnn_gtune_dale_relu.py"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class IndependentNetwork:
    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.parent.name.replace("TBPTT_bio_", "").replace("_valid", "") + "_" + self.path.stem
        db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        row = dict(db.execute("SELECT * FROM currentRNNParameters ORDER BY id DESC LIMIT 1").fetchone())
        db.close()
        a = {k: pickle.loads(v) for k, v in row.items() if isinstance(v, bytes)}
        assert (row["N"], row["PPC_N"], row["PULexh_N"]) == (500, 400, 70)
        assert (row["ppc_rate"], row["pul_rate"], row["trn_rate"]) == ("relu", "relu", "alpha_relu")
        self.W = np.maximum(np.asarray(a["raw_Wr"], dtype=float), 0) * np.asarray(a["SynapseMask"], dtype=float)
        self.B = self.W[:400, 400:470].copy()
        self.input = np.asarray(a["Win"], dtype=float).ravel()
        self.output = np.asarray(a["Wout"], dtype=float).ravel()
        self.bias = float(np.asarray(a["b_out"]).item())
        lo, hi = a["tausRange"]
        z = np.asarray(a["raw_tauS"], dtype=float).ravel()
        self.alpha = 1 / (lo + (hi - lo) / (1 + np.exp(-z)))
        assert np.all(self.B >= 0)

    @staticmethod
    def rates(x):
        rates = np.maximum(x, 0)
        rates[..., 470:] = np.maximum(10 * (x[..., 470:] - 9), 0)
        return np.minimum(rates, 20)

    def step(self, state, signal):
        drive = self.rates(state) @ self.W.T + np.asarray(signal)[..., None] * self.input
        return state + self.alpha * (drive - state)


def contexts(n_base, seed, q_values=(0, 2, 18, 20)):
    """Independent implementation of the frozen lawful sequential sampling rule."""
    rng = np.random.default_rng(seed)
    q_values = tuple(q_values)
    K = len(q_values)
    u = np.zeros((n_base * K, 2, 2200))
    counts = np.empty((n_base * K, 2, 6), dtype=int)
    x0 = np.empty((n_base * K, 500))
    allowed = np.r_[np.arange(10), np.arange(11, 21)]
    for b in range(n_base):
        while True:
            old = rng.choice(allowed, size=4)
            if abs(np.sum(old - 10)) > 10:
                break
        if np.sum(old - 10) < 0:
            old = 20 - old
        for j, q in enumerate(old):
            pips = np.r_[np.ones(q), -np.ones(20 - q)]
            rng.shuffle(pips)
            for i in range(K):
                u[b * K + i, 0, j * 400 + 50:j * 400 + 110] = np.repeat(pips, 3)
                u[b * K + i, 1, j * 400 + 50:j * 400 + 110] = np.repeat(-pips, 3)
                counts[b * K + i, :, j] = [q, 20 - q]
        ordering = rng.permutation(20)
        ambiguous = -np.ones(20)
        ambiguous[rng.permutation(20)[:10]] = 1
        initial = rng.normal(0, 1 / 500, size=500)
        for i, q in enumerate(q_values):
            pips = -np.ones(20)
            pips[ordering[:q]] = 1
            k = b * K + i
            u[k, :, 1650:1710] = np.repeat(pips, 3)
            u[k, :, 2050:2110] = np.repeat(ambiguous, 3)
            counts[k, :, 4] = q
            counts[k, :, 5] = 10
            x0[k] = initial
    assert np.all(np.sum(counts[:, 0, :5] - 10, axis=-1) > 0)
    assert np.all(np.sum(counts[:, 1, :5] - 10, axis=-1) < 0)
    assert np.all(u[:, 0, 1400:] == u[:, 1, 1400:])
    return dict(u=u, counts=counts, x0=x0, base_context=np.repeat(np.arange(n_base), K), seed=np.array(seed))


def initial_and_donors(net, ctx, stop=1600):
    n = len(ctx["u"])
    x = np.repeat(ctx["x0"][:, None], 2, axis=1)
    for t in range(1400):
        x = net.step(x, ctx["u"][:, :, t])
    initial = x.copy()
    donor = np.empty((200, n, 2, 400))
    for k in range(200):
        donor[k] = net.rates(x)[..., 400:470] @ net.B.T
        x = net.step(x, ctx["u"][:, :, 1400 + k])
    return initial, donor


CONDITIONS = [
    ("natural", "intact", "natural"),
    ("intact_full_own", "intact", "full_own"),
    ("intact_full_opposite", "intact", "full_opposite"),
    ("intact_rank1_own", "intact", "rank1_own"),
    ("intact_rank1_opposite", "intact", "rank1_opposite"),
    ("all_mean_full_own", "all_mean", "full_own"),
    ("all_mean_full_opposite", "all_mean", "full_opposite"),
    ("all_mean_rank1_own", "all_mean", "rank1_own"),
    ("all_mean_rank1_opposite", "all_mean", "rank1_opposite"),
    ("all_mean_mean", "all_mean", "mean"),
]


def fit_axis(net, train):
    _, donors = initial_and_donors(net, train)
    difference = donors[:, :, 0] - donors[:, :, 1]
    mat = difference.reshape(-1, 400)
    values, vectors = np.linalg.eigh(mat.T @ mat)
    axis = vectors[:, -1]
    axis[np.all(net.B == 0, axis=1)] = 0
    axis /= np.linalg.norm(axis)
    if np.mean(mat @ axis) < 0:
        axis *= -1
    return axis, values[::-1], float(np.sum((mat @ axis) ** 2) / np.sum(mat ** 2))


def evaluate(net, ctx, axis):
    initial, donor = initial_and_donors(net, ctx)
    n = len(initial)
    m = len(CONDITIONS)
    mu = donor.mean(axis=2)
    d = donor[:, :, 0] - donor[:, :, 1]
    projection = (d @ axis)[..., None] * axis
    # Pairwise common attenuation preserves exact spatial rank one and prevents
    # negative excitatory currents. It is determined without inspecting outputs.
    ratios = np.divide(2 * mu, np.abs(projection), out=np.full_like(mu, np.inf), where=np.abs(projection) > 1e-14)
    feasibility = np.minimum(1, np.min(ratios, axis=-1))
    feasibility = np.maximum(feasibility, 0)
    projected_safe = projection * feasibility[..., None]
    coefficient = (d @ axis) * feasibility
    x = np.repeat(initial[:, None], m, axis=1)
    for j, (_, state, _) in enumerate(CONDITIONS):
        if state == "all_mean":
            x[:, j] = initial.mean(axis=1, keepdims=True)
    y = np.empty((800, n, m, 2))
    region = np.zeros((n, m, 2, 2, 3))
    minimum_current = np.inf
    same_state_mean_error = 0.0
    sign = np.array([1, -1])[None, :, None]
    for k in range(800):
        r = net.rates(x)
        y[k] = r @ net.output + net.bias
        response = 0 if 320 <= k < 400 else (1 if 720 <= k < 800 else None)
        if response is not None:
            for region_id, sl in enumerate((slice(0, 400), slice(400, 470), slice(470, 500))):
                region[..., response, region_id] += (r[..., sl] @ net.output[sl]) / 80
        drive = (r.reshape(-1, 500) @ net.W.T).reshape(n, m, 2, 500)
        if k < 200:
            p = projected_safe[k]
            targets = {
                "full_own": donor[k],
                "full_opposite": donor[k, :, ::-1],
                "rank1_own": mu[k, :, None] + 0.5 * sign * p[:, None],
                "rank1_opposite": mu[k, :, None] - 0.5 * sign * p[:, None],
                "mean": np.repeat(mu[k, :, None], 2, axis=1),
            }
            minimum_current = min(minimum_current, min(float(z.min()) for z in targets.values()))
            for j, (_, _, source) in enumerate(CONDITIONS):
                if source != "natural":
                    arrival = r[:, j, :, 400:470] @ net.B.T
                    drive[:, j, :, :400] += targets[source] - arrival
        drive += ctx["u"][:, None, :, 1400 + k, None] * net.input
        x = x + net.alpha * (drive - x)
        same_state_mean_error = max(same_state_mean_error, float(np.max(np.abs(x[:, -1, 0] - x[:, -1, 1]))))
    sham = float(np.max(np.abs(y[:, :, 0] - y[:, :, 1])))
    windows = [slice(320, 400), slice(720, 800)]
    margin = np.stack([y[w].mean(axis=0) for w in windows], axis=-1)
    ymin = np.stack([y[w].min(axis=0) for w in windows], axis=-1)
    ymax = np.stack([y[w].max(axis=0) for w in windows], axis=-1)
    checks = {
        "own_replay_max_abs_output_error": sham,
        "components_max_abs_error": float(np.max(np.abs(region.sum(axis=-1) + net.bias - margin))),
        "all_mean_mean_pair_state_max_abs_error": same_state_mean_error,
        "minimum_replayed_current": minimum_current,
        "rank1_current_reconstruction_max_error": float(np.max(np.abs(projected_safe - coefficient[..., None] * axis))),
    }
    assert sham < 1e-9, checks
    assert checks["components_max_abs_error"] < 1e-9, checks
    assert same_state_mean_error < 1e-9, checks
    assert minimum_current >= -1e-10, checks
    assert checks["rank1_current_reconstruction_max_error"] < 1e-9, checks
    reduction = {
        "heldout_rank1_energy_fraction_unattenuated": float(np.sum(projection ** 2) / np.sum(d ** 2)),
        "heldout_rank1_energy_fraction_replayed": float(np.sum(projected_safe ** 2) / np.sum(d ** 2)),
        "feasibility_attenuation_fraction": float(np.mean(feasibility < 1 - 1e-12)),
        "feasibility_minimum": float(feasibility.min()),
        "feasibility_median": float(np.median(feasibility)),
        "axis_absolute_cosine_output_PPC": float(abs(axis @ net.output[:400]) / np.linalg.norm(net.output[:400])),
    }
    arrays = dict(margin=margin, response_min=ymin, response_max=ymax, components=region,
                  output_trace=y, source_axis=axis, source_coefficient=coefficient,
                  feasibility=feasibility, donor_difference=d, source_mean=mu)
    return arrays, checks, reduction


def summarize(arrays, ctx):
    margin = arrays["margin"]
    ymin, ymax = arrays["response_min"], arrays["response_max"]
    expected = np.stack([np.sign(ctx["counts"][:, :, 4] - 10), np.broadcast_to([1, -1], (len(margin), 2))], axis=-1)
    rows = []
    for j, (name, state, source) in enumerate(CONDITIONS):
        z = margin[:, j]
        lo, hi = ymin[:, j], ymax[:, j]
        correct = np.where(expected > 0, (hi >= 0.75) & (lo >= -0.75), (lo <= -0.75) & (hi <= 0.75))
        opposite = np.where(expected < 0, (hi >= 0.75) & (lo >= -0.75), (lo <= -0.75) & (hi <= 0.75))
        neither = ~(correct | opposite)
        row = dict(condition=name, state=state, source=source)
        for task_id, task in enumerate(("current", "history")):
            row.update({
                task + "_sign_accuracy": float(np.mean((z[:, :, task_id] > 0) == (expected[:, :, task_id] > 0))),
                task + "_author_correct": float(np.mean(correct[:, :, task_id])),
                task + "_author_opposite": float(np.mean(opposite[:, :, task_id])),
                task + "_author_neither": float(np.mean(neither[:, :, task_id])),
                task + "_signed_history_margin": float(np.mean(z[:, :, task_id] * np.array([1, -1]))),
            })
        rows.append(row)
    return rows


def independent_one_step_contract(net):
    # This check compares the independent loader/update with root's implementation
    # only after both implementations have been specified. All novel outputs use
    # this file's independent evaluator.
    import sys
    sys.path.insert(0, str(R3 / "code"))
    from rnn_core import Model
    reference = Model(net.path)
    rng = np.random.default_rng(709031)
    x = rng.normal(5, 12, (5, 500))
    stimulus = np.array([-1, -0.5, 0, 0.5, 1])
    errors = {
        "weight_max_abs_error": float(np.max(np.abs(net.W - reference.W))),
        "time_constant_alpha_max_abs_error": float(np.max(np.abs(net.alpha - reference.alpha))),
        "rates_max_abs_error": float(np.max(np.abs(net.rates(x) - reference.rate(x)))),
        "one_step_max_abs_error": float(np.max(np.abs(net.step(x, stimulus) - reference.step(x, stimulus)))),
    }
    assert max(errors.values()) < 1e-10, errors
    return errors


def run_model(job):
    path, trainpath, testpath, outdir = job
    tick = time.time()
    net = IndependentNetwork(path)
    train, test = dict(np.load(trainpath)), dict(np.load(testpath))
    contract = independent_one_step_contract(net)
    axis, eigenvalues, train_energy = fit_axis(net, train)
    arrays, checks, reduction = evaluate(net, test, axis)
    arrays["training_spatial_eigenvalues"] = eigenvalues
    arrays["counts"] = test["counts"]
    arrays["base_context"] = test["base_context"]
    outdir = Path(outdir)
    np.savez_compressed(outdir / (net.name + ".npz"), **arrays)
    row = dict(model=net.name, checkpoint_sha256=sha(path), train_rank1_energy_fraction=train_energy,
               contract=contract, checks=checks, reduction=reduction, conditions=summarize(arrays, test), seconds=time.time()-tick)
    (outdir / (net.name + ".json")).write_text(json.dumps(row, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    out = HERE / "rank_one_heldout"
    out.mkdir(exist_ok=True)
    train = contexts(12, 83103, (2,))
    test = contexts(8, 83104)
    trainpath, testpath = out / "training_contexts.npz", out / "heldout_contexts.npz"
    np.savez_compressed(trainpath, **train)
    np.savez_compressed(testpath, **test)
    paths = sorted(DATA.glob("*/*.db"))
    assert len(paths) == 9
    protocol = dict(
        written_utc=datetime.now(timezone.utc).isoformat(), status="frozen_before_new_model_results",
        design="Spatial rank-one source reduction, basis from independent training contexts; all released checkpoints retained",
        n_training_base=12, training_seed=83103, n_heldout_base=8, heldout_seed=83104,
        q_values=[0, 2, 18, 20], intervention_updates=[1400, 1600], current_response=[1720, 1800], history_response=[2120, 2200],
        conditions=CONDITIONS, code_sha256=sha(__file__), original_source_sha256=sha(SOURCE),
        axis_fit="Leading eigenvector of uncentered covariance of paired donor differences over training contexts and 200 intervention steps",
        nonnegative_control="Common scalar attenuation of the rank-one difference, per context/time, only when pair-mean plus or minus half projection would become negative",
        endpoints="All continuous margins, mean signs, and author correct/opposite/neither criteria; no output-based condition selection",
        limitation="One spatial source axis with a time-varying coefficient does not assert one-dimensional whole-network dynamics",
    )
    protocolpath = out / "protocol.freeze.json"
    if protocolpath.exists():
        raise RuntimeError("Keep prior protocol/results; choose a new directory for any revised run")
    protocolpath.write_text(json.dumps(protocol, indent=2))
    results = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_model, (str(p), str(trainpath), str(testpath), str(out))) for p in paths]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            (out / "summary.json").write_text(json.dumps(sorted(results, key=lambda v: v["model"]), indent=2))
            selected = [r for r in result["conditions"] if r["condition"] in ("natural", "intact_full_opposite", "intact_rank1_opposite", "all_mean_rank1_own")]
            print(json.dumps(dict(model=result["model"], seconds=result["seconds"], train_energy=result["train_rank1_energy_fraction"], reduction=result["reduction"], checks=result["checks"], conditions=selected)), flush=True)
    (out / "completion.json").write_text(json.dumps(dict(status="complete", seconds=time.time()-start, models=len(results), code_sha256=sha(__file__)), indent=2))


if __name__ == "__main__":
    main()
