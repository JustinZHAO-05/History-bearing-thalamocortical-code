"""Independent numerical audit of retained R3 inputs and endpoints."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
R3 = HERE.parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def native(zmin, zmax, target):
    correct = np.where(target > 0, (zmax >= .75) & (zmin >= -.75), (zmin <= -.75) & (zmax <= .75))
    opposite = np.where(target < 0, (zmax >= .75) & (zmin >= -.75), (zmin <= -.75) & (zmax <= .75))
    return dict(correct=correct, opposite=opposite, neither=~(correct | opposite))


def history_audit():
    base = R3 / "results/history_confirmation_v1"
    ctx = dict(np.load(base / "contexts.npz"))
    conds = json.loads((base / "conditions.json").read_text())
    index = {v["name"]: i for i, v in enumerate(conds)}
    counts = ctx["counts"]
    observed = np.stack([(ctx["u"][:, :, t*400+50:t*400+110:3] > 0).sum(axis=-1) for t in range(6)], axis=-1)
    assert np.array_equal(observed, counts)
    assert np.all(counts[:, 0, :5].sum(axis=-1) > 50)
    assert np.all(counts[:, 1, :5].sum(axis=-1) < 50)
    conflict = counts[:, 0, 4] < 10
    target = np.broadcast_to([1, -1], (len(counts), 2))
    results = []
    for path in sorted(base.glob("*NUM*.npz")):
        z = np.load(path)
        y = z["margin"]
        stored = json.loads(path.with_suffix(".json").read_text())
        agreement = 0.0
        for j, row in enumerate(stored["conditions"]):
            criteria = native(z["response_min"][:, j], z["response_max"][:, j], target)
            sign_accuracy = np.mean((y[:, j] > 0) == (target > 0))
            agreement = max(agreement, abs(float(sign_accuracy) - row["history_sign_accuracy"]), abs(float(criteria["correct"].mean()) - row["history_author_success"]))
        ni, oi = index["natural"], index["intact_opposite_200"]
        a, b = y[:, ni], y[:, oi]
        component_delta = ((z["components"][:, ni] - z["components"][:, oi]) * target[..., None]).mean(axis=(0, 1))
        margin_delta = float(((a-b)*target).mean())
        rows = {}
        for label, mask in (("all", np.ones(len(counts), dtype=bool)), ("latest_conflicts_with_integrated_history", conflict), ("latest_agrees_with_integrated_history", ~conflict)):
            rows[label] = {}
            for condition in ("natural", "intact_opposite_200"):
                j = index[condition]
                criteria = native(z["response_min"][mask, j], z["response_max"][mask, j], target[mask])
                rows[label][condition] = dict(
                    n_contexts=int(mask.sum()),
                    history_sign_accuracy=float(np.mean((y[mask, j] > 0) == (target[mask] > 0))),
                    signed_history_margin=float(np.mean(y[mask, j] * target[mask])),
                    **{k: float(v.mean()) for k, v in criteria.items()},
                )
        results.append(dict(model=path.stem, endpoint_max_error=agreement,
                            native_minus_opposite_signed_margin=margin_delta,
                            component_delta_PPC_PUL_TRN=component_delta.tolist(),
                            component_delta_fraction_PPC_PUL_TRN=(component_delta/margin_delta).tolist(),
                            subgroup_results=rows, npz_sha256=sha(path)))
    assert max(r["endpoint_max_error"] for r in results) < 1e-15
    return dict(n_contexts=len(counts), latest_conflict_contexts=int(conflict.sum()),
                input_count_contract_passed=True, models=results)


def sequential_audit():
    base = R3 / "results/sequential_access_pilot"
    ctx = dict(np.load(base / "contexts.npz"))
    conditions = json.loads((base / "conditions.json").read_text())
    index = {c["name"]: i for i, c in enumerate(conditions)}
    counts = ctx["counts"]
    target = np.stack([np.sign(counts[:, :, 4]-10), np.broadcast_to([1, -1], (len(counts), 2))], axis=-1)
    observed = np.stack([(ctx["u"][:, :, t*400+50:t*400+110:3] > 0).sum(axis=-1) for t in range(6)], axis=-1)
    assert np.array_equal(observed, counts)
    assert np.all(counts[:, 0, :5].sum(axis=-1) > 50)
    assert np.all(counts[:, 1, :5].sum(axis=-1) < 50)
    results = []
    for path in sorted(base.glob("*NUM*.npz")):
        z = np.load(path)
        y = z["margin"]
        a, b = y[:, index["natural"]], y[:, index["intact_opposite_200"]]
        stored = json.loads(path.with_suffix(".json").read_text())
        error = 0.0
        for j, row in enumerate(stored["conditions"]):
            criteria = native(z["response_min"][:, j], z["response_max"][:, j], target)
            signs = (y[:, j] > 0) == (target > 0)
            for t, name in enumerate(("current", "history")):
                error = max(error, abs(float(signs[:, :, t].mean()) - row[name+"_sign_accuracy"]),
                            abs(float(criteria["correct"][:, :, t].mean()) - row[name+"_author_success"]))
        delta = a-b
        paired = (a[:, 0]-a[:, 1])/2
        perstrength = []
        for q in (0, 2, 18, 20):
            mask = counts[:, 0, 4] == q
            perstrength.append(dict(q=q, n_base=int(mask.sum()),
                                    mean_absolute_source_effect_current=float(abs(delta[mask, :, 0]).mean()),
                                    mean_absolute_source_effect_history=float(abs(delta[mask, :, 1]).mean())))
        results.append(dict(model=path.stem, endpoint_max_error=error,
                            natural_mean_absolute_paired_history_half_difference_current_history=abs(paired).mean(axis=0).tolist(),
                            mean_absolute_source_effect_current_history=abs(delta).mean(axis=(0, 1)).tolist(),
                            maximum_absolute_current_source_effect=float(abs(delta[:, :, 0]).max()),
                            per_strength=perstrength, npz_sha256=sha(path)))
    assert max(r["endpoint_max_error"] for r in results) < 1e-15
    return dict(n_base=len(np.unique(ctx["base_context"])), input_count_contract_passed=True, models=results)


def main():
    result = dict(history_confirmation=history_audit(), sequential_pilot=sequential_audit(), code_sha256=sha(__file__))
    (HERE / "saved_output_audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(
        history_contexts=result["history_confirmation"]["n_contexts"],
        latest_conflict_contexts=result["history_confirmation"]["latest_conflict_contexts"],
        history_latest_conflict_accuracy={r["model"]: r["subgroup_results"]["latest_conflicts_with_integrated_history"]["natural"]["history_sign_accuracy"] for r in result["history_confirmation"]["models"]},
        max_endpoint_error=max(r["endpoint_max_error"] for kind in ("history_confirmation", "sequential_pilot") for r in result[kind]["models"]),
    )), flush=True)


if __name__ == "__main__":
    main()
