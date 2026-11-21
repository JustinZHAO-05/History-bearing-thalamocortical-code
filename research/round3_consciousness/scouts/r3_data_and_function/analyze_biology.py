#!/usr/bin/env python3
"""Independent descriptive reproduction of Hosford & Mei et al. 2026 Fig 1/4.

Data: Zenodo 19465617; CC BY 4.0. All primary inference uses mice within
intervention. This is a reanalysis of published intervention data, not a held-out
validation of a newly specified theory. Run from any working directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "pydeps"))
import h5py
import numpy as np
from scipy import stats

SOURCES = {
    "PPC_suppression": "Figure 1/PPCSuppEffectsSourceF1.mat",
    "PUL_suppression": "Figure 1/PULSuppEffectsSourceF1.mat",
    "TRN_PUL_activation": "Figure 4/4BtoD/TRNActEffectBehavSourceF4.mat",
}
PERF_KEYS = ("regPerfNL", "regPerfLas", "catchPerfNL", "catchPerfLas")
MOUSE_KEYS = ("regNLMouse", "regLasMouse", "catchNLMouse", "catchLasMouse")


def decode(h, obj):
    if isinstance(obj, h5py.Group):
        return {k: decode(h, x) for k, x in obj.items()}
    if obj.attrs.get("MATLAB_empty", False):
        return np.empty((0,))
    a = obj[()]
    kind = obj.attrs.get("MATLAB_class", b"")
    if kind == b"char":
        return "".join(chr(int(x)) for x in a.ravel())
    if kind == b"cell":
        values = [decode(h, h[x]) for x in a.ravel()]
        return values[0] if a.size == 1 else values
    return a


def read_source(label, relative):
    path = BASE / "source_data" / relative
    with h5py.File(path, "r") as h:
        refs = h["ZmatSet"][()]
        names = h["SessNameSet"][()]
        sessions = []
        for idx in np.ndindex(refs.shape):
            d = h[refs[idx]]
            if d.attrs.get("MATLAB_empty", False):
                continue
            z = d[()]
            assert z.ndim == 2 and z.shape[0] == 6
            sessions.append({
                "intervention": label,
                "mouse": idx[1] + 1,
                "within_mouse_session": idx[0] + 1,
                "name": decode(h, h[names[idx]]),
                "z": z,
            })
        author_mouse = h["mouseRowVal"][()].ravel().astype(int)
        assert np.array_equal(author_mouse, [s["mouse"] for s in sessions])
        colnames = decode(h, h["colNameSet"])
        # colNameSet can contain shortened names. Preserve it for review rather
        # than assert spelling identity with the full filename in SessNameSet.
        author_perf = np.stack([h[k][()].ravel() for k in PERF_KEYS], axis=1)
        author_mice = np.stack([h[k][()].ravel() for k in MOUSE_KEYS], axis=1)
        coefficients = {k: np.asarray(decode(h, h[k])).ravel()
                        for k in ("sessNLStim", "sessLasStim", "sessNLHist",
                                  "sessLasHist", "sessNLDelHist", "sessLasDelHist")}
        assert all(len(v) == len(sessions) for v in coefficients.values())
    return sessions, author_perf, author_mice, coefficients, colnames


def trial_performance(z):
    # These definitions reproduce every one of the four stored per-session
    # performance arrays exactly in all three datasets. Catch/inference labels
    # must use the explicit flag, not every sensory 50:50 mixture.
    history = (z[5] == 1) & np.isclose(z[3], .5)
    laser = z[4] == 1
    correct = z[0] == 1  # Source values .5 are not counted as correct.
    masks = ((~history) & (~laser), (~history) & laser,
             history & (~laser), history & laser)
    counts = np.array([np.sum(mask) for mask in masks], dtype=int)
    successes = np.array([np.sum(correct & mask) for mask in masks], dtype=int)
    rates = np.divide(successes, counts, out=np.full(4, np.nan), where=counts > 0)
    return rates, counts, successes


def paired_stats(d, bootstrap_indices):
    d = np.asarray(d, dtype=float)
    assert np.isfinite(d).all()
    b = d[bootstrap_indices].mean(axis=1)
    nonzero = d[d != 0]
    return {
        "n_mice": len(d),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "sd": float(d.std(ddof=1)),
        "bootstrap95_mean": np.quantile(b, [.025, .975]).tolist(),
        "wilcoxon_exact_two_sided_p": float(stats.wilcoxon(d, method="exact").pvalue),
        "sign_two_sided_p": float(stats.binomtest(np.sum(nonzero > 0), len(nonzero), .5).pvalue),
        "n_positive": int(np.sum(d > 0)),
        "n_negative": int(np.sum(d < 0)),
        "n_zero": int(np.sum(d == 0)),
        "values": d.tolist(),
    }


def main():
    output = BASE / "results"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261111)
    all_rows, all_sessions = [], []
    results = {
        "analysis": "frozen_v0_descriptive_replication",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": 20261111,
        "bootstrap_resamples": 100000,
        "independence": "Mouse, separately for each intervention. Cross-intervention identity not assumed.",
        "performance_units": "fraction correct; multiply differences by 100 for percentage points",
        "groups": {},
    }
    for label, relative in SOURCES.items():
        sessions, author_perf, author_mice, coeffs, names = read_source(label, relative)
        mice = np.array([s["mouse"] for s in sessions])
        computed = [trial_performance(s["z"]) for s in sessions]
        rates = np.stack([p[0] for p in computed])
        counts = np.stack([p[1] for p in computed])
        successes = np.stack([p[2] for p in computed])
        assert np.allclose(rates, author_perf, rtol=0, atol=1e-14, equal_nan=True)
        mouse_values = np.stack([np.nanmean(rates[mice == m], axis=0)
                                 for m in sorted(set(mice))])
        assert np.allclose(mouse_values, author_mice, rtol=0, atol=1e-14, equal_nan=True)
        n = len(mouse_values)
        boot = rng.integers(n, size=(100000, n))
        delta_current = mouse_values[:, 1] - mouse_values[:, 0]
        delta_history = mouse_values[:, 3] - mouse_values[:, 2]
        rho, rho_p = stats.spearmanr(delta_current, -delta_history)
        coefficient_results = {}
        for v in ("Stim", "Hist", "DelHist"):
            d = coeffs["sessLas" + v] - coeffs["sessNL" + v]
            # Both aggregations are reported; published source values can be
            # extreme, and sessions are repeated observations of the same mouse.
            means = np.array([np.mean(d[mice == m]) for m in sorted(set(mice))])
            medians = np.array([np.median(d[mice == m]) for m in sorted(set(mice))])
            coefficient_results[v] = {
                "session_delta_min_max": [float(d.min()), float(d.max())],
                "mouse_mean_delta": paired_stats(means, boot),
                "mouse_median_delta": paired_stats(medians, boot),
                "interpretation": "Reaggregated author-cached rolling logistic coefficients; original rolling fit not independently recreated.",
            }
        g = {
            "n_mice": n,
            "n_sessions": len(sessions),
            "n_trials": int(counts.sum()),
            "trial_counts_current_NL_Laser_history_NL_Laser": counts.sum(axis=0).tolist(),
            "accuracy_equal_mouse_current_NL_Laser_history_NL_Laser": mouse_values.mean(axis=0).tolist(),
            "accuracy_equal_session_current_NL_Laser_history_NL_Laser": np.nanmean(rates, axis=0).tolist(),
            "accuracy_pooled_trials_current_NL_Laser_history_NL_Laser": (successes.sum(axis=0) / counts.sum(axis=0)).tolist(),
            "delta_current": paired_stats(delta_current, boot),
            "delta_history": paired_stats(delta_history, boot),
            "interaction_current_minus_history": paired_stats(delta_current - delta_history, boot),
            "exploratory_history_cost_current_benefit_spearman": {"rho": float(rho), "p": float(rho_p)},
            "cached_coefficient_reaggregation": coefficient_results,
            "audit": {
                "session_performance_max_absolute_error": float(np.nanmax(abs(rates - author_perf))),
                "mouse_performance_max_absolute_error": float(np.nanmax(abs(mouse_values - author_mice))),
                "mouse_grouping_validated_with_mouseRowVal": True,
                "source_sha256": hashlib.sha256((BASE / "source_data" / relative).read_bytes()).hexdigest(),
            },
        }
        results["groups"][label] = g
        for m, row in enumerate(mouse_values, start=1):
            all_rows.append(dict(intervention=label, mouse=m,
                n_sessions=int(np.sum(mice == m)),
                n_trials=int(counts[mice == m].sum()),
                current_NL=row[0], current_laser=row[1], history_NL=row[2], history_laser=row[3],
                delta_current=row[1]-row[0], delta_history=row[3]-row[2],
                interaction=(row[1]-row[0])-(row[3]-row[2])))
        for i, s in enumerate(sessions):
            record = {k:v for k,v in s.items() if k != "z"}
            for j, k in enumerate(("current_NL", "current_laser", "history_NL", "history_laser")):
                record[k] = rates[i,j]
                record[k+"_n"] = counts[i,j]
                record[k+"_correct"] = successes[i,j]
            all_sessions.append(record)
        print(label, "n", n, "sessions", len(sessions), "trials", int(counts.sum()),
              "delta current", g["delta_current"]["mean"],
              "delta history", g["delta_history"]["mean"],
              "interaction", g["interaction_current_minus_history"]["mean"])
    (output / "biology_v0_results.json").write_text(json.dumps(results, indent=2) + "\n")
    for filename, records in (("biology_mouse_summary.csv", all_rows),
                               ("biology_session_summary.csv", all_sessions)):
        with (output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    return results


if __name__ == "__main__":
    main()
