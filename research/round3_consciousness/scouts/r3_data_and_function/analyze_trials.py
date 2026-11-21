#!/usr/bin/env python3
"""Frozen-v1 exploratory trial analysis; see trial_analysis_v1.md for decisions."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

import numpy as np
from scipy import optimize, special, stats

from analyze_biology import BASE, SOURCES, read_source, paired_stats


def lagmean(a, length=5):
    out = np.full(len(a), np.nan)
    # Direct slices ensure the current trial is excluded and session boundaries
    # cannot contaminate the history predictor.
    for t in range(length, len(a)):
        out[t] = np.mean(a[t-length:t])
    return out


def features(z):
    s = 2*z[3]-1
    history = (z[5] == 1) & np.isclose(z[3], .5)
    return {
        "s": s,
        "h": lagmean(s),
        "h_target": lagmean(2*z[1]-1),
        "L": z[4],
        "correct": z[0] == 1,
        "target": z[1],
        "choice": z[2],
        "history_trial": history,
        "valid_choice": np.isin(z[2], [0,1]),
        "valid_target": np.isin(z[1], [0,1]),
        "coherent_feedback": (z[0] == 1) == (z[1] == z[2]),
        "previous_choice": np.r_[np.nan, 2*z[2,:-1]-1],
        "previous_feedback": np.r_[np.nan, 2*(z[0,:-1] == 1)-1],
    }


def conflict_counts(sessions, mode):
    rows = []
    for session in sessions:
        f = features(session["z"])
        h = f["h_target"] if mode == "target_history" else f["h"]
        v = (~f["history_trial"]) & (np.abs(f["s"]) > 1e-12) & (np.abs(h) > 1e-12)
        v &= np.isfinite(h) & f["valid_choice"]
        if mode != "include_target_mismatch":
            v &= f["target"] == (f["s"] > 0)
        congruent = f["s"]*h > 0
        masks = [v & congruent & (f["L"] == 0), v & congruent & (f["L"] == 1),
                 v & (~congruent) & (f["L"] == 0), v & (~congruent) & (f["L"] == 1)]
        n = np.array([m.sum() for m in masks], dtype=int)
        k = np.array([(m & f["correct"]).sum() for m in masks], dtype=int)
        rows.append({"mouse": session["mouse"], "name": session["name"], "n": n, "k": k})
    return rows


def summarize_conflict(rows, rng, equal_session=False):
    mouse_rows = []
    for m in sorted(set(r["mouse"] for r in rows)):
        rr = [r for r in rows if r["mouse"] == m]
        if equal_session:
            rr = [r for r in rr if np.all(r["n"] > 0)]
        if not rr:
            continue
        n = np.stack([r["n"] for r in rr]).sum(axis=0)
        k = np.stack([r["k"] for r in rr]).sum(axis=0)
        if np.any(n == 0):
            continue
        rates = (np.stack([r["k"] / r["n"] for r in rr]).mean(axis=0)
                 if equal_session else k/n)
        mouse_rows.append({"mouse": m, "n_sessions": len(rr), "counts": n.tolist(),
                           "rates": rates.tolist(),
                           "delta_congruent": float(rates[1]-rates[0]),
                           "delta_conflict": float(rates[3]-rates[2]),
                           "interaction": float((rates[3]-rates[2])-(rates[1]-rates[0]))})
    n = len(mouse_rows)
    boot = rng.integers(n, size=(100000, n))
    return {"n_mice": n, "total_eligible_trials": int(sum(sum(r["counts"]) for r in mouse_rows)),
            "order": "congruent_NL, congruent_laser, conflict_NL, conflict_laser",
            "aggregation": "equal-session within mouse" if equal_session else "trial-pool within mouse",
            "mice": mouse_rows,
            **{k: paired_stats(np.array([r[k] for r in mouse_rows]), boot)
               for k in ("delta_congruent", "delta_conflict", "interaction")}}


def fit_logit(X, y, n_intercepts, penalty=0.):
    penalty_vector = np.r_[np.zeros(n_intercepts), np.full(X.shape[1]-n_intercepts, penalty)]
    def objective(b):
        eta = X @ b
        loss = np.logaddexp(0, eta).sum() - y @ eta + .5*np.sum(penalty_vector*b*b)
        grad = X.T @ (special.expit(eta)-y) + penalty_vector*b
        return float(loss), grad
    result = optimize.minimize(objective, np.zeros(X.shape[1]), method="L-BFGS-B", jac=True,
                               options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 2000,
                                        "maxls": 40, "maxcor": 30})
    # A likelihood near 2000 can satisfy a relative function tolerance while
    # gradients remain around 2e-4. Newton polishing solves the SAME objective
    # accurately before classifying a fit as unstable or imposing a penalty.
    polish_steps = 0
    for _ in range(20):
        loss, gradient_vector = objective(result.x)
        if np.max(np.abs(gradient_vector)) < 1e-8:
            break
        prob = special.expit(X @ result.x)
        curvature = X.T @ (X*(prob*(1-prob))[:,None]) + np.diag(penalty_vector)
        try:
            direction = np.linalg.solve(curvature, gradient_vector)
        except np.linalg.LinAlgError:
            break
        step = 1.
        accepted = False
        for _ in range(30):
            candidate = result.x - step*direction
            candidate_loss, candidate_gradient = objective(candidate)
            if candidate_loss <= loss + 1e-10 and np.max(abs(candidate_gradient)) < np.max(abs(gradient_vector)):
                result.x = candidate
                result.fun = candidate_loss
                polish_steps += 1
                accepted = True
                break
            step *= .5
        if not accepted:
            break
    probability = special.expit(X @ result.x)
    hessian = X.T @ (X*(probability*(1-probability))[:,None]) + np.diag(penalty_vector)
    finite = np.isfinite(result.x).all()
    gradient = float(np.max(np.abs(objective(result.x)[1])))
    conditioned = float(np.linalg.cond(hessian))
    stable = finite and gradient < 1e-4 and conditioned < 1e12 and np.max(abs(result.x)) < 50
    covariance = np.linalg.pinv(hessian)
    return {"coefficients": result.x.tolist(), "approx_within_mouse_se": np.sqrt(np.diag(covariance)).tolist(),
            "n_trials": len(y), "n_parameters": X.shape[1], "converged": bool(result.success),
            "stable": bool(stable), "message": str(result.message),
            "same_objective_newton_polish_steps": polish_steps,
            "max_abs_gradient": gradient, "hessian_condition": conditioned,
            "negative_log_likelihood_plus_penalty": float(result.fun), "penalty": penalty,
            "fraction_fitted_p_outside_001_099": float(np.mean((probability < .01) | (probability > .99)))}


def model_design(sessions, nuisance=False):
    blocks, outcomes, identities = [], [], []
    for i, session in enumerate(sessions):
        f = features(session["z"])
        v = (~f["history_trial"]) & np.isfinite(f["h"]) & f["valid_target"] & f["valid_choice"] & f["coherent_feedback"]
        if nuisance:
            v &= np.isfinite(f["previous_choice"]) & np.isin(f["previous_choice"], [-1,1])
        s,h,L = f["s"][v], f["h"][v], f["L"][v]
        cols = [s,h,L,s*L,h*L]
        if nuisance:
            c,r = f["previous_choice"][v], f["previous_feedback"][v]
            cols.extend([c,r,c*L,r*L])
        blocks.append(np.column_stack(cols))
        outcomes.append(f["choice"][v])
        identities.extend([i]*int(v.sum()))
    identities = np.array(identities)
    intercepts = np.eye(len(sessions))[identities]
    X = np.column_stack([intercepts, np.vstack(blocks)])
    y = np.concatenate(outcomes)
    return X, y, len(sessions)


def main():
    rng = np.random.default_rng(20261112)
    results = {"created_utc": datetime.now(timezone.utc).isoformat(), "analysis": "frozen_v1_exploratory",
               "history_length": 5, "groups": {}, "readme": "See trial_analysis_v1.md for exclusions and interpretation."}
    all_session_tables = []
    coefficient_rows = []
    for label, source in SOURCES.items():
        sessions, _, _, _, _ = read_source(label, source)
        group = {"conflict": {}, "choice": {}, "schema": {}}
        all_z = np.concatenate([s["z"] for s in sessions], axis=1)
        f = features(all_z)  # schema counts only; no histories used from concatenation
        group["schema"] = {
            "total_trials": int(all_z.shape[1]),
            "feedback_nonbinary_n": int(np.sum(~np.isin(all_z[0], [0,1]))),
            "choice_nonbinary_or_nan_n": int(np.sum(~f["valid_choice"])),
            "target_nonbinary_n": int(np.sum(~f["valid_target"])),
            "feedback_identity_disagreements": int(np.sum(~f["coherent_feedback"])),
            "stimulus_nonzero_target_disagreements": int(np.sum((np.abs(f["s"])>1e-12)&(f["target"]!=(f["s"]>0)))),
            "all_50_50_trials": int(np.sum(np.isclose(all_z[3],.5))),
            "explicit_history_trials": int(np.sum(f["history_trial"])),
        }
        for mode in ("primary", "include_target_mismatch", "target_history"):
            rr = conflict_counts(sessions, mode)
            group["conflict"][mode] = summarize_conflict(rr, rng)
            if mode == "primary":
                group["conflict"]["equal_session"] = summarize_conflict(rr, rng, equal_session=True)
                for r in rr:
                    for j, cell in enumerate(("congruent_NL", "congruent_laser", "conflict_NL", "conflict_laser")):
                        all_session_tables.append({"intervention":label,"mouse":r["mouse"],"session":r["name"],"cell":cell,"n":r["n"][j],"correct":r["k"][j]})
        for nuisance in (False, True):
            name = "with_choice_reward_controls" if nuisance else "primary"
            fitted, designs = [], []
            for m in sorted(set(s["mouse"] for s in sessions)):
                X,y,ni = model_design([s for s in sessions if s["mouse"]==m], nuisance)
                designs.append((m,X,y,ni))
                fitted.append(fit_logit(X,y,ni))
            # If any fit unstable, the predeclared weak penalty is applied to all
            # mice in that intervention/model, avoiding outcome-specific tuning.
            use_penalty = any(not fit["stable"] for fit in fitted)
            if use_penalty:
                fitted = [fit_logit(X,y,ni,1e-4) for m,X,y,ni in designs]
            rows = []
            names = ["stimulus", "history", "laser_bias", "laser_stimulus", "laser_history"]
            if nuisance:
                names += ["previous_choice", "previous_feedback", "laser_previous_choice", "laser_previous_feedback"]
            for (m,X,y,ni), fit in zip(designs,fitted):
                row = {"mouse":m, "session_intercepts":fit["coefficients"][:ni],
                       "coefficients":dict(zip(names,fit["coefficients"][ni:])),
                       "fit":{k:v for k,v in fit.items() if k!="coefficients"}}
                rows.append(row)
                coefficient_rows.append({"intervention":label,"model":name,"mouse":m,"n_trials":len(y),"penalty":fit["penalty"],**row["coefficients"]})
            n=len(rows);boot=rng.integers(n,size=(100000,n))
            summaries = {c:paired_stats(np.array([r["coefficients"][c] for r in rows]),boot) for c in names}
            group["choice"][name] = {"all_mice_penalized":use_penalty,"mice":rows,"coefficient_summaries":summaries}
        results["groups"][label] = group
        for mode,g in group["conflict"].items():
            r=g["interaction"]
            print(label,mode,"conflict benefit minus congruent benefit",r["mean"],r["bootstrap95_mean"],r["wilcoxon_exact_two_sided_p"])
        r=group["choice"]["primary"]["coefficient_summaries"]
        print(label,"choice laser interactions stimulus/history",r["laser_stimulus"]["mean"],r["laser_history"]["mean"])
    out=BASE/"results"
    (out/"biology_v1_trial_results.json").write_text(json.dumps(results,indent=2)+"\n")
    for filename,records in [("biology_conflict_session_cells.csv",all_session_tables),
                              ("biology_choice_coefficients.csv",coefficient_rows)]:
        fields=list(dict.fromkeys(k for r in records for k in r))
        with (out/filename).open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(records)


if __name__ == "__main__":
    main()
