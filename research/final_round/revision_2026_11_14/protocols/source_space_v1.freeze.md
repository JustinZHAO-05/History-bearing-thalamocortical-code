# Source-space history replay — prospective v1 protocol

Frozen before inspecting any novel source-space behavioral outcomes. Created on 2026-11-14.

## Core question
Can historical differences carried by 70 bounded PUL-source activities, when passed through the original 400x70 trained PUL→PPC matrix, reconstruct a signed historical contrast? Are independent alternative source directions comparably effective when output-difference energy is matched?

## Predefined conditions and data
Use all nine fixed published checkpoints. All model weights, native 5-ms dynamics, trained readout and cell masks remain unchanged. Use 12 training base histories (seed 94001; strong fifth trial q=2 only) to learn per-checkpoint leading left principal direction v of signed PUL rate difference, without behavioral selection. Use 12 independently generated test base histories (seed 94002), crossed with q=2 and q=18 at original fifth position, paired opposite old histories, common future. Perturb during 200 updates [1400,1600), 1 s before the strong fifth trial, and read windows updates [1720,1800) and [2120,2200). Generate three fixed Gaussian source directions (seeds 94010+model_index*20+j), orthogonalize to v. No outcome-dependent candidate selection.

## Source-reachable replay
From each native donor pair r_T^+,r_T^- in 70 PUL nodes, let mu=(r_T^++r_T^-)/2, d=r_T^+-r_T^-, and B=W_PPC,PUL. A replayed signed difference d' must satisfy each 0<=mu_i +/- d'_i/2<=20. For learned direction q=v or each random q, use raw signed amplitude (d·v)q, tune a nonnegative scalar using **only** source trajectories, B and matching to ||B d||, then clip to [-2 min(mu_i,20-mu_i),2 min(mu_i,20-mu_i)]. The optimizer records actual ||B d'||/||B d|| and marks conditions unable to attain 1 within 1%. It keeps them; behavioral endpoints do not decide feasibility. Pair background mu and 400-dimensional baseline for all compared target currents are exactly matched.

A mathematical limitation: an input d' in the constrained feasible box can be generated as B d' but need not be a freely evolving PUL state. This is a bounded source **edge-clamp** test. It does not establish natural endogenous dynamics can generate the imposed source trajectory.

## Replay conditions (in this order)
0 natural intact; 1 all-state pair average + natural own-source rate replay; 2 all-state pair average + pair mean-source replay; 3 all-state pair average + learned one-mode source replay; 4-6 all-state pair average + three orthogonal random-source replay controls; 7 all-state pair average + natural opposite-source rate replay.

## Endpoints
Primary: signed historical half-pair contrast, native historical commitment, mean absolute source-space energy match, mean source amplitude and direction, source feasibility coverage. Current task: native strong-stimulus correct commitment, sign, signed sensory response margin. Fully retain within-pair new time traces of y and rate-difference norms, and all training/test seeds and SHA digests. Units are base histories within each published checkpoint. No pooled biological p-values across nine parameter files of incompletely known training genealogy.

## Interpretation
Predefined comparison: full original and the learned source code versus three unrelated sampled source directions, with explicit achieved energy and any matching residual. If less than half the time/context points meet 1% energy matching, report feasibility failure and treat unequal energy as confounding; do not proclaim a causal geometry advantage. Current-function matching is a separate necessary outcome, not a post-hoc selection criterion.
