# Final Round biological carryover audit, version 1

Frozen before execution, 2026-11-11. Inputs are immutable Round 3 original Figure 1/4 MAT source arrays. New outputs belong only to final_round/audit/cognitive. No new outcome-based exclusion or checkpoint selection.

## Question

Does the reduction of historical evidence influence during current PPC/PUL suppression survive adjustment for preceding intertrial interventions and observed choice/reward history? Randomized current laser is analyzed as current treatment; previous laser may modify the memory state on which it acts.

## Common eligibility

As the corrected Round 3 all-binary GLM: regular (unflagged history) trials after five observations, finite preceding-five-stimulus mean, binary target and choice, finite binary laser. No feedback-coherence or target/stimulus agreement filter in GLMs. Session boundaries reset all lags. Previous-choice controls require a preceding binary choice. All variants are fit to the same eligible set per mouse unless explicitly the no-recent-laser subset.

## Declared calculations

1. Independently reconstruct every source session/mouse cached accuracy summary and independently recompute the primary conflict-accuracy contrasts.
2. Describe current-laser rates and balance of stimulus, history, absolute stimulus, previous laser, mean preceding-five laser, previous choice, previous feedback. Report animal-wise contrasts, no claims of randomized allocation from a nonsignificant test.
3. Fit baseline per-mouse logistic choice model with session intercepts and s, h, L, sL, hL, without coherence selection.
4. Fit lag-adjusted model: baseline plus centered previous-one laser (p1) and centered mean laser at lags 2–5 (p25), each interacting with 1, s, h, L, sL, hL. Add previous signed choice and previous feedback, each centered within mouse, with their current-laser interactions. Centering retains β_hL as the average conditional history interaction on the observed preceding-laser distribution. This is an adjusted descriptive choice model, not a unique biological mediation model.
5. Fit baseline within trials having no laser in the previous five trials. This subset is specified by pre-current-treatment variables. Report all retained animals and per-cell counts; preserve instability rather than silently discard a mouse.
6. Obtain cross-mouse means and bootstrap percentile 95% intervals (100000 resamples; seed 20261122), exact signed-rank p-values as descriptive secondary results, and all animal coefficients. Primary decision: whether the direction and magnitude of the historical interaction persist after lag adjustment. The no-recent-laser subset is a supporting sensitivity with expected lower precision.

## Numerical validation

Use the existing likelihood optimizer as a separately callable routine with exact objective, require maximum gradient <1e-4, finite coefficients with |β|<50 and Hessian condition <1e12. If any fit within intervention/specification is unstable, apply the predeclared common penalty 1e-4 to all animals in that intervention/specification, and report. All convergence diagnostics, exclusions, complete fits and raw derived trial table are retained. The baseline results should reproduce the prior all-binary model up to any explicitly counted previous-choice eligibility difference.

## Interpretation rules

Persistence supports robustness to recent intervention exposure; marked attenuation directs the final story toward history-state dependence. A weaker no-recent-laser interval changes precision, not automatically the adjusted mechanism. Any discrepancy in cached outcomes/schema is resolved before modeling. Data are reused and this audit is retrospective; the internal protocol freeze protects against specification changes after seeing this calculation.
