# Final Round biological carryover audit, amendment 2

Date 2026-11-11. Written after completing v1 and before the following new computations. All v1 outputs, including unstable subset fits and the initial empty-cell implementation error, are retained.

## Observations motivating amendment

The primary and lag-adjusted full-sample fits are stable. Strict absence of laser over the preceding five trials leaves very small samples for several early mice and completely separates some session intercepts. The original coefficient-only ridge does not regularize such nuisance intercepts. Also, actual source laser series show repeated patterns/blocks and heterogeneous rates, making slow within-session covariation a material alternative to the paper's summary random-interleaving description.

## Declared new checks

1. Quantify source laser rates, lag-1 through lag-5 autocorrelation, transition probabilities, consecutive-run lengths and current-vs-control stimulus/history balance. Preserve session-wise tables and source timeline examples from the first source session for every mouse. These describe observed assignment sequences, without alleging an underlying randomization algorithm.
2. Fit the v1 lag-adjusted model plus per-session linear and quadratic normalized trial-index terms, and their interactions with current stimulus and five-stimulus history. Normalized time is t/(n_session−1) mapped to [−1,1], with both polynomial columns centered on the eligible trials within the session. No time-basis tuning is allowed. This allows choice bias, sensory slope and history slope to drift smoothly within each session while retaining a common current laser-by-history coefficient per mouse. Common 1e-4 non-intercept penalty is used for every mouse if any unpenalized fit within the cohort is unstable; every diagnostic is retained.
3. Resolve the no-prior-laser subset's separation using Firth/Jeffreys-bias-reduced logistic likelihood for every mouse and intervention, not selected individual fits. The design, eligibility and session intercepts remain unchanged. Optimize negative log likelihood minus 0.5 log det(X'WX), with analytic gradient independently checked by central finite differences. Report data rank and fitted diagnostics. If rank deficient or a fit remains unstable, label this subset inconclusive for that cohort rather than removing animals. This conditional subset has limited overlap and lower precision even after numerical regularization.

## Analysis and interpretation

Use 100000 animal bootstrap draws with seed 20261123. Report effect estimates, original and adjusted contrasts, exact signed-rank values, sample sizes and stability. This is a transparent retrospective sensitivity extension. Persistence after lag/time adjustment strengthens the functional history-weight interpretation under measured sequence covariates; source schedules limit the design information available for unqualified randomized-effect statements. Preserve heterogeneous mouse results, including any interval spanning zero.
