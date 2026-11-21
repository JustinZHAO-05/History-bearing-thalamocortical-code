# Historical messages and receiver transformations

15 November 2026. This amendment follows the scientific review of the integrated
manuscript. The source-axis correction reuses the original fitting and test
histories and stored axes. New receiver experiments use separate calibration
and test histories. All nine released parameter files and the native readout
remain fixed.

## Bounded source-axis correction

For each stored axis, evaluate the full path clip(k q, -c, c), k >= 0,
with the temporal sign of the learned scalar coefficient retained. Each
clipping interval has a quadratic receiving energy. Locate the first interval
reaching the natural receiving norm and solve within it. If that norm lies
above the entire path, use the breakpoint with maximal receiving norm.
Zero-target cases use the zero difference. Multiple solutions use the first
crossing; behavioral outcomes enter after construction. Report the exact-path
1% norm matching fraction, the realized message, and its clipping geometry.
Recompute all eight original replay conditions on seed 94002, 12 bases crossed
with two strong stimuli. Preserve the original finite-grid results separately.

## Low-dose receiver experiment

Twelve calibration bases (seed 99101) and 48 new test bases (99102) cross
positive-pip counts 2,10,18. Recorded own-source replay and full-state averaging
are identical across conditions through update 1710. At that update:

* Common inward displacement is a dose times the preceding experiment's
  half-displacement toward [2,18] in the common PPC state.
* Eight Gaussian unit vectors are drawn once per checkpoint (seed 99300+index),
  fixed across histories and strengths, and scaled to the same displacement
  norm in each context. The paired state difference remains identical.
* A separate arm attenuates the full 500-unit recurrent-drive difference for
  20 updates by a fraction lambda, retaining its pair mean and external input.

Calibration evaluates doses/fractions 0.001,0.003,0.01,0.03,0.1,0.3,1 plus
zero, using only the current-response window. The displacement family takes
the largest dose for which the inward direction and every one of the eight
random directions both satisfy the following criterion at each strong stimulus:
the paired 95% interval for signed mean sensory change lies within +/-0.02
native output units and native commitment loss relative to own is at most 0.01.
The recurrent attenuation family selects its largest passing fraction by the
same criterion. Zero is retained when no positive dose qualifies. No later
historical response enters calibration.

On the separate 48-base sample, evaluate own, the selected inward displacement,
all eight selected equal-norm random displacements, and selected recurrent
attenuation. Retain every checkpoint and direction. Report selected dose,
calibration and test sensory effects, current commitments, immediate state and
rate differences, current/later historical contrasts, readout alignment and
time consistency. Confidence intervals resample whole base histories, keeping
strengths, paired signs and directions nested within each base. Test data do
not change the selected doses. Random directions are interpreted as the fixed
sampled comparison family.

Partition activation pairs into below threshold, linear, upper bound, lower
crossing, upper crossing and both-boundary crossing. Report both
|readout| * native slope * |state difference| weights and actual absolute
readout-contribution weights, separately by region and strength. These locate
the nonlinearities acting on historical differences.

## Prediction and temporal interpretation

Existing 12-reference/24-test histories supply the full 2x2 comparison of
evidence-specific versus pre-sensory gains and present versus absent recurrent
transformation. A reference mean response is a separate comparator. Report
pooled, within-evidence, and reference-template-residual prediction quality,
with paired base-history uncertainty in error differences. These predictions
condition on the measured test starting state and reference gain trajectories.

Keep the completed whole-trajectory source experiment as a test of temporally
coherent, function-preserving alternatives. Document its composite fixed
spatial reflections, per-context roughness and natural-message fallback.
Spatial displacement and temporal organization are interpreted jointly.
