# History-bearing thalamocortical signals

Analysis and simulation code for *History-bearing thalamocortical signals reconstruct and redirect later decisions in a circuit model* by Yanzhe Zhao and Yongzhang Ji. [Preprint, version 2](https://doi.org/10.5281/zenodo.24418936)

The repository includes the mouse choice analyses, fixed-weight circuit simulations, source replay and redistribution, receiver perturbations, and held-out prediction comparisons.

## Setup and published inputs

Use Python 3.12 on Linux for simulations that use the fork multiprocessing start method.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python prepare_inputs.py all
```

The input preparation script obtains the original model and behavioral archives and verifies their checksums. Use `models` or `behavior` to prepare each input group independently. [Published source data and checkpoints](https://doi.org/10.5281/zenodo.19465617) · [Original model implementation](https://github.com/DistribCogLab-1/history_based_rnn)

## Mouse choice analyses

```bash
python research/round3_consciousness/scouts/r3_data_and_function/portable_reproduce.py biology
python research/final_round/audit/cognitive/reproduce_cognitive_audit.py --out runs/mouse --skip-rnn-array-check
```

The first command produces the choice-model results. The second runs the carryover and treatment-overlap analyses. Its model-array check uses saved circuit responses from separate simulations.

## Fixed-weight circuit simulations

Use a fresh output directory for each task.

```bash
python reproduce.py history --out runs/history
python reproduce.py sequential --out research/round3_consciousness/results/sequential_confirmation_v1
python reproduce.py finite --out runs/finite
python reproduce.py direction --out runs/direction
python reproduce.py native-classes --out runs/native-classes
python reproduce.py position --out runs/position
```

The finite calculation reads the saved sequential contexts. The launcher also exposes the rank-one, uplift, clipping, and residual add-back controls.

## Source matching, receiver perturbations, and prediction

Run these stages from the repository root in order. The later stages read source and receiver arrays produced by the earlier stages. The original model checkpoints stay fixed. Numerical outputs are stored in the corresponding study directories under `results/`.

```bash
python research/final_round/revision_2026_11_14/code/source_space_v1.py --workers 2
python research/final_round/revision_2026_11_15/experiments.py --workers 2
python research/final_round/revision_2026_11_15/analyze.py
python research/final_round/revision_2026_11_15/prediction_comparison.py
python research/final_round/revision_2026_11_15_r2/experiments.py axis --workers 2
python research/final_round/revision_2026_11_15_r2/experiments.py receiver --workers 2
python research/final_round/revision_2026_11_15_r2/prediction.py
python research/final_round/revision_2026_11_15_r2/analyze.py
python research/final_round/revision_2026_11_15_r2/check_science.py
python review_prediction_scores.py
```

`review_prediction_scores.py --results PATH` can also recalculate the five prediction scores from the nine trajectory arrays and summary table packaged in Extended Data 1. See the protocols beside the scripts for sampling and intervention definitions. Plotting and saved-array audit scripts use the respective run outputs and the source tables or figure assets named in those scripts.
