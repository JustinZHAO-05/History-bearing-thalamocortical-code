from pathlib import Path
import argparse
import json
import numpy as np

default = Path(__file__).resolve().parent / 'research/final_round/revision_2026_11_15_r2/results'
parser = argparse.ArgumentParser(description="Recalculate the five held-out predictions.")
parser.add_argument("--results", type=Path, default=default, help="Folder containing prediction arrays and summary.")
folder = parser.parse_args().results
names = ('evidence_recurrence', 'pre_recurrence', 'evidence_leak', 'pre_leak', 'reference_template')
records = json.loads((folder / 'prediction_summary.json').read_text(encoding='utf-8'))
by_model = {row['model']: row['scores'] for row in records}
for file in sorted(folder.glob('prediction_*.npz')):
    model = file.stem.removeprefix('prediction_')
    with np.load(file, allow_pickle=False) as data:
        actual = data['actual']
        forecasts = data['prediction']
        errors = ((forecasts - actual[..., None])**2)
        model_sse = errors.sum(axis=(0,1))
        pooled_rmse = np.sqrt(errors.mean(axis=(0,1)))
        pooled_r2 = 1 - model_sse / np.sum((actual - actual.mean())**2)
        skill = 1 - model_sse / model_sse[4]
        for index, name in enumerate(names):
            score = by_model[model][name]
            assert np.allclose(
                [pooled_rmse[index], pooled_r2[index], skill[index]],
                [score['rmse'], score['r2'], score['template_relative_skill']],
                rtol=1e-12, atol=1e-12,
            ), (model, name)
    print(model, 'RMSE', ', '.join(f'{k}={v:.4f}' for k, v in zip(names, pooled_rmse)))
assert len(by_model) == 9

assert {path.stem.removeprefix('prediction_') for path in folder.glob('prediction_*.npz')} == set(by_model)
