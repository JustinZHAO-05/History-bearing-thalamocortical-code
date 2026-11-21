import json, pathlib,numpy as np,hashlib,csv
root=pathlib.Path(__file__).resolve().parents[1]
files=sorted(root.glob('results/*_RateRNNstructNUM*.json'))
source=[]
for p in files:source.append(json.loads(p.read_text()))
assert len(source)==9,len(source)
ref=[json.loads(p.read_text()) for p in sorted((root/'results/reflection_v2').glob('*_RateRNNstructNUM*.json'))]
assert len(ref)==9,len(ref)
(root/'results/summary.json').write_text(json.dumps(source,indent=2))
(root/'results/completion.json').write_text(json.dumps(dict(status='complete',models=9),indent=2))
(root/'results/reflection_v2/summary.json').write_text(json.dumps(ref,indent=2))
(root/'results/reflection_v2/completion.json').write_text(json.dumps(dict(status='complete',models=9),indent=2))
rows=[]
for x in source:
    name=x['model'].replace('Gtune_RateRNNstructNUM','G').replace('sussano_RateRNNstructNUM','S')
    r=next(a for a in ref if a['model']==x['model'])
    ap=np.load(root/'results/reflection_v2'/f"{x['model']}.npz")
    H=ap['history_signed'];base=ap['base'];b=np.stack([H[base==k].mean(axis=0) for k in range(12)])
    rng=np.random.default_rng(96200+len(rows));ix=rng.integers(0,12,size=(20000,12));diff=b[:,1,None]-b[:,3:5]
    avg=diff.mean(axis=1);dbar=avg.mean();ci=np.quantile(avg[ix].mean(axis=1),[.025,.975])
    means=x['metrics']['history_signed_margin']['means'];rmeans=r['outcomes']['history_signed']['mean'];cur=r['outcomes']['current_success']['mean']
    row=dict(model=name,source_learned_variance=x['training_source_first_axis_energy'],full_history=means['allmean_own'],single_source_history=means['allmean_learned'],
         source_random_abs_mean=np.mean([abs(means[f'allmean_random{j}']) for j in range(3)]),
         learned_1pct_energy_matched=x['geometry'][0]['matching_fraction_1pct'],
         random_energy_match_median=float(np.median([x['geometry'][j]['matching_fraction_1pct'] for j in (1,2,3)])),
         current_learned=x['metrics']['current_success']['means']['allmean_learned'],
         exact_map_feasible=float(np.mean([j['feasible_fraction'] for j in r['geometry']])),
         exact_receiver_change=float(np.mean([j['mean_receiver_relative_change'] for j in r['geometry']])),
         exact_full_H=rmeans['allmean_own'],exact_map_H=float(np.mean([rmeans[f'allmean_map{j}'] for j in range(2)])),
         exact_change_H=float(dbar),exact_change_H_CI_low=float(ci[0]),exact_change_H_CI_high=float(ci[1]),
         exact_map_current_success=float(np.mean([cur[f'allmean_map{j}'] for j in range(2)])))
    rows.append(row)
with open(root/'results/source_space_findings.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
agg=dict(n_checkpoints=9,learned_axis_source_energy_range=[float(min(z['source_learned_variance'] for z in rows)),float(max(z['source_learned_variance'] for z in rows))],
      allmean_full_hist_min=float(min(z['full_history'] for z in rows)),allmean_full_hist_max=float(max(z['full_history'] for z in rows)),
      learned_hist_min=float(min(z['single_source_history'] for z in rows)),learned_hist_max=float(max(z['single_source_history'] for z in rows)),
      learned_matching_frac_range=[float(min(z['learned_1pct_energy_matched'] for z in rows)),float(max(z['learned_1pct_energy_matched'] for z in rows))],
      random_matching_frac_median=float(np.median([g['matching_fraction_1pct'] for x in source for g in x['geometry'][1:]])),
      random_arms_1pct_matching_ge_50pct=int(sum(g['matching_fraction_1pct']>=0.5 for x in source for g in x['geometry'][1:])),
      exact_model_feasible_range=[float(min(z['exact_map_feasible'] for z in rows)),float(max(z['exact_map_feasible'] for z in rows))],
      exact_source_diff_range=[float(min(z['exact_receiver_change'] for z in rows)),float(max(z['exact_receiver_change'] for z in rows))],
      exact_H_change_range=[float(min(z['exact_change_H'] for z in rows)),float(max(z['exact_change_H'] for z in rows))],
      all_current_success_learned=float(min(z['current_learned'] for z in rows)),
      all_current_success_exact=float(min(z['exact_map_current_success'] for z in rows)),
      note='Nine released model checkpoints; inferential resampling is over 12 independently drawn base histories within each model, with two current strengths paired. Maps are within-model geometric controls, not new biological subjects.')
(root/'results/source_space_findings.json').write_text(json.dumps(dict(summary=agg,rows=rows),indent=2))
print(json.dumps(agg,indent=2))
print('\nMODEL LEVEL')
for z in rows:print(z['model'], 'rank1',round(z['source_learned_variance'],3),'full',round(z['full_history'],3),'src1',round(z['single_source_history'],3),'exact difference',round(z['exact_change_H'],4),'CI',round(z['exact_change_H_CI_low'],4),round(z['exact_change_H_CI_high'],4),'delta',round(z['exact_receiver_change'],3),'cur',round(z['exact_map_current_success'],3))
