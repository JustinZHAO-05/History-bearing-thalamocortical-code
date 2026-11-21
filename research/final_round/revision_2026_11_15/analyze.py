"""Summarize saved outcomes and verify them from independently loaded arrays."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,sys,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;R=Path(os.environ.get('NOV15_RESULTS',str(ROOT/'results')))
sys.path.insert(0,str(ROOT.parents[1]/'round3_consciousness/code'))
from rnn_core import Model
def interval(a,base):
 b=np.array([a[base==k].mean(axis=0) for k in np.unique(base)])
 ix=np.random.default_rng(98001).integers(0,len(b),(10000,len(b)))
 return dict(mean=b.mean(axis=0).tolist(),ci=np.quantile(b[ix].mean(axis=1),[.025,.975],axis=0).T.tolist())
def main():
 models=sorted((ROOT.parents[1]/'round3_consciousness/data/hosford_history_rnn').glob('*/*.db'))
 source=[];receiver=[];checks=[]
 for p in models:
  model=Model(p);name=model.name
  s=json.loads((R/f'source_{name}.json').read_text());z=np.load(R/f'source_{name}.npz')
  B=model.W[:400,400:470];E=model.arrays['PPCexh_idx'];I=model.arrays['PPCinh_idx'];A=np.array([B[E].sum(0),B[I].sum(0)])
  d=z['natural_difference'];maps=z['maps'];mu=z['source_mean'];norm=np.sum((d@B.T)**2,axis=-1)
  ec=[];ac=[];bounds=[]
  for m in maps:
   e=np.sum((m@B.T)**2,axis=-1);ec.append(float(np.max(np.divide(abs(e-norm),norm,out=np.zeros_like(norm),where=norm>1e-15))))
   ac.append(float(abs((m-d)@A.T).max()));bounds.append(float((abs(m)-2*np.minimum(mu,20-mu)).max()))
  y=z['output_trace'];H=(y[720:800,:,:,0]-y[720:800,:,:,1]).mean(0)/2
  assert np.max(abs(H-z['H']))<1e-12
  assert max(ec)<1e-8 and max(ac)<1e-7 and max(bounds)<1e-8
  for j in range(3):
   ci=s['outcomes']['sensory_change']['ci'][j]
   s['geometry'][j]['sensory_equivalent']=ci[0]>-.02 and ci[1]<.02
   s['geometry'][j]['history_loss_positive']=s['outcomes']['history_loss']['ci'][j][0]>0
  source.append(s)
  rec=json.loads((R/f'receiver_{name}.json').read_text());v=np.load(R/f'receiver_{name}.npz');yy=v['output_trace'];base=v['base'];by={}
  for qi,q in enumerate((2,10,18)):
   mask=np.arange(len(base))%3==qi;curH=(yy[320:400][:,mask][...,0]-yy[320:400][:,mask][...,1]).mean(0)/2
   h=v['H'][mask];sens=v['sensory'][mask];current=v['current'][mask]
   by[str(q)]=dict(current_H=interval(curH,base[mask]),later_H=interval(h,base[mask]),later_change=interval(h-h[:,0,None],base[mask]),current_H_change=interval(curH-curH[:,0,None],base[mask]))
   if q!=10:by[str(q)].update(sensory=interval(sens,base[mask]),current=interval(current,base[mask]))
   normr=v['rate_norm'][320:400][:,mask].mean(0);al=v['alignment'][320:400][:,mask].mean(0)
   by[str(q)]['rate_norm']=interval(normr,base[mask]);by[str(q)]['alignment']=interval(al,base[mask])
  rec['responses']=by;receiver.append(rec)
  checks.append(dict(model=name,energy_error=max(ec),ei_error=max(ac),box_error=max(bounds),output_reaggregation_error=float(abs(H-z['H']).max()),weights_sha256=model.sha256))
 out=dict(source=source,receiver=receiver,checks=checks)
 (R/'scientific_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
 rows=[]
 for s in source:
  for j,g in enumerate(s['geometry']):rows.append(dict(model=s['model'],map=s['names'][j+3],change=g['relative_change'],history_loss=s['outcomes']['history_loss']['mean'][j],history_low=s['outcomes']['history_loss']['ci'][j][0],history_high=s['outcomes']['history_loss']['ci'][j][1],sensory_change=s['outcomes']['sensory_change']['mean'][j],sensory_low=s['outcomes']['sensory_change']['ci'][j][0],sensory_high=s['outcomes']['sensory_change']['ci'][j][1],sensory_equivalent=g['sensory_equivalent'],current=s['outcomes']['current']['mean'][j+3],roughness_ratio=g['roughness']/max(g['natural_roughness'],1e-20)))
 with (R/'source_function.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 print(json.dumps(dict(models=len(source),max_energy=max(x['energy_error'] for x in checks),max_ei=max(x['ei_error'] for x in checks),source_rows=rows,receiver=[dict(model=x['model'],pred=x['predictions'],responses=x['responses']) for x in receiver]),indent=2))
if __name__=='__main__':main()
