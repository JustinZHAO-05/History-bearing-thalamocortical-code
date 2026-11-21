"""Scientific tables from all corrected source and low-dose response arrays."""
from pathlib import Path
import json,csv
import numpy as np
import experiments as ex
ROOT=Path(__file__).resolve().parent;OUT=ex.OUT;OLD=ROOT.parent/'revision_2026_11_15/results'
def csvout(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 source=[];low=[];activation=[];sensory=[];audit=[];receiver_rows=[]
 for path in sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db')):
  net=ex.v1.eng.IndependentNetwork(path);name=net.name
  a=json.loads((OUT/f'axis_{name}.json').read_text());z=np.load(OUT/f'axis_{name}.npz');base=z['base_context'];H=z['output_trace'][720:800].mean(0);H=(H[:,:,0]-H[:,:,1])/2
  assert np.max(abs(H-z['history_signed_margin']))<1e-12
  for j,g in enumerate(a['geometry']):
   source.append(dict(model=name,**g,history=a['metrics']['history_signed_margin']['mean'][j+3],history_low=a['metrics']['history_signed_margin']['ci'][j+3][0],history_high=a['metrics']['history_signed_margin']['ci'][j+3][1],current=a['metrics']['current_success']['mean'][j+3]))
  for j in [0,1,2,7]:assert np.max(abs(z['output_trace'][:,:,j]-np.load(ROOT.parent/'revision_2026_11_14/results'/f'{name}.npz')['output_trace'][:,:,j]))<1e-11
  v=np.load(OUT/f'lowdose_{name}.npz');meta=json.loads((OUT/f'lowdose_{name}.json').read_text());base=v['base'];y=v['output_trace'];h=(y[410:490,:,:,0]-y[410:490,:,:,1]).mean(0)/2
  assert np.max(abs(h-v['H']))<1e-12
  minimum=y[410:490].min(0);maximum=y[410:490].max(0);signs=np.array([1.,-1.])[None,None]
  history_commit=np.where(signs>0,(maximum>=.75)&(minimum>=-.75),(minimum<=-.75)&(maximum<=.75)).mean(-1)
  opposite_commit=np.where(signs<0,(maximum>=.75)&(minimum>=-.75),(minimum<=-.75)&(maximum<=.75)).mean(-1)
  by={}
  for qi,q in enumerate((2,10,18)):
   keep=np.arange(len(base))%3==qi;by[str(q)]={}
   for key,arr in [('later_H_change',h-h[:,0,None]),('current_H_change',v['current_H']-v['current_H'][:,0,None]),('sensory_change',v['sensory']-v['sensory'][:,0,None]),('commitment_loss',v['current'][:,0,None]-v['current']),('history_native_commitment',history_commit),('history_opposite_commitment',opposite_commit)]:by[str(q)][key]=ex.interval(arr[keep],base[keep])
   # Exact retention/alignment/time identity at each context, each window.
   for label,sl in [('current',slice(10,90)),('later',slice(410,490))]:
    dy=y[sl,:,:,0]-y[sl,:,:,1];R=np.sqrt(v['rate_norm_squared'][sl].mean(0));rmsout=np.sqrt((dy*dy).mean(0));A=np.divide(rmsout,np.linalg.norm(net.output)*R,out=np.zeros_like(R),where=R>1e-20);C=np.divide(abs(dy.mean(0)),rmsout,out=np.zeros_like(R),where=rmsout>1e-20)
    assert np.max(abs(abs(dy.mean(0))/2-.5*np.linalg.norm(net.output)*R*A*C))<1e-12
    for key,arr in [('R',R),('A',A),('C',C)]:by[str(q)][f'{label}_{key}']=ex.interval(arr[keep],base[keep])
   w=v['activation_weights'][qi]
   for ri,region in enumerate(('PPC','PUL','TRN')):
    for arm in range(11):
     for ci,category in enumerate(meta['activation_categories']):
      activation.append(dict(model=name,q=q,region=region,arm=arm,category=category,potential_fraction=float(w[ri,arm,ci,0]/max(w[ri,arm,:,0].sum(),1e-30)),absolute_readout_fraction=float(w[ri,arm,ci,1]/max(w[ri,arm,:,1].sum(),1e-30)),signed_readout_weight=float(w[ri,arm,ci,2])))
  strong=np.arange(len(base))%3!=1;b=base[strong]
  native=dict(later_change=ex.interval((h-h[:,0,None])[strong],b),inward_minus_random=ex.interval((h[:,1]-h[:,2:10].mean(1))[strong],b),recurrent_change=ex.interval((h[:,-1]-h[:,0])[strong],b),random_change=ex.interval((h[:,2:10].mean(1)-h[:,0])[strong],b),history_commitment=ex.interval(history_commit[strong],b),history_commitment_change=ex.interval((history_commit-history_commit[:,0,None])[strong],b))
  entry=dict(model=name,selected_displacement=meta['selected_displacement'],selected_attenuation=meta['selected_attenuation'],mean_actual_shift=meta['mean_actual_shift'],responses=by,native=native,calibration=meta)
  low.append(entry)
  for q in ('2','18'):
   ci=np.array(by[q]['sensory_change']['ci']);loss=np.array(by[q]['commitment_loss']['mean']);entry.setdefault('test_equivalence',{})[q]=dict(sensory_all=bool(np.all((ci[:,0]>-.02)&(ci[:,1]<.02))),commitment_all=bool(np.all(loss<=.01+1e-12)),largest_sensory_interval=float(abs(ci).max()),largest_commitment_loss=float(loss.max()))
  for qi,q in enumerate((2,10,18)):
   for j,arm in enumerate(meta['arms']):
    rr=by[str(q)];receiver_rows.append(dict(model=name,q=q,arm=j,name=arm['name'],dose=arm['dose'],later_H_change=rr['later_H_change']['mean'][j],later_low=rr['later_H_change']['ci'][j][0],later_high=rr['later_H_change']['ci'][j][1],sensory_change=rr['sensory_change']['mean'][j],sensory_low=rr['sensory_change']['ci'][j][0],sensory_high=rr['sensory_change']['ci'][j][1],current_commitment=meta['responses'][str(q)]['current']['mean'][j],later_commitment=rr['history_native_commitment']['mean'][j],current_R=rr['current_R']['mean'][j],current_A=rr['current_A']['mean'][j],current_C=rr['current_C']['mean'][j],later_R=rr['later_R']['mean'][j],later_A=rr['later_A']['mean'][j],later_C=rr['later_C']['mean'][j]))
  for row in receiver_rows:
   if row['q']==10:
    for key in ('sensory_change','sensory_low','sensory_high','current_commitment'):row[key]=''
  # The preceding 48-base bounded-source experiment, resolved by stimulus.
  zz=np.load(OLD/f'source_{name}.npz');bb=zz['base'];s=zz['sensory']
  for qi,q in enumerate((0,2,18,20)):
   keep=np.arange(len(bb))%4==qi
   for j in (3,4,5):
    iv=ex.interval(s[keep,j]-s[keep,1],bb[keep]);sensory.append(dict(model=name,q=q,map=j,mean=iv['mean'],low=iv['ci'][0],high=iv['ci'][1]))
  audit.append(dict(model=name,weights_sha256=ex.previous.Model(path).sha256,source_mean_bound=float(np.max(abs(z['source_candidate_difference'])-2*np.minimum(z['source_mean'],20-z['source_mean']))),initial_delta_change=float(np.max(abs(v['initial_state_difference']-v['initial_state_difference'][:,0,None])))))
 summary=dict(axis=[json.loads((OUT/f'axis_{ex.v1.eng.IndependentNetwork(p).name}.json').read_text()) for p in sorted((ex.v1.R3/'data/hosford_history_rnn').glob('*/*.db'))],lowdose=low,prediction=json.loads((OUT/'prediction_summary.json').read_text()),audit=audit)
 ex.dump(OUT/'scientific_summary.json',summary);csvout('source_axis_geometry.csv',source);csvout('activation_weights.csv',activation);csvout('source_sensory_by_strength.csv',sensory);csvout('receiver_function.csv',receiver_rows)
 print(json.dumps([dict(model=r['model'],dose=r['selected_displacement'],attenuation=r['selected_attenuation'],native=r['native'],test_equivalence=r['test_equivalence']) for r in low],indent=2))
if __name__=='__main__':main()
