#!/usr/bin/env python3
"""Independent reaggregation of frozen Round-3 arrays; never rewrites inputs."""
from pathlib import Path
import json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;R3=HERE.parents[2]/'round3_consciousness'

def native(lo,hi,target):
    return np.where(target>0,(hi>=.75)&(lo>=-.75),(lo<=-.75)&(hi<=.75))

def main():
    out={'inputs':{},'history':[],'sequential':[],'finite':[]}
    for kind,folder in [('history','history_confirmation_v1'),('sequential','sequential_confirmation_v1')]:
        path=R3/'results'/folder;ctx=np.load(path/'contexts.npz');conds=json.loads((path/'conditions.json').read_text());names=[c['name'] for c in conds]
        out['inputs'][kind]={'context_sha256':hashlib.sha256((path/'contexts.npz').read_bytes()).hexdigest(),'n_contexts':len(ctx['u']),'n_conditions':len(conds)}
        if kind=='sequential':
            u=ctx['u'];bases=ctx['base_context'];mx=0
            for b in set(bases):
                uu=u[bases==b,:,:1650];mx=max(mx,float(np.max(abs(uu-uu[0]))))
            out['inputs'][kind]['max_same_base_prefix_discrepancy']=mx
            out['inputs'][kind]['shared_future_plus_minus_max_difference']=float(np.max(abs(u[:,0,1400:]-u[:,1,1400:])))
            target=np.sign(ctx['counts'][:,:,4]-10)
        for file in sorted(path.glob('*NUM*.npz')):
            z=np.load(file);y=z['margin'];lo=z['response_min'];hi=z['response_max'];row={'model':file.stem,'conditions':{}}
            for name in ['natural','intact_opposite_100','intact_opposite_200','PPC_mean_own_100','PPC_mean_mean_100']:
                if name not in names:continue
                j=names.index(name);y_j=y[:,j];rlo=lo[:,j];rhi=hi[:,j]
                if kind=='history':row['conditions'][name]={'H':float(np.mean((y_j[:,0]-y_j[:,1])/2))}
                else:
                    H=(y_j[:,0]-y_j[:,1])/2;hist=np.broadcast_to([1,-1],target.shape)
                    row['conditions'][name]=dict(H_current=float(H[:,0].mean()),H_history=float(H[:,1].mean()),current_mean_sign_accuracy=float(((y_j[:,:,0]>0)==(target>0)).mean()),current_native_commitment=float(native(rlo[:,:,0],rhi[:,:,0],target).mean()),original_history_mean_sign_accuracy=float(((y_j[:,:,1]>0)==(hist>0)).mean()),original_history_native_commitment=float(native(rlo[:,:,1],rhi[:,:,1],hist).mean()),opposite_history_native_commitment=float(native(rlo[:,:,1],rhi[:,:,1],-hist).mean()))
            if kind=='sequential':
                a=y[:,names.index('natural')];b=y[:,names.index('intact_opposite_200')]
                row['mean_absolute_change_current_history']=np.mean(abs(a-b),axis=(0,1)).tolist()
                comp=z['components'];dc=np.mean(comp[:,names.index('natural'),:,:, :]-comp[:,names.index('intact_opposite_200'),:,:,:],axis=0)
                # Signed historical contrast uses pair differences, not raw averaging across opposite signs.
                dc=.5*np.mean((comp[:,names.index('natural'),0,1]-comp[:,names.index('natural'),1,1])-(comp[:,names.index('intact_opposite_200'),0,1]-comp[:,names.index('intact_opposite_200'),1,1]),axis=0)
                row['regional_signed_history_effect']=dc.tolist();row['regional_signed_fraction']=(dc/dc.sum()).tolist()
            out[kind].append(row)
    p=R3/'results/finite_history_calculus_v1';fnames=[r['name'] for r in json.loads((p/'conditions.json').read_text())];seq=R3/'results/sequential_confirmation_v1';snames=[r['name'] for r in json.loads((seq/'conditions.json').read_text())]
    for file in sorted(p.glob('*NUM*.npz')):
        z=np.load(file);s=np.load(seq/file.name)
        reference=np.stack([s['margin'][:64,snames.index(c),0]-s['margin'][:64,snames.index(c),1] for c in fnames],axis=1)
        out['finite'].append(dict(model=file.stem,component_closure_max=float(np.max(abs(z['component_pair_differences'].sum(-1)-z['output_pair_difference']))),sequential_recalculation_max=float(np.max(abs(reference-z['output_pair_difference'])))))
    out['summary']={
      'history_opposite_1s_range':[min(r['conditions']['intact_opposite_200']['H'] for r in out['history']),max(r['conditions']['intact_opposite_200']['H'] for r in out['history'])],
      'sequential_current_native_min':min(r['conditions']['intact_opposite_200']['current_native_commitment'] for r in out['sequential']),
      'sequential_current_native_max':max(r['conditions']['intact_opposite_200']['current_native_commitment'] for r in out['sequential']),
      'sequential_history_sign_median_natural':float(np.median([r['conditions']['natural']['original_history_mean_sign_accuracy'] for r in out['sequential']])),
      'sequential_history_sign_median_opposite':float(np.median([r['conditions']['intact_opposite_200']['original_history_mean_sign_accuracy'] for r in out['sequential']])),
      'finite_component_max':max(r['component_closure_max'] for r in out['finite']),
      'finite_sequential_max':max(r['sequential_recalculation_max'] for r in out['finite'])}
    (HERE/'independent_array_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['summary'],indent=2))

if __name__=='__main__':main()
