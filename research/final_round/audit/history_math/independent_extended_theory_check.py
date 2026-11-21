"""Read-only finite-account and analytic saturation checks for Extended Theory."""
from pathlib import Path
import numpy as np
import hashlib,json,datetime

OUT=Path(__file__).resolve().parent
RESEARCH=Path(__file__).resolve().parents[3]
R3=RESEARCH/'round3_consciousness'
rng=np.random.default_rng(19331)
source=[];records=[]
for f in sorted((R3/'results/finite_history_calculus_v1').glob('*.npz')):
    if f.stem=='contexts':continue
    z=np.load(f)
    dy=z['output_pair_difference']
    comp=z['component_pair_differences']
    rms=z['instantaneous_output_difference_rms']
    alignment=z['readout_alignment']
    # Coherence is independent of native output scaling and bounds by one.
    coherence=np.divide(dy,rms,out=np.zeros_like(dy),where=rms>0)
    assert np.max(abs(coherence))<=1+1e-12
    err=float(np.max(abs(comp.sum(-1)-dy)))
    assert err<1e-10
    records.append({'model':f.stem,'output_sum_error':err,
                    'max_absolute_temporal_coherence':float(abs(coherence).max()),
                    'max_absolute_window_alignment':float(abs(alignment).max()),
                    'context_count':len(dy),'condition_count':dy.shape[1]})
    source.append({'file':str(f.relative_to(RESEARCH)), 'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})

# The precise analytic smooth expression uses a half contrast.
z=rng.uniform(-8,8,5000);a=rng.uniform(0,3,5000)
actual=(np.tanh(z+a)-np.tanh(z-a))/2
closed=np.sinh(2*a)/(np.cosh(2*z)+np.cosh(2*a))
smooth_error=float(np.max(abs(actual-closed)))
assert smooth_error<1e-14

# Exact finite secants, including all three activation pieces and zero deltas.
x=rng.uniform(-30,40,(1000,500));xp=rng.uniform(-30,40,(1000,500))
xp[::4]=x[::4]
def rate(y):
    r=np.maximum(y,0);r[:,470:]=np.maximum(10*(y[:,470:]-9),0)
    return np.minimum(r,20)
diff=x-xp;dr=rate(x)-rate(xp)
mid=(x+xp)/2
fb=((mid>0)&(mid<20)).astype(float)
fb[:,470:]=10*((mid[:,470:]>9)&(mid[:,470:]<11))
sec=np.divide(dr,diff,out=fb,where=diff!=0)
secerr=float(np.max(abs(sec*diff-dr)))
assert secerr<1e-13

logical=[]
for history in (-1,1):
    for write in (-1,1):
        for current in (-1,1):
            h=.85*history*write;s=2*current
            y=float(np.clip(h+s,0,1)-np.clip(-h-s,0,1))
            h_after=h+.025*s
            yl=float(np.clip(h_after,0,1)-np.clip(-h_after,0,1))
            assert y*current>=.75 and yl*history*write>=.75
            logical.append({'history':history,'source_write':write,'current':current,'current_output':y,'later_output':yl})
report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Read-only archived finite components; independent random-point analytic and activation tests. No new trained-network simulation or fitted model.',
        'source_files':source,'finite_component_checks':records,
        'max_saved_component_output_error':max(v['output_sum_error'] for v in records),
        'smooth_saturation_points':len(z),'smooth_saturation_max_error':smooth_error,
        'secant_coordinate_cases':int(diff.size),'secant_identity_max_error':secerr,
        'secant_range':[float(sec.min()),float(sec.max())],
        'scalar_memory_eight_cases':logical}
(OUT/'independent_extended_theory_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['source_files','finite_component_checks','scalar_memory_eight_cases']},indent=2))
