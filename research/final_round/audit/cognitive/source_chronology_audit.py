#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
import numpy as np
import biology_carryover_audit as audit
HERE=Path(__file__).resolve().parent

def main():
 out={}
 for label,src in audit.SOURCES.items():
  sessions,*_=audit.read_source(label,src);rows=[];mrows=[]
  for ss in sessions:
   z=ss['z'];s=2*z[3]-1;n=len(s);h=np.mean(np.stack([audit.lag(s,k) for k in range(1,6)]),axis=0)
   mask=(z[5]==1)&np.isclose(z[3],.5)&np.isin(z[1],[0,1])&np.isfinite(h)&(abs(h)>1e-12)
   ix=np.flatnonzero(mask);bad=ix[z[1,ix]!=(h[ix]>0)]
   # Compare declared preceding-five and shifted windows, a descriptive chronology check.
   shifted={}
   for off in (-2,-1,0,1,2):
    pred=[];truth=[]
    for t in ix:
     a=t-5+off;b=t+off
     if a>=0 and b<=n:
      v=float(s[a:b].mean())
      if abs(v)>1e-12:pred.append(v>0);truth.append(z[1,t])
    shifted[str(off)]=dict(n=len(pred),agreements=int(np.sum(np.array(pred)==truth)))
   rows.append(dict(mouse=ss['mouse'],session=ss['name'],history_n=len(ix),history_matches=len(ix)-len(bad),mismatch_indices=bad.tolist(),shifted_windows=shifted))
  for m in sorted(set(ss['mouse'] for ss in sessions)):
   rr=[r for r in rows if r['mouse']==m];trs=np.zeros((2,2),int);rates=[];max0=[]
   for ss in sessions:
    if ss['mouse']!=m:continue
    l=ss['z'][4].astype(int);np.add.at(trs,(l[:-1],l[1:]),1)
    rates.extend(l.tolist());cuts=np.r_[0,np.flatnonzero(np.diff(l))+1,len(l)];lens=np.diff(cuts);lev=l[cuts[:-1]];max0.append(int(max(lens[lev==0])))
   conditional=(trs[:,1]/trs.sum(1)).tolist()
   mrows.append(dict(mouse=m,n_history=sum(r['history_n'] for r in rr),n_history_agreements=sum(r['history_matches'] for r in rr),laser_rate=float(np.mean(rates)),transition_counts=trs.tolist(),prob_laser_given_previous_0_1=conditional,transition_difference=conditional[1]-conditional[0],max_nonlaser_run=max(max0)))
  sourcepath=audit.OLD/'source_data'/src
  out[label]=dict(source_sha256=hashlib.sha256(sourcepath.read_bytes()).hexdigest(),n_history=sum(r['history_n'] for r in rows),n_history_agreements=sum(r['history_matches'] for r in rows),mouse_summary=mrows,session_rows=rows)
 (HERE/'source_chronology_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 for label,g in out.items():print(label,g['n_history_agreements'],g['n_history'],g['n_history_agreements']/g['n_history'],[(r['mouse'],round(r['transition_difference'],3),r['max_nonlaser_run']) for r in g['mouse_summary']])

if __name__=='__main__':main()
