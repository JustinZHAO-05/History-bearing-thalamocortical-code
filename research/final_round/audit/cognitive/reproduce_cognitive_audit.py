#!/usr/bin/env python3
"""Run frozen analyses into a new directory without overwriting archived results.

Example:
  python reproduce_cognitive_audit.py --out /tmp/final_cognitive_audit
Optional --r3-root changes only source paths; sibling R3 package is default.
"""
from pathlib import Path
import argparse,importlib.util,sys,shutil,json,hashlib

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def clean_jsons(out):
    # Intermediate failed fits may retain unavailable asymptotic SEs. Their
    # values are made explicit nulls in a separate strict JSON export; original
    # executor outputs stay untouched for provenance. Final accepted numerical
    # summaries are finite.
    def convert(x):
        if isinstance(x,float):
            import math
            return x if math.isfinite(x) else None
        if isinstance(x,list):return [convert(v) for v in x]
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        return x
    d=out/'strict_json';d.mkdir()
    for p in out.glob('*.json'):
        data=json.loads(p.read_text());(d/p.name).write_text(json.dumps(convert(data),indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--r3-root',type=Path,default=HERE.parents[2]/'round3_consciousness');ap.add_argument('--skip-rnn-array-check',action='store_true');args=ap.parse_args();out=Path(args.out).resolve();r3=args.r3_root.resolve()
    if out.exists() and any(out.iterdir()):raise SystemExit('Refusing to overwrite a nonempty output directory: '+str(out))
    out.mkdir(parents=True,exist_ok=True);old=r3/'scouts/r3_data_and_function'
    if not old.is_dir():raise SystemExit('Missing R3 biological source directory: '+str(old))
    sys.path[:0]=[str(old),str(old/'pydeps'),str(HERE)]
    for name in ['biology_carryover_protocol.md','biology_carryover_protocol_v2.md']:
        shutil.copyfile(HERE/name,out/name)
    bio=load('analyze_biology',old/'analyze_biology.py');bio.BASE=old
    load('analyze_trials',old/'analyze_trials.py')
    audit=load('biology_carryover_audit',HERE/'biology_carryover_audit.py');audit.HERE=out;audit.R3=r3;audit.OLD=old;audit.main()
    v2=load('biology_carryover_v2',HERE/'biology_carryover_v2.py');v2.HERE=out;v2.main()
    chrono=load('source_chronology_audit',HERE/'source_chronology_audit.py');chrono.HERE=out;chrono.main()
    if not args.skip_rnn_array_check:
        arrays=load('independent_array_audit',HERE/'independent_array_audit.py');arrays.HERE=out;arrays.R3=r3;arrays.main()
    clean_jsons(out)
    manifest={'adapter':'path/output override only; scientific functions unmodified','r3_root':str(r3),'output':str(out),'scripts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'biology_carryover_audit.py',HERE/'biology_carryover_v2.py',HERE/'source_chronology_audit.py',HERE/'independent_array_audit.py']}}
    (out/'execution_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('AUDIT REPRODUCTION COMPLETE',out,flush=True)

if __name__=='__main__':main()
