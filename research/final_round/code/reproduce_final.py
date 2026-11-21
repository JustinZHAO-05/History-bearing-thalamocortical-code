#!/usr/bin/env python3
"""Portable, fresh-output launchers for the final paper. No archived result is overwritten."""
from __future__ import annotations
import os
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse, hashlib, importlib, importlib.util, json, multiprocessing, shutil, subprocess, sys, time

FINAL=Path(__file__).resolve().parents[1]
R3=FINAL.parent/'round3_consciousness'
PACKAGE=FINAL.parents[1]

def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod

def fresh(path):
    out=Path(path).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('Choose an empty output directory; archived data will not be overwritten: '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    return out

def prepare_models(model_dir=None):
    scout=R3/'scouts/r3_data_and_function'
    portable=load('portable_reproduce',scout/'portable_reproduce.py')
    destination=Path(model_dir).expanduser().resolve() if model_dir else R3/'data/hosford_history_rnn'
    inputs=[destination/x for x in portable.MODEL_FILES]
    if not all(p.is_file() for p in inputs):
        portable.extract_verified_models(R3/'data/hosford_model_source.zip',destination)
    manifest=json.loads((scout/'portable_input_manifest.json').read_text())['model_database_sha256']
    for file in inputs:
        if sha(file)!=manifest[file.relative_to(destination).as_posix()]:
            raise ValueError('Checkpoint checksum mismatch: '+str(file))
    if set(destination.glob('*/*.db'))!=set(inputs):
        raise ValueError('Expected exactly the nine released checkpoints.')
    os.environ['R3_HOSFORD_DATA']=str(destination)
    os.environ['FINAL_R3_ROOT']=str(R3)
    sys.path[:0]=[str(R3/'code'),str(R3/'reviews/independent_audit'),str(FINAL/'code')]
    engine=importlib.import_module('rank_one_control')
    engine.DATA=destination;engine.SOURCE=destination/portable.AUTHOR_SOURCE
    engine.R3=R3
    return destination,engine,sorted(inputs)

def configure_fork():
    if 'fork' not in multiprocessing.get_all_start_methods():
        raise RuntimeError('The frozen multiprocessing implementations require a Unix fork start method. Use the documented Linux environment.')
    multiprocessing.set_start_method('fork',force=True)

def compare_arrays(actual,reference):
    import numpy as np
    checked=[]
    with np.load(actual) as a,np.load(reference) as b:
        for key in b.files:
            if key not in a.files:raise AssertionError('Missing reproduced array '+key)
            exact=np.array_equal(a[key],b[key],equal_nan=True)
            error=float(np.max(np.abs(a[key]-b[key]))) if a[key].size and a[key].dtype.kind in 'fciu' else None
            checked.append(dict(key=key,shape=list(a[key].shape),exact=bool(exact),max_abs_error=error))
            if not exact and not (error is not None and error<1e-11):
                raise AssertionError((str(actual),key,error))
    return checked

def geometry_task(args,out,subset=False):
    import numpy as np
    _,engine,paths=prepare_models(args.model_dir)
    mapping={'direction':('source_geometry_control','source_geometry_v1'),
             'native-classes':('fixed_aggregate_geometry_v2','fixed_aggregate_geometry_v2')}
    module_name,folder=mapping[args.experiment]
    mod=importlib.import_module(module_name);mod.OUT=out
    if subset:
        index=args.model_index;path=paths[index]
        shutil.copyfile(FINAL/'results'/folder/'contexts.npz',out/'contexts.npz')
        result=mod.run(index,path)
        comparisons=compare_arrays(out/(result['model']+'.npz'),FINAL/'results'/folder/(result['model']+'.npz'))
        payload=dict(task='relocated full-condition single-checkpoint reproduction',experiment=args.experiment,
            model=result['model'],checkpoint_sha256=sha(path),reference_directory=str(FINAL/'results'/folder),
            code_sha256=sha(mod.__file__),checks=comparisons,all_saved_reference_arrays_reproduced=True,
            scope='The selected checkpoint uses all original contexts, conditions and seeds. This check does not rerun the other eight checkpoints.')
        (out/'reproduction_check.json').write_text(json.dumps(payload,indent=2)+'\n')
    else:
        sys.argv=[mod.__file__,'--workers',str(args.workers)];mod.main()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('task',choices=['verify','prepare','direction','native-classes','check-simulation','biology','history','sequential','finite','position','rank-one','rank-one-uplift','scalar-clipped','residual-addback','theory','figures'])
    ap.add_argument('--out',type=Path,required=True,help='An empty/new output directory.')
    ap.add_argument('--workers',type=int,choices=[1,2],default=2)
    ap.add_argument('--model-dir',type=Path)
    ap.add_argument('--experiment',choices=['direction','native-classes'],default='native-classes')
    ap.add_argument('--model-index',type=int,choices=range(9),default=0)
    args=ap.parse_args();out=fresh(args.out);configure_fork();tic=time.time()
    if args.task=='verify':
        manifest=PACKAGE/'MANIFEST.json'
        records=[]
        if manifest.is_file():
            for row in json.loads(manifest.read_text())['files']:
                p=PACKAGE/row['path'];actual=sha(p)
                if actual!=row['sha256']:raise AssertionError('Package checksum mismatch: '+str(p))
                records.append(row['path'])
        _,engine,paths=prepare_models(args.model_dir)
        contracts=[dict(model=engine.IndependentNetwork(p).name,checkpoint_sha256=sha(p),checks=engine.independent_one_step_contract(engine.IndependentNetwork(p))) for p in paths]
        array_audit=load('final_saved_array_audit',FINAL/'audit/cognitive/independent_array_audit.py');array_audit.R3=R3;array_audit.HERE=out;array_audit.main()
        result=dict(status='complete',manifest_present=manifest.is_file(),package_files_verified=len(records),native_model_contracts=contracts,
            scope='File hashes, all nine frozen model one-step contracts and independent paper endpoint/finite-closure reaggregation. Full trajectory simulations are separate tasks.')
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    elif args.task=='prepare':
        directory,_,paths=prepare_models(args.model_dir)
        (out/'prepared.json').write_text(json.dumps(dict(model_dir=str(directory),models={str(p.relative_to(directory)):sha(p) for p in paths}),indent=2)+'\n')
    elif args.task in ('direction','native-classes'):
        args.experiment=args.task;geometry_task(args,out)
    elif args.task=='check-simulation':geometry_task(args,out,subset=True)
    elif args.task=='biology':
        subprocess.run([sys.executable,str(FINAL/'audit/cognitive/reproduce_cognitive_audit.py'),'--out',str(out),'--r3-root',str(R3)],check=True)
    elif args.task in ('history','sequential','finite'):
        prepare_models(args.model_dir)
        mapping={'history':('cognitive_causal',['--task','history','--n','96','--seed','83011']),
                 'sequential':('sequential_access',['--n','48','--seed','83012']),
                 'finite':('finite_history_calculus',['--contexts',str(R3/'results/sequential_confirmation_v1/contexts.npz'),'--base-n','16'])}
        name,arguments=mapping[args.task];mod=importlib.import_module(name)
        sys.argv=[mod.__file__,*arguments,'--out',str(out),'--workers',str(args.workers)];mod.main()
    elif args.task=='position':
        import numpy as np,csv
        from concurrent.futures import ProcessPoolExecutor,as_completed
        _,_,paths=prepare_models(args.model_dir)
        mod=load('position_mixture_stress',R3/'scouts/r3_data_and_function/position_mixture_stress.py')
        context=mod.make_contexts();cp=out/'contexts.npz';np.savez_compressed(cp,**context)
        (out/'conditions.json').write_text(json.dumps(mod.CONDITIONS,indent=2))
        manifest=dict(status='running',wrapper='Calls unchanged make_contexts and one_model',n_base=12,seed=83031,
            q5=list(mod.QS),q6=list(mod.QS),code_sha256=sha(mod.__file__),contexts_sha256=sha(cp),
            scope='Crossed fifth/sixth stimulus mixtures; off-training cells have no accuracy target.')
        (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
        results=[]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(mod.one_model,(str(p),str(cp),str(out))) for p in paths]):results.append(future.result())
        results.sort(key=lambda x:x['model']);(out/'summary.json').write_text(json.dumps(results,indent=2))
        rows=[r for x in results for r in x['cell_rows']]
        with (out/'cell_summary.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest.update(status='complete',seconds=time.time()-tic);(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2))
    elif args.task in ('rank-one','rank-one-uplift','scalar-clipped','residual-addback'):
        _,engine,_=prepare_models(args.model_dir)
        names={'rank-one':'rank_one_control','rank-one-uplift':'rank_one_uplift_control','scalar-clipped':'scalar_clipped_control','residual-addback':'residual_addback_confirmation'}
        mod=importlib.import_module(names[args.task])
        if args.task=='rank-one':
            mod.HERE=out;sys.argv=[mod.__file__,'--workers',str(args.workers)]
        else:mod.OUT=out;sys.argv=[mod.__file__]
        mod.main()
    elif args.task=='theory':
        history=FINAL/'audit/history_math'
        mod=load('independent_final_theory',history/'independent_theory_check.py');mod.HERE=out;mod.R2=FINAL.parent/'round2_consciousness';mod.main()
        work=out/'working_tree';mathout=work/'research/final_round/audit/history_math';mathout.mkdir(parents=True)
        shutil.copyfile(history/'independent_extended_theory_check.py',mathout/'independent_extended_theory_check.py')
        r3out=work/'research/round3_consciousness/results/finite_history_calculus_v1';r3out.mkdir(parents=True)
        for p in (R3/'results/finite_history_calculus_v1').glob('*.npz'):shutil.copyfile(p,r3out/p.name)
        subprocess.run([sys.executable,str(mathout/'independent_extended_theory_check.py')],check=True)
        shutil.copyfile(mathout/'independent_extended_theory_check.json',out/'independent_extended_theory_check.json')
    elif args.task=='figures':
        # Import the stored presentation functions, retain their input paths and
        # direct every generated file to the fresh directory.
        sys.path[:0]=[str(R3/'code'),str(FINAL/'code')]
        mod=load('paper_r3_figures',R3/'code/plot_research_figures.py');mod.OUT=out
        mod.carriers();mod.sequential();mod.expression();mod.weak_boundary()
        mod=load('paper_direction_figure',FINAL/'code/plot_source_geometry.py');mod.OUT=out;mod.main()
        # Scripts that plot at import time run from a small, relocated working
        # tree containing their unchanged code and read-only copied JSON inputs.
        work=out/'working_tree';fwork=work/'research/final_round';rwork=work/'research/round3_consciousness'
        bio=fwork/'audit/cognitive';bio.mkdir(parents=True)
        for name in ['plot_final_biology.py','plot_carryover_audit.py','biology_carryover_results.json','biology_carryover_v2_results.json','source_chronology_audit.json']:
            source=FINAL/'audit/cognitive'/name
            if source.is_file():shutil.copyfile(source,bio/name)
        old=rwork/'scouts/r3_data_and_function/results';old.mkdir(parents=True)
        shutil.copyfile(R3/'scouts/r3_data_and_function/results/biology_v0_results.json',old/'biology_v0_results.json')
        (fwork/'figures').mkdir()
        for script in ['plot_final_biology.py','plot_carryover_audit.py']:
            subprocess.run([sys.executable,str(bio/script)],check=True)
        for p in (fwork/'figures').glob('*'):shutil.copyfile(p,out/p.name)
        for extension in ['pdf','svg','png']:
            p=bio/('Figure_S_Carryover_Audit.'+extension)
            shutil.copyfile(p,out/('Figure_1-1_Schedule_Controls.'+extension))
        # Earlier model-based biological decompositions use their stored fitted coefficients.
        scout=rwork/'scouts/r3_data_and_function'
        shutil.copyfile(R3/'scouts/r3_data_and_function/plot_biology.py',scout/'plot_biology.py')
        for name in ['biology_v1_trial_results.json','biology_v2_mechanism_results.json']:
            shutil.copyfile(R3/'scouts/r3_data_and_function/results'/name,old/name)
        subprocess.run([sys.executable,str(scout/'plot_biology.py')],check=True)
        for source,target in [('biology_choice_mechanism','Figure_1-2_Behavioral_Mechanism'),('biology_conditional_history_use','Figure_1-3_Conditional_History')]:
            for extension in ['pdf','svg','png']:shutil.copyfile(old/(source+'.'+extension),out/(target+'.'+extension))
        # The stress-test analyzer writes summaries and figures into its result directory.
        pos=out/'position_data';shutil.copytree(R3/'results/position_mixture_stress',pos)
        module=load('paper_position_figure',R3/'scouts/r3_data_and_function/analyze_position_mixture_stress.py');module.OUT=pos;module.main()
        for source,target in [('position_mixture_expression','Figure_7_Position_and_Evidence'),('position5_state_vs_expression','Figure_7-1_State_Expression_Ratios')]:
            for extension in ['pdf','svg','png']:shutil.copyfile(pos/(source+'.'+extension),out/(target+'.'+extension))
        # Rebuild amplitude/residual comparison from its archived response extrema.
        residual=out/'residual_data';shutil.copytree(R3/'reviews/independent_audit/residual_addback_fresh_v1',residual)
        module=load('paper_residual_figure',R3/'reviews/independent_audit/build_residual_report.py');module.HERE=out;module.RESULTS=residual
        import matplotlib.figure
        original_savefig=matplotlib.figure.Figure.savefig
        def save_with_pdf(figure,path,*aa,**kk):
            original_savefig(figure,path,*aa,**kk)
            if Path(path).suffix=='.svg':original_savefig(figure,out/'Figure_4-1_Source_Magnitude_Control.pdf',*aa,**kk)
        matplotlib.figure.Figure.savefig=save_with_pdf
        try:module.main()
        finally:matplotlib.figure.Figure.savefig=original_savefig
        for extension in ['svg','png']:(out/('residual_addback_figure.'+extension)).rename(out/('Figure_4-1_Source_Magnitude_Control.'+extension))
        # Remaining final plotting files are discovered by explicit names below;
        # all are included in the release together with their numerical sources.
        for name in ['plot_fixed_aggregate.py','plot_saturation.py']:
            file=FINAL/'code'/name
            if file.is_file():
                module=load('paper_native_class_figure',file)
                module.OUT=out
                if hasattr(module,'main'):module.main()
        # Figure renumbering in the final paper changes no graphical content.
        for old,new in [('Figure_4_Retention_and_Expression','Figure_6_Retention_and_Expression'),('Supplement_1_Weak_Stimulus_Boundary','Figure_3-1_Weak_Stimulus_Boundary')]:
            for extension in ['pdf','svg','png']:
                p=out/(old+'.'+extension)
                if p.exists():p.rename(out/(new+'.'+extension))
        (out/'figure_reproduction_scope.json').write_text(json.dumps(dict(status='complete',generated=[p.name for p in out.iterdir() if p.is_file()],
            note='Final mapping and complete original plot-source paths are also listed in FIGURE_REPRODUCTION.md.'),indent=2))
    (out/'launcher_record.json').write_text(json.dumps(dict(task=args.task,started_from_cwd=str(Path.cwd()),launcher_sha256=sha(__file__),seconds=time.time()-tic,final_root=str(FINAL),r3_root=str(R3)),indent=2)+'\n')
    print(json.dumps(dict(status='complete',task=args.task,output=str(out),seconds=time.time()-tic)),flush=True)

if __name__=='__main__':main()
