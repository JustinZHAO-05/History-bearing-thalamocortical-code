#!/usr/bin/env python3
"""Launch frozen analyses from their preserved Round3 layout, from any cwd.

This adapter resolves input paths and sets the existing R3_HOSFORD_DATA hook.
It changes no equations, random seeds, analysis selections, or frozen scripts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
R3 = HERE.parents[1]
ARCHIVE_SHA256 = "aeb78c3c04c0b69909960e2edb380d55f2a5b44a473495886e26459df9385f2e"
MODEL_FILES = [
    f"TBPTT_bio_Gtune_valid/RateRNNstructNUM{n}.db" for n in (20, 22, 23, 24, 25)
] + [f"TBPTT_bio_sussano_valid/RateRNNstructNUM{n}.db" for n in range(4)]
AUTHOR_SOURCE = "CustomRnn_gtune_dale_relu.py"
AUDIT_SCRIPTS = ["rank_one_control", "rank_one_uplift_control", "scalar_clipped_control",
                 "residual_addback_confirmation", "manuscript_audit_checks"]
BIOLOGY_INPUTS = [
    "Figure 1/PPCSuppEffectsSourceF1.mat",
    "Figure 1/PULSuppEffectsSourceF1.mat",
    "Figure 4/4BtoD/TRNActEffectBehavSourceF4.mat",
]


def extract_verified_models(archive: Path, destination: Path) -> None:
    """Extract nine databases and the source used for provenance checks."""
    if not archive.is_file():
        raise FileNotFoundError(
            f"Place the official model archive at {archive}, or provide --model-archive "
            "or --model-dir. Source: https://zenodo.org/records/19465617"
        )
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError(f"Official model archive SHA256 mismatch: {digest}")
    with zipfile.ZipFile(archive) as source:
        for relative in MODEL_FILES + [AUTHOR_SOURCE]:
            member = "rnn_model/" + relative
            safe = PurePosixPath(relative)
            if safe.is_absolute() or ".." in safe.parts:
                raise ValueError(f"Unsafe archive member: {member}")
            target = destination.joinpath(*safe.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(member))


def audit_smoke_worker(path: str) -> dict:
    """Exercise the unchanged independent loader in an actual forked worker."""
    import rank_one_control as audit
    net = audit.IndependentNetwork(path)
    return {"model": net.name, "model_sha256": audit.sha(path),
            "worker_DATA": str(audit.DATA), "worker_SOURCE": str(audit.SOURCE),
            "author_source_sha256": audit.sha(audit.SOURCE),
            "transition_contract": audit.independent_one_step_contract(net)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["biology", "position_mixture", "audit"])
    parser.add_argument("--model-archive", type=Path,
                        default=R3 / "data/hosford_model_source.zip")
    parser.add_argument("--model-dir", type=Path,
                        default=R3 / "data/hosford_history_rnn")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--audit-script", choices=AUDIT_SCRIPTS, default="rank_one_control")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--audit-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       MKL_NUM_THREADS="1", MPLBACKEND="Agg")

    def run(script: str, *arguments: str) -> None:
        path = HERE / script
        if not path.is_file():
            raise FileNotFoundError(path)
        subprocess.run([sys.executable, str(path), *arguments], env=environment, check=True)

    if args.task == "biology":
        inputs = [HERE / "source_data" / relative for relative in BIOLOGY_INPUTS]
        if not all(path.is_file() for path in inputs):
            run("fetch_biology_sources.py")
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(path)
        scripts = ["analyze_biology.py", "analyze_trials.py", "analyze_mechanism_v2.py",
                   "resolve_review.py", "plot_biology.py"]
        print(json.dumps({"task": args.task, "root": str(R3),
                          "required_input_files": [str(p) for p in inputs]}, indent=2), flush=True)
        if not args.prepare_only:
            for script in scripts:
                run(script)
    else:
        directory = args.model_dir.expanduser().resolve()
        inputs = [directory / relative for relative in MODEL_FILES]
        if not all(path.is_file() for path in inputs):
            extract_verified_models(args.model_archive.expanduser().resolve(), directory)
        expected = set(inputs)
        actual = set(directory.glob("*/*.db"))
        if actual != expected:
            raise ValueError(f"Expected exactly nine released checkpoints in {directory}")
        audit_manifest = HERE / "portable_input_manifest.json"
        if audit_manifest.is_file():
            hashes = json.loads(audit_manifest.read_text())["model_database_sha256"]
            for path in inputs:
                relative = path.relative_to(directory).as_posix()
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != hashes[relative]:
                    raise ValueError(f"Model checksum mismatch: {path}")
        environment["R3_HOSFORD_DATA"] = str(directory)
        print(json.dumps({"task": args.task, "root": str(R3),
                          "R3_HOSFORD_DATA": str(directory), "models": len(inputs)}, indent=2), flush=True)
        if args.task == "audit":
            source = directory / AUTHOR_SOURCE
            if not source.is_file():
                extract_verified_models(args.model_archive.expanduser().resolve(), directory)
            # The unchanged review module has two historical path constants.
            # Rebind these constants before importing dependent review modules.
            # Their Linux fork workers inherit the explicitly configured module.
            os.environ.update(environment)
            import multiprocessing
            if "fork" not in multiprocessing.get_all_start_methods():
                raise RuntimeError("This frozen audit adapter requires Linux/macOS fork workers")
            multiprocessing.set_start_method("fork", force=True)
            sys.path.insert(0, str(R3 / "reviews/independent_audit"))
            audit = importlib.import_module("rank_one_control")
            audit.DATA = directory
            audit.SOURCE = source
            if args.smoke_test:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    records = list(pool.map(audit_smoke_worker, [str(p) for p in inputs]))
                report = {"status": "complete", "workers": args.workers,
                          "start_method": multiprocessing.get_start_method(),
                          "n_models": len(records), "models": records,
                          "scope": "Path/checksum/model-loading/one-step smoke check; no rank experiment rerun"}
                (HERE / "audit_portability_smoke.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
            elif not args.prepare_only:
                module = importlib.import_module(args.audit_script)
                sys.argv = [str(Path(module.__file__))] + args.audit_args
                module.main()
        elif not args.prepare_only:
            run("position_mixture_stress.py", "--workers", str(args.workers))
            run("analyze_position_mixture_stress.py")


if __name__ == "__main__":
    main()
