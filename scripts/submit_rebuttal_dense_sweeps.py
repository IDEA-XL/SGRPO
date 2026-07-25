#!/usr/bin/env python3
"""Submit the complete five-seed dense sweep dependency graph on Pudong."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from build_rebuttal_dense_sweep_specs import (
    DE_NOVO_EXPERIMENTS,
    MMGENMOL_EXPERIMENTS,
    PROGEN2_EXPERIMENTS,
    REPO_REMOTE_ROOT,
    RUN_OUTPUT_ROOT,
    SEEDS,
)


SLURM_ROOT = Path("/opt/gridview/slurm/bin")
SBATCH = SLURM_ROOT / "sbatch"
SPEC_ROOT = RUN_OUTPUT_ROOT / "specs"
LOG_ROOT = RUN_OUTPUT_ROOT / "logs"
SUBMISSION_MANIFEST_PATH = RUN_OUTPUT_ROOT / "submission_manifest.json"

POCKET_MANIFEST_PATH = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix/data/"
    "gdrive_exact_only/crossdocked_pocket10_pocket_prefix_manifest.pt"
)
CROSSDOCKED_ROOT = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/test_set/test_set"
)
DOCKING_CACHE_DIR = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/"
    "pocket_prefix_crossdocked_5500ckpt/docking_cache"
)
DOCKING_CONDA_PREFIX = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/conda_env"
)
QVINA_PATH = REPO_REMOTE_ROOT / "scripts/exps/lead/docking/qvina02"
PROGEN2_ASSET_DIRS = (
    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/python_overlay/openfold",
    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/temberture_official",
    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/prot_bert_bfd",
    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/proteinsol_official",
)
PROGEN2_ASSET_FILES = (
    RUN_OUTPUT_ROOT.parents[0] / "progen2_official/tokenizer.json",
    RUN_OUTPUT_ROOT.parents[0] / "progen2_official/prompts_unconditional.txt",
    RUN_OUTPUT_ROOT.parents[0] / "progen2_models/cuda_toolchain_cu124/lib/libcudart.so.12",
)


def _require_files(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(str(path) for path in missing))


def _require_dirs(paths: Iterable[Path]) -> None:
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Missing required directories:\n" + "\n".join(str(path) for path in missing)
        )


def _clean_submission_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CONFIG_PATH",
        "DOCKING_ROOT",
        "MODE",
        "OUTPUT_DIR",
        "OUTPUT_ROOT",
        "REWARD_NAME",
        "RUN_ROOT",
        "SEED",
        "TASK_ID",
        "TASKS_PATH",
    ):
        environment.pop(name, None)
    return environment


def _atomic_write_manifest(payload: dict) -> None:
    temporary_path = SUBMISSION_MANIFEST_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(SUBMISSION_MANIFEST_PATH)


def _submit(
    manifest: dict,
    name: str,
    *,
    script: Path | None = None,
    options: Iterable[str] = (),
    wrap: str | None = None,
) -> str:
    if (script is None) == (wrap is None):
        raise ValueError("Exactly one of script and wrap must be set")
    command = [str(SBATCH), "--parsable", "--exclude=server13", *options]
    if script is not None:
        command.append(str(script))
    else:
        command.extend(["--wrap", str(wrap)])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=_clean_submission_environment(),
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse Slurm job ID from: {result.stdout!r}")
    manifest["jobs"][name] = {
        "job_id": job_id,
        "command": command,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_manifest(manifest)
    print(f"{name}={job_id}", flush=True)
    return job_id


def _export_options(values: dict[str, object]) -> str:
    for key, value in values.items():
        if "," in str(value):
            raise ValueError(f"Slurm export value for {key} contains a comma: {value}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in values.items())


def _preflight() -> str:
    if RUN_OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Run root already exists; refusing to mix or overwrite results: {RUN_OUTPUT_ROOT}"
        )
    if not REPO_REMOTE_ROOT.joinpath(".git").is_dir():
        raise FileNotFoundError(f"Missing Git worktree: {REPO_REMOTE_ROOT}")
    if not SBATCH.is_file() or not os.access(SBATCH, os.X_OK):
        raise FileNotFoundError(f"Missing executable sbatch: {SBATCH}")
    checkpoint_files = [
        *(Path(item["checkpoint_path"]) for item in DE_NOVO_EXPERIMENTS),
        *(Path(item["checkpoint_path"]) for item in MMGENMOL_EXPERIMENTS),
    ]
    checkpoint_dirs = [Path(item["checkpoint_dir"]) for item in PROGEN2_EXPERIMENTS]
    _require_files(
        [
            *checkpoint_files,
            POCKET_MANIFEST_PATH,
            QVINA_PATH,
            *PROGEN2_ASSET_FILES,
        ]
    )
    _require_dirs(
        [
            *checkpoint_dirs,
            CROSSDOCKED_ROOT,
            DOCKING_CACHE_DIR / "vina_dock",
            DOCKING_CONDA_PREFIX,
            *PROGEN2_ASSET_DIRS,
        ]
    )
    receptor_pqr_count = len(list((DOCKING_CACHE_DIR / "vina_dock").rglob("*.pqr")))
    receptor_pdbqt_count = len(list((DOCKING_CACHE_DIR / "vina_dock").rglob("*.pdbqt")))
    if receptor_pqr_count != 100 or receptor_pdbqt_count != 100:
        raise RuntimeError(
            "Expected a complete 100-pocket Vina receptor cache, found "
            f"{receptor_pqr_count} PQR and {receptor_pdbqt_count} PDBQT files"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_REMOTE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> None:
    git_commit = _preflight()
    LOG_ROOT.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO_REMOTE_ROOT / "scripts/build_rebuttal_dense_sweep_specs.py"),
            "--spec-root",
            str(SPEC_ROOT),
        ],
        cwd=REPO_REMOTE_ROOT,
        check=True,
    )
    spec_manifest = json.loads((SPEC_ROOT / "manifest.json").read_text())
    expected_counts = {
        "denovo_tasks": 60,
        "mmgenmol_tasks_per_seed": 30,
        "progen2_tasks_per_seed": 92,
    }
    for key, expected in expected_counts.items():
        if spec_manifest.get(key) != expected:
            raise RuntimeError(f"Unexpected {key}: {spec_manifest.get(key)} vs {expected}")

    manifest = {
        "status": "submitting",
        "git_commit": git_commit,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(RUN_OUTPUT_ROOT),
        "spec_manifest": spec_manifest,
        "jobs": {},
    }
    _atomic_write_manifest(manifest)

    try:
        denovo_job = _submit(
            manifest,
            "denovo_array",
            script=REPO_REMOTE_ROOT / "scripts/slurm/rebuttal_dense_denovo_array_1gpu.sbatch",
            options=("--array=0-59",),
        )
        denovo_aggregate_job = _submit(
            manifest,
            "denovo_aggregate",
            script=REPO_REMOTE_ROOT
            / "scripts/slurm/rebuttal_dense_denovo_aggregate_cpu.sbatch",
            options=(f"--dependency=afterok:{denovo_job}",),
        )

        terminal_jobs = [denovo_aggregate_job]
        for seed in SEEDS:
            tasks_path = SPEC_ROOT / f"mmgenmol/seed{seed}.tsv"
            docking_root = RUN_OUTPUT_ROOT / "mmgenmol" / f"seed{seed}" / "docking"
            aggregate_root = RUN_OUTPUT_ROOT / "mmgenmol" / f"seed{seed}" / "aggregate"
            generation_job = _submit(
                manifest,
                f"mmgenmol_seed{seed}_generation",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/rebuttal_dense_mmgenmol_generate_1gpu.sbatch",
                options=(
                    "--array=0-29",
                    f"--export={_export_options({'TASKS_PATH': tasks_path, 'SEED': seed})}",
                ),
            )
            docking_job = _submit(
                manifest,
                f"mmgenmol_seed{seed}_docking",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/dock_mmgenmol_sweep_vina_array_64cpu.sbatch",
                options=(
                    "--array=0-29",
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbmmd{seed}",
                    f"--output={LOG_ROOT}/mm_dock_seed{seed}_%A_%a.out",
                    f"--error={LOG_ROOT}/mm_dock_seed{seed}_%A_%a.err",
                    f"--export={_export_options({'TASKS_PATH': tasks_path, 'OUTPUT_ROOT': docking_root, 'DOCKING_CACHE_DIR': DOCKING_CACHE_DIR})}",
                ),
            )
            aggregate_job = _submit(
                manifest,
                f"mmgenmol_seed{seed}_aggregate",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/rebuttal_dense_mmgenmol_aggregate_cpu.sbatch",
                options=(
                    f"--dependency=afterok:{docking_job}",
                    f"--job-name=rbmma{seed}",
                    f"--export={_export_options({'TASKS_PATH': tasks_path, 'DOCKING_ROOT': docking_root, 'OUTPUT_DIR': aggregate_root})}",
                ),
            )
            terminal_jobs.append(aggregate_job)

        for seed in SEEDS:
            config_path = SPEC_ROOT / f"progen2/seed{seed}.yaml"
            common_gpu_script = REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_gpu.sbatch"
            generation_job = _submit(
                manifest,
                f"progen2_seed{seed}_generation",
                script=common_gpu_script,
                options=(
                    "--array=0-91",
                    f"--job-name=rbp2g{seed}",
                    f"--output={LOG_ROOT}/p2_generate_seed{seed}_%A_%a.out",
                    f"--error={LOG_ROOT}/p2_generate_seed{seed}_%A_%a.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path, 'MODE': 'generate-task'})}",
                ),
            )
            naturalness_job = _submit(
                manifest,
                f"progen2_seed{seed}_naturalness",
                script=common_gpu_script,
                options=(
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbp2n{seed}",
                    f"--output={LOG_ROOT}/p2_naturalness_seed{seed}_%j.out",
                    f"--error={LOG_ROOT}/p2_naturalness_seed{seed}_%j.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path, 'MODE': 'score-packed-gpu-reward', 'REWARD_NAME': 'naturalness'})}",
                ),
            )
            stability_job = _submit(
                manifest,
                f"progen2_seed{seed}_stability",
                script=common_gpu_script,
                options=(
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbp2s{seed}",
                    f"--output={LOG_ROOT}/p2_stability_seed{seed}_%j.out",
                    f"--error={LOG_ROOT}/p2_stability_seed{seed}_%j.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path, 'MODE': 'score-packed-gpu-reward', 'REWARD_NAME': 'stability'})}",
                ),
            )
            foldability_job = _submit(
                manifest,
                f"progen2_seed{seed}_foldability",
                script=common_gpu_script,
                options=(
                    "--array=0-91",
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbp2f{seed}",
                    f"--output={LOG_ROOT}/p2_foldability_seed{seed}_%A_%a.out",
                    f"--error={LOG_ROOT}/p2_foldability_seed{seed}_%A_%a.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path, 'MODE': 'score-point-reward-task', 'REWARD_NAME': 'foldability'})}",
                ),
            )
            developability_job = _submit(
                manifest,
                f"progen2_seed{seed}_developability",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/run_progen2_sweep_developability_cpu.sbatch",
                options=(
                    "--array=0-91",
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbp2d{seed}",
                    f"--output={LOG_ROOT}/p2_developability_seed{seed}_%A_%a.out",
                    f"--error={LOG_ROOT}/p2_developability_seed{seed}_%A_%a.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path})}",
                ),
            )
            diversity_job = _submit(
                manifest,
                f"progen2_seed{seed}_diversity",
                script=REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_diversity_cpu.sbatch",
                options=(
                    "--array=0-91",
                    f"--dependency=afterok:{generation_job}",
                    f"--job-name=rbp2v{seed}",
                    f"--output={LOG_ROOT}/p2_diversity_seed{seed}_%A_%a.out",
                    f"--error={LOG_ROOT}/p2_diversity_seed{seed}_%A_%a.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path})}",
                ),
            )
            score_dependencies = ":".join(
                [
                    naturalness_job,
                    stability_job,
                    foldability_job,
                    developability_job,
                    diversity_job,
                ]
            )
            aggregate_job = _submit(
                manifest,
                f"progen2_seed{seed}_aggregate",
                script=REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_aggregate_cpu.sbatch",
                options=(
                    f"--dependency=afterok:{score_dependencies}",
                    f"--job-name=rbp2a{seed}",
                    f"--output={LOG_ROOT}/p2_aggregate_seed{seed}_%j.out",
                    f"--error={LOG_ROOT}/p2_aggregate_seed{seed}_%j.err",
                    f"--export={_export_options({'CONFIG_PATH': config_path})}",
                ),
            )
            terminal_jobs.append(aggregate_job)

        final_dependency = ":".join(terminal_jobs)
        completion_job = _submit(
            manifest,
            "all_complete",
            options=(
                "--partition=cpu",
                "--nodes=1",
                "--cpus-per-task=1",
                "--mem=1G",
                "--time=00:10:00",
                "--job-name=rbdone",
                f"--dependency=afterok:{final_dependency}",
                f"--output={LOG_ROOT}/all_complete_%j.out",
                f"--error={LOG_ROOT}/all_complete_%j.err",
            ),
            wrap=f"date -Is > {RUN_OUTPUT_ROOT}/COMPLETE",
        )
        manifest["status"] = "submitted"
        manifest["completion_job_id"] = completion_job
        manifest["finished_submitting_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_manifest(manifest)
    except Exception as exc:
        manifest["status"] = "submission_failed"
        manifest["submission_error"] = repr(exc)
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_manifest(manifest)
        raise


if __name__ == "__main__":
    main()
