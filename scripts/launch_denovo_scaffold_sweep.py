#!/usr/bin/env python3
"""Resolve five completed training jobs and launch the scaffold sweep DAG."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/genmol")
RUNS_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs")
SWEEP_ROOT = RUNS_ROOT / "denovo_scaffold_sweep"
SBATCH = Path("/opt/gridview/slurm/bin/sbatch")
SACCT = Path("/opt/gridview/slurm/bin/sacct")

CONFIG_STEMS = {
    "grpo": (
        "cpgrpo_denovo_scaffold_grpo_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0"
    ),
    "hbd": (
        "cpgrpo_denovo_scaffold_hbd_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_st09_sc04_ms2000_stl0"
    ),
    "sgrpo": (
        "cpgrpo_denovo_scaffold_sgrpo_ng64_sg8_bs2048_lr5e-5_beta5e-3_"
        "gw09_rewardsum_loo_gc_ms2000_stl0"
    ),
    "dmb": (
        "cpgrpo_denovo_scaffold_dmb_ng1024_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0"
    ),
    "entropy": (
        "cpgrpo_denovo_scaffold_entropy001_ng512_bs2048_lr5e-5_beta5e-3_"
        "ni1_gc_ms2000_stl0"
    ),
}


def _positive_job_id(value: str) -> int:
    try:
        job_id = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer job ID, got {value!r}") from exc
    if job_id <= 0:
        raise argparse.ArgumentTypeError(f"Job ID must be positive, got {job_id}")
    return job_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in CONFIG_STEMS:
        parser.add_argument(
            f"--{name}-job-id",
            type=_positive_job_id,
            required=True,
        )
    return parser.parse_args()


def _require_completed_jobs(job_ids: dict[str, int]) -> None:
    if len(set(job_ids.values())) != len(job_ids):
        raise ValueError(f"Training job IDs must be distinct: {job_ids}")
    result = subprocess.run(
        [
            str(SACCT),
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(str(job_id) for job_id in job_ids.values()),
            "--format=JobIDRaw,State,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = {}
    expected = {str(job_id) for job_id in job_ids.values()}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, state, exit_code = line.split("|", 2)
        if job_id in expected:
            states[job_id] = (state.split()[0].rstrip("+"), exit_code)
    failures = {
        name: states.get(str(job_id))
        for name, job_id in job_ids.items()
        if states.get(str(job_id)) != ("COMPLETED", "0:0")
    }
    if failures:
        raise RuntimeError(f"Scaffold training jobs are not all complete: {failures}")


def _resolve_checkpoint(name: str, job_id: int) -> Path:
    run_dir = (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / f"{CONFIG_STEMS[name]}_slurm{job_id}"
    )
    checkpoint = run_dir / "checkpoint-002000/model.ckpt"
    required = (
        checkpoint,
        run_dir / "checkpoint-002000/trainer_state.json",
        run_dir / "train_results.json",
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(
            f"Incomplete final checkpoint for {name} job {job_id}:\n"
            + "\n".join(str(path) for path in missing)
        )
    return checkpoint


def _submit(script: Path, *options: str) -> str:
    command = [
        str(SBATCH),
        "--parsable",
        "--exclude=server13",
        *options,
        str(script),
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse job ID from {result.stdout!r}")
    return job_id


def main() -> None:
    args = _parse_args()
    if not SBATCH.is_file() or not SACCT.is_file():
        raise FileNotFoundError("Required Slurm executables are missing")
    if SWEEP_ROOT.exists():
        raise FileExistsError(
            f"Sweep root already exists; refusing to mix results: {SWEEP_ROOT}"
        )

    job_ids = {
        name: int(getattr(args, f"{name}_job_id"))
        for name in CONFIG_STEMS
    }
    _require_completed_jobs(job_ids)
    checkpoints = {
        name: _resolve_checkpoint(name, job_id)
        for name, job_id in job_ids.items()
    }

    build_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_denovo_scaffold_sweep_specs.py"),
        "--run-root",
        str(SWEEP_ROOT),
    ]
    for name, checkpoint in checkpoints.items():
        build_command.extend([f"--{name}-checkpoint", str(checkpoint)])
    subprocess.run(build_command, cwd=REPO_ROOT, check=True)
    (SWEEP_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    controller_job_id = _submit(
        REPO_ROOT / "scripts/slurm/drive_denovo_scaffold_sweep_cpu.sbatch"
    )
    render_job_id = _submit(
        REPO_ROOT / "scripts/slurm/render_denovo_scaffold_results_cpu.sbatch",
        f"--dependency=afterok:{controller_job_id}",
    )
    manifest = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "training_job_ids": job_ids,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
        "controller_job_id": controller_job_id,
        "render_job_id": render_job_id,
    }
    (SWEEP_ROOT / "pipeline_submission.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"controller_job_id={controller_job_id}")
    print(f"render_job_id={render_job_id}")


if __name__ == "__main__":
    main()
