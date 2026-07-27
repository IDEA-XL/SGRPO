#!/usr/bin/env python3
"""Resolve scaffold checkpoints and launch an incremental sweep DAG."""

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

SGRPO_CONFIG_STEM = (
    "cpgrpo_denovo_scaffold_sgrpo_ng64_sg8_bs2048_lr5e-5_beta5e-3_"
    "gw09_rewardsum_loo_gc_ms2000_stl0"
)
REUSED_CHECKPOINTS = {
    "grpo": (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / (
            "cpgrpo_denovo_ng512_bs1024_lr5e-5_beta5e-3_ni1_"
            "ms2000_20260422_161812"
        )
        / "checkpoint-002000/model.ckpt"
    ),
    "entropy": (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / (
            "cpgrpo_denovo_entropy001_ng512_bs1024_lr5e-5_beta5e-3_"
            "ni1_ms2000_stl0_20260726_155242"
        )
        / "checkpoint-002000/model.ckpt"
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
    parser.add_argument("--sgrpo-job-id", type=_positive_job_id, required=True)
    return parser.parse_args()


def _require_viable_job(job_id: int) -> tuple[str, str]:
    result = subprocess.run(
        [
            str(SACCT),
            "-X",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    state = None
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        record_job_id, record_state, exit_code = line.split("|", 2)
        if record_job_id == str(job_id):
            state = (record_state.split()[0].rstrip("+"), exit_code)
    if (
        state is None
        or state[0] not in {"PENDING", "RUNNING", "COMPLETED"}
        or (state[0] == "COMPLETED" and state[1] != "0:0")
    ):
        raise RuntimeError(
            f"Scaffold SGRPO training job {job_id} is not viable: {state}"
        )
    return state


def _validate_checkpoint(checkpoint: Path, label: str) -> Path:
    run_dir = checkpoint.parents[1]
    required = (
        checkpoint,
        checkpoint.parent / "trainer_state.json",
        run_dir / "train_results.json",
    )
    missing = [
        path for path in required if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            f"Incomplete final checkpoint for {label}:\n"
            + "\n".join(str(path) for path in missing)
        )
    return checkpoint


def _resolve_sgrpo_checkpoint(job_id: int) -> Path:
    run_dir = (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / f"{SGRPO_CONFIG_STEM}_slurm{job_id}"
    )
    checkpoint = run_dir / "checkpoint-002000/model.ckpt"
    return _validate_checkpoint(checkpoint, f"scaffold SGRPO job {job_id}")


def _expected_sgrpo_checkpoint(job_id: int) -> Path:
    return (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / f"{SGRPO_CONFIG_STEM}_slurm{job_id}"
        / "checkpoint-002000/model.ckpt"
    )


def _submit(script: Path, *options: str) -> str:
    command = [
        str(SBATCH),
        "--parsable",
        "--exclude=server13,server59",
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

    training_state = _require_viable_job(args.sgrpo_job_id)
    checkpoints = {
        name: _validate_checkpoint(path, f"reused {name}")
        for name, path in REUSED_CHECKPOINTS.items()
    }
    checkpoints["sgrpo"] = (
        _resolve_sgrpo_checkpoint(args.sgrpo_job_id)
        if training_state[0] == "COMPLETED"
        else _expected_sgrpo_checkpoint(args.sgrpo_job_id)
    )

    build_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_denovo_scaffold_sweep_specs.py"),
        "--run-root",
        str(SWEEP_ROOT),
        "--allow-pending-checkpoints",
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
        "training_job_ids": {"sgrpo": args.sgrpo_job_id},
        "training_job_states": {"sgrpo": list(training_state)},
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
