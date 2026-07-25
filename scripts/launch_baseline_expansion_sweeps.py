#!/usr/bin/env python3
"""Resolve completed baseline checkpoints and launch their sweep controller."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/genmol")
RUNS_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs")
SWEEP_ROOT = RUNS_ROOT / "baseline_expansion_sweep"
SBATCH = Path("/opt/gridview/slurm/bin/sbatch")
SACCT = Path("/opt/gridview/slurm/bin/sacct")

DENOVO_DMB_CONFIG = (
    "cpgrpo_denovo_dmb_ng1024_bs1024_lr5e-5_beta5e-3_ni1_ms2000_stl0"
)
DENOVO_ENTROPY_CONFIG = (
    "cpgrpo_denovo_entropy001_ng512_bs1024_lr5e-5_beta5e-3_ni1_ms2000_stl0"
)
MMGENMOL_DMB_CONFIG = (
    "cpgrpo_denovo_pocket_prefix_dmb_ng384_bs384_lr5e-5_beta5e-3_ni1_"
    "q03_sa02_unidock05_stl0"
)
MMGENMOL_ENTROPY_CONFIG = (
    "cpgrpo_denovo_pocket_prefix_entropy001_ng192_bs384_lr5e-5_beta5e-3_"
    "ni1_q03_sa02_unidock05_stl0"
)
PROGEN2_DMB_CONFIG = "progen2_dmb_grpo_ng192_bs2_len256_rbs16_ms100"
PROGEN2_ENTROPY_CONFIG = "progen2_entropy001_grpo_ng96_bs2_len256_rbs16_ms100"


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
    parser.add_argument("--denovo-dmb-job-id", type=_positive_job_id, required=True)
    parser.add_argument("--denovo-entropy-job-id", type=_positive_job_id, required=True)
    parser.add_argument("--mmgenmol-dmb-job-id", type=_positive_job_id, required=True)
    parser.add_argument("--mmgenmol-entropy-job-id", type=_positive_job_id, required=True)
    parser.add_argument("--progen2-dmb-job-id", type=_positive_job_id, required=True)
    parser.add_argument("--progen2-entropy-job-id", type=_positive_job_id, required=True)
    return parser.parse_args()


def _resolve_molecule_checkpoint(
    *,
    task_root: Path,
    config_stem: str,
    checkpoint_step: int,
) -> Path:
    relative_checkpoint = Path(
        f"checkpoint-{checkpoint_step:06d}/model.ckpt"
    )
    matches = sorted(
        run_dir / relative_checkpoint
        for run_dir in task_root.glob(f"{config_stem}_*")
        if (run_dir / relative_checkpoint).is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one completed checkpoint for {config_stem}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _resolve_progen2_checkpoint(
    *,
    config_stem: str,
    job_id: int,
    checkpoint_step: int,
) -> Path:
    checkpoint = (
        RUNS_ROOT
        / "progen2_sgrpo"
        / f"{config_stem}_slurm{job_id}"
        / f"checkpoint-{checkpoint_step:06d}"
    )
    required = (
        checkpoint / "trainer_state.pt",
        checkpoint / "model.safetensors",
        checkpoint / "config.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Incomplete ProGen2 checkpoint {checkpoint}; missing {missing}"
        )
    return checkpoint


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
    expected_ids = {str(job_id) for job_id in job_ids.values()}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, state, exit_code = line.split("|", 2)
        if job_id in expected_ids:
            states[job_id] = (state.split()[0].rstrip("+"), exit_code)
    failures = {
        name: states.get(str(job_id))
        for name, job_id in job_ids.items()
        if states.get(str(job_id)) != ("COMPLETED", "0:0")
    }
    if failures:
        raise RuntimeError(
            f"Baseline training jobs are not all successfully completed: {failures}"
        )


def main() -> None:
    args = _parse_args()
    if not SBATCH.is_file():
        raise FileNotFoundError(f"Missing Slurm sbatch executable: {SBATCH}")
    if not SACCT.is_file():
        raise FileNotFoundError(f"Missing Slurm sacct executable: {SACCT}")
    if (SWEEP_ROOT / "controller_state.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing sweep controller state: {SWEEP_ROOT}"
        )

    _require_completed_jobs(
        {
            "denovo_dmb": args.denovo_dmb_job_id,
            "denovo_entropy": args.denovo_entropy_job_id,
            "mmgenmol_dmb": args.mmgenmol_dmb_job_id,
            "mmgenmol_entropy": args.mmgenmol_entropy_job_id,
            "progen2_dmb": args.progen2_dmb_job_id,
            "progen2_entropy": args.progen2_entropy_job_id,
        }
    )
    checkpoints = {
        "denovo_dmb": _resolve_molecule_checkpoint(
            task_root=RUNS_ROOT / "cpgrpo_denovo",
            config_stem=DENOVO_DMB_CONFIG,
            checkpoint_step=2000,
        ),
        "denovo_entropy": _resolve_molecule_checkpoint(
            task_root=RUNS_ROOT / "cpgrpo_denovo",
            config_stem=DENOVO_ENTROPY_CONFIG,
            checkpoint_step=2000,
        ),
        "mmgenmol_dmb": _resolve_molecule_checkpoint(
            task_root=RUNS_ROOT / "cpgrpo_denovo_pocket_prefix",
            config_stem=MMGENMOL_DMB_CONFIG,
            checkpoint_step=1000,
        ),
        "mmgenmol_entropy": _resolve_molecule_checkpoint(
            task_root=RUNS_ROOT / "cpgrpo_denovo_pocket_prefix",
            config_stem=MMGENMOL_ENTROPY_CONFIG,
            checkpoint_step=1000,
        ),
        "progen2_dmb": _resolve_progen2_checkpoint(
            config_stem=PROGEN2_DMB_CONFIG,
            job_id=args.progen2_dmb_job_id,
            checkpoint_step=100,
        ),
        "progen2_entropy": _resolve_progen2_checkpoint(
            config_stem=PROGEN2_ENTROPY_CONFIG,
            job_id=args.progen2_entropy_job_id,
            checkpoint_step=100,
        ),
    }

    build_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/build_baseline_expansion_sweep_specs.py"),
        "--run-root",
        str(SWEEP_ROOT),
        "--denovo-dmb-checkpoint",
        str(checkpoints["denovo_dmb"]),
        "--denovo-entropy-checkpoint",
        str(checkpoints["denovo_entropy"]),
        "--mmgenmol-dmb-checkpoint",
        str(checkpoints["mmgenmol_dmb"]),
        "--mmgenmol-entropy-checkpoint",
        str(checkpoints["mmgenmol_entropy"]),
        "--progen2-dmb-checkpoint",
        str(checkpoints["progen2_dmb"]),
        "--progen2-entropy-checkpoint",
        str(checkpoints["progen2_entropy"]),
    ]
    subprocess.run(build_command, cwd=REPO_ROOT, check=True)
    (SWEEP_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    submit_command = [
        str(SBATCH),
        "--parsable",
        "--exclude=server13",
        str(
            REPO_ROOT
            / "scripts/slurm/drive_baseline_expansion_sweeps_cpu.sbatch"
        ),
    ]
    result = subprocess.run(
        submit_command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    controller_job_id = result.stdout.strip().split(";", 1)[0]
    if not controller_job_id.isdigit():
        raise RuntimeError(
            f"Could not parse controller job ID from {result.stdout!r}"
        )
    print("Resolved checkpoints:")
    for name, checkpoint in checkpoints.items():
        print(f"  {name}: {checkpoint}")
    print(f"Submitted baseline expansion sweep controller: {controller_job_id}")


if __name__ == "__main__":
    main()
