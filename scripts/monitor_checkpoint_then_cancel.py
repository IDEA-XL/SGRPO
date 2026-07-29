#!/usr/bin/env python3
"""Cancel a Slurm training job after a complete checkpoint is stable."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESIZING",
    "SUSPENDED",
}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer: {value}")
    return parsed


def _job_state(sacct: Path, job_id: int) -> str:
    result = subprocess.run(
        [
            str(sacct),
            "-X",
            "-j",
            str(job_id),
            "--format=State",
            "-n",
            "-P",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    states = [
        line.strip().split()[0].rstrip("+")
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if not states:
        raise RuntimeError(f"Slurm accounting has no state for job {job_id}")
    return states[0]


def _required_checkpoint_files(
    checkpoint_dir: Path,
    expected_ranks: int,
) -> tuple[Path, ...]:
    accelerator_dir = checkpoint_dir / "accelerator_state"
    files = [
        checkpoint_dir / "model.ckpt",
        checkpoint_dir / "reference_backbone.pt",
        checkpoint_dir / "trainer_state.json",
        accelerator_dir / "latest",
        accelerator_dir / "scheduler.bin",
        accelerator_dir / "zero_to_fp32.py",
        accelerator_dir / "pytorch_model/mp_rank_00_model_states.pt",
    ]
    files.extend(
        accelerator_dir / f"random_states_{rank}.pkl"
        for rank in range(expected_ranks)
    )
    files.extend(
        accelerator_dir
        / "pytorch_model"
        / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        for rank in range(expected_ranks)
    )
    return tuple(files)


def _checkpoint_signature(
    checkpoint_dir: Path,
    *,
    target_step: int,
    expected_ranks: int,
    min_model_bytes: int = 100_000_000,
    min_reference_bytes: int = 100_000_000,
) -> tuple[tuple[str, int, int], ...] | None:
    required = _required_checkpoint_files(checkpoint_dir, expected_ranks)
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None

    trainer_state_path = checkpoint_dir / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text())
    if int(trainer_state.get("global_step", -1)) != target_step:
        raise RuntimeError(
            f"checkpoint trainer step mismatch in {trainer_state_path}: "
            f"{trainer_state.get('global_step')!r} vs {target_step}"
        )

    model_size = (checkpoint_dir / "model.ckpt").stat().st_size
    reference_size = (checkpoint_dir / "reference_backbone.pt").stat().st_size
    if model_size < min_model_bytes or reference_size < min_reference_bytes:
        raise RuntimeError(
            f"checkpoint tensor files are unexpectedly small: "
            f"model={model_size}, reference={reference_size}"
        )

    return tuple(
        (
            str(path.relative_to(checkpoint_dir)),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in required
    )


def _write_receipt(
    path: Path,
    *,
    job_id: int,
    checkpoint_dir: Path,
    target_step: int,
    state_before_cancel: str,
    final_state: str,
    signature: tuple[tuple[str, int, int], ...],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite monitor receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "checkpoint_dir": str(checkpoint_dir),
                "final_state": final_state,
                "job_id": job_id,
                "signature": signature,
                "state_before_cancel": state_before_cancel,
                "target_step": target_step,
                "validated_at_epoch": time.time(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", type=_positive_int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=_positive_int, required=True)
    parser.add_argument("--expected-ranks", type=_positive_int, default=8)
    parser.add_argument("--poll-seconds", type=_positive_int, default=30)
    parser.add_argument("--stable-polls", type=_positive_int, default=2)
    parser.add_argument(
        "--min-model-bytes",
        type=_positive_int,
        default=100_000_000,
    )
    parser.add_argument(
        "--min-reference-bytes",
        type=_positive_int,
        default=100_000_000,
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--slurm-root",
        type=Path,
        default=Path("/opt/gridview/slurm/bin"),
    )
    args = parser.parse_args()

    sacct = args.slurm_root / "sacct"
    scancel = args.slurm_root / "scancel"
    for executable in (sacct, scancel):
        if not executable.is_file():
            raise FileNotFoundError(f"missing Slurm executable: {executable}")

    checkpoint_dir = (
        args.run_dir / f"checkpoint-{args.target_step:06d}"
    )
    previous_signature = None
    stable_count = 0
    while True:
        state = _job_state(sacct, args.job_id)
        signature = _checkpoint_signature(
            checkpoint_dir,
            target_step=args.target_step,
            expected_ranks=args.expected_ranks,
            min_model_bytes=args.min_model_bytes,
            min_reference_bytes=args.min_reference_bytes,
        )
        if signature is not None:
            stable_count = (
                stable_count + 1 if signature == previous_signature else 1
            )
            previous_signature = signature
            if stable_count >= args.stable_polls:
                state_before_cancel = state
                if state in ACTIVE_STATES:
                    subprocess.run(
                        [str(scancel), str(args.job_id)],
                        check=True,
                    )
                    for _ in range(60):
                        state = _job_state(sacct, args.job_id)
                        if state not in ACTIVE_STATES:
                            break
                        time.sleep(2)
                    else:
                        raise RuntimeError(
                            f"job {args.job_id} remained active after cancellation"
                        )
                _write_receipt(
                    args.receipt,
                    job_id=args.job_id,
                    checkpoint_dir=checkpoint_dir,
                    target_step=args.target_step,
                    state_before_cancel=state_before_cancel,
                    final_state=state,
                    signature=signature,
                )
                print(
                    f"validated {checkpoint_dir}; "
                    f"job {args.job_id} final state={state}",
                    flush=True,
                )
                return
        elif state not in ACTIVE_STATES:
            raise RuntimeError(
                f"job {args.job_id} reached {state} before a complete "
                f"checkpoint at step {args.target_step}"
            )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
