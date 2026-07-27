#!/usr/bin/env python3
"""Drive motif-extension training and sweeps within the Pudong GPU-job cap."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/genmol")
WORKSPACE_ROOT = REPO_ROOT.parent
RUNS_ROOT = WORKSPACE_ROOT / "runs"
TRAIN_ROOT = RUNS_ROOT / "cpgrpo_motif_extension"
PIPELINE_ROOT = RUNS_ROOT / "motif_extension"
SWEEP_ROOT = PIPELINE_ROOT / "sweep_5seed"
STATE_PATH = PIPELINE_ROOT / "pipeline_state.json"
LOCK_PATH = PIPELINE_ROOT / "pipeline.lock"
COMPLETE_PATH = SWEEP_ROOT / "COMPLETE"
EXPANDED_RESULTS_PATH = (
    WORKSPACE_ROOT / "nips26/rebuttal/expanded-sweep-results.md"
)

SLURM_ROOT = Path("/opt/gridview/slurm/bin")
SBATCH = SLURM_ROOT / "sbatch"
SQUEUE = SLURM_ROOT / "squeue"
SACCT = SLURM_ROOT / "sacct"
TRAIN_SCRIPT = REPO_ROOT / "scripts/slurm/motif_extension_8gpu_train.sbatch"
SWEEP_SCRIPT = REPO_ROOT / "scripts/slurm/motif_extension_sweep_1gpu.sbatch"

SMOKE_STEM = "cpgrpo_motif_extension_sgrpo_smoke2"
SMOKE_FINAL_STEP = 2
TRAIN_FINAL_STEP = 2000
SEEDS = (42, 43, 44, 45, 46)
FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
ACTIVE_STATES = {
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "RUNNING",
    "RESIZING",
    "SUSPENDED",
}
ACCOUNTING_GRACE_SECONDS = 300


@dataclass(frozen=True)
class TrainingSpec:
    key: str
    job_name: str
    config_path: Path

    @property
    def config_stem(self) -> str:
        return self.config_path.stem


TRAINING_SPECS = (
    TrainingSpec(
        "grpo",
        "motif_grpo",
        REPO_ROOT
        / "configs/"
        "cpgrpo_motif_extension_grpo_ng512_bs1024_lr5e-5_"
        "beta5e-3_ms2000_stl0.yaml",
    ),
    TrainingSpec(
        "dmb",
        "motif_dmb",
        REPO_ROOT
        / "configs/"
        "cpgrpo_motif_extension_dmb_ng1024_bs1024_lr5e-5_"
        "beta5e-3_ms2000_stl0.yaml",
    ),
    TrainingSpec(
        "entropy",
        "motif_entropy",
        REPO_ROOT
        / "configs/"
        "cpgrpo_motif_extension_entropy001_ng512_bs1024_lr5e-5_"
        "beta5e-3_ms2000_stl0.yaml",
    ),
    TrainingSpec(
        "sgrpo",
        "motif_sgrpo",
        REPO_ROOT
        / "configs/"
        "cpgrpo_motif_extension_sgrpo_ng64_sg8_bs1024_lr5e-5_"
        "beta5e-3_gw09_rewardsum_loo_ms2000_stl0.yaml",
    ),
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer: {value}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-job-id", type=_positive_int, required=True)
    parser.add_argument("--gpu-submit-limit", type=_positive_int, default=40)
    parser.add_argument("--poll-seconds", type=_positive_int, default=10)
    parser.add_argument(
        "--first-step-grace-seconds",
        type=_positive_int,
        default=3600,
    )
    parser.add_argument(
        "--stale-seconds",
        type=_positive_int,
        default=1800,
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    *,
    cwd: Path = REPO_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _normalize_state(value: str) -> str:
    return value.split()[0].rstrip("+")


def _job_record(job_id: int) -> dict:
    result = _run(
        [
            str(SACCT),
            "-X",
            "-n",
            "-P",
            "-j",
            str(job_id),
            "--format=JobIDRaw,State,ExitCode,ElapsedRaw",
        ]
    )
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        raw_job_id, raw_state, exit_code, elapsed = line.split("|", 3)
        if raw_job_id == str(job_id):
            return {
                "state": _normalize_state(raw_state),
                "exit_code": exit_code,
                "elapsed_seconds": int(elapsed or 0),
            }
    return {
        "state": "UNKNOWN",
        "exit_code": "",
        "elapsed_seconds": 0,
    }


def _gpu_submitted_count() -> int:
    result = _run(
        [
            str(SQUEUE),
            "-r",
            "-h",
            "-u",
            os.environ["USER"],
            "-o",
            "%P",
        ]
    )
    return sum(
        line.strip() == "gpu"
        for line in result.stdout.splitlines()
        if line.strip()
    )


def _submit(options: list[str], script: Path) -> int:
    result = _run(
        [
            str(SBATCH),
            "--parsable",
            "--exclude=server13,server46,server59",
            *options,
            str(script),
        ]
    )
    raw_job_id = result.stdout.strip().split(";", 1)[0]
    if not raw_job_id.isdigit():
        raise RuntimeError(
            f"Could not parse Slurm job ID from {result.stdout!r}"
        )
    return int(raw_job_id)


def _read_jsonl_metrics(path: Path, final_step: int) -> dict:
    if not path.is_file():
        return {"max_step": 0, "latest": {}}
    rows_by_step = {}
    lines = path.read_text(errors="replace").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                break
            raise
        step = int(row.get("step", 0))
        if step <= 0 or step > final_step:
            raise RuntimeError(f"Invalid training step {step} in {path}")
        required = (
            "reward_mean",
            "grad_norm",
            "valid_fraction",
            "motif_extension/candidate_valid_fraction",
            "motif_extension/candidate_retention_fraction",
        )
        for key in required:
            value = float(row[key])
            if not math.isfinite(value):
                raise RuntimeError(
                    f"Non-finite metric {key}={value} at step {step}"
                )
        for key in (
            "valid_fraction",
            "motif_extension/candidate_valid_fraction",
            "motif_extension/candidate_retention_fraction",
        ):
            value = float(row[key])
            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Metric {key}={value} is outside [0, 1] at step {step}"
                )
        rows_by_step[step] = row
    max_step = max(rows_by_step, default=0)
    return {
        "max_step": max_step,
        "latest": rows_by_step.get(max_step, {}),
    }


def _validate_checkpoint(run_dir: Path, final_step: int) -> Path:
    checkpoint_dir = run_dir / f"checkpoint-{final_step:06d}"
    model_path = checkpoint_dir / "model.ckpt"
    checkpoint_state_path = checkpoint_dir / "trainer_state.json"
    train_results_path = run_dir / "train_results.json"
    required = (model_path, checkpoint_state_path, train_results_path)
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(
            "Incomplete final checkpoint:\n" + "\n".join(missing)
        )
    checkpoint_state = json.loads(checkpoint_state_path.read_text())
    train_results = json.loads(train_results_path.read_text())
    if checkpoint_state.get("global_step") != final_step:
        raise RuntimeError(
            f"Checkpoint state is not at step {final_step}: "
            f"{checkpoint_state_path}"
        )
    if train_results.get("step") != final_step:
        raise RuntimeError(
            f"Train results are not at step {final_step}: "
            f"{train_results_path}"
        )
    return model_path


def _smoke_run_dir(smoke_job_id: int) -> Path:
    return TRAIN_ROOT / f"{SMOKE_STEM}_slurm{smoke_job_id}"


def _formal_run_dir(spec: TrainingSpec, job_id: int) -> Path:
    return TRAIN_ROOT / f"{spec.config_stem}_slurm{job_id}"


def _validate_smoke(smoke_job_id: int) -> None:
    run_dir = _smoke_run_dir(smoke_job_id)
    metrics = _read_jsonl_metrics(
        run_dir / "metrics.jsonl",
        SMOKE_FINAL_STEP,
    )
    if metrics["max_step"] != SMOKE_FINAL_STEP:
        raise RuntimeError(
            f"Motif smoke ended at step {metrics['max_step']}, "
            f"expected {SMOKE_FINAL_STEP}"
        )
    _validate_checkpoint(run_dir, SMOKE_FINAL_STEP)


def _load_or_create_state(smoke_job_id: int) -> dict:
    expected_configs = {
        spec.key: str(spec.config_path)
        for spec in TRAINING_SPECS
    }
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
        if state.get("smoke_job_id") != smoke_job_id:
            raise RuntimeError(
                "Existing motif pipeline state uses a different smoke job"
            )
        if state.get("training_configs") != expected_configs:
            raise RuntimeError(
                "Existing motif pipeline state uses different training configs"
            )
        state["status"] = "running"
        state.pop("failure", None)
        state.pop("failed_at_epoch", None)
        state.setdefault("training_submitted_at", {})
        _atomic_json(STATE_PATH, state)
        return state
    git_commit = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    state = {
        "schema_version": 1,
        "status": "running",
        "git_commit": git_commit,
        "smoke_job_id": smoke_job_id,
        "training_configs": expected_configs,
        "training_jobs": {},
        "training_submitted_at": {},
        "training_progress": {},
        "sweep_jobs": {},
        "started_at_epoch": time.time(),
    }
    _atomic_json(STATE_PATH, state)
    return state


def _record_training_progress(
    state: dict,
    spec: TrainingSpec,
    job_id: int,
    record: dict,
    *,
    first_step_grace_seconds: int,
    stale_seconds: int,
) -> dict:
    run_dir = _formal_run_dir(spec, job_id)
    metrics = _read_jsonl_metrics(
        run_dir / "metrics.jsonl",
        TRAIN_FINAL_STEP,
    )
    now = time.time()
    progress = state["training_progress"].setdefault(
        spec.key,
        {
            "last_step": 0,
            "last_progress_epoch": now,
        },
    )
    if metrics["max_step"] > int(progress["last_step"]):
        progress["last_step"] = metrics["max_step"]
        progress["last_progress_epoch"] = now
        _atomic_json(STATE_PATH, state)
    if record["state"] == "RUNNING":
        if (
            metrics["max_step"] == 0
            and record["elapsed_seconds"] > first_step_grace_seconds
        ):
            raise RuntimeError(
                f"{spec.key} training has no metric after "
                f"{record['elapsed_seconds']} seconds"
            )
        if (
            metrics["max_step"] > 0
            and now - float(progress["last_progress_epoch"]) > stale_seconds
        ):
            raise RuntimeError(
                f"{spec.key} training is stale at step {metrics['max_step']}"
            )
    return metrics


def _submit_ready_training(state: dict, gpu_submit_limit: int) -> bool:
    for spec in TRAINING_SPECS:
        if spec.key in state["training_jobs"]:
            continue
        if _gpu_submitted_count() >= gpu_submit_limit:
            return False
        job_id = _submit(
            [
                f"--job-name={spec.job_name}",
                f"--export=ALL,CONFIG_PATH={spec.config_path}",
            ],
            TRAIN_SCRIPT,
        )
        state["training_jobs"][spec.key] = job_id
        state["training_submitted_at"][spec.key] = time.time()
        state["training_progress"][spec.key] = {
            "last_step": 0,
            "last_progress_epoch": time.time(),
        }
        _atomic_json(STATE_PATH, state)
        print(
            f"submitted training key={spec.key} job={job_id}",
            flush=True,
        )
        return True
    return False


def _checkpoint_paths(state: dict) -> dict[str, Path]:
    if set(state["training_jobs"]) != {spec.key for spec in TRAINING_SPECS}:
        raise RuntimeError("Cannot resolve checkpoints before all jobs exist")
    paths = {
        "original": (
            REPO_ROOT
            / "checkpoints/genmol_v2_v1.0/model_v2.ckpt"
        )
    }
    for spec in TRAINING_SPECS:
        paths[spec.key] = (
            _formal_run_dir(
                spec,
                int(state["training_jobs"][spec.key]),
            )
            / f"checkpoint-{TRAIN_FINAL_STEP:06d}/model.ckpt"
        )
    return paths


def _ensure_manifest(state: dict) -> dict:
    manifest_path = SWEEP_ROOT / "manifest.json"
    checkpoints = _checkpoint_paths(state)
    if not manifest_path.is_file():
        if SWEEP_ROOT.exists():
            raise RuntimeError(
                f"Sweep root exists without a manifest: {SWEEP_ROOT}"
            )
        command = [
            sys.executable,
            "scripts/build_motif_extension_sweep_specs.py",
            "--run-root",
            str(SWEEP_ROOT),
            "--original-checkpoint",
            str(checkpoints["original"]),
            "--grpo-checkpoint",
            str(checkpoints["grpo"]),
            "--dmb-checkpoint",
            str(checkpoints["dmb"]),
            "--entropy-checkpoint",
            str(checkpoints["entropy"]),
            "--sgrpo-checkpoint",
            str(checkpoints["sgrpo"]),
            "--allow-pending-checkpoints",
        ]
        _run(command)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task_count") != 25:
        raise RuntimeError("Motif sweep manifest must contain 25 tasks")
    expected_checkpoints = {str(path) for path in checkpoints.values()}
    actual_checkpoints = {
        str(task["checkpoint_path"])
        for task in manifest["tasks"]
    }
    if actual_checkpoints != expected_checkpoints:
        raise RuntimeError("Motif sweep manifest checkpoint paths changed")
    state["manifest_path"] = str(manifest_path)
    _atomic_json(STATE_PATH, state)
    return manifest


def _summary_valid(task: dict) -> bool:
    output_dir = Path(task["output_dir"])
    summary_path = output_dir / "summary.json"
    rows_path = output_dir / "rows.jsonl"
    if not summary_path.is_file() or not rows_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        metadata = summary["metadata"]
        results = summary["results"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return (
        metadata.get("experiment") == task["experiment"]
        and int(metadata.get("seed", -1)) == int(task["seed"])
        and int(metadata.get("row_count", -1)) == 10_000
        and int(metadata.get("motif_count", -1)) == 10
        and int(metadata.get("samples_per_motif", -1)) == 100
        and metadata.get("rows_sha256") == _sha256(rows_path)
        and isinstance(results, list)
        and len(results) == 10
    )


def _checkpoint_ready_for_task(task: dict, state: dict) -> bool:
    checkpoint = Path(task["checkpoint_path"])
    if task["experiment"] == "motif_original_genmol_v2":
        return checkpoint.is_file() and checkpoint.stat().st_size > 0
    key_by_experiment = {
        "motif_grpo_2000": "grpo",
        "motif_dmb_2000": "dmb",
        "motif_entropy_2000": "entropy",
        "motif_sgrpo_2000": "sgrpo",
    }
    key = key_by_experiment[task["experiment"]]
    spec = next(item for item in TRAINING_SPECS if item.key == key)
    job_id = int(state["training_jobs"][key])
    try:
        validated = _validate_checkpoint(
            _formal_run_dir(spec, job_id),
            TRAIN_FINAL_STEP,
        )
    except (FileNotFoundError, RuntimeError):
        return False
    return validated == checkpoint


def _refresh_sweep_jobs(state: dict, manifest: dict) -> None:
    tasks_by_index = {
        str(task["task_index"]): task
        for task in manifest["tasks"]
    }
    for task_index, task in tasks_by_index.items():
        if (
            state["sweep_jobs"].get(task_index, {}).get("status")
            == "complete"
        ):
            continue
        if _summary_valid(task):
            state["sweep_jobs"].setdefault(task_index, {})
            state["sweep_jobs"][task_index]["status"] = "complete"
    for task_index, submission in list(state["sweep_jobs"].items()):
        if submission.get("status") == "complete":
            continue
        task = tasks_by_index[task_index]
        record = _job_record(int(submission["job_id"]))
        submission["slurm_state"] = record["state"]
        submission["exit_code"] = record["exit_code"]
        if record["state"] in FAILURE_STATES:
            raise RuntimeError(
                f"Motif sweep task {task_index} failed in job "
                f"{submission['job_id']}: {record}"
            )
        if record["state"] == "COMPLETED":
            raise RuntimeError(
                f"Motif sweep task {task_index} completed without a valid "
                f"summary: {task['output_dir']}"
            )
        if (
            record["state"] == "UNKNOWN"
            and time.time() - float(submission["submitted_at_epoch"])
            > ACCOUNTING_GRACE_SECONDS
        ):
            raise RuntimeError(
                f"Motif sweep task {task_index} is absent from accounting "
                f"after {ACCOUNTING_GRACE_SECONDS} seconds"
            )
    _atomic_json(STATE_PATH, state)


def _submit_ready_sweep(
    state: dict,
    manifest: dict,
    gpu_submit_limit: int,
) -> bool:
    if _gpu_submitted_count() >= gpu_submit_limit:
        return False
    for task in manifest["tasks"]:
        task_index = str(task["task_index"])
        if task_index in state["sweep_jobs"]:
            continue
        output_dir = Path(task["output_dir"])
        if output_dir.exists():
            raise RuntimeError(
                "Incomplete untracked motif sweep output requires diagnosis: "
                f"{output_dir}"
            )
        if not _checkpoint_ready_for_task(task, state):
            continue
        job_id = _submit(
            [
                f"--array={task_index}",
                (
                    "--export=ALL,MANIFEST_PATH="
                    f"{state['manifest_path']}"
                ),
            ],
            SWEEP_SCRIPT,
        )
        state["sweep_jobs"][task_index] = {
            "job_id": job_id,
            "status": "submitted",
            "submitted_at_epoch": time.time(),
        }
        _atomic_json(STATE_PATH, state)
        print(
            f"submitted sweep task={task_index} job={job_id}",
            flush=True,
        )
        return True
    return False


def _all_sweeps_complete(state: dict, manifest: dict) -> bool:
    return all(
        state["sweep_jobs"].get(str(task["task_index"]), {}).get("status")
        == "complete"
        for task in manifest["tasks"]
    )


def _render_results() -> None:
    output_dir = SWEEP_ROOT / "results_summary"
    _run(
        [
            sys.executable,
            "vis-code/render_motif_extension_results.py",
            "--run-root",
            str(SWEEP_ROOT),
            "--output-dir",
            str(output_dir),
            "--expanded-results-path",
            str(EXPANDED_RESULTS_PATH),
        ]
    )
    required = (
        output_dir / "figure2-motif-extension.pdf",
        output_dir / "motif-extension-results.md",
        EXPANDED_RESULTS_PATH,
    )
    missing = [
        str(path)
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Motif renderer omitted required outputs:\n" + "\n".join(missing)
        )


def _training_summary(state: dict) -> str:
    parts = []
    for spec in TRAINING_SPECS:
        job_id = state["training_jobs"].get(spec.key)
        progress = state["training_progress"].get(spec.key, {})
        parts.append(
            f"{spec.key}=job{job_id or '-'}:"
            f"step{progress.get('last_step', 0)}"
        )
    return ",".join(parts)


def main() -> None:
    args = _parse_args()
    for path in (SBATCH, SQUEUE, SACCT, TRAIN_SCRIPT, SWEEP_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(
            "Another motif-extension pipeline controller is running"
        ) from exc

    state = _load_or_create_state(args.smoke_job_id)
    try:
        while True:
            smoke = _job_record(args.smoke_job_id)
            state["smoke_state"] = smoke
            if smoke["state"] in FAILURE_STATES:
                raise RuntimeError(
                    f"Motif smoke job {args.smoke_job_id} failed: {smoke}"
                )
            if smoke["state"] != "COMPLETED":
                if smoke["state"] not in ACTIVE_STATES:
                    raise RuntimeError(
                        f"Unknown motif smoke state: {smoke}"
                    )
                _atomic_json(STATE_PATH, state)
                print(
                    f"waiting smoke={smoke['state']} "
                    f"gpu_submitted={_gpu_submitted_count()}",
                    flush=True,
                )
                time.sleep(args.poll_seconds)
                continue
            if smoke["exit_code"] != "0:0":
                raise RuntimeError(
                    f"Motif smoke exited unsuccessfully: {smoke}"
                )
            _validate_smoke(args.smoke_job_id)
            state["smoke_validated"] = True

            submitted_training = _submit_ready_training(
                state,
                args.gpu_submit_limit,
            )
            training_complete = {}
            for spec in TRAINING_SPECS:
                job_id = state["training_jobs"].get(spec.key)
                if job_id is None:
                    training_complete[spec.key] = False
                    continue
                record = _job_record(int(job_id))
                if record["state"] in FAILURE_STATES:
                    raise RuntimeError(
                        f"{spec.key} training job {job_id} failed: {record}"
                    )
                metrics = _record_training_progress(
                    state,
                    spec,
                    int(job_id),
                    record,
                    first_step_grace_seconds=(
                        args.first_step_grace_seconds
                    ),
                    stale_seconds=args.stale_seconds,
                )
                if record["state"] == "COMPLETED":
                    if record["exit_code"] != "0:0":
                        raise RuntimeError(
                            f"{spec.key} training exited unsuccessfully: "
                            f"{record}"
                        )
                    _validate_checkpoint(
                        _formal_run_dir(spec, int(job_id)),
                        TRAIN_FINAL_STEP,
                    )
                    if metrics["max_step"] != TRAIN_FINAL_STEP:
                        raise RuntimeError(
                            f"{spec.key} metrics ended at "
                            f"{metrics['max_step']}"
                        )
                    training_complete[spec.key] = True
                else:
                    submitted_at = float(
                        state["training_submitted_at"].get(spec.key, 0.0)
                    )
                    accounting_delayed = (
                        record["state"] == "UNKNOWN"
                        and time.time() - submitted_at
                        <= ACCOUNTING_GRACE_SECONDS
                    )
                    if (
                        record["state"] not in ACTIVE_STATES
                        and not accounting_delayed
                    ):
                        raise RuntimeError(
                            f"Unknown {spec.key} job state: {record}"
                        )
                    training_complete[spec.key] = False

            manifest = None
            if len(state["training_jobs"]) == len(TRAINING_SPECS):
                manifest = _ensure_manifest(state)
                _refresh_sweep_jobs(state, manifest)
                _submit_ready_sweep(
                    state,
                    manifest,
                    args.gpu_submit_limit,
                )
                if _all_sweeps_complete(state, manifest):
                    _render_results()
                    COMPLETE_PATH.write_text(
                        time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n"
                    )
                    state["status"] = "complete"
                    state["completed_at_epoch"] = time.time()
                    state["training_complete"] = training_complete
                    _atomic_json(STATE_PATH, state)
                    print("motif pipeline complete", flush=True)
                    return

            state["training_complete"] = training_complete
            _atomic_json(STATE_PATH, state)
            complete_sweeps = (
                0
                if manifest is None
                else sum(
                    state["sweep_jobs"].get(
                        str(task["task_index"]),
                        {},
                    ).get("status")
                    == "complete"
                    for task in manifest["tasks"]
                )
            )
            print(
                f"controller_poll gpu_submitted={_gpu_submitted_count()} "
                f"training={_training_summary(state)} "
                f"sweeps={complete_sweeps}/25 "
                f"submitted_training={submitted_training}",
                flush=True,
            )
            time.sleep(args.poll_seconds)
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = repr(exc)
        state["failed_at_epoch"] = time.time()
        _atomic_json(STATE_PATH, state)
        raise


if __name__ == "__main__":
    main()
