#!/usr/bin/env python3
"""Monitor Scaffold-SGRPO training and its dependent sweep pipeline."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import monitor_baseline_expansion as baseline_monitor


RUNS_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs")
TRAIN_CONFIG_STEM = (
    "cpgrpo_denovo_scaffold_sgrpo_ng64_sg8_bs2048_lr5e-5_beta5e-3_"
    "gw09_rewardsum_loo_gc_ms2000_stl0"
)
EXPECTED_FINAL_STEP = 2000
SWEEP_ROOT = RUNS_ROOT / "denovo_scaffold_sweep"
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "denovo_scaffold_monitor"
CONTROLLER_PATTERN = re.compile(r"controller_job_id=(\d+)")
RENDER_PATTERN = re.compile(r"render_job_id=(\d+)")
METRICS_READ_ATTEMPTS = 3
METRICS_READ_RETRY_SECONDS = 0.1


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {parsed}")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-job-id", type=_positive_int, required=True)
    parser.add_argument("--launcher-job-id", type=_positive_int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-seconds", type=_positive_int, default=120)
    parser.add_argument("--stale-seconds", type=_positive_int, default=1800)
    parser.add_argument("--first-step-grace-seconds", type=_positive_int, default=1800)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metrics_path(training_job_id: int) -> Path:
    return (
        RUNS_ROOT
        / "cpgrpo_denovo"
        / f"{TRAIN_CONFIG_STEM}_slurm{training_job_id}"
        / "metrics.jsonl"
    )


def _finite(row: dict, key: str, *, context: str) -> float:
    if key not in row:
        raise RuntimeError(f"{context} is missing required metric {key!r}")
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"{context} has non-finite {key}={value}")
    return value


def _read_metrics(
    path: Path,
    *,
    allow_partial_tail: bool,
) -> dict:
    for attempt in range(METRICS_READ_ATTEMPTS):
        try:
            return _read_metrics_snapshot(
                path,
                allow_partial_tail=allow_partial_tail,
            )
        except RuntimeError as exc:
            transient_read = (
                allow_partial_tail
                and str(exc).startswith("Invalid JSON in ")
                and attempt + 1 < METRICS_READ_ATTEMPTS
            )
            if not transient_read:
                raise
            time.sleep(METRICS_READ_RETRY_SECONDS)
    raise AssertionError("Unreachable metrics-read retry state")


def _read_metrics_snapshot(
    path: Path,
    *,
    allow_partial_tail: bool,
) -> dict:
    if not path.is_file():
        return {
            "path": str(path),
            "max_step": 0,
            "first_ten_verified": False,
            "latest": {},
        }
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    nonempty_lines = [
        line_number
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]
    last_nonempty = nonempty_lines[-1] if nonempty_lines else None
    rows_by_step = {}
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if allow_partial_tail and line_number == last_nonempty:
                break
            raise RuntimeError(
                f"Invalid JSON in {path}:{line_number}: {exc}"
            ) from exc
        step = int(row.get("step", 0))
        if step <= 0 or step > EXPECTED_FINAL_STEP:
            raise RuntimeError(f"Invalid step in {path}:{line_number}: {step}")
        context = f"scaffold training step {step}"
        reward = _finite(row, "reward_mean", context=context)
        grad_norm = _finite(row, "grad_norm", context=context)
        scaffold_diversity = _finite(
            row,
            "group_reward/diversity_mean",
            context=context,
        )
        soft_reward = _finite(row, "rewards/soft_mean", context=context)
        valid_fraction = _finite(row, "valid_fraction", context=context)
        if not 0.0 <= reward <= 1.0:
            raise RuntimeError(f"{context} has reward_mean={reward} outside [0, 1]")
        if grad_norm < 0.0:
            raise RuntimeError(f"{context} has negative grad_norm={grad_norm}")
        if not 0.0 <= scaffold_diversity <= 1.0:
            raise RuntimeError(
                f"{context} has scaffold diversity={scaffold_diversity} "
                "outside [0, 1]"
            )
        if not 0.0 <= soft_reward <= 1.0:
            raise RuntimeError(
                f"{context} has rewards/soft_mean={soft_reward} outside [0, 1]"
            )
        if not 0.0 <= valid_fraction <= 1.0:
            raise RuntimeError(
                f"{context} has valid_fraction={valid_fraction} outside [0, 1]"
            )
        rows_by_step[step] = row
    max_step = max(rows_by_step, default=0)
    return {
        "path": str(path),
        "max_step": max_step,
        "first_ten_verified": 1 in rows_by_step and 10 in rows_by_step,
        "latest": rows_by_step.get(max_step, {}),
    }


def _final_checkpoint_complete(training_job_id: int) -> bool:
    run_dir = _metrics_path(training_job_id).parent
    checkpoint_dir = run_dir / "checkpoint-002000"
    required = (
        checkpoint_dir / "model.ckpt",
        checkpoint_dir / "trainer_state.json",
        run_dir / "train_results.json",
    )
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        return False
    try:
        checkpoint_state = json.loads(required[1].read_text())
        train_results = json.loads(required[2].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(checkpoint_state, dict)
        and checkpoint_state.get("global_step") == EXPECTED_FINAL_STEP
        and isinstance(train_results, dict)
        and train_results.get("step") == EXPECTED_FINAL_STEP
    )


def _discover_pipeline_jobs(launcher_record: dict) -> tuple[int | None, int | None]:
    text = baseline_monitor._read_tail(launcher_record["stdout"])
    controller_match = CONTROLLER_PATTERN.search(text)
    render_match = RENDER_PATTERN.search(text)
    controller = (
        int(controller_match.group(1)) if controller_match is not None else None
    )
    render = int(render_match.group(1)) if render_match is not None else None
    if (controller is None) != (render is None):
        raise RuntimeError(
            "Scaffold launcher output contains only one dependent job ID"
        )
    return controller, render


def _load_runtime(path: Path) -> dict:
    if not path.is_file():
        return {
            "last_step": 0,
            "last_progress_epoch": time.time(),
            "controller_job_id": None,
            "render_job_id": None,
        }
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"Runtime state must be a JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _append_event(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _pipeline_outputs_complete() -> bool:
    required = (
        SWEEP_ROOT / "COMPLETE",
        SWEEP_ROOT / "results/figure2-scaffold-diversity.pdf",
        SWEEP_ROOT / "results/scaffold-diversity-results.md",
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _poll(
    *,
    training_job_id: int,
    launcher_job_id: int,
    runtime: dict,
    stale_seconds: int,
    first_step_grace_seconds: int,
    events_path: Path,
) -> tuple[dict, bool, bool]:
    known_ids = [training_job_id, launcher_job_id]
    for key in ("controller_job_id", "render_job_id"):
        if runtime.get(key) is not None:
            known_ids.append(int(runtime[key]))
    records = baseline_monitor._query_jobs(known_ids)
    training = records[training_job_id]
    launcher = records[launcher_job_id]
    metrics = _read_metrics(
        _metrics_path(training_job_id),
        allow_partial_tail=training["state"]
        not in baseline_monitor.TERMINAL_STATES,
    )
    now = time.time()
    active_alerts = []
    failed = False

    previous_step = int(runtime.get("last_step", 0))
    if metrics["max_step"] > previous_step:
        runtime["last_step"] = metrics["max_step"]
        runtime["last_progress_epoch"] = now
        _append_event(
            events_path,
            f"PROGRESS scaffold_sgrpo step={metrics['max_step']}",
        )
    last_progress = float(runtime.get("last_progress_epoch", now))
    if training["state"] == "RUNNING":
        if (
            metrics["max_step"] == 0
            and training["elapsed_seconds"] > first_step_grace_seconds
        ):
            active_alerts.append(
                "Scaffold-SGRPO exceeded first-step grace without a metric"
            )
        if metrics["max_step"] > 0 and now - last_progress > stale_seconds:
            active_alerts.append(
                f"Scaffold-SGRPO is stale at step {metrics['max_step']}"
            )
    if training["state"] in baseline_monitor.FAILURE_STATES:
        active_alerts.append(
            f"Scaffold-SGRPO job {training_job_id} failed: "
            f"{training['state']} exit={training['exit_code']}"
        )
        failed = True
    if training["state"] == "COMPLETED":
        if (
            training["exit_code"] != "0:0"
            or metrics["max_step"] != EXPECTED_FINAL_STEP
            or not _final_checkpoint_complete(training_job_id)
        ):
            active_alerts.append(
                "Scaffold-SGRPO completed without a validated step-2000 checkpoint"
            )
            failed = True

    actionable_errors = (
        []
        if training["state"] == "COMPLETED"
        and training["exit_code"] == "0:0"
        and metrics["max_step"] == EXPECTED_FINAL_STEP
        and _final_checkpoint_complete(training_job_id)
        else baseline_monitor._log_errors(training)
    )
    if actionable_errors:
        active_alerts.append(
            f"Scaffold-SGRPO log error signatures: {actionable_errors}"
        )

    discovered_controller, discovered_render = _discover_pipeline_jobs(launcher)
    if discovered_controller is not None:
        for key, discovered in (
            ("controller_job_id", discovered_controller),
            ("render_job_id", discovered_render),
        ):
            existing = runtime.get(key)
            if existing not in (None, discovered):
                raise RuntimeError(
                    f"{key} changed from {existing} to {discovered}"
                )
            runtime[key] = discovered

    dependent_status = {}
    for label, key in (
        ("controller", "controller_job_id"),
        ("render", "render_job_id"),
    ):
        job_id = runtime.get(key)
        if job_id is None:
            dependent_status[label] = None
            continue
        record = records.get(int(job_id))
        if record is None:
            record = baseline_monitor._query_jobs([int(job_id)])[int(job_id)]
        dependent_status[label] = {"job_id": int(job_id), **record}
        if record["state"] in baseline_monitor.FAILURE_STATES:
            active_alerts.append(
                f"Scaffold {label} job {job_id} failed: "
                f"{record['state']} exit={record['exit_code']}"
            )
            failed = True

    if launcher["state"] in baseline_monitor.FAILURE_STATES:
        active_alerts.append(
            f"Scaffold launcher job {launcher_job_id} failed: "
            f"{launcher['state']} exit={launcher['exit_code']}"
        )
        failed = True
    if (
        launcher["state"] == "COMPLETED"
        and launcher["exit_code"] == "0:0"
        and discovered_controller is None
    ):
        active_alerts.append(
            "Scaffold launcher completed without reporting controller/render job IDs"
        )
        failed = True

    pipeline_complete = (
        dependent_status["render"] is not None
        and dependent_status["render"]["state"] == "COMPLETED"
        and dependent_status["render"]["exit_code"] == "0:0"
        and _pipeline_outputs_complete()
    )
    if (
        dependent_status["render"] is not None
        and dependent_status["render"]["state"] == "COMPLETED"
        and dependent_status["render"]["exit_code"] == "0:0"
        and not pipeline_complete
    ):
        active_alerts.append(
            "Scaffold render completed without all validated output artifacts"
        )
        failed = True
    status = {
        "updated_at": _utc_now(),
        "training": {
            "job_id": training_job_id,
            **training,
            **metrics,
            "final_checkpoint_complete": _final_checkpoint_complete(
                training_job_id
            ),
        },
        "launcher": {"job_id": launcher_job_id, **launcher},
        **dependent_status,
        "pipeline_complete": pipeline_complete,
        "active_alerts": active_alerts,
    }
    return status, pipeline_complete, failed


def main() -> None:
    args = _parse_args()
    if args.training_job_id == args.launcher_job_id:
        raise ValueError("Training and launcher job IDs must differ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    runtime_path = args.output_dir / "runtime.json"
    events_path = args.output_dir / "events.log"
    attention_path = args.output_dir / "ATTENTION_REQUIRED"
    runtime = _load_runtime(runtime_path)
    _append_event(events_path, "MONITOR started")

    while True:
        status, complete, failed = _poll(
            training_job_id=args.training_job_id,
            launcher_job_id=args.launcher_job_id,
            runtime=runtime,
            stale_seconds=args.stale_seconds,
            first_step_grace_seconds=args.first_step_grace_seconds,
            events_path=events_path,
        )
        _atomic_json(status_path, status)
        _atomic_json(runtime_path, runtime)
        if status["active_alerts"]:
            attention_path.write_text("\n".join(status["active_alerts"]) + "\n")
        elif attention_path.exists():
            attention_path.unlink()
        if complete:
            _append_event(events_path, "MONITOR pipeline complete")
            return
        if failed:
            raise RuntimeError(
                f"Scaffold pipeline failed; inspect {attention_path}"
            )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
