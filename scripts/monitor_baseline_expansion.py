#!/usr/bin/env python3
"""Monitor baseline training jobs and their dependent sweep pipeline."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SACCT = Path("/opt/gridview/slurm/bin/sacct")
SCONTROL = Path("/opt/gridview/slurm/bin/scontrol")
RUNS_ROOT = Path("/public/home/xinwuye/ai4s-tool-joint-train/runs")
DEFAULT_OUTPUT_DIR = RUNS_ROOT / "baseline_expansion_monitor"
SWEEP_ROOT = RUNS_ROOT / "baseline_expansion_sweep"

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
FAILURE_STATES = TERMINAL_STATES - {"COMPLETED"}
ERROR_PATTERNS = (
    re.compile(r"Traceback \(most recent call last\)"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"ChildFailedError"),
    re.compile(r"NCCL.*(?:error|failure)", re.IGNORECASE),
    re.compile(r"segmentation fault", re.IGNORECASE),
)
CONTROLLER_PATTERN = re.compile(
    r"Submitted baseline expansion sweep controller:\s*(\d+)"
)


@dataclass(frozen=True)
class JobSpec:
    name: str
    job_id: int
    metrics_glob: str
    method: str
    expected_final_step: int
    expected_candidate_count: int | None = None
    expected_selected_count: int | None = None
    selection_includes_invalid_candidates: bool = False


JOB_TEMPLATES = {
    "denovo_dmb": {
        "metrics_glob": str(
            RUNS_ROOT
            / "cpgrpo_denovo"
            / (
                "cpgrpo_denovo_dmb_ng1024_bs1024_lr5e-5_beta5e-3_ni1_"
                "ms2000_stl0_*/metrics.jsonl"
            )
        ),
        "method": "dmb",
        "expected_final_step": 2000,
        "expected_candidate_count": 2048,
        "expected_selected_count": 1024,
    },
    "denovo_entropy": {
        "metrics_glob": str(
            RUNS_ROOT
            / "cpgrpo_denovo"
            / (
                "cpgrpo_denovo_entropy001_ng512_bs1024_lr5e-5_beta5e-3_"
                "ni1_ms2000_stl0_*/metrics.jsonl"
            )
        ),
        "method": "entropy",
        "expected_final_step": 2000,
    },
    "mmgenmol_dmb": {
        "metrics_glob": str(
            RUNS_ROOT
            / "cpgrpo_denovo_pocket_prefix"
            / (
                "cpgrpo_denovo_pocket_prefix_dmb_ng384_bs384_lr5e-5_"
                "beta5e-3_ni1_q03_sa02_unidock05_stl0_*/metrics.jsonl"
            )
        ),
        "method": "dmb",
        "expected_final_step": 1000,
        "expected_candidate_count": 768,
        "expected_selected_count": 384,
    },
    "mmgenmol_entropy": {
        "metrics_glob": str(
            RUNS_ROOT
            / "cpgrpo_denovo_pocket_prefix"
            / (
                "cpgrpo_denovo_pocket_prefix_entropy001_ng192_bs384_"
                "lr5e-5_beta5e-3_ni1_q03_sa02_unidock05_stl0_*/metrics.jsonl"
            )
        ),
        "method": "entropy",
        "expected_final_step": 1000,
    },
    "progen2_dmb": {
        "metrics_glob": str(
            RUNS_ROOT
            / "progen2_sgrpo"
            / "progen2_dmb_grpo_ng192_bs2_len256_rbs16_ms100_slurm{job_id}"
            / "metrics.jsonl"
        ),
        "method": "dmb",
        "expected_final_step": 100,
        "expected_candidate_count": 384,
        "expected_selected_count": 192,
        "selection_includes_invalid_candidates": True,
    },
    "progen2_entropy": {
        "metrics_glob": str(
            RUNS_ROOT
            / "progen2_sgrpo"
            / "progen2_entropy001_grpo_ng96_bs2_len256_rbs16_ms100_slurm{job_id}"
            / "metrics.jsonl"
        ),
        "method": "entropy",
        "expected_final_step": 100,
    },
}


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
    for name in JOB_TEMPLATES:
        parser.add_argument(
            f"--{name.replace('_', '-')}-job-id",
            type=_positive_int,
            required=True,
        )
    parser.add_argument("--launcher-job-id", type=_positive_int, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--poll-seconds", type=_positive_int, default=120)
    parser.add_argument("--stale-seconds", type=_positive_int, default=900)
    parser.add_argument("--first-step-grace-seconds", type=_positive_int, default=1500)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    os.replace(temporary, path)


def _append_event(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{_utc_now()} {message}\n")
        handle.flush()


def _normalize_state(value: str) -> str:
    return value.split()[0].rstrip("+")


def _query_job_io_paths(job_id: int) -> tuple[str, str]:
    result = subprocess.run(
        [str(SCONTROL), "show", "job", "-o", str(job_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "", ""
    stdout_match = re.search(r"(?:^|\s)StdOut=(\S+)", result.stdout)
    stderr_match = re.search(r"(?:^|\s)StdErr=(\S+)", result.stdout)
    stdout = stdout_match.group(1) if stdout_match is not None else ""
    stderr = stderr_match.group(1) if stderr_match is not None else ""
    return stdout, stderr


def _query_jobs(job_ids: list[int]) -> dict[int, dict]:
    result = subprocess.run(
        [
            str(SACCT),
            "-X",
            "-n",
            "-P",
            "-j",
            ",".join(str(job_id) for job_id in job_ids),
            (
                "--format=JobIDRaw,State%32,ExitCode,ElapsedRaw,Start,"
                "StdOut%500,StdErr%500"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = {str(job_id) for job_id in job_ids}
    records: dict[int, dict] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) != 7:
            raise RuntimeError(f"Unexpected sacct row: {line!r}")
        job_id, state, exit_code, elapsed_raw, start, stdout, stderr = (
            field.strip() for field in fields
        )
        if job_id not in expected:
            continue
        if not stdout or not stderr:
            live_stdout, live_stderr = _query_job_io_paths(int(job_id))
            stdout = stdout or live_stdout
            stderr = stderr or live_stderr
        records[int(job_id)] = {
            "state": _normalize_state(state),
            "exit_code": exit_code,
            "elapsed_seconds": int(elapsed_raw or 0),
            "start": start,
            "stdout": stdout,
            "stderr": stderr,
        }
    missing = sorted(set(job_ids) - set(records))
    if missing:
        raise RuntimeError(f"sacct did not return requested jobs: {missing}")
    return records


def _resolve_metrics_path(spec: JobSpec) -> Path | None:
    matches = sorted(Path(path) for path in glob.glob(spec.metrics_glob))
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one metrics file for {spec.name}, found {matches}"
        )
    return matches[0] if matches else None


def _finite_metric(row: dict, key: str, *, context: str) -> float:
    if key not in row:
        raise RuntimeError(f"{context} is missing required metric {key!r}")
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"{context} has non-finite {key}={value}")
    return value


def _read_metrics(
    spec: JobSpec,
    *,
    allow_partial_tail: bool = False,
) -> dict:
    path = _resolve_metrics_path(spec)
    if path is None:
        return {
            "path": None,
            "max_step": 0,
            "completed_row_count": 0,
            "first_ten_verified": False,
            "latest": {},
        }

    rows_by_step: dict[int, dict] = {}
    completed_row_count = 0
    lines = path.read_text(errors="replace").splitlines(keepends=True)
    nonempty_line_numbers = [
        line_number
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]
    last_nonempty_line_number = (
        nonempty_line_numbers[-1] if nonempty_line_numbers else None
    )
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if (
                allow_partial_tail
                and line_number == last_nonempty_line_number
            ):
                break
            raise RuntimeError(
                f"Invalid JSON in {path}:{line_number}: {exc}"
            ) from exc
        step = int(row.get("step", 0))
        if step <= 0:
            raise RuntimeError(f"Invalid step in {path}:{line_number}: {step}")
        completed_row_count += 1
        rows_by_step[step] = row

    for step, row in rows_by_step.items():
        context = f"{spec.name} step {step}"
        _finite_metric(row, "reward_mean", context=context)
        _finite_metric(row, "grad_norm", context=context)
        if "loss" in row:
            _finite_metric(row, "loss", context=context)
        if spec.method == "entropy":
            entropy = _finite_metric(
                row,
                "entropy/normalized_mean",
                context=context,
            )
            if entropy < 0.0:
                raise RuntimeError(f"{context} has negative normalized entropy")
        elif spec.method == "dmb":
            candidate_count = _finite_metric(
                row,
                "diverse_minibatch/candidate_count",
                context=context,
            )
            selected_count = _finite_metric(
                row,
                "diverse_minibatch/selected_count",
                context=context,
            )
            valid_candidate_count = _finite_metric(
                row,
                "diverse_minibatch/valid_candidate_count",
                context=context,
            )
            target_count = _finite_metric(
                row,
                "diverse_minibatch/target_optimization_count",
                context=context,
            )
            shortfall_count = _finite_metric(
                row,
                "diverse_minibatch/shortfall_count",
                context=context,
            )
            if not math.isclose(
                candidate_count,
                float(spec.expected_candidate_count),
                rel_tol=1.0e-6,
                abs_tol=1.0e-4,
            ):
                raise RuntimeError(
                    f"{context} candidate_count={candidate_count}, expected "
                    f"{spec.expected_candidate_count}"
                )
            if not math.isclose(
                target_count,
                float(spec.expected_selected_count),
                rel_tol=1.0e-6,
                abs_tol=1.0e-4,
            ):
                raise RuntimeError(
                    f"{context} target_count={target_count}, expected "
                    f"{spec.expected_selected_count}"
                )
            if (
                valid_candidate_count < 0.0
                or valid_candidate_count > candidate_count + 1.0e-4
            ):
                raise RuntimeError(
                    f"{context} has invalid valid_candidate_count="
                    f"{valid_candidate_count} for candidate_count={candidate_count}"
                )
            if spec.selection_includes_invalid_candidates:
                selected_valid_count = _finite_metric(
                    row,
                    "diverse_minibatch/selected_valid_count",
                    context=context,
                )
                selected_invalid_count = _finite_metric(
                    row,
                    "diverse_minibatch/selected_invalid_count",
                    context=context,
                )
                if not math.isclose(
                    selected_count,
                    target_count,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-4,
                ):
                    raise RuntimeError(
                        f"{context} selected_count={selected_count}, expected fixed "
                        f"target_count={target_count}"
                    )
                if (
                    selected_valid_count < 0.0
                    or selected_valid_count
                    > min(valid_candidate_count, selected_count) + 1.0e-4
                ):
                    raise RuntimeError(
                        f"{context} has invalid selected_valid_count="
                        f"{selected_valid_count}"
                    )
                if (
                    selected_invalid_count < 0.0
                    or selected_invalid_count
                    > candidate_count - valid_candidate_count + 1.0e-4
                    or not math.isclose(
                        selected_invalid_count,
                        selected_count - selected_valid_count,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-4,
                    )
                ):
                    raise RuntimeError(
                        f"{context} has inconsistent selected_invalid_count="
                        f"{selected_invalid_count}"
                    )
                if selected_valid_count <= 0.0:
                    raise RuntimeError(
                        f"{context} has no valid selected candidates and therefore "
                        "no validity-correcting GRPO signal"
                    )
            elif (
                selected_count < 0.0
                or selected_count
                > min(valid_candidate_count, target_count) + 1.0e-4
            ):
                raise RuntimeError(
                    f"{context} has invalid selected_count={selected_count} for "
                    f"valid_candidate_count={valid_candidate_count} and "
                    f"target_count={target_count}"
                )
            expected_shortfall = target_count - selected_count
            if not math.isclose(
                shortfall_count,
                expected_shortfall,
                rel_tol=1.0e-6,
                abs_tol=1.0e-4,
            ):
                raise RuntimeError(
                    f"{context} has inconsistent shortfall_count={shortfall_count}; "
                    f"expected {expected_shortfall}"
                )
        else:
            raise RuntimeError(f"Unknown method {spec.method!r}")

    max_step = max(rows_by_step, default=0)
    return {
        "path": str(path),
        "max_step": max_step,
        "completed_row_count": completed_row_count,
        "first_ten_verified": 1 in rows_by_step and 10 in rows_by_step,
        "latest": rows_by_step.get(max_step, {}),
    }


def _read_tail(path_value: str, max_bytes: int = 2 * 1024 * 1024) -> str:
    if not path_value:
        return ""
    path = Path(path_value)
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def _log_errors(record: dict) -> list[str]:
    text = "\n".join(
        (_read_tail(record["stdout"]), _read_tail(record["stderr"]))
    )
    return [
        pattern.pattern
        for pattern in ERROR_PATTERNS
        if pattern.search(text) is not None
    ]


def _actionable_log_errors(
    record: dict,
    metrics: dict,
    spec: JobSpec,
) -> list[str]:
    output_is_complete = (
        record["state"] == "COMPLETED"
        and record["exit_code"] == "0:0"
        and metrics["max_step"] == spec.expected_final_step
    )
    return [] if output_is_complete else _log_errors(record)


def _build_specs(args: argparse.Namespace) -> list[JobSpec]:
    specs = []
    job_ids = []
    for name, template in JOB_TEMPLATES.items():
        job_id = getattr(args, f"{name}_job_id")
        job_ids.append(job_id)
        specs.append(
            JobSpec(
                name=name,
                job_id=job_id,
                metrics_glob=template["metrics_glob"].format(job_id=job_id),
                method=template["method"],
                expected_final_step=template["expected_final_step"],
                expected_candidate_count=template.get(
                    "expected_candidate_count"
                ),
                expected_selected_count=template.get(
                    "expected_selected_count"
                ),
                selection_includes_invalid_candidates=template.get(
                    "selection_includes_invalid_candidates",
                    False,
                ),
            )
        )
    if len(set(job_ids)) != len(job_ids):
        raise ValueError(f"Training job IDs must be distinct: {job_ids}")
    if args.launcher_job_id in job_ids:
        raise ValueError("Launcher job ID must differ from training job IDs")
    return specs


def _load_runtime(path: Path) -> dict:
    if not path.exists():
        return {"jobs": {}, "alerts": [], "controller_job_id": None}
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Runtime state must be an object: {path}")
    payload.setdefault("jobs", {})
    payload.setdefault("alerts", [])
    payload.setdefault("controller_job_id", None)
    return payload


def _record_alert(runtime: dict, events_path: Path, message: str) -> None:
    if message in runtime["alerts"]:
        return
    runtime["alerts"].append(message)
    _append_event(events_path, f"ALERT {message}")


def _resolve_alerts_with_prefix(
    runtime: dict,
    events_path: Path,
    prefix: str,
) -> None:
    resolved = [
        message
        for message in runtime["alerts"]
        if message.startswith(prefix)
    ]
    if not resolved:
        return
    runtime["alerts"] = [
        message
        for message in runtime["alerts"]
        if not message.startswith(prefix)
    ]
    for message in resolved:
        _append_event(events_path, f"RESOLVED {message}")


def _metrics_progressed(
    *,
    previous_step: int,
    current_step: int,
    previous_completed_row_count: int,
    current_completed_row_count: int,
) -> bool:
    return (
        current_step > previous_step
        or current_completed_row_count > previous_completed_row_count
    )


def _controller_job_id(launcher_record: dict) -> int | None:
    text = _read_tail(launcher_record["stdout"])
    match = CONTROLLER_PATTERN.search(text)
    return int(match.group(1)) if match is not None else None


def _updated_progress_epoch(
    *,
    previous_state: str | None,
    current_state: str,
    previous_step: int,
    current_step: int,
    previous_completed_row_count: int,
    current_completed_row_count: int,
    previous_epoch: float,
    now: float,
) -> float:
    if _metrics_progressed(
        previous_step=previous_step,
        current_step=current_step,
        previous_completed_row_count=previous_completed_row_count,
        current_completed_row_count=current_completed_row_count,
    ):
        return now
    if current_state == "RUNNING" and previous_state != "RUNNING":
        return now
    return previous_epoch


def _poll(
    *,
    specs: list[JobSpec],
    launcher_job_id: int,
    runtime: dict,
    events_path: Path,
    stale_seconds: int,
    first_step_grace_seconds: int,
) -> tuple[dict, bool, bool]:
    now = time.time()
    records = _query_jobs(
        [spec.job_id for spec in specs]
        + [launcher_job_id]
        + (
            [int(runtime["controller_job_id"])]
            if runtime.get("controller_job_id") is not None
            else []
        )
    )
    status_jobs = {}
    training_failed = False

    for spec in specs:
        record = records[spec.job_id]
        metrics = _read_metrics(
            spec,
            allow_partial_tail=record["state"] not in TERMINAL_STATES,
        )
        previous = runtime["jobs"].get(spec.name, {})
        previous_state = previous.get("state")
        previous_step = int(previous.get("max_step", 0))
        previous_completed_row_count = int(
            previous.get("completed_row_count", 0)
        )
        previous_first_ten_verified = bool(
            previous.get("first_ten_verified", False)
        )
        metrics_progressed = _metrics_progressed(
            previous_step=previous_step,
            current_step=metrics["max_step"],
            previous_completed_row_count=previous_completed_row_count,
            current_completed_row_count=metrics["completed_row_count"],
        )
        last_progress_epoch = float(previous.get("last_progress_epoch", now))
        last_progress_epoch = _updated_progress_epoch(
            previous_state=previous_state,
            current_state=record["state"],
            previous_step=previous_step,
            current_step=metrics["max_step"],
            previous_completed_row_count=previous_completed_row_count,
            current_completed_row_count=metrics["completed_row_count"],
            previous_epoch=last_progress_epoch,
            now=now,
        )
        if metrics_progressed:
            _append_event(
                events_path,
                (
                    f"PROGRESS {spec.name} step={metrics['max_step']} "
                    f"rows={metrics['completed_row_count']}"
                ),
            )
            _resolve_alerts_with_prefix(
                runtime,
                events_path,
                f"{spec.name} has not advanced beyond step ",
            )
        if record["state"] != previous_state:
            _append_event(
                events_path,
                f"STATE {spec.name} {previous_state}->{record['state']}",
            )
        if metrics["first_ten_verified"] and not previous_first_ten_verified:
            _append_event(
                events_path,
                f"VERIFIED {spec.name} first_ten_steps",
            )

        errors = _actionable_log_errors(record, metrics, spec)
        if errors:
            _record_alert(
                runtime,
                events_path,
                f"{spec.name} log error signatures: {errors}",
            )
        if record["state"] in FAILURE_STATES:
            training_failed = True
            _record_alert(
                runtime,
                events_path,
                (
                    f"{spec.name} job {spec.job_id} ended in {record['state']} "
                    f"with exit {record['exit_code']}"
                ),
            )
        if record["state"] == "COMPLETED" and record["exit_code"] != "0:0":
            training_failed = True
            _record_alert(
                runtime,
                events_path,
                (
                    f"{spec.name} job {spec.job_id} is COMPLETED but has "
                    f"exit {record['exit_code']}"
                ),
            )
        if (
            record["state"] == "COMPLETED"
            and metrics["max_step"] != spec.expected_final_step
        ):
            training_failed = True
            _record_alert(
                runtime,
                events_path,
                (
                    f"{spec.name} completed with metrics step "
                    f"{metrics['max_step']}, expected {spec.expected_final_step}"
                ),
            )
        if record["state"] == "RUNNING":
            if (
                metrics["max_step"] == 0
                and record["elapsed_seconds"] > first_step_grace_seconds
            ):
                _record_alert(
                    runtime,
                    events_path,
                    (
                        f"{spec.name} exceeded the first-step grace period "
                        "without a completed step"
                    ),
                )
            if (
                metrics["max_step"] > 0
                and now - last_progress_epoch > stale_seconds
            ):
                _record_alert(
                    runtime,
                    events_path,
                    (
                        f"{spec.name} has not advanced beyond step "
                        f"{metrics['max_step']} within the stale-progress threshold"
                    ),
                )

        job_status = {
            **record,
            **metrics,
            "last_progress_epoch": last_progress_epoch,
            "log_error_signatures": errors,
        }
        status_jobs[spec.name] = job_status
        runtime["jobs"][spec.name] = {
            "state": record["state"],
            "max_step": metrics["max_step"],
            "completed_row_count": metrics["completed_row_count"],
            "first_ten_verified": metrics["first_ten_verified"],
            "last_progress_epoch": last_progress_epoch,
        }

    launcher = records[launcher_job_id]
    if launcher["state"] in FAILURE_STATES:
        _record_alert(
            runtime,
            events_path,
            (
                f"launcher job {launcher_job_id} ended in {launcher['state']} "
                f"with exit {launcher['exit_code']}"
            ),
        )
    discovered_controller = _controller_job_id(launcher)
    if discovered_controller is not None:
        existing = runtime.get("controller_job_id")
        if existing not in (None, discovered_controller):
            raise RuntimeError(
                f"Controller job changed from {existing} to {discovered_controller}"
            )
        if existing is None:
            runtime["controller_job_id"] = discovered_controller
            _append_event(
                events_path,
                f"CONTROLLER submitted job={discovered_controller}",
            )

    controller_status = None
    controller_job_id = runtime.get("controller_job_id")
    if controller_job_id is not None:
        if controller_job_id not in records:
            controller_record = _query_jobs([int(controller_job_id)])[
                int(controller_job_id)
            ]
        else:
            controller_record = records[int(controller_job_id)]
        controller_status = controller_record
        if controller_record["state"] in FAILURE_STATES:
            _record_alert(
                runtime,
                events_path,
                (
                    f"sweep controller job {controller_job_id} ended in "
                    f"{controller_record['state']} with exit "
                    f"{controller_record['exit_code']}"
                ),
            )

    sweep_complete = (SWEEP_ROOT / "COMPLETE").is_file()
    status = {
        "updated_at": _utc_now(),
        "jobs": status_jobs,
        "launcher": {"job_id": launcher_job_id, **launcher},
        "controller": (
            None
            if controller_status is None
            else {"job_id": int(controller_job_id), **controller_status}
        ),
        "controller_state_path": str(SWEEP_ROOT / "controller_state.json"),
        "sweep_complete": sweep_complete,
        "alerts": list(runtime["alerts"]),
    }
    pipeline_failed = training_failed or launcher["state"] in FAILURE_STATES
    if controller_status is not None:
        pipeline_failed = (
            pipeline_failed or controller_status["state"] in FAILURE_STATES
        )
    return status, sweep_complete, pipeline_failed


def main() -> None:
    args = _parse_args()
    if not SACCT.is_file():
        raise FileNotFoundError(f"Missing Slurm accounting executable: {SACCT}")
    if not SCONTROL.is_file():
        raise FileNotFoundError(f"Missing Slurm control executable: {SCONTROL}")
    specs = _build_specs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    runtime_path = args.output_dir / "runtime.json"
    events_path = args.output_dir / "events.log"
    attention_path = args.output_dir / "ATTENTION_REQUIRED"
    runtime = _load_runtime(runtime_path)

    _append_event(events_path, "MONITOR started")
    while True:
        try:
            status, sweep_complete, pipeline_failed = _poll(
                specs=specs,
                launcher_job_id=args.launcher_job_id,
                runtime=runtime,
                events_path=events_path,
                stale_seconds=args.stale_seconds,
                first_step_grace_seconds=args.first_step_grace_seconds,
            )
        except Exception as exc:
            _record_alert(
                runtime,
                events_path,
                f"monitor exception: {type(exc).__name__}: {exc}",
            )
            _atomic_json(runtime_path, runtime)
            attention_path.write_text(f"{_utc_now()} {type(exc).__name__}: {exc}\n")
            raise

        _atomic_json(status_path, status)
        _atomic_json(runtime_path, runtime)
        if status["alerts"]:
            attention_path.write_text("\n".join(status["alerts"]) + "\n")
        elif attention_path.exists():
            attention_path.unlink()

        if sweep_complete:
            _append_event(events_path, "MONITOR pipeline complete")
            return
        if pipeline_failed:
            raise RuntimeError(
                f"Baseline expansion pipeline failed; inspect {attention_path}"
            )
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
