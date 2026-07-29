#!/usr/bin/env python3
"""Drive the dense rebuttal sweep DAG within Pudong's GPU submit-job limit."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import yaml

from build_rebuttal_dense_sweep_specs import (
    REPO_REMOTE_ROOT,
    RUN_OUTPUT_ROOT as DEFAULT_RUN_OUTPUT_ROOT,
    SEEDS,
)


SLURM_ROOT = Path("/opt/gridview/slurm/bin")
SBATCH = SLURM_ROOT / "sbatch"
SQUEUE = SLURM_ROOT / "squeue"
SACCT = SLURM_ROOT / "sacct"
RUN_OUTPUT_ROOT = Path(
    os.environ.get("REBUTTAL_RUN_OUTPUT_ROOT", str(DEFAULT_RUN_OUTPUT_ROOT))
)
SPEC_ROOT = RUN_OUTPUT_ROOT / "specs"
LOG_ROOT = RUN_OUTPUT_ROOT / "logs"
STATE_PATH = RUN_OUTPUT_ROOT / "controller_state.json"
LOCK_PATH = RUN_OUTPUT_ROOT / "controller.lock"
COMPLETE_PATH = RUN_OUTPUT_ROOT / "COMPLETE"
GPU_MAX_SUBMITTED_JOBS = 40
GPU_TASKS_PER_GROUP_PER_ROUND = 4
POLL_SECONDS = 60
COMPLETED_OUTPUT_GRACE_SECONDS = 180
DOMAIN_ENV = "REBUTTAL_DOMAINS"
PROGEN2_PACKED_BY_EXPERIMENT_ENV = "PROGEN2_PACKED_BY_EXPERIMENT"
SUPPORTED_DOMAINS = ("denovo", "mmgenmol", "progen2")

DOCKING_CACHE_DIR = Path(
    "/public/home/xinwuye/ai4s-tool-joint-train/runs/pocket_prefix_eval/"
    "pocket_prefix_crossdocked_5500ckpt/docking_cache"
)


Validator = Callable[[], bool]


def _always_ready() -> bool:
    return True


@dataclass(frozen=True)
class TaskSpec:
    key: str
    group: str
    array_id: int | None
    prerequisites: tuple[str, ...]
    validator: Validator
    readiness_validator: Validator = _always_ready


@dataclass(frozen=True)
class GroupSpec:
    name: str
    resource: str
    script: Path
    job_name: str
    output_pattern: Path
    error_pattern: Path
    time_limit: str | None = None
    exports: tuple[tuple[str, str], ...] = ()


def _json_has_length(path: Path, expected: int, *, field: str | None = None) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if field is not None:
        if not isinstance(value, dict) or field not in value:
            return False
        value = value[field]
    return isinstance(value, list) and len(value) == expected


def _json_is_nonempty_dict(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(value, dict) and bool(value)


def _jsonl_has_rows(path: Path, expected: int) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open() as handle:
            count = sum(1 for line in handle if line.strip())
    except OSError:
        return False
    return count == expected


def _point_reward_output_valid(
    generation_path: Path,
    reward_path: Path,
    reward_name: str,
    *,
    expected_generation_rows: int = 512,
) -> bool:
    if reward_name not in {"foldability", "developability"}:
        raise ValueError(f"Unsupported point reward: {reward_name!r}")
    if not generation_path.is_file() or not reward_path.is_file():
        return False
    try:
        generation_rows = [
            json.loads(line)
            for line in generation_path.read_text().splitlines()
            if line.strip()
        ]
        reward_rows = [
            json.loads(line)
            for line in reward_path.read_text().splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    if len(generation_rows) != expected_generation_rows:
        return False

    generation_indices = set()
    expected_reward_indices = set()
    for row in generation_rows:
        if not isinstance(row, dict):
            return False
        sample_index = row.get("sample_index")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index < 0
            or sample_index in generation_indices
            or not isinstance(row.get("is_valid"), bool)
        ):
            return False
        generation_indices.add(sample_index)
        if row["is_valid"]:
            expected_reward_indices.add(sample_index)

    actual_reward_indices = set()
    for row in reward_rows:
        if not isinstance(row, dict):
            return False
        sample_index = row.get("sample_index")
        reward_value = row.get(reward_name)
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index in actual_reward_indices
            or isinstance(reward_value, bool)
            or not isinstance(reward_value, (int, float))
            or not math.isfinite(float(reward_value))
        ):
            return False
        actual_reward_indices.add(sample_index)
    return actual_reward_indices == expected_reward_indices


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _packed_reward_experiment_path(
    config: dict,
    reward_name: str,
    experiment_name: str,
) -> Path:
    if (
        not experiment_name
        or Path(experiment_name).name != experiment_name
        or experiment_name in {".", ".."}
    ):
        raise ValueError(
            f"Unsafe experiment name for packed reward path: {experiment_name!r}"
        )
    key = f"packed_{reward_name}_scores_path"
    if reward_name not in {"naturalness", "stability"} or key not in config:
        raise ValueError(f"Unsupported packed reward: {reward_name!r}")
    base_path = Path(config[key])
    return (
        base_path.parent
        / "by_experiment"
        / experiment_name
        / base_path.name
    )


def _packed_reward_output_valid(
    output_path: Path,
    reward_name: str,
    experiment_names: tuple[str, ...],
    task_ids: tuple[int, ...],
) -> bool:
    summary_path = Path(f"{output_path}.summary.json")
    if not _nonempty_file(output_path) or not _nonempty_file(summary_path):
        return False
    try:
        summary = json.loads(summary_path.read_text())
        output_bytes = output_path.stat().st_size
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(summary, dict)
        and summary.get("schema_version") == 1
        and summary.get("reward_name") == reward_name
        and summary.get("experiments") == list(experiment_names)
        and summary.get("task_ids") == list(task_ids)
        and isinstance(summary.get("num_output_rows"), int)
        and summary["num_output_rows"] > 0
        and summary.get("output_bytes") == output_bytes
    )


def _molecule_checkpoint_ready(path: Path) -> bool:
    if not _nonempty_file(path):
        return False
    if not path.parent.name.startswith("checkpoint-"):
        return True
    try:
        expected_step = int(path.parent.name.removeprefix("checkpoint-"))
        checkpoint_state = json.loads(
            (path.parent / "trainer_state.json").read_text()
        )
        train_results = json.loads(
            (path.parents[1] / "train_results.json").read_text()
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(checkpoint_state, dict)
        and checkpoint_state.get("global_step") == expected_step
        and isinstance(train_results, dict)
        and train_results.get("step") == expected_step
    )


def _progen2_checkpoint_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not _nonempty_file(path / "config.json"):
        return False
    if not any(
        _nonempty_file(path / name)
        for name in ("model.safetensors", "pytorch_model.bin")
    ):
        return False
    return (
        not path.name.startswith("checkpoint-")
        or _nonempty_file(path / "trainer_state.pt")
    )


def _all_valid(validators: Iterable[Validator]) -> Validator:
    frozen = tuple(validators)
    return lambda: all(validator() for validator in frozen)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError(f"No task rows in {path}")
    return rows


def _export_option(exports: tuple[tuple[str, str], ...]) -> str:
    for key, value in exports:
        if "," in value:
            raise ValueError(f"Slurm export value for {key} contains a comma: {value}")
    return "ALL," + ",".join(f"{key}={value}" for key, value in exports)


def _clean_submission_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CONFIG_PATH",
        "DOCKING_ROOT",
        "EXPERIMENT_NAME",
        "MODE",
        "OUTPUT_DIR",
        "OUTPUT_ROOT",
        "REWARD_NAME",
        "RUN_ROOT",
        "SEED",
        "TASK_ID",
        "TASKS_PATH",
        DOMAIN_ENV,
    ):
        environment.pop(name, None)
    return environment


def _atomic_write_state(state: dict) -> None:
    temporary_path = STATE_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(STATE_PATH)


def _add_group(groups: dict[str, GroupSpec], group: GroupSpec) -> None:
    if group.name in groups:
        raise ValueError(f"Duplicate group: {group.name}")
    if group.resource not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported resource: {group.resource}")
    groups[group.name] = group


def _add_task(tasks: dict[str, TaskSpec], task: TaskSpec) -> None:
    if task.key in tasks:
        raise ValueError(f"Duplicate task: {task.key}")
    tasks[task.key] = task


def _task_domain(task_key: str) -> str:
    if task_key.startswith("denovo:"):
        return "denovo"
    if task_key.startswith("mm:"):
        return "mmgenmol"
    if task_key.startswith("p2:"):
        return "progen2"
    raise ValueError(f"Cannot determine domain for task: {task_key}")


def _active_domains() -> tuple[str, ...]:
    raw_value = os.environ.get(DOMAIN_ENV)
    if raw_value is None:
        return SUPPORTED_DOMAINS
    domains = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    if not domains:
        raise ValueError(f"{DOMAIN_ENV} must contain at least one domain")
    invalid = sorted(set(domains) - set(SUPPORTED_DOMAINS))
    if invalid:
        raise ValueError(
            f"Unsupported {DOMAIN_ENV} values: {invalid}; expected {SUPPORTED_DOMAINS}"
        )
    if len(set(domains)) != len(domains):
        raise ValueError(f"{DOMAIN_ENV} contains duplicate domains: {domains}")
    return domains


def _split_progen2_packed_rewards() -> bool:
    raw_value = os.environ.get(PROGEN2_PACKED_BY_EXPERIMENT_ENV, "0")
    if raw_value not in {"0", "1"}:
        raise ValueError(
            f"{PROGEN2_PACKED_BY_EXPERIMENT_ENV} must be 0 or 1, "
            f"got {raw_value!r}"
        )
    return raw_value == "1"


def _build_dag() -> tuple[dict[str, GroupSpec], dict[str, TaskSpec], tuple[str, ...]]:
    groups: dict[str, GroupSpec] = {}
    tasks: dict[str, TaskSpec] = {}

    denovo_rows = _read_tsv(SPEC_ROOT / "denovo/tasks.tsv")
    denovo_rows_per_seed = {
        seed: [row for row in denovo_rows if int(row["seed"]) == seed]
        for seed in SEEDS
    }
    if any(not rows for rows in denovo_rows_per_seed.values()):
        raise ValueError("Every de novo seed must have at least one experiment")
    denovo_experiment_counts = {
        len(rows) for rows in denovo_rows_per_seed.values()
    }
    if len(denovo_experiment_counts) != 1:
        raise ValueError(
            f"De novo experiment counts differ across seeds: {denovo_experiment_counts}"
        )
    denovo_expected_experiments = next(iter(denovo_experiment_counts))
    _add_group(
        groups,
        GroupSpec(
            name="denovo",
            resource="gpu",
            script=REPO_REMOTE_ROOT / "scripts/slurm/rebuttal_dense_denovo_array_1gpu.sbatch",
            job_name="rbdenovo",
            output_pattern=LOG_ROOT / "denovo_%A_%a.out",
            error_pattern=LOG_ROOT / "denovo_%A_%a.err",
            time_limit="01:00:00",
            exports=(("TASKS_PATH", str(SPEC_ROOT / "denovo/tasks.tsv")),),
        ),
    )
    denovo_keys = []
    denovo_expected_rows = {seed: 0 for seed in SEEDS}
    for row in denovo_rows:
        task_id = int(row["task_id"])
        seed = int(row["seed"])
        config = yaml.safe_load(Path(row["config_path"]).read_text())
        output_path = Path(config["output_json_path"])
        checkpoint_path = Path(config["experiments"][0]["checkpoint_path"])
        expected_points = len(config["randomness_temperature_pairs"])
        if expected_points <= 0:
            raise ValueError(f"Empty de novo sweep in {row['config_path']}")
        denovo_expected_rows[seed] += expected_points
        key = f"denovo:{task_id}"
        denovo_keys.append(key)
        _add_task(
            tasks,
            TaskSpec(
                key=key,
                group="denovo",
                array_id=task_id,
                prerequisites=(),
                validator=lambda path=output_path, expected=expected_points: _json_has_length(
                    path, expected
                ),
                readiness_validator=lambda path=checkpoint_path: _molecule_checkpoint_ready(
                    path
                ),
            ),
        )

    denovo_aggregate_paths = tuple(
        RUN_OUTPUT_ROOT / "denovo" / f"seed{seed}" / "aggregate/denovo_dense.json"
        for seed in SEEDS
    )
    _add_group(
        groups,
        GroupSpec(
            name="denovo_aggregate",
            resource="cpu",
            script=REPO_REMOTE_ROOT
            / "scripts/slurm/rebuttal_dense_denovo_aggregate_cpu.sbatch",
            job_name="rbdenagg",
            output_pattern=LOG_ROOT / "denovo_aggregate_%j.out",
            error_pattern=LOG_ROOT / "denovo_aggregate_%j.err",
            exports=(
                ("RUN_ROOT", str(RUN_OUTPUT_ROOT)),
                ("EXPECTED_EXPERIMENTS", str(denovo_expected_experiments)),
            ),
        ),
    )
    _add_task(
        tasks,
        TaskSpec(
            key="denovo:aggregate",
            group="denovo_aggregate",
            array_id=None,
            prerequisites=tuple(denovo_keys),
            validator=_all_valid(
                lambda path=path, expected=denovo_expected_rows[seed]: _json_has_length(
                    path, expected
                )
                for seed, path in zip(SEEDS, denovo_aggregate_paths)
            ),
        ),
    )

    mm_aggregate_keys = []
    for seed in SEEDS:
        tasks_path = SPEC_ROOT / f"mmgenmol/seed{seed}.tsv"
        rows = _read_tsv(tasks_path)
        generation_group = f"mm_generation_{seed}"
        docking_group = f"mm_docking_{seed}"
        aggregate_group = f"mm_aggregate_{seed}"
        docking_root = RUN_OUTPUT_ROOT / "mmgenmol" / f"seed{seed}" / "docking"
        aggregate_root = RUN_OUTPUT_ROOT / "mmgenmol" / f"seed{seed}" / "aggregate"
        _add_group(
            groups,
            GroupSpec(
                name=generation_group,
                resource="gpu",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/rebuttal_dense_mmgenmol_generate_1gpu.sbatch",
                job_name=f"rbmmg{seed}",
                output_pattern=LOG_ROOT / f"mm_generate_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"mm_generate_seed{seed}_%A_%a.err",
                time_limit="00:15:00",
                exports=(("TASKS_PATH", str(tasks_path)), ("SEED", str(seed))),
            ),
        )
        _add_group(
            groups,
            GroupSpec(
                name=docking_group,
                resource="cpu",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/dock_mmgenmol_sweep_vina_array_64cpu.sbatch",
                job_name=f"rbmmd{seed}",
                output_pattern=LOG_ROOT / f"mm_dock_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"mm_dock_seed{seed}_%A_%a.err",
                exports=(
                    ("TASKS_PATH", str(tasks_path)),
                    ("OUTPUT_ROOT", str(docking_root)),
                    ("DOCKING_CACHE_DIR", str(DOCKING_CACHE_DIR)),
                ),
            ),
        )
        _add_group(
            groups,
            GroupSpec(
                name=aggregate_group,
                resource="cpu",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/rebuttal_dense_mmgenmol_aggregate_cpu.sbatch",
                job_name=f"rbmma{seed}",
                output_pattern=LOG_ROOT / f"mm_aggregate_seed{seed}_%j.out",
                error_pattern=LOG_ROOT / f"mm_aggregate_seed{seed}_%j.err",
                exports=(
                    ("TASKS_PATH", str(tasks_path)),
                    ("DOCKING_ROOT", str(docking_root)),
                    ("OUTPUT_DIR", str(aggregate_root)),
                    ("EXPECTED_NUM_TASKS", str(len(rows))),
                ),
            ),
        )
        docking_keys = []
        for row in rows:
            task_id = int(row["task_id"])
            generation_key = f"mm:{seed}:generation:{task_id}"
            docking_key = f"mm:{seed}:docking:{task_id}"
            generated_path = Path(row["output_path"])
            docking_dir = (
                docking_root
                / row["model_name"]
                / f"{row['sweep_type']}_{row['sweep_value']}"
            )
            records_path = docking_dir / "docking.records.jsonl"
            summary_path = docking_dir / "docking.summary.json"
            docking_keys.append(docking_key)
            _add_task(
                tasks,
                TaskSpec(
                    key=generation_key,
                    group=generation_group,
                    array_id=task_id,
                    prerequisites=(),
                    validator=lambda path=generated_path: _jsonl_has_rows(path, 1600),
                    readiness_validator=lambda path=Path(
                        row["checkpoint_path"]
                    ): _molecule_checkpoint_ready(path),
                ),
            )
            _add_task(
                tasks,
                TaskSpec(
                    key=docking_key,
                    group=docking_group,
                    array_id=task_id,
                    prerequisites=(generation_key,),
                    validator=lambda records=records_path, summary=summary_path: (
                        _jsonl_has_rows(records, 1600) and _json_is_nonempty_dict(summary)
                    ),
                ),
            )
        aggregate_key = f"mm:{seed}:aggregate"
        mm_aggregate_keys.append(aggregate_key)
        _add_task(
            tasks,
            TaskSpec(
                key=aggregate_key,
                group=aggregate_group,
                array_id=None,
                prerequisites=tuple(docking_keys),
                validator=lambda path=aggregate_root
                / "mmgenmol_dense.json", expected=len(rows): _json_has_length(
                    path, expected
                ),
            ),
        )

    progen2_aggregate_keys = []
    split_progen2_packed = _split_progen2_packed_rewards()
    for seed in SEEDS:
        config_path = SPEC_ROOT / f"progen2/seed{seed}.yaml"
        config = yaml.safe_load(config_path.read_text())
        rows = _read_tsv(SPEC_ROOT / f"progen2/seed{seed}_tasks.tsv")
        generation_group = f"progen2_generation_{seed}"
        naturalness_group = f"progen2_naturalness_{seed}"
        stability_group = f"progen2_stability_{seed}"
        foldability_group = f"progen2_foldability_{seed}"
        developability_group = f"progen2_developability_{seed}"
        diversity_group = f"progen2_diversity_{seed}"
        aggregate_group = f"progen2_aggregate_{seed}"
        gpu_script = REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_gpu.sbatch"
        _add_group(
            groups,
            GroupSpec(
                name=generation_group,
                resource="gpu",
                script=gpu_script,
                job_name=f"rbp2g{seed}",
                output_pattern=LOG_ROOT / f"p2_generate_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"p2_generate_seed{seed}_%A_%a.err",
                time_limit="01:00:00",
                exports=(("CONFIG_PATH", str(config_path)), ("MODE", "generate-task")),
            ),
        )
        if not split_progen2_packed:
            _add_group(
                groups,
                GroupSpec(
                    name=naturalness_group,
                    resource="gpu",
                    script=gpu_script,
                    job_name=f"rbp2n{seed}",
                    output_pattern=LOG_ROOT / f"p2_naturalness_seed{seed}_%j.out",
                    error_pattern=LOG_ROOT / f"p2_naturalness_seed{seed}_%j.err",
                    time_limit="01:00:00",
                    exports=(
                        ("CONFIG_PATH", str(config_path)),
                        ("MODE", "score-packed-gpu-reward"),
                        ("REWARD_NAME", "naturalness"),
                    ),
                ),
            )
            _add_group(
                groups,
                GroupSpec(
                    name=stability_group,
                    resource="gpu",
                    script=gpu_script,
                    job_name=f"rbp2s{seed}",
                    output_pattern=LOG_ROOT / f"p2_stability_seed{seed}_%j.out",
                    error_pattern=LOG_ROOT / f"p2_stability_seed{seed}_%j.err",
                    time_limit="01:00:00",
                    exports=(
                        ("CONFIG_PATH", str(config_path)),
                        ("MODE", "score-packed-gpu-reward"),
                        ("REWARD_NAME", "stability"),
                    ),
                ),
            )
        _add_group(
            groups,
            GroupSpec(
                name=foldability_group,
                resource="gpu",
                script=gpu_script,
                job_name=f"rbp2f{seed}",
                output_pattern=LOG_ROOT / f"p2_foldability_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"p2_foldability_seed{seed}_%A_%a.err",
                time_limit="01:00:00",
                exports=(
                    ("CONFIG_PATH", str(config_path)),
                    ("MODE", "score-point-reward-task"),
                    ("REWARD_NAME", "foldability"),
                ),
            ),
        )
        _add_group(
            groups,
            GroupSpec(
                name=developability_group,
                resource="cpu",
                script=REPO_REMOTE_ROOT
                / "scripts/slurm/run_progen2_sweep_developability_cpu.sbatch",
                job_name=f"rbp2d{seed}",
                output_pattern=LOG_ROOT / f"p2_developability_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"p2_developability_seed{seed}_%A_%a.err",
                exports=(("CONFIG_PATH", str(config_path)),),
            ),
        )
        _add_group(
            groups,
            GroupSpec(
                name=diversity_group,
                resource="cpu",
                script=REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_diversity_cpu.sbatch",
                job_name=f"rbp2v{seed}",
                output_pattern=LOG_ROOT / f"p2_diversity_seed{seed}_%A_%a.out",
                error_pattern=LOG_ROOT / f"p2_diversity_seed{seed}_%A_%a.err",
                exports=(("CONFIG_PATH", str(config_path)),),
            ),
        )
        _add_group(
            groups,
            GroupSpec(
                name=aggregate_group,
                resource="cpu",
                script=REPO_REMOTE_ROOT / "scripts/slurm/run_progen2_sweep_aggregate_cpu.sbatch",
                job_name=f"rbp2a{seed}",
                output_pattern=LOG_ROOT / f"p2_aggregate_seed{seed}_%j.out",
                error_pattern=LOG_ROOT / f"p2_aggregate_seed{seed}_%j.err",
                exports=(("CONFIG_PATH", str(config_path)),),
            ),
        )

        generation_keys = []
        generation_keys_by_experiment: dict[str, list[str]] = {}
        foldability_keys = []
        developability_keys = []
        diversity_keys = []
        for row in rows:
            task_id = int(row["task_id"])
            generation_key = f"p2:{seed}:generation:{task_id}"
            foldability_key = f"p2:{seed}:foldability:{task_id}"
            developability_key = f"p2:{seed}:developability:{task_id}"
            diversity_key = f"p2:{seed}:diversity:{task_id}"
            generation_keys.append(generation_key)
            generation_keys_by_experiment.setdefault(
                row["experiment"], []
            ).append(generation_key)
            foldability_keys.append(foldability_key)
            developability_keys.append(developability_key)
            diversity_keys.append(diversity_key)
            _add_task(
                tasks,
                TaskSpec(
                    key=generation_key,
                    group=generation_group,
                    array_id=task_id,
                    prerequisites=(),
                    validator=lambda path=Path(
                        row["generation_rows_path"]
                    ): _jsonl_has_rows(path, 512),
                    readiness_validator=lambda path=Path(
                        row["checkpoint_dir"]
                    ): _progen2_checkpoint_ready(path),
                ),
            )
            _add_task(
                tasks,
                TaskSpec(
                    key=foldability_key,
                    group=foldability_group,
                    array_id=task_id,
                    prerequisites=(generation_key,),
                    validator=lambda generation_path=Path(
                        row["generation_rows_path"]
                    ), reward_path=Path(
                        row["foldability_scores_path"]
                    ): _point_reward_output_valid(
                        generation_path,
                        reward_path,
                        "foldability",
                    ),
                ),
            )
            _add_task(
                tasks,
                TaskSpec(
                    key=developability_key,
                    group=developability_group,
                    array_id=task_id,
                    prerequisites=(generation_key,),
                    validator=lambda generation_path=Path(
                        row["generation_rows_path"]
                    ), reward_path=Path(
                        row["developability_scores_path"]
                    ): _point_reward_output_valid(
                        generation_path,
                        reward_path,
                        "developability",
                    ),
                ),
            )
            _add_task(
                tasks,
                TaskSpec(
                    key=diversity_key,
                    group=diversity_group,
                    array_id=task_id,
                    prerequisites=(generation_key,),
                    validator=lambda path=Path(
                        row["diversity_scores_path"]
                    ): _json_is_nonempty_dict(path),
                ),
            )

        if split_progen2_packed:
            experiment_names = tuple(
                experiment["name"] for experiment in config["experiments"]
            )
            if set(experiment_names) != set(generation_keys_by_experiment):
                raise ValueError(
                    "ProGen2 configured experiments do not match task rows: "
                    f"config={experiment_names} "
                    f"tasks={tuple(generation_keys_by_experiment)}"
                )
            packed_reward_keys = []
            for experiment_index, experiment_name in enumerate(experiment_names):
                experiment_rows = [
                    row for row in rows if row["experiment"] == experiment_name
                ]
                experiment_task_ids = tuple(
                    sorted(int(row["task_id"]) for row in experiment_rows)
                )
                for reward_name, reward_code in (
                    ("naturalness", "n"),
                    ("stability", "s"),
                ):
                    group_name = (
                        f"progen2_{reward_name}_{seed}_{experiment_name}"
                    )
                    output_path = _packed_reward_experiment_path(
                        config,
                        reward_name,
                        experiment_name,
                    )
                    _add_group(
                        groups,
                        GroupSpec(
                            name=group_name,
                            resource="gpu",
                            script=gpu_script,
                            job_name=(
                                f"rbp2{reward_code}{seed}e{experiment_index}"
                            ),
                            output_pattern=LOG_ROOT
                            / (
                                f"p2_{reward_name}_seed{seed}_"
                                f"{experiment_name}_%j.out"
                            ),
                            error_pattern=LOG_ROOT
                            / (
                                f"p2_{reward_name}_seed{seed}_"
                                f"{experiment_name}_%j.err"
                            ),
                            time_limit="01:00:00",
                            exports=(
                                ("CONFIG_PATH", str(config_path)),
                                ("MODE", "score-packed-gpu-reward"),
                                ("REWARD_NAME", reward_name),
                                ("EXPERIMENT_NAME", experiment_name),
                            ),
                        ),
                    )
                    reward_key = (
                        f"p2:{seed}:{reward_name}:{experiment_name}"
                    )
                    packed_reward_keys.append(reward_key)
                    _add_task(
                        tasks,
                        TaskSpec(
                            key=reward_key,
                            group=group_name,
                            array_id=None,
                            prerequisites=tuple(
                                generation_keys_by_experiment[experiment_name]
                            ),
                            validator=lambda path=output_path,
                            reward=reward_name,
                            experiment=experiment_name,
                            ids=experiment_task_ids: _packed_reward_output_valid(
                                path,
                                reward,
                                (experiment,),
                                ids,
                            ),
                        ),
                    )

            packed_merge_group = f"progen2_packed_merge_{seed}"
            _add_group(
                groups,
                GroupSpec(
                    name=packed_merge_group,
                    resource="cpu",
                    script=REPO_REMOTE_ROOT
                    / "scripts/slurm/run_progen2_sweep_merge_cpu.sbatch",
                    job_name=f"rbp2m{seed}",
                    output_pattern=LOG_ROOT
                    / f"p2_packed_merge_seed{seed}_%j.out",
                    error_pattern=LOG_ROOT
                    / f"p2_packed_merge_seed{seed}_%j.err",
                    exports=(("CONFIG_PATH", str(config_path)),),
                ),
            )
            packed_merge_key = f"p2:{seed}:packed_merge"
            all_task_ids = tuple(sorted(int(row["task_id"]) for row in rows))
            _add_task(
                tasks,
                TaskSpec(
                    key=packed_merge_key,
                    group=packed_merge_group,
                    array_id=None,
                    prerequisites=tuple(packed_reward_keys),
                    validator=_all_valid(
                        (
                            lambda reward=reward_name,
                            path=Path(config[f"packed_{reward_name}_scores_path"]),
                            experiments=experiment_names,
                            ids=all_task_ids: _packed_reward_output_valid(
                                path,
                                reward,
                                experiments,
                                ids,
                            )
                        )
                        for reward_name in ("naturalness", "stability")
                    ),
                ),
            )
            packed_prerequisites = (packed_merge_key,)
        else:
            naturalness_key = f"p2:{seed}:naturalness"
            stability_key = f"p2:{seed}:stability"
            _add_task(
                tasks,
                TaskSpec(
                    key=naturalness_key,
                    group=naturalness_group,
                    array_id=None,
                    prerequisites=tuple(generation_keys),
                    validator=lambda path=Path(
                        config["packed_naturalness_scores_path"]
                    ): _nonempty_file(path),
                ),
            )
            _add_task(
                tasks,
                TaskSpec(
                    key=stability_key,
                    group=stability_group,
                    array_id=None,
                    prerequisites=tuple(generation_keys),
                    validator=lambda path=Path(
                        config["packed_stability_scores_path"]
                    ): _nonempty_file(path),
                ),
            )
            packed_prerequisites = (naturalness_key, stability_key)
        aggregate_key = f"p2:{seed}:aggregate"
        progen2_aggregate_keys.append(aggregate_key)
        expected_progen2_rows = len(rows)
        _add_task(
            tasks,
            TaskSpec(
                key=aggregate_key,
                group=aggregate_group,
                array_id=None,
                prerequisites=(
                    *packed_prerequisites,
                    *foldability_keys,
                    *developability_keys,
                    *diversity_keys,
                ),
                validator=lambda path=Path(
                    config["output_json_path"]
                ), expected=expected_progen2_rows: _json_has_length(
                    path, expected, field="results"
                ),
            ),
        )

    terminal_keys = (
        "denovo:aggregate",
        *mm_aggregate_keys,
        *progen2_aggregate_keys,
    )
    for task in tasks.values():
        if task.group not in groups:
            raise ValueError(f"Task {task.key} references unknown group {task.group}")
        for prerequisite in task.prerequisites:
            if prerequisite not in tasks:
                raise ValueError(f"Task {task.key} references unknown prerequisite {prerequisite}")
    return groups, tasks, terminal_keys


def _load_state(tasks: dict[str, TaskSpec]) -> dict:
    if STATE_PATH.is_file():
        state = json.loads(STATE_PATH.read_text())
        if state.get("task_keys") != sorted(tasks):
            raise RuntimeError("Controller task graph differs from the saved controller state")
        return state
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_REMOTE_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    state = {
        "status": "running",
        "git_commit": result.stdout.strip(),
        "started_at_epoch": time.time(),
        "task_keys": sorted(tasks),
        "tasks": {},
        "submissions": [],
    }
    _atomic_write_state(state)
    return state


def _active_jobs() -> tuple[set[tuple[str, int | None]], int]:
    result = subprocess.run(
        [
            str(SQUEUE),
            "-r",
            "-h",
            "-u",
            os.environ["USER"],
            "-o",
            "%F|%K|%P",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    active: set[tuple[str, int | None]] = set()
    gpu_count = 0
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        job_id, array_value, partition = line.strip().split("|", 2)
        array_id = None if array_value == "N/A" else int(array_value)
        active.add((job_id, array_id))
        if partition == "gpu":
            gpu_count += 1
    return active, gpu_count


def _accounting_states(job_ids: Iterable[str]) -> dict[tuple[str, int | None], tuple[str, str]]:
    states: dict[tuple[str, int | None], tuple[str, str]] = {}
    for job_id in sorted(set(job_ids)):
        result = subprocess.run(
            [
                str(SACCT),
                "-n",
                "-P",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            logical_id, raw_state, exit_code = line.split("|", 2)
            if "." in logical_id:
                continue
            if "_" in logical_id:
                base_id, array_value = logical_id.split("_", 1)
                if not array_value.isdigit():
                    continue
                key = (base_id, int(array_value))
            elif logical_id.isdigit():
                key = (logical_id, None)
            else:
                continue
            state_name = raw_state.split()[0].rstrip("+")
            states[key] = (state_name, exit_code)
    return states


def _refresh_task_states(
    state: dict,
    tasks: dict[str, TaskSpec],
    active_jobs: set[tuple[str, int | None]],
) -> None:
    submitted_job_ids = [
        state["tasks"][key]["job_id"]
        for key in tasks
        if (entry := state["tasks"].get(key))
        if entry.get("status") == "submitted"
    ]
    accounting = _accounting_states(submitted_job_ids)
    terminal_failures = {
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
    now = time.time()
    for key, task in tasks.items():
        entry = state["tasks"].get(key)
        if task.validator():
            state["tasks"][key] = {
                **(entry or {}),
                "status": "complete",
                "completed_at_epoch": now,
            }
            continue
        if entry is None or entry.get("status") != "submitted":
            continue
        job_key = (entry["job_id"], entry["array_id"])
        if job_key in active_jobs:
            continue
        job_state = accounting.get(job_key)
        if job_state is None:
            if now - float(entry["submitted_at_epoch"]) > 300:
                raise RuntimeError(f"Job disappeared without accounting state or output: {key} {job_key}")
            continue
        state_name, exit_code = job_state
        if state_name == "COMPLETED":
            grace_started_at = entry.get("output_validation_grace_started_at_epoch")
            if grace_started_at is None:
                entry["output_validation_grace_started_at_epoch"] = now
                print(
                    f"waiting for completed job output: task={key} job={job_key}",
                    flush=True,
                )
                continue
            grace_elapsed = now - float(grace_started_at)
            if grace_elapsed < COMPLETED_OUTPUT_GRACE_SECONDS:
                continue
            raise RuntimeError(
                "Job completed but expected output validation failed after "
                f"{grace_elapsed:.0f}s grace: {key} {job_key}"
            )
        if state_name in terminal_failures:
            raise RuntimeError(
                f"Job failed: task={key} job={job_key} state={state_name} exit={exit_code}"
            )


def _submit_group(
    state: dict,
    group: GroupSpec,
    selected_tasks: list[TaskSpec],
) -> bool:
    if not selected_tasks:
        return True
    array_ids = [task.array_id for task in selected_tasks]
    if any(array_id is None for array_id in array_ids):
        if len(selected_tasks) != 1 or array_ids != [None]:
            raise ValueError(f"Non-array group must contain one task: {group.name}")
    command = [
        str(SBATCH),
        "--parsable",
        "--exclude=server13,server59",
        f"--job-name={group.job_name}",
        f"--output={group.output_pattern}",
        f"--error={group.error_pattern}",
    ]
    if array_ids != [None]:
        command.append("--array=" + ",".join(str(array_id) for array_id in array_ids))
    if group.time_limit is not None:
        command.append(f"--time={group.time_limit}")
    if group.exports:
        command.append(f"--export={_export_option(group.exports)}")
    command.append(str(group.script))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_clean_submission_environment(),
    )
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}"
        if "QOSMaxSubmitJobPerUserLimit" in combined and group.resource == "gpu":
            print(f"GPU submit slots changed concurrently; retrying later: {combined.strip()}", flush=True)
            return False
        raise RuntimeError(
            f"sbatch failed for {group.name}: returncode={result.returncode}\n{combined}"
        )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse job ID for {group.name}: {result.stdout!r}")
    submitted_at = time.time()
    for task in selected_tasks:
        state["tasks"][task.key] = {
            "status": "submitted",
            "job_id": job_id,
            "array_id": task.array_id,
            "submitted_at_epoch": submitted_at,
        }
    state["submissions"].append(
        {
            "group": group.name,
            "job_id": job_id,
            "task_keys": [task.key for task in selected_tasks],
            "command": command,
            "submitted_at_epoch": submitted_at,
        }
    )
    _atomic_write_state(state)
    print(
        f"submitted group={group.name} job={job_id} elements={len(selected_tasks)}",
        flush=True,
    )
    return True


def _ready_tasks(
    state: dict,
    tasks: dict[str, TaskSpec],
) -> dict[str, list[TaskSpec]]:
    ready: dict[str, list[TaskSpec]] = {}
    completed = {
        key
        for key, entry in state["tasks"].items()
        if entry.get("status") == "complete"
    }
    for key, task in tasks.items():
        entry = state["tasks"].get(key)
        if entry is not None and entry.get("status") in {"submitted", "complete"}:
            continue
        if (
            all(prerequisite in completed for prerequisite in task.prerequisites)
            and task.readiness_validator()
        ):
            ready.setdefault(task.group, []).append(task)
    for group_tasks in ready.values():
        group_tasks.sort(key=lambda task: (-1 if task.array_id is None else task.array_id))
    return ready


def _schedule_cpu_tasks(
    state: dict,
    groups: dict[str, GroupSpec],
    ready: dict[str, list[TaskSpec]],
) -> None:
    for group_name, group in groups.items():
        if group.resource != "cpu":
            continue
        selected = ready.get(group_name, [])
        if selected:
            _submit_group(state, group, selected)


def _select_gpu_tasks(
    groups: dict[str, GroupSpec],
    ready: dict[str, list[TaskSpec]],
    capacity: int,
    start_group: str | None,
) -> dict[str, list[TaskSpec]]:
    gpu_group_names = [
        group_name
        for group_name, group in groups.items()
        if group.resource == "gpu"
    ]
    if start_group is not None:
        if start_group not in gpu_group_names:
            raise ValueError(f"Unknown GPU scheduling cursor: {start_group}")
        start_index = gpu_group_names.index(start_group)
        gpu_group_names = (
            gpu_group_names[start_index:] + gpu_group_names[:start_index]
        )
    queues = {
        group_name: list(ready.get(group_name, []))
        for group_name in gpu_group_names
        if ready.get(group_name)
    }
    selected: dict[str, list[TaskSpec]] = {}
    while capacity > 0 and queues:
        progressed = False
        for group_name in list(queues):
            take = min(GPU_TASKS_PER_GROUP_PER_ROUND, capacity, len(queues[group_name]))
            if take:
                selected.setdefault(group_name, []).extend(queues[group_name][:take])
                del queues[group_name][:take]
                capacity -= take
                progressed = True
            if not queues[group_name]:
                del queues[group_name]
            if capacity == 0:
                break
        if not progressed:
            break
    return selected


def _next_gpu_group(
    groups: dict[str, GroupSpec],
    current_group: str,
) -> str:
    gpu_group_names = [
        group_name
        for group_name, group in groups.items()
        if group.resource == "gpu"
    ]
    if current_group not in gpu_group_names:
        raise ValueError(f"Unknown GPU group: {current_group}")
    current_index = gpu_group_names.index(current_group)
    return gpu_group_names[(current_index + 1) % len(gpu_group_names)]


def _schedule_gpu_tasks(
    state: dict,
    groups: dict[str, GroupSpec],
    ready: dict[str, list[TaskSpec]],
    gpu_submitted_count: int,
) -> None:
    capacity = max(0, GPU_MAX_SUBMITTED_JOBS - gpu_submitted_count)
    selected_by_group = _select_gpu_tasks(
        groups,
        ready,
        capacity,
        state.get("gpu_group_cursor"),
    )
    for group_name, selected in selected_by_group.items():
        if not _submit_group(state, groups[group_name], selected):
            break
        state["gpu_group_cursor"] = _next_gpu_group(groups, group_name)
        _atomic_write_state(state)


def _progress_summary(state: dict, tasks: dict[str, TaskSpec]) -> str:
    counts = {"complete": 0, "submitted": 0, "pending": 0}
    for key in tasks:
        status = state["tasks"].get(key, {}).get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return " ".join(f"{name}={value}" for name, value in counts.items())


def main() -> None:
    if not SPEC_ROOT.joinpath("manifest.json").is_file():
        raise FileNotFoundError(f"Missing generated sweep specs: {SPEC_ROOT}")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another rebuttal sweep controller is already running") from exc

    groups, all_tasks, all_terminal_keys = _build_dag()
    state = _load_state(all_tasks)
    active_domains = _active_domains()
    tasks = {
        key: task
        for key, task in all_tasks.items()
        if _task_domain(key) in active_domains
    }
    terminal_keys = tuple(
        key for key in all_terminal_keys if _task_domain(key) in active_domains
    )
    if not tasks or not terminal_keys:
        raise RuntimeError(f"No tasks selected for domains: {active_domains}")
    state["status"] = "running"
    state["active_domains"] = list(active_domains)
    state["controller_restarted_at_epoch"] = time.time()
    _atomic_write_state(state)
    print(
        f"controller_start domains={','.join(active_domains)} "
        f"groups={len(groups)} tasks={len(tasks)} "
        f"gpu_submit_limit={GPU_MAX_SUBMITTED_JOBS}",
        flush=True,
    )

    try:
        while True:
            active_jobs, gpu_submitted_count = _active_jobs()
            _refresh_task_states(state, tasks, active_jobs)
            _atomic_write_state(state)
            if all(
                state["tasks"].get(key, {}).get("status") == "complete"
                for key in terminal_keys
            ):
                COMPLETE_PATH.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z") + "\n")
                state["status"] = "complete"
                state["completed_at_epoch"] = time.time()
                _atomic_write_state(state)
                print(f"controller_complete {_progress_summary(state, tasks)}", flush=True)
                return

            ready = _ready_tasks(state, tasks)
            _schedule_cpu_tasks(state, groups, ready)
            active_jobs, gpu_submitted_count = _active_jobs()
            ready = _ready_tasks(state, tasks)
            _schedule_gpu_tasks(state, groups, ready, gpu_submitted_count)
            print(
                f"controller_poll gpu_submitted={gpu_submitted_count} "
                f"{_progress_summary(state, tasks)}",
                flush=True,
            )
            time.sleep(POLL_SECONDS)
    except Exception as exc:
        state["status"] = "failed"
        state["failure"] = repr(exc)
        state["failed_at_epoch"] = time.time()
        _atomic_write_state(state)
        raise


if __name__ == "__main__":
    main()
