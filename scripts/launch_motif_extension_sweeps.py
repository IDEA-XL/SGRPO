#!/usr/bin/env python3
"""Submit ready, incomplete motif-extension sweep tasks as one Slurm array."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--sbatch",
        default="/opt/gridview/slurm/bin/sbatch",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    ready = []
    incomplete = []
    for task in manifest["tasks"]:
        checkpoint = Path(task["checkpoint_path"])
        summary = Path(task["output_dir"]) / "summary.json"
        output_dir = Path(task["output_dir"])
        if summary.is_file():
            continue
        if output_dir.exists():
            incomplete.append(str(output_dir))
            continue
        if checkpoint.is_file() and checkpoint.stat().st_size > 0:
            ready.append(int(task["task_index"]))
    if incomplete:
        raise RuntimeError(
            "incomplete output directories require diagnosis before resubmission:\n"
            + "\n".join(incomplete)
        )
    if not ready:
        print("No ready incomplete tasks")
        return
    array_spec = ",".join(str(index) for index in ready)
    command = [
        args.sbatch,
        f"--array={array_spec}",
        f"--export=ALL,MANIFEST_PATH={args.manifest}",
        "scripts/slurm/motif_extension_sweep_1gpu.sbatch",
    ]
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout.strip())


if __name__ == "__main__":
    main()
