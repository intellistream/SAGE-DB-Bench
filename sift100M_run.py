#!/usr/bin/env python3
"""Run three StreamSeed-off and three StreamSeed-on SIFT100M trials."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = REPO_ROOT / "bench" / "algorithms" / "streamseed_hybrid" / "config.yaml"
RESULT_PARENT = REPO_ROOT / "results" / "sift100M" / "streamseed_hybrid"
RESULT_SOURCE = RESULT_PARENT / "ef-120"
TRIALS = (1, 2, 3)


@dataclass(frozen=True)
class ExperimentMode:
    label: str
    config_value: str
    result_suffix: str

    def result_name(self, trial: int) -> str:
        return f"test{trial}{self.result_suffix}"


MODES = (
    ExperimentMode("off", "off", ""),
    ExperimentMode("on", "streamseed_core", "-on"),
)


class RunInterrupted(Exception):
    pass


def _handle_signal(signum: int, _frame: object) -> None:
    raise RunInterrupted(f"received signal {signum}")


def _indented_block_bounds(
    lines: list[str], key: str, indent: int, start: int = 0, end: int | None = None
) -> tuple[int, int]:
    if end is None:
        end = len(lines)
    prefix = " " * indent + key + ":"
    block_start = None

    for index in range(start, end):
        if lines[index].rstrip("\r\n") == prefix:
            block_start = index
            break

    if block_start is None:
        raise ValueError(f"configuration block not found: {prefix}")

    block_end = end
    for index in range(block_start + 1, end):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent <= indent:
            block_end = index
            break
    return block_start, block_end


def _validate_mode(content: str, expected_mode: str) -> None:
    document = yaml.safe_load(content)
    query_args_text = document["sift100M"]["streamseed_hybrid"]["run-groups"][
        "base"
    ]["query-args"]
    query_args = json.loads(query_args_text)
    if not isinstance(query_args, list) or len(query_args) != 1:
        raise ValueError("expected exactly one sift100M streamseed_hybrid query-args entry")
    actual_mode = query_args[0].get("streamseed_mode")
    if actual_mode != expected_mode:
        raise ValueError(
            f"sift100M streamseed_mode validation failed: "
            f"expected {expected_mode!r}, got {actual_mode!r}"
        )


def _render_config(original: str, mode: str) -> str:
    lines = original.splitlines(keepends=True)
    dataset_start, dataset_end = _indented_block_bounds(lines, "sift100M", 0)
    algorithm_start, algorithm_end = _indented_block_bounds(
        lines,
        "streamseed_hybrid",
        2,
        start=dataset_start + 1,
        end=dataset_end,
    )

    mode_pattern = re.compile(r'("streamseed_mode"\s*:\s*")[^"]+("\s*[,}])')
    replacement_count = 0
    for index in range(algorithm_start + 1, algorithm_end):
        replaced_line, count = mode_pattern.subn(rf'\g<1>{mode}\g<2>', lines[index])
        if count:
            lines[index] = replaced_line
            replacement_count += count

    if replacement_count != 1:
        raise ValueError(
            "expected exactly one streamseed_mode under "
            f"sift100M/streamseed_hybrid, replaced {replacement_count}"
        )

    rendered = "".join(lines)
    _validate_mode(rendered, mode)
    return rendered


def _atomic_write(path: Path, content: str) -> None:
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _benchmark_command() -> list[str]:
    return [
        sys.executable,
        "run_benchmark.py",
        "--algorithm",
        "streamseed_hybrid",
        "--dataset",
        "sift100M",
        "--runbook",
        "runbooks/general_experiment/general_experiment.yaml",
        "--enable-cache-profiling",
    ]


def _destinations() -> list[Path]:
    return [
        RESULT_PARENT / experiment_mode.result_name(trial)
        for experiment_mode in MODES
        for trial in TRIALS
    ]


def _preflight() -> None:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"algorithm config does not exist: {CONFIG_PATH}")
    if RESULT_SOURCE.exists():
        raise FileExistsError(
            f"result source already exists: {RESULT_SOURCE}\n"
            "Move or rename it before starting so it cannot be mistaken for a new result."
        )

    existing = [path for path in _destinations() if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "the following destinations already exist; refusing to overwrite:\n"
            f"{formatted}"
        )


def _print_plan() -> None:
    print("OMP_NUM_THREADS=4 " + " ".join(_benchmark_command()))
    for experiment_mode in MODES:
        for trial in TRIALS:
            print(
                f"mode={experiment_mode.config_value} trial={trial} "
                f"-> {experiment_mode.result_name(trial)}"
            )


def run_experiments() -> None:
    _preflight()
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)

    original = CONFIG_PATH.read_text(encoding="utf-8")
    rendered_configs = {
        experiment_mode.config_value: _render_config(
            original, experiment_mode.config_value
        )
        for experiment_mode in MODES
    }

    backup = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.sift100M_run.backup")
    if backup.exists():
        raise FileExistsError(
            f"backup already exists: {backup}\n"
            "A previous run may have been interrupted; inspect and restore it first."
        )

    command = _benchmark_command()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"
    shutil.copy2(CONFIG_PATH, backup)

    try:
        for experiment_mode in MODES:
            rendered = rendered_configs[experiment_mode.config_value]
            _atomic_write(CONFIG_PATH, rendered)
            print(
                f"\nConfigured sift100M streamseed_mode="
                f"{experiment_mode.config_value}",
                flush=True,
            )

            for trial in TRIALS:
                destination = RESULT_PARENT / experiment_mode.result_name(trial)
                print(
                    f"\n[{experiment_mode.label}, trial {trial}/{len(TRIALS)}]",
                    flush=True,
                )
                print("Running: OMP_NUM_THREADS=4 " + " ".join(command), flush=True)

                subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=environment,
                    check=True,
                )

                if not RESULT_SOURCE.is_dir():
                    raise FileNotFoundError(
                        "benchmark succeeded but result directory was not created: "
                        f"{RESULT_SOURCE}"
                    )
                if destination.exists():
                    raise FileExistsError(f"destination appeared during run: {destination}")
                RESULT_SOURCE.rename(destination)
                print(f"Saved: {destination}", flush=True)
    finally:
        if backup.exists():
            shutil.copy2(backup, CONFIG_PATH)
            backup.unlink()
            print(f"\nRestored algorithm config: {CONFIG_PATH}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run SIFT100M streamseed_hybrid off/on experiments three times each."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate both configurations and print all six runs without executing them",
    )
    args = parser.parse_args()

    original = CONFIG_PATH.read_text(encoding="utf-8")
    for experiment_mode in MODES:
        _render_config(original, experiment_mode.config_value)

    if args.dry_run:
        _print_plan()
        return 0

    for handled_signal in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(handled_signal, _handle_signal)

    try:
        run_experiments()
    except (RunInterrupted, KeyboardInterrupt) as error:
        detail = str(error) or "keyboard interrupt"
        print(f"\nExperiment interrupted: {detail}", file=sys.stderr)
        return 130
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"\nExperiment failed: {error}", file=sys.stderr)
        if RESULT_SOURCE.exists():
            print(
                f"Partial or failed-run output was left untouched at {RESULT_SOURCE}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
