#!/usr/bin/env python3
"""Run the SIFT query-repetition sweep for streamseed_hybrid."""

from __future__ import annotations

import argparse
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
RUNBOOK = (
    REPO_ROOT
    / "runbooks"
    / "general_experiment"
    / "general_experiment_simple2.yaml"
)
RESULT_PARENT = REPO_ROOT / "results" / "sift" / "streamseed_hybrid"
RESULT_SOURCE = RESULT_PARENT / "ef-120"
TRIALS = 3


@dataclass(frozen=True)
class RepetitionConfig:
    repetition: int
    pool_size: int
    sample_size: int
    sample_mode: str

    def result_name(self, trial: int) -> str:
        return (
            f"test{trial}_different_repetition_{self.repetition}_"
            f"{self.pool_size}_{self.sample_size}_{self.sample_mode}"
        )


CONFIGS = (
    RepetitionConfig(20, 30000, 6000, "per_batch"),
    RepetitionConfig(25, 24000, 6000, "per_batch"),
    RepetitionConfig(50, 12000, 6000, "per_batch"),
    RepetitionConfig(75, 8000, 6000, "per_batch"),
    RepetitionConfig(100, 30000, 6000, "fixed"),
)


class SweepInterrupted(Exception):
    pass


def _handle_signal(signum: int, _frame: object) -> None:
    raise SweepInterrupted(f"received signal {signum}")


def _sift_block_bounds(lines: list[str]) -> tuple[int, int]:
    start = None
    for index, line in enumerate(lines):
        if line.rstrip("\r\n") == "sift:":
            start = index
            break

    if start is None:
        raise ValueError("top-level 'sift:' section was not found in the runbook")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            end = index
            break
    return start, end


def _render_runbook(original: str, config: RepetitionConfig) -> str:
    lines = original.splitlines(keepends=True)
    start, end = _sift_block_bounds(lines)
    replacements = {
        "queryPoolSize": str(config.pool_size),
        "querySampleSize": str(config.sample_size),
        "querySampleMode": config.sample_mode,
    }
    replaced = {key: 0 for key in replacements}
    key_pattern = re.compile(
        r"^(?P<indent>\s*)(?P<key>queryPoolSize|querySampleSize|querySampleMode):.*?"
        r"(?P<newline>\r?\n)?$"
    )

    for index in range(start + 1, end):
        match = key_pattern.match(lines[index])
        if not match:
            continue
        key = match.group("key")
        newline = match.group("newline") or ""
        lines[index] = f"{match.group('indent')}{key}: {replacements[key]}{newline}"
        replaced[key] += 1

    invalid = {key: count for key, count in replaced.items() if count != 1}
    if invalid:
        raise ValueError(f"expected each SIFT query setting exactly once, got {invalid}")

    rendered = "".join(lines)
    parsed = yaml.safe_load(rendered)
    batch_insert = parsed["sift"][3]
    expected = {
        "queryPoolSize": config.pool_size,
        "querySampleSize": config.sample_size,
        "querySampleMode": config.sample_mode,
    }
    actual = {key: batch_insert.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"runbook validation failed: expected {expected}, got {actual}")
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
        "sift",
        "--runbook",
        "runbooks/general_experiment/general_experiment_simple2.yaml",
        "--enable-cache-profiling",
    ]


def _all_destinations() -> list[Path]:
    return [
        RESULT_PARENT / config.result_name(trial)
        for config in CONFIGS
        for trial in range(1, TRIALS + 1)
    ]


def _preflight() -> None:
    if not RUNBOOK.is_file():
        raise FileNotFoundError(f"runbook does not exist: {RUNBOOK}")
    if RESULT_SOURCE.exists():
        raise FileExistsError(
            f"result source already exists: {RESULT_SOURCE}\n"
            "Move or rename it before starting so it cannot be mistaken for a new result."
        )
    existing = [path for path in _all_destinations() if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "the following destination directories already exist; refusing to overwrite:\n"
            f"{formatted}"
        )


def _print_plan() -> None:
    command = "OMP_NUM_THREADS=4 " + " ".join(_benchmark_command())
    print(command)
    for config in CONFIGS:
        for trial in range(1, TRIALS + 1):
            print(
                f"repetition={config.repetition}% trial={trial} "
                f"pool={config.pool_size} sample={config.sample_size} "
                f"mode={config.sample_mode} -> {config.result_name(trial)}"
            )


def run_sweep() -> None:
    _preflight()
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)

    original = RUNBOOK.read_text(encoding="utf-8")
    backup = RUNBOOK.with_name(f"{RUNBOOK.name}.repetition_sweep.backup")
    if backup.exists():
        raise FileExistsError(
            f"backup already exists: {backup}\n"
            "A previous sweep may have been interrupted; inspect and restore it first."
        )

    shutil.copy2(RUNBOOK, backup)
    command = _benchmark_command()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"

    try:
        for config in CONFIGS:
            rendered = _render_runbook(original, config)
            for trial in range(1, TRIALS + 1):
                destination = RESULT_PARENT / config.result_name(trial)
                _atomic_write(RUNBOOK, rendered)
                print(
                    f"\n[{config.repetition}% repetition, trial {trial}/{TRIALS}] "
                    f"pool={config.pool_size}, sample={config.sample_size}, "
                    f"mode={config.sample_mode}",
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
                        f"benchmark succeeded but result directory was not created: "
                        f"{RESULT_SOURCE}"
                    )
                if destination.exists():
                    raise FileExistsError(f"destination appeared during the run: {destination}")
                RESULT_SOURCE.rename(destination)
                print(f"Saved: {destination}", flush=True)
    finally:
        if backup.exists():
            shutil.copy2(backup, RUNBOOK)
            backup.unlink()
            print(f"\nRestored runbook: {RUNBOOK}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run 20/25/50/75/100% SIFT query-repetition experiments."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the runbook and print all 15 runs without executing them",
    )
    args = parser.parse_args()

    for config in CONFIGS:
        _render_runbook(RUNBOOK.read_text(encoding="utf-8"), config)

    if args.dry_run:
        _print_plan()
        return 0

    for handled_signal in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(handled_signal, _handle_signal)

    try:
        run_sweep()
    except SweepInterrupted as error:
        print(f"\nSweep interrupted: {error}", file=sys.stderr)
        return 130
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"\nSweep failed: {error}", file=sys.stderr)
        if RESULT_SOURCE.exists():
            print(
                f"Partial or failed-run output was left untouched at {RESULT_SOURCE}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
