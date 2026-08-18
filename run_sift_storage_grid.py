#!/usr/bin/env python3
"""Run the SIFT StreamSeed two-storage capacity grid."""

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
CONFIG_FILE = (
    REPO_ROOT / "bench" / "algorithms" / "streamseed_hybrid" / "config.yaml"
)
RUNBOOK = REPO_ROOT / "runbooks" / "general_experiment" / "general_experiment.yaml"
RESULT_PARENT = REPO_ROOT / "results" / "sift" / "streamseed_hybrid"
RESULT_SOURCE = RESULT_PARENT / "ef-120"
TRIALS = 3


@dataclass(frozen=True)
class StorageConfig:
    hot_capacity: int
    table_slots: int
    slot_capacity: int


# H=0 is not used because the current C++ implementation maps non-positive Hot
# capacity to its default value (512). H=512 is therefore the most
# Semantic-heavy configuration that can be requested without changing C++.
HOT_CAPACITIES = (2000, 5000, 7000, 9000)
TABLE_SLOTS = (50, 100, 500, 1000)
SLOT_CAPACITIES = (30, 50, 70, 100)

# Full H x S x m grid for configurations that still leave part of the fixed
# 10,000-query pool in Semantic storage.
FACTORIAL_GRID = tuple(
    StorageConfig(hot_capacity, table_slots, slot_capacity)
    for hot_capacity in HOT_CAPACITIES
    for table_slots in TABLE_SLOTS
    for slot_capacity in SLOT_CAPACITIES
)

# H=10,000 covers the query pool, so S and m would be redundant there.
HOT_ONLY_REFERENCE = (StorageConfig(10000, 500, 70),)
CONFIGURATIONS = FACTORIAL_GRID + HOT_ONLY_REFERENCE

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_FIELD_PATTERNS = {
    "hint_hot_capacity": re.compile(
        rf'(?P<prefix>"hint_hot_capacity"\s*:\s*){_NUMBER}'
    ),
    "hint_table_slots": re.compile(
        rf'(?P<prefix>"hint_table_slots"\s*:\s*){_NUMBER}'
    ),
    "hint_slot_capacity": re.compile(
        rf'(?P<prefix>"hint_slot_capacity"\s*:\s*){_NUMBER}'
    ),
}


class SweepInterrupted(Exception):
    pass


def _handle_signal(signum: int, _frame: object) -> None:
    raise SweepInterrupted(f"received signal {signum}")


def _section_bounds(
    lines: list[str], section: str, start: int, end: int
) -> tuple[int, int]:
    found = None
    indent = None
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(section)}:\s*(?:#.*)?$")

    for index in range(start, end):
        match = pattern.match(lines[index].rstrip("\r\n"))
        if match:
            found = index
            indent = len(match.group("indent").expandtabs(8))
            break

    if found is None or indent is None:
        raise ValueError(f"section '{section}:' was not found")

    section_end = end
    for index in range(found + 1, end):
        stripped = lines[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        current_indent = len(lines[index]) - len(lines[index].lstrip())
        if current_indent <= indent:
            section_end = index
            break
    return found, section_end


def _streamseed_block_bounds(lines: list[str]) -> tuple[int, int]:
    sift_start, sift_end = _section_bounds(lines, "sift", 0, len(lines))
    return _section_bounds(lines, "streamseed_hybrid", sift_start + 1, sift_end)


def _render_config(original: str, config: StorageConfig) -> str:
    lines = original.splitlines(keepends=True)
    start, end = _streamseed_block_bounds(lines)
    block = "".join(lines[start:end])
    values = {
        "hint_hot_capacity": config.hot_capacity,
        "hint_table_slots": config.table_slots,
        "hint_slot_capacity": config.slot_capacity,
    }

    for field, value in values.items():
        block, replacements = _FIELD_PATTERNS[field].subn(
            lambda match, replacement=value: (
                f"{match.group('prefix')}{replacement}"
            ),
            block,
        )
        if replacements != 1:
            raise ValueError(
                f"expected exactly one {field} in sift.streamseed_hybrid, "
                f"found {replacements}"
            )

    rendered = "".join(lines[:start]) + block + "".join(lines[end:])
    parsed = yaml.safe_load(rendered)
    query_args_raw = parsed["sift"]["streamseed_hybrid"]["run-groups"]["base"][
        "query-args"
    ]
    query_args = (
        json.loads(query_args_raw)
        if isinstance(query_args_raw, str)
        else query_args_raw
    )
    if not isinstance(query_args, list) or len(query_args) != 1:
        raise ValueError("expected exactly one base query-args entry")

    actual = query_args[0]
    expected = {
        "hint_hot_capacity": config.hot_capacity,
        "hint_table_slots": config.table_slots,
        "hint_slot_capacity": config.slot_capacity,
    }
    for field, value in expected.items():
        if int(actual[field]) != value:
            raise ValueError(
                f"config validation failed: expected {field}={value}, "
                f"got {actual[field]}"
            )
    if int(actual.get("hint_semantic_enabled", 0)) != 1:
        raise ValueError(
            "hint_semantic_enabled must be 1 for the storage grid experiment"
        )
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
        "runbooks/general_experiment/general_experiment.yaml",
        "--enable-cache-profiling",
    ]


def _result_name(trial: int, config: StorageConfig) -> str:
    return (
        f"test{trial}-storage-h{config.hot_capacity}"
        f"-s{config.table_slots}-m{config.slot_capacity}"
    )


def _all_destinations() -> list[Path]:
    return [
        RESULT_PARENT / _result_name(trial, config)
        for config in CONFIGURATIONS
        for trial in range(1, TRIALS + 1)
    ]


def _preflight(original: str) -> None:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"algorithm config does not exist: {CONFIG_FILE}")
    if not RUNBOOK.is_file():
        raise FileNotFoundError(f"runbook does not exist: {RUNBOOK}")
    if len(CONFIGURATIONS) != 65:
        raise ValueError(
            f"expected 61 unique storage configurations, got {len(CONFIGURATIONS)}"
        )
    for config in CONFIGURATIONS:
        _render_config(original, config)
    if RESULT_SOURCE.exists():
        raise FileExistsError(
            f"result source already exists: {RESULT_SOURCE}\n"
            "Move or rename it before starting so it cannot be mistaken for a new result."
        )
    existing = [path for path in _all_destinations() if path.exists()]
    if existing:
        formatted = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "the following destination directories already exist; "
            f"refusing to overwrite:\n{formatted}"
        )


def _print_plan() -> None:
    print("OMP_NUM_THREADS=4 " + " ".join(_benchmark_command()))
    for config in CONFIGURATIONS:
        for trial in range(1, TRIALS + 1):
            print(
                f"H={config.hot_capacity} S={config.table_slots} "
                f"m={config.slot_capacity} trial={trial}/{TRIALS} "
                f"-> {_result_name(trial, config)}"
            )


def run_sweep() -> None:
    original = CONFIG_FILE.read_text(encoding="utf-8")
    _preflight(original)
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)

    backup = CONFIG_FILE.with_name(f"{CONFIG_FILE.name}.storage_grid.backup")
    if backup.exists():
        raise FileExistsError(
            f"backup already exists: {backup}\n"
            "A previous sweep may have been interrupted; inspect and restore it first."
        )

    shutil.copy2(CONFIG_FILE, backup)
    command = _benchmark_command()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4"

    try:
        for config in CONFIGURATIONS:
            rendered = _render_config(original, config)
            for trial in range(1, TRIALS + 1):
                destination = RESULT_PARENT / _result_name(trial, config)
                _atomic_write(CONFIG_FILE, rendered)
                print(
                    f"\n[H={config.hot_capacity}, S={config.table_slots}, "
                    f"m={config.slot_capacity}, trial {trial}/{TRIALS}]",
                    flush=True,
                )
                print(
                    "Running: OMP_NUM_THREADS=4 " + " ".join(command),
                    flush=True,
                )

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
                    raise FileExistsError(
                        f"destination appeared during the run: {destination}"
                    )
                RESULT_SOURCE.rename(destination)
                print(f"Saved: {destination}", flush=True)
    finally:
        if backup.exists():
            shutil.copy2(backup, CONFIG_FILE)
            backup.unlink()
            print(f"\nRestored algorithm config: {CONFIG_FILE}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run 61 StreamSeed storage configurations three times each on SIFT."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print all 183 runs without executing them",
    )
    args = parser.parse_args()

    try:
        original = CONFIG_FILE.read_text(encoding="utf-8")
        if args.dry_run:
            _preflight(original)
            _print_plan()
            return 0

        for handled_signal in (signal.SIGHUP, signal.SIGTERM):
            signal.signal(handled_signal, _handle_signal)
        run_sweep()
    except (KeyboardInterrupt, SweepInterrupted) as error:
        print(f"\nSweep interrupted: {error}", file=sys.stderr)
        return 130
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        subprocess.CalledProcessError,
    ) as error:
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
