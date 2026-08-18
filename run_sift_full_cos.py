#!/usr/bin/env python3
"""Run the SIFT StreamSeed consistency-gate sweep."""

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
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = (
    REPO_ROOT / "bench" / "algorithms" / "streamseed_hybrid" / "config.yaml"
)
RUNBOOK = REPO_ROOT / "runbooks" / "general_experiment" / "general_experiment.yaml"
RESULT_PARENT = REPO_ROOT / "results" / "sift" / "streamseed_hybrid"
RESULT_SOURCE = RESULT_PARENT / "ef-120"
CONS_GATES = ("0.3", "0.5", "0.6", "0.8", "0.9", "1.0")
TRIALS = 2

EXPECTED_BUILD_ARGS = {
    "efConstruction": 120,
    "indexkey": "HNSWIncremental32",
    "verbose": True,
}

# hint_cons_gate is deliberately omitted because it is the swept parameter.
EXPECTED_QUERY_BASELINE = {
    "ef": 120,
    "streamseed_mode": "streamseed_two_storage",
    "hint_level1_only": 1,
    "hint_adaptive_gate_mode": 1,
    "hint_hops": 1,
    "hint_max_candidates": 500,
    "hint_gate": -1.0,
    "hint_qual_gate": -1.0,
    "hint_gate_m_quantile": 0.25,
    "hint_gate_o_quantile": 0.30,
    "hint_gate_min_samples": 128,
    "hint_table_slots": 10000,
    "hint_slot_capacity": 70,
    "hint_hot_capacity": 7000,
    "hint_probe_count": 1,
    "hint_retrieval_threshold": 0.70,
    "hint_signature_weight": 0.90,
    "hint_boundary_gap_profile": 0,
    "hint_semantic_enabled": 1,
    "hint_promotion_hits": 100000,
    "hint_demotion_window": 20000,
}

_CONS_PATTERN = re.compile(
    r'(?P<prefix>"hint_cons_gate"\s*:\s*)'
    r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
)


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


def _base_args(config_text: str) -> tuple[dict[str, object], dict[str, object]]:
    parsed = yaml.safe_load(config_text)
    base = parsed["sift"]["streamseed_hybrid"]["run-groups"]["base"]
    build_raw = base["args"]
    query_raw = base["query-args"]
    build_args = json.loads(build_raw) if isinstance(build_raw, str) else build_raw
    query_args = json.loads(query_raw) if isinstance(query_raw, str) else query_raw
    if not isinstance(build_args, list) or len(build_args) != 1:
        raise ValueError("expected exactly one base build args entry")
    if not isinstance(query_args, list) or len(query_args) != 1:
        raise ValueError("expected exactly one base query-args entry")
    return build_args[0], query_args[0]


def _validate_baseline(config_text: str) -> None:
    build_args, query_args = _base_args(config_text)
    differences = []
    for key, expected in EXPECTED_BUILD_ARGS.items():
        actual = build_args.get(key)
        if actual != expected:
            differences.append(f"build.{key}: expected {expected!r}, got {actual!r}")
    for key, expected in EXPECTED_QUERY_BASELINE.items():
        actual = query_args.get(key)
        if actual != expected:
            differences.append(f"query.{key}: expected {expected!r}, got {actual!r}")
    if "hint_cons_gate" not in query_args:
        differences.append("query.hint_cons_gate: missing")
    if differences:
        formatted = "\n".join(f"  {difference}" for difference in differences)
        raise ValueError(f"SIFT streamseed_hybrid baseline mismatch:\n{formatted}")


def _render_config(original: str, cons_gate: str) -> str:
    lines = original.splitlines(keepends=True)
    start, end = _streamseed_block_bounds(lines)
    block = "".join(lines[start:end])
    rendered_block, replacements = _CONS_PATTERN.subn(
        lambda match: f"{match.group('prefix')}{cons_gate}", block
    )
    if replacements != 1:
        raise ValueError(
            "expected exactly one hint_cons_gate in "
            f"sift.streamseed_hybrid, found {replacements}"
        )

    rendered = "".join(lines[:start]) + rendered_block + "".join(lines[end:])
    _validate_baseline(rendered)
    _, query_args = _base_args(rendered)
    actual = float(query_args["hint_cons_gate"])
    if actual != float(cons_gate):
        raise ValueError(
            f"config validation failed: expected hint_cons_gate={cons_gate}, got {actual}"
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


def _result_name(trial: int, cons_gate: str) -> str:
    return f"test{trial}-cos{cons_gate}"


def _all_destinations() -> list[Path]:
    return [
        RESULT_PARENT / _result_name(trial, cons_gate)
        for cons_gate in CONS_GATES
        for trial in range(1, TRIALS + 1)
    ]


def _preflight(original: str) -> None:
    if not CONFIG_FILE.is_file():
        raise FileNotFoundError(f"algorithm config does not exist: {CONFIG_FILE}")
    if not RUNBOOK.is_file():
        raise FileNotFoundError(f"runbook does not exist: {RUNBOOK}")
    _validate_baseline(original)
    for cons_gate in CONS_GATES:
        _render_config(original, cons_gate)
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
    print("OMP_NUM_THREADS=4 " + " ".join(_benchmark_command()))
    for cons_gate in CONS_GATES:
        for trial in range(1, TRIALS + 1):
            print(
                f"hint_cons_gate={cons_gate} trial={trial}/{TRIALS} "
                f"-> {_result_name(trial, cons_gate)}"
            )


def run_sweep() -> None:
    original = CONFIG_FILE.read_text(encoding="utf-8")
    _preflight(original)
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)

    backup = CONFIG_FILE.with_name(f"{CONFIG_FILE.name}.full_cos_sweep.backup")
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
        for cons_gate in CONS_GATES:
            rendered = _render_config(original, cons_gate)
            for trial in range(1, TRIALS + 1):
                destination = RESULT_PARENT / _result_name(trial, cons_gate)
                _atomic_write(CONFIG_FILE, rendered)
                print(
                    f"\n[hint_cons_gate={cons_gate}, trial {trial}/{TRIALS}]",
                    flush=True,
                )
                print(
                    "Running: OMP_NUM_THREADS=4 " + " ".join(command),
                    flush=True,
                )
                subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)

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
        description="Run six StreamSeed consistency-gate values three times each on SIFT."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and print all 18 runs without executing them",
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
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
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
