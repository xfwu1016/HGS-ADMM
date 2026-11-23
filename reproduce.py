"""Run validation, complete manuscript experiments, or output generation.

All numerical algorithms and experiment settings remain in the original
scripts under code/. This entry point supplies directories and execution order.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parent
STEPS = {
    "solver-checks": ("validate_solver.py", []),
    "input-checks": ("check_solver_inputs.py", []),
    "budget-checks": ("check_pilot_budget_theory.py", []),
    "analytic": ("analytic_partition.py", []),
    "partition-checks": ("verify_partition_examples.py", []),
    "main": ("run_experiments.py", []),
    "consensus": ("consensus_baseline.py", []),
    "sensitivity": ("penalty_sensitivity.py", []),
    "pilot": ("pilot_budget_experiment.py", []),
    "pilot-box": ("pilot_box.py", []),
    "pilot-length": ("pilot_length.py", []),
    "pilot-summary": ("pilot_budget_experiment.py", ["--rebuild-summaries"]),
    "tables": ("make_tables.py", []),
    "figures": ("make_figures.py", []),
}
VALIDATE = ["solver-checks", "input-checks", "budget-checks", "analytic", "partition-checks"]
FULL = VALIDATE + ["main", "consensus", "sensitivity", "pilot", "pilot-box", "pilot-length", "tables", "figures"]
POSTPROCESS = ["pilot-summary", "pilot-box", "tables", "figures"]


def check_dependencies():
    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        package, required = line.split("==")
        try:
            installed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            raise SystemExit(
                f"Missing {package}. Run: python -m pip install -r requirements.txt"
            )
        versions[package] = installed
        if installed != required:
            print(f"Version note: {package}={installed}; manuscript environment uses {required}.", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("Version note: the manuscript environment uses Python 3.12.14.", flush=True)
    return versions


def read_rows(name):
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify_full_outputs():
    expected = {
        "runs.csv": 138,
        "consensus_baseline.csv": 30,
        "penalty_sensitivity.csv": 27,
        "pilot_budget_runs.csv": 135,
        "pilot_budget_pilots.csv": 45,
        "pilot_length.csv": 36,
        "pilot_box.csv": 45,
        "analytic_partition.csv": 12,
        "partition_same_objective.csv": 4,
        "partition_examples.csv": 12,
    }
    for name, count in expected.items():
        rows = read_rows(name)
        if len(rows) != count:
            raise RuntimeError(f"{name}: expected {count} records, found {len(rows)}")
        if name in {"runs.csv", "consensus_baseline.csv", "penalty_sensitivity.csv", "pilot_budget_runs.csv", "pilot_length.csv"}:
            if any(row["converged"] != "True" for row in rows):
                raise RuntimeError(f"{name}: at least one main solve did not reach its target")
    for stem in ["fig1_factor_structure", "fig2_convergence", "fig3_communication"]:
        for extension in ["pdf", "png", "svg"]:
            path = ROOT / "figures" / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing figure output: {path.name}")
    for name in ["hybrid_comparison", "consensus_comparison", "pilot_budget_results",
                 "real_results", "stopping_results", "analytic_results", "additional_results"]:
        path = ROOT / "tables" / f"{name}.tex"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing numerical table: {path.name}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate", action="store_true", help="Run numerical checks and exact examples; not the full study")
    mode.add_argument("--all", action="store_true", help="Run every experiment and generate all numerical tables/figures")
    mode.add_argument("--postprocess", action="store_true", help="Rebuild summaries/tables/figures from an already completed full run")
    mode.add_argument("--step", choices=list(STEPS), help="Run one named stage; earlier dependencies must already exist")
    parser.add_argument("--jobs", type=int, default=1, help="Independent pilot jobs (default 1); every solver uses one BLAS thread")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    steps = VALIDATE if args.validate else FULL if args.all else POSTPROCESS if args.postprocess else [args.step]
    versions = check_dependencies()
    for directory in ["results", "figures", "tables", "logs"]:
        (ROOT / directory).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    record = {"started_utc": stamp, "python": platform.python_version(),
              "platform": platform.platform(), "dependencies": versions,
              "pilot_process_jobs": args.jobs, "blas_threads": 1, "steps": []}
    summary_path = ROOT / "logs" / f"run_{stamp}.json"
    start = perf_counter()
    for index, name in enumerate(steps, 1):
        script, extra = STEPS[name]
        command = [sys.executable, str(ROOT / "code" / script), *extra]
        if name == "pilot":
            command += ["--workers", str(args.jobs)]
        print(f"[{index}/{len(steps)}] {name}", flush=True)
        step_start = perf_counter()
        log_path = ROOT / "logs" / f"{stamp}_{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        record["steps"].append({"name": name, "script": script, "arguments": command[2:],
                                "elapsed_seconds": perf_counter() - step_start,
                                "returncode": returncode, "log": str(log_path.relative_to(ROOT))})
        record["elapsed_seconds"] = perf_counter() - start
        record["status"] = "running" if returncode == 0 else "failed"
        summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if returncode != 0:
            raise SystemExit(f"Stopped at {name}. See {log_path}. Earlier completed outputs remain available.")
    if args.all:
        record["full_output_record_counts"] = verify_full_outputs()
    record["status"] = "passed"
    record["elapsed_seconds"] = perf_counter() - start
    summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(steps)} stages in {record['elapsed_seconds']:.1f} seconds.", flush=True)
    print(f"Run record: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
