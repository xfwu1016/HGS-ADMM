"""Run validation, complete manuscript experiments, or output generation.

All numerical algorithms and experiment settings remain in the original
scripts under code/. This entry point supplies directories and execution order.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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
    "reduced-checks": ("check_reduced_state.py", []),
    "solver-checks": ("validate_solver.py", []),
    "input-checks": ("check_solver_inputs.py", []),
    "budget-checks": ("check_pilot_budget_theory.py", []),
    "analytic": ("analytic_partition.py", []),
    "partition-checks": ("verify_partition_examples.py", []),
    "oracle-check": ("oracle_stopping_example.py", []),
    "replicate-calibration": ("replicate_calibration.py", []),
    "residual-gap-bridge": ("residual_gap_bridge.py", []),
    "regrouping": ("regrouping_sensitivity.py", []),
    "layout-choice": ("layout_choice.py", []),
    "fixed16-bound-audit": ("fixed16_bound_audit.py", []),
    "heldout-metric-race": ("heldout_metric_race.py", []),
    "regrouping-summary": ("regrouping_sensitivity.py", ["--summarize-only"]),
    "layout-summary": ("layout_choice.py", ["--summarize-only"]),
    "publication-excerpts": ("publication_excerpts.py", []),
    "main": ("run_experiments.py", []),
    "consensus": ("consensus_baseline.py", []),
    "sensitivity": ("penalty_sensitivity.py", []),
    "pilot": ("pilot_budget_experiment.py", []),
    "pilot-box": ("pilot_box.py", []),
    "pilot-length": ("pilot_length.py", []),
    "pilot-continuation": ("pilot_continuation.py", []),
    "matched-control": ("matched_output_control.py", [
        "--code-root", ".", "--archive-results", "results",
        "--output", "results/matched_output_control.csv",
        "--retained-output", "results/retained_output_control.csv",
    ]),
    "pilot-summary": ("pilot_budget_experiment.py", ["--rebuild-summaries"]),
    "fused-dual-ranges": ("fused_dual_ranges.py", []),
    "rank-diagnostics": ("referee_diagnostics.py", []),
    "traffic-audit": ("referee_traffic.py", []),
    "loose-stopping": ("stopping_sensitivity.py", []),
    "lasso-scale": ("lasso_stopping_scale.py", []),
    "centralized-benchmark": ("centralized_benchmark.py", []),
    "repeated-real": ("repeated_real_splits.py", []),
    "repeated-real-summary": ("repeated_real_splits.py", ["--summarize-only"]),
    "tcp-replay": ("tcp_transport_replay.py", []),
    "tcp-diabetes-lan": ("tcp_distributed_solver.py", ["--datasets", "diabetes", "--seeds", "2026", "2027", "2028", "2029", "2030", "--profiles", "lan", "--output", "results/tcp_diabetes_lan_repeated.csv"]),
    "tcp-diabetes-wan": ("tcp_distributed_solver.py", ["--datasets", "diabetes", "--seeds", "2026", "--profiles", "wan", "--output", "results/tcp_diabetes.csv"]),
    "tcp-breast-lan": ("tcp_distributed_solver.py", ["--datasets", "breast_cancer", "--seeds", "2026", "--profiles", "lan", "--output", "results/tcp_breast_cancer.csv"]),
    "tcp-merge": ("merge_tcp_distributed_results.py", []),
    "extension-outputs": ("extension_outputs.py", []),
    "stopping-summary": ("stopping_summaries.py", []),
    "tables": ("make_tables.py", []),
    "figures": ("make_figures.py", []),
    "workflow-figure": ("make_workflow.py", ["--outdir", "figures"]),
}
MINIMAL = ["budget-checks", "analytic", "partition-checks", "reduced-checks", "oracle-check"]
VALIDATE = ["reduced-checks", "solver-checks", "input-checks", "budget-checks", "analytic", "partition-checks", "oracle-check"]
SUMMARY_STEPS = ["rank-diagnostics", "traffic-audit", "stopping-summary"]
PUBLICATION_STEPS = ["publication-excerpts", "tables", "extension-outputs", "figures", "workflow-figure"]
FULL = VALIDATE + ["main", "fused-dual-ranges", "consensus", "sensitivity", "pilot", "pilot-box",
                   "pilot-length", "pilot-continuation", "matched-control",
                   "loose-stopping", "lasso-scale", "replicate-calibration", "residual-gap-bridge",
                   "centralized-benchmark", "regrouping", "repeated-real", "tcp-diabetes-lan", "tcp-diabetes-wan", "tcp-breast-lan", "tcp-merge"] + SUMMARY_STEPS + ["layout-choice", "fixed16-bound-audit", "heldout-metric-race"] + PUBLICATION_STEPS
POSTPROCESS = ["pilot-summary", "pilot-box", "repeated-real-summary"] + SUMMARY_STEPS + ["layout-summary", "fixed16-bound-audit", "regrouping-summary"] + PUBLICATION_STEPS


def check_dependencies(minimal=False):
    versions = {}
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        package, required = line.split("==")
        if minimal and package != "numpy":
            continue
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


def verify_matched_control(rows):
    original = {}
    for name in ("pilot_continuation_runs.csv", "pilot_continuation_public.csv"):
        for saved in read_rows(name):
            if saved["method"] == "unit_continue":
                key = (saved["model"], saved["seed"], saved["row_parts"], saved["column_parts"])
                original[key] = saved
    observed = set()
    for row in rows:
        key = (row["model"], row["seed"], row["sample_blocks"], row["feature_blocks"])
        if key in observed or key not in original:
            raise ValueError(f"Unexpected matched-control configuration {key}")
        observed.add(key)
        for arm in ("learned", "uniform"):
            if not 0 <= float(row[f"{arm}_relative_gap"]) <= float(row["gap_target"]):
                raise ValueError(f"Invalid stopping gap for {arm} in {key}")
            allocation = int(row["learned_allocation_scalars"]) if arm == "learned" else 0
            payload = (int(row[f"{arm}_iterations"]) * int(row["scalars_per_iteration"])
                       + allocation + int(row["refresh_scalars"])
                       + int(row[f"{arm}_checks"]) * int(row["scalars_per_gap_check_bound"]))
            if payload != int(row[f"{arm}_scalars"]):
                raise ValueError(f"Inconsistent payload for {arm} in {key}")
        if row["learned_scalars"] != original[key]["payload_with_certificate_bound"]:
            raise ValueError(f"Learned-weight payload differs from the continuation run for {key}")
    if observed != set(original) or len(observed) != 49:
        raise ValueError("Incomplete matched-control comparison")


def verify_retained_control(rows):
    previous = read_rows("matched_output_control.csv")
    if len(rows) != len(previous):
        raise ValueError("Retained-state and two-arm controls differ in size")
    for saved, row in zip(previous, rows):
        if any(row[key] != value for key, value in saved.items()):
            raise ValueError("Previously reported arm differs in the retained-state control")
        allocation = int(row["centered_allocation_scalars"])
        total = (int(row["centered_iterations"]) * int(row["scalars_per_iteration"])
                 + allocation + int(row["refresh_scalars"])
                 + int(row["centered_checks"]) * int(row["scalars_per_gap_check_bound"]))
        if (allocation != int(row["learned_allocation_scalars"]) or total != int(row["centered_scalars"])
                or not 0 <= float(row["centered_relative_gap"]) <= float(row["gap_target"])):
            raise ValueError("Retained-state stopping or communication accounting differs")


def verify_full_outputs():
    expected = {
        "runs.csv": 138,
        "repeated_real_splits.csv": 40,
        "tcp_distributed_solver.csv": 14,
        "replicate_calibration.csv": 5,
        "replicate_calibration_runs.csv": 30,
        "residual_gap_bridge_summary.csv": 6,
        "oracle_stopping_example.csv": 80,
        "layout_choice_fixed16.csv": 90,
        "fixed16_bound_audit.csv": 90,
        "fixed16_bound_ranks.csv": 30,
        "heldout_metric_race.csv": 72,
        "heldout_metric_race_summary.csv": 7,
        "heldout_geometry_choice.csv": 24,
        "matched_storage_tuning.csv": 72,
        "matched_storage_test.csv": 48,
        "layout_choice_cases.csv": 79,
        "layout_choice_weight_pairs.csv": 45,
        "layout_choice_summary.csv": 19,
        "regrouping_runs.csv": 80,
        "regrouping_paired.csv": 60,
        "regrouping_permutations.csv": 900,
        "regrouping_summary.csv": 12,
        "fused_dual_ranges.csv": 40,
        "fused_dual_fixed16_extra.csv": 20,
        "fused_dual_primary_pairs.csv": 140,
        "fused_dual_fixed16_pairs.csv": 75,
        "stopping_loose_fixed.csv": 136,
        "stopping_loose_consensus.csv": 30,
        "stopping_loose_continuation.csv": 245,
        "stopping_loose_matched.csv": 49,
        "lasso_stopping_scale.csv": 40,
        "centralized_benchmark.csv": 60,
        "consensus_baseline.csv": 30,
        "penalty_sensitivity.csv": 27,
        "pilot_budget_runs.csv": 135,
        "pilot_budget_pilots.csv": 45,
        "pilot_length.csv": 36,
        "pilot_box.csv": 45,
        "pilot_continuation_runs.csv": 225,
        "pilot_continuation_public.csv": 20,
        "matched_output_control.csv": 49,
        "retained_output_control.csv": 49,
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
        if name == "matched_output_control.csv":
            verify_matched_control(rows)
        if name == "tcp_distributed_solver.csv":
            sys.path.insert(0, str(ROOT / 'code'))
            from merge_tcp_distributed_results import validate_records
            validate_records(rows)
        if name == "retained_output_control.csv":
            verify_retained_control(rows)
        if name.startswith("pilot_continuation_"):
            if any(float(row["relative_gap"]) > float(row["gap_target"]) for row in rows):
                raise RuntimeError(f"{name}: at least one method missed its common gap target")
    for stem in ["fig1_factor_structure", "fig2_convergence", "fig3_communication", "fig_alg1_workflow"]:
        for extension in ["pdf", "png", "svg"]:
            path = ROOT / "figures" / f"{stem}.{extension}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing figure output: {path.name}")
    for name in ["hybrid_comparison", "consensus_comparison", "pilot_budget_results",
                 "real_results", "stopping_results", "analytic_results", "additional_results",
                 "continuation_results", "rank_diagnostics", "paired_strata",
                 "hybrid_accounting", "consensus_accounting", "model_settings",
                 "tolerance_rankings", "full_synthetic", "appendix_diagnostics",
                 "layout_choice", "supp_fused_widths", "experiment_timing_text",
                 "supp_fixed16_bound_audit", "supp_sensitivity_tables", "experiment_residual_validation",
                 "heldout_metric_race", "matched_storage_test", "residual_gap_bridge",
                 "replicate_calibration_rows", "table_repeated_real_splits",
                 "table_tcp_distributed", "table_tcp_distributed_repeated"]:
        path = ROOT / "tables" / f"{name}.tex"
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing numerical table: {path.name}")
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-minimal", action="store_true", help="Check exact examples, budgets and reduced-state identities using only NumPy")
    mode.add_argument("--validate", action="store_true", help="Run numerical checks and exact examples; not the full study")
    mode.add_argument("--all", action="store_true", help="Run every experiment and generate all numerical tables/figures")
    mode.add_argument("--postprocess", action="store_true", help="Rebuild summaries/tables/figures from an already completed full run")
    mode.add_argument("--step", choices=list(STEPS), help="Run one named stage; earlier dependencies must already exist")
    parser.add_argument("--jobs", type=int, default=1, help="Independent pilot jobs (default 1); every solver uses one BLAS thread")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    steps = MINIMAL if args.validate_minimal else VALIDATE if args.validate else FULL if args.all else POSTPROCESS if args.postprocess else [args.step]
    versions = check_dependencies(minimal=args.validate_minimal or args.step in MINIMAL)
    for directory in ["results", "figures", "tables", "logs"]:
        (ROOT / directory).mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    check_output = None
    if args.validate or args.validate_minimal:
        check_output = ROOT / "results_check" / stamp
        check_output.mkdir(parents=True)
        env["CSDA_CHECK_OUTPUT"] = str(check_output)
        print(f"Writing validation outputs to {check_output}", flush=True)
    else:
        env.pop("CSDA_CHECK_OUTPUT", None)
    record = {"started_utc": stamp, "python": platform.python_version(),
              "platform": platform.platform(), "dependencies": versions,
              "pilot_process_jobs": args.jobs, "blas_threads": 1, "steps": []}
    if check_output is not None:
        record["check_output"] = str(check_output.relative_to(ROOT))
    summary_path = ROOT / "logs" / f"run_{stamp}.json"
    start = perf_counter()
    for index, name in enumerate(steps, 1):
        script, extra = STEPS[name]
        script_path = ROOT / "code" / script
        script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
        command = [sys.executable, str(script_path), *extra]
        if name == "residual-gap-bridge":
            command += ["--jobs", str(args.jobs)]
        if name in {"pilot", "pilot-continuation", "matched-control", "loose-stopping"}:
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
        record["steps"].append({"name": name, "script": script, "script_sha256": script_hash, "arguments": command[2:],
                                "elapsed_seconds": perf_counter() - step_start,
                                "returncode": returncode, "log": str(log_path.relative_to(ROOT))})
        record["elapsed_seconds"] = perf_counter() - start
        record["status"] = "running" if returncode == 0 else "failed"
        summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if returncode != 0:
            raise SystemExit(f"Stopped at {name}. See {log_path}. Earlier completed outputs remain available.")
    if args.all:
        record["full_output_record_counts"] = verify_full_outputs()
    elif args.step == "matched-control":
        verify_matched_control(read_rows("matched_output_control.csv"))
        verify_retained_control(read_rows("retained_output_control.csv"))
    record["status"] = "passed"
    record["elapsed_seconds"] = perf_counter() - start
    summary_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(steps)} stages in {record['elapsed_seconds']:.1f} seconds.", flush=True)
    print(f"Run record: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
