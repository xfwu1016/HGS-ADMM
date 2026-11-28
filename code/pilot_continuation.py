"""Paired comparison of restart and continuation after output-weight selection.

This experiment uses the original synthetic samples, layouts, and gap target.
The switch keeps the feasible consensus vector and unscaled multiplier. All
settings are applied to all cases. Counts include the pilot and allocation.
"""

from __future__ import annotations

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np

from hybrid_solver import GraphSplitting, make_regression
from stopping_checks import is_checkpoint
from pilot_budget_experiment import DESIGN, make_pilot_weights, write_csv
from run_experiments import prepare_real_case, synthetic_case, test_metrics


ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_TOL = DESIGN["main_relative_gap"]
PUBLIC_TOL = 1e-7
PUBLIC_LAYOUTS = ((4, 4), (8, 4))
PUBLIC_CASES = ("diabetes", "breast_cancer")
CAP = DESIGN["main_iteration_cap"]
CHECK = DESIGN["main_check_every"]


def switch_output_metric(solver, weights):
    """Change weights once while preserving the original primal and dual state."""
    target = GraphSplitting(
        solver.problem, rho=solver.rho, weights="pilot", custom_weights=weights
    )
    unscaled_dual = solver.rho * solver.W * solver.u
    target.eta[:] = solver.eta
    target.u[:] = unscaled_dual / (target.rho * target.W)
    np.testing.assert_allclose(
        target.rho * target.W * target.u, unscaled_dual, rtol=5e-15, atol=1e-15
    )
    primal_error, dual_error = target.invariant_error()
    if max(primal_error, dual_error) > 1e-9:
        raise ArithmeticError("Metric transition violated a state invariant")
    return target


def step(solver):
    local = solver.prox(solver.eta - solver.u)
    updated = solver.project(local)
    solver.u += local - updated
    solver.eta = updated


def checkpoint(solver):
    primal, dual, gap = solver.certificate()
    return primal, dual, gap, gap / max(1.0, abs(primal))


def run_prefix(problem, metric, stop_at, tolerance):
    start = perf_counter()
    solver = GraphSplitting(problem, rho=DESIGN["rho"], weights=metric)
    setup_seconds = perf_counter() - start
    trace = []
    work_start = perf_counter()
    for k in range(1, stop_at + 1):
        step(solver)
        if is_checkpoint(k, CHECK, stop_at):
            trace.append((k, *checkpoint(solver)))
            if trace[-1][-1] <= tolerance:
                break
    return solver, trace, setup_seconds, perf_counter() - work_start


def solve_from(solver, history, remaining, tolerance):
    start = perf_counter()
    completed = history[-1][0] if history else 0
    new_checks = []
    for offset in range(1, remaining + 1):
        step(solver)
        k = completed + offset
        if is_checkpoint(k, CHECK, completed + remaining):
            new_checks.append((k, *checkpoint(solver)))
            if new_checks[-1][-1] <= tolerance:
                return history + new_checks, perf_counter() - start
    raise RuntimeError("The solve did not reach the common objective gap target")


def record(solver, history, method, prefix_steps, setup_seconds,
           solve_seconds, tolerance, allocation_seconds=0, refactor_changes=0,
           allocated=False, switched=False):
    steps, objective, dual, gap, relative_gap = history[-1]
    if relative_gap > tolerance or any(point[-1] <= tolerance for point in history[:-1]):
        raise RuntimeError("Invalid first-certified checkpoint")
    checkpoints = len(history)
    iteration_payload = solver.communication_per_iteration
    # Each worker uploads a local prediction norm and an operator norm, then
    # receives one selected weight. The gap arrives via the certificate.
    allocation = 3 * len(solver.edges) if allocated else 0
    # Charge a complete refresh of coordinator-to-worker vectors conservatively.
    transition = iteration_payload // 2 if switched else 0
    certificate_per_check = iteration_payload + 3 * (
        len(solver.problem.factors) + len(solver.problem.block_sizes)
    )
    ordinary = steps * iteration_payload + allocation + transition
    primal_error, dual_error = solver.invariant_error()
    return dict(method=method, total_iterations=steps, prefix_iterations=prefix_steps,
                optimized_iterations=steps - prefix_steps, objective=objective, dual=dual,
                absolute_gap=gap, relative_gap=relative_gap, gap_target=tolerance,
                iteration_payload=iteration_payload, allocation_payload=allocation,
                transition_payload=transition,
                optimization_and_allocation_payload=ordinary, checkpoints=checkpoints,
                certificate_payload_bound=checkpoints * certificate_per_check,
                payload_with_certificate_bound=ordinary + checkpoints * certificate_per_check,
                factorization_passes=2 if allocated else 1,
                refactorized_local_systems=len(solver.edges) if allocated else 0,
                materially_changed_local_systems=refactor_changes,
                setup_seconds=setup_seconds, solve_seconds=solve_seconds,
                allocation_seconds=allocation_seconds,
                primal_invariant_error=primal_error, dual_invariant_error=dual_error)


def evaluate_layout(case, model, m, j, prefix_length, *, tolerance=None):
    seed = case["seed"]
    if tolerance is None:
        tolerance = PUBLIC_TOL if case["kind"] == "real" else SYNTHETIC_TOL
    problem = make_regression(case["X"], case["y"], m, j, **case["args"])
    records = []

    def add_result(solver, result):
        beta = np.concatenate([solver.eta[s] for s in solver.xs])
        records.append(dict(result, **test_metrics(beta, case)))

    for metric in ("unit", "budget"):
        prefix, history, setup, work = run_prefix(problem, metric, prefix_length, tolerance)
        if history[-1][-1] <= tolerance:
            add_result(prefix, record(prefix, history, metric, 0, setup, work, tolerance))
            early_methods = [metric + "_continue"]
            if metric == "unit":
                early_methods.append("unit_restart")
            for method in early_methods:
                add_result(prefix, record(prefix, history, method, history[-1][0], setup, work, tolerance))
            continue

        p_result = dict(coef=np.concatenate([prefix.eta[s] for s in prefix.xs]),
                        gap=history[-1][3])
        weights, diagnostics = make_pilot_weights(problem, p_result)

        material_changes = int(np.count_nonzero(
            ~np.isclose(prefix.b, weights["b"], rtol=1e-12, atol=0)
            | ~np.isclose(prefix.d, weights["d"], rtol=1e-12, atol=0)
        ))
        start = perf_counter()
        continued = switch_output_metric(prefix, weights)
        continue_setup = perf_counter() - start
        cont_history, continuation_work = solve_from(
            continued, history, CAP - prefix_length, tolerance
        )
        add_result(continued, record(
            continued, cont_history, metric + "_continue", prefix_length,
            setup + continue_setup, work + continuation_work, tolerance,
            diagnostics["allocation_seconds"], material_changes, allocated=True, switched=True
        ))

        if metric == "unit":
            start = perf_counter()
            restarted = GraphSplitting(
                problem, rho=DESIGN["rho"], weights="pilot", custom_weights=weights
            )
            restart_setup = perf_counter() - start
            restart_history, restart_work = solve_from(restarted, [], CAP, tolerance)
            charged_history = history + [(k + prefix_length, *values) for k, *values in restart_history]
            add_result(restarted, record(
                restarted, charged_history, "unit_restart", prefix_length,
                setup + restart_setup, work + restart_work, tolerance,
                diagnostics["allocation_seconds"], material_changes, allocated=True
            ))

        baseline_history, baseline_work = solve_from(prefix, history, CAP - prefix_length, tolerance)
        add_result(prefix, record(
            prefix, baseline_history, metric, 0, setup, work + baseline_work, tolerance
        ))

    if len(records) != 5:
        raise AssertionError("Each case must include five comparators")
    return [dict(model=model, case=case["case"], data=case["kind"], seed=seed, row_parts=m, column_parts=j,
                 requested_prefix_iterations=prefix_length, **r) for r in records]


def evaluate(task):
    model, seed, m, j, prefix_length = task
    return evaluate_layout(synthetic_case(model, seed), model, m, j, prefix_length)


def evaluate_public(which):
    case = prepare_real_case(which, save_split=False)
    return [row for m, j in PUBLIC_LAYOUTS
            for row in evaluate_layout(case, which, m, j, DESIGN["pilot_iterations"])]


def summaries(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["model"], row["row_parts"], row["column_parts"], row["requested_prefix_iterations"])
        groups[key].append(row)
    metrics = ("total_iterations", "optimization_and_allocation_payload", "payload_with_certificate_bound")
    for (model, m, j, prefix), group in sorted(groups.items()):
        for method in sorted({r["method"] for r in group}):
            selected = [r for r in group if r["method"] == method]
            summary = dict(model=model, row_parts=m, column_parts=j,
                           requested_prefix_iterations=prefix, method=method,
                           runs=len(selected))
            for field in metrics:
                v = np.array([r[field] for r in selected])
                summary[field + "_mean"] = float(v.mean())
                summary[field + "_sd"] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
            yield summary


def paired_checks(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = (row["case"], row["row_parts"], row["column_parts"], row["requested_prefix_iterations"])
        groups[key][row["method"]] = row
    count = len(groups)
    if any(len(group) != 5 for group in groups.values()):
        raise AssertionError("A paired case is incomplete")
    summary = {"paired_cases": count, "certified_solves": len(rows),
               "maximum_gap_ratio": max(r["relative_gap"] / r["gap_target"] for r in rows),
               "maximum_state_invariant_error": max(max(r["primal_invariant_error"], r["dual_invariant_error"])
                                                    for r in rows)}
    for field in ("optimization_and_allocation_payload", "payload_with_certificate_bound"):
        summary[field] = {}
        for method in ("unit_restart", "unit_continue", "budget_continue"):
            for baseline in ("unit_restart", "unit", "budget", "better_fixed"):
                if method == baseline:
                    continue
                rates = []
                for group in groups.values():
                    base = (min(group["unit"][field], group["budget"][field])
                            if baseline == "better_fixed" else group[baseline][field])
                    rates.append(group[method][field] / base - 1)
                summary[field][method + "_vs_" + baseline] = {
                    "fewer_scalars": sum(rate < 0 for rate in rates),
                    "equal_scalars": sum(rate == 0 for rate in rates),
                    "more_scalars": sum(rate > 0 for rate in rates),
                    "median_percent_change": 100 * float(np.median(rates)),
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=[DESIGN["pilot_iterations"]])
    parser.add_argument("--seeds", type=int, nargs="+", default=DESIGN["seeds"])
    parser.add_argument("--output-stem", default="pilot_continuation")
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    if any(length <= 0 or length >= CAP or length % CHECK for length in args.lengths):
        parser.error("prefix lengths must be positive checkpoint multiples below the iteration cap")
    tasks = [(model, seed, m, j, length) for model in DESIGN["models"]
             for seed in args.seeds for m, j in DESIGN["layouts"] for length in args.lengths]
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = [row for case in pool.map(evaluate, tasks) for row in case]
        real_rows = [] if args.synthetic_only else [row for case in pool.map(evaluate_public, PUBLIC_CASES)
                                                      for row in case]
    write_csv(results / (args.output_stem + "_runs.csv"), rows)
    write_csv(results / (args.output_stem + "_summary.csv"), list(summaries(rows)))
    summary = {"synthetic": paired_checks(rows)}
    if real_rows:
        write_csv(results / (args.output_stem + "_public.csv"), real_rows)
        summary["public"] = paired_checks(real_rows)
    (results / (args.output_stem + "_checks.json")).write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Completed {len(tasks)} synthetic paired cases and {len(real_rows) // 5} public-data pairs.", flush=True)


if __name__ == "__main__":
    main()
