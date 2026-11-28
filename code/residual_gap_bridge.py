"""Audit the residual-to-gap bound on the 90 fixed-weight, 16-block fits.

Run with `python code/residual_gap_bridge.py` from the reproduction package.
Every checkpoint follows the original projected-anchor stopping schedule. Both
coefficient outputs are evaluated with the same proximal factor candidate.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from hybrid_solver import GraphSplitting, make_regression
from pilot_budget_experiment import DESIGN
from run_experiments import synthetic_case
from stopping_checks import is_checkpoint


ROOT = Path(__file__).resolve().parents[1]
ACCURACY = DESIGN["main_relative_gap"]
NUMERICAL_TOLERANCE = 1e-11
MODELS = {"fused": "Squared fused", "huber": "Huber", "logistic": "Logistic"}
OUTPUTS = ("consistent", "proximal")


def incidence_constants(solver):
    """Compute the proposition's constants in the solver's scaled coordinates."""
    factor_count = len(solver.problem.factors)
    gamma2 = 1.0 / solver.a.copy()
    extra_proximal_gamma2 = np.zeros(factor_count)
    delta2 = solver.c.copy()
    for edge, (a, j, block) in enumerate(solver.edges):
        norm2 = np.linalg.norm(block, ord=2) ** 2
        gamma2[a] += 1.0 / solver.b[edge] + norm2 / solver.d[edge]
        extra_proximal_gamma2[a] += norm2 / solver.c[j]
        delta2[j] += solver.d[edge] + (solver.a[a] + solver.b[edge]) * norm2

    smooth = np.zeros(factor_count)
    nonsmooth = np.zeros(factor_count)
    for a, factor in enumerate(solver.problem.factors):
        if factor.kind in ("squared", "huber"):
            smooth[a] = factor.scale
        elif factor.kind == "logistic":
            smooth[a] = factor.scale * np.max(factor.response**2) / 4.0
        elif factor.kind == "l1":
            nonsmooth[a] = factor.scale * np.sqrt(factor.size)
        else:
            raise ValueError(f"Unspecified Lipschitz constant for {factor.kind}")
    return gamma2, extra_proximal_gamma2, np.sqrt(delta2), smooth, nonsmooth


def gap_ceiling(solver, constants, primal_residual, dual_residual, proximal):
    gamma2, extra_gamma2, delta, smooth, nonsmooth = constants
    ridge = solver.problem.ridge
    if proximal:
        gamma2 = gamma2 + extra_gamma2
    u = np.sqrt(gamma2 / solver.rho) * primal_residual
    z = np.sqrt(solver.rho) * delta * dual_residual
    v = np.zeros_like(solver.c) if proximal else primal_residual / np.sqrt(solver.rho * solver.c)
    penalty_lipschitz = solver.problem.l1 * np.sqrt(solver.problem.block_sizes)
    return float(
        np.dot(smooth, u**2) / 2.0
        + 2.0 * np.dot(nonsmooth, u)
        + 2.0 * np.dot(penalty_lipschitz, v)
        + np.sum((z + ridge * v) ** 2) / (2.0 * ridge)
    )


def checkpoints(solver, constants, model, seed, sample_blocks, feature_blocks, metric):
    rows = []
    maxiter = DESIGN["main_iteration_cap"]
    for k in range(1, maxiter + 1):
        previous_eta = solver.eta
        proximal_input = previous_eta - solver.u
        xi = solver.prox(proximal_input)
        solver.eta = solver.project(xi)
        primal_step = xi - solver.eta
        solver.u += primal_step
        if not is_checkpoint(k, DESIGN["main_check_every"], maxiter):
            continue

        primal_residual = float(np.sqrt(solver.rho * np.dot(solver.W, primal_step**2)))
        dual_residual = float(np.sqrt(solver.rho * np.dot(solver.W, (previous_eta - solver.eta)**2)))
        q = solver.rho * solver.W * (proximal_input - xi)
        theta = [q[slot] for slot in solver.ys]
        projection_error = max(
            np.max(np.abs(factor.project_dual(value) - value))
            for factor, value in zip(solver.problem.factors, theta)
        )
        if projection_error > NUMERICAL_TOLERANCE:
            raise ArithmeticError(f"Proximal candidate outside factor dual domain at iteration {k}")

        dual_bound = solver.problem.dual_bound(theta)
        anchor_objective, _, anchor_gap = solver.certificate()
        row = dict(model=model, seed=seed, sample_blocks=sample_blocks,
                   feature_blocks=feature_blocks, metric=metric, iteration=k,
                   primal_residual=primal_residual, dual_residual=dual_residual,
                   proximal_candidate_domain_correction=float(projection_error),
                   anchor_relative_gap=anchor_gap / max(1.0, abs(anchor_objective)))
        for output, state in zip(OUTPUTS, (solver.eta, xi)):
            objective = solver.problem.objective([state[slot] for slot in solver.xs])
            gap = objective - dual_bound
            ceiling = gap_ceiling(solver, constants, primal_residual, dual_residual, output == "proximal")
            scale = max(1.0, abs(objective))
            tolerance = NUMERICAL_TOLERANCE * max(scale, abs(dual_bound), ceiling)
            if not np.isfinite(gap + ceiling) or gap < -tolerance or gap - ceiling > tolerance:
                raise ArithmeticError(
                    f"Gap bridge check failed for {model}, seed {seed}, {sample_blocks}×{feature_blocks}, "
                    f"{metric}, {output}, iteration {k}: gap={gap}, bound={ceiling}"
                )
            row.update({
                f"{output}_objective": objective,
                f"{output}_gap": gap,
                f"{output}_gap_bound": ceiling,
                f"{output}_relative_gap_bound": ceiling / scale,
            })
        rows.append(row)
        if row["anchor_relative_gap"] <= ACCURACY:
            return rows
    raise ArithmeticError(f"Original certificate did not reach its target for {model}, seed {seed}")


def run_sample(task):
    model, seed = task
    with threadpool_limits(limits=1):
        case = synthetic_case(model, seed, n=DESIGN["n"], p=DESIGN["p"])
        fits = []
        for sample_blocks, feature_blocks in DESIGN["layouts"]:
            problem = make_regression(case["X"], case["y"], sample_blocks, feature_blocks, **case["args"])
            for metric in ("unit", "budget"):
                solver = GraphSplitting(problem, rho=DESIGN["rho"], weights=metric)
                fits.append(checkpoints(solver, incidence_constants(solver), model, seed,
                                        sample_blocks, feature_blocks, metric))
        return fits


def configuration_statistics(fits):
    statistics = []
    for model in MODELS:
        model_fits = [fit for fit in fits if fit[-1]["model"] == model]
        for output in OUTPUTS:
            finals = [fit[-1] for fit in model_fits]
            ratios = [
                row[f"{output}_gap_bound"] / row[f"{output}_gap"]
                for row in finals
                if row[f"{output}_gap"] > NUMERICAL_TOLERANCE
            ]
            statistics.append(dict(
                model=model, output=output, fits=len(model_fits),
                checkpoints=sum(map(len, model_fits)),
                terminal_ratios=len(ratios),
                median_terminal_bound_over_gap=float(np.median(ratios)),
                max_terminal_bound_over_gap=max(ratios),
                max_gap_minus_bound=max(
                    row[f"{output}_gap"] - row[f"{output}_gap_bound"]
                    for fit in model_fits for row in fit
                ),
                target_reached_by_bound=sum(
                    any(row[f"{output}_relative_gap_bound"] <= ACCURACY for row in fit)
                    for fit in model_fits
                ),
            ))
    return statistics


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path, summary):
    rows = [
        "% Generated by code/residual_gap_bridge.py.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Numerical check of the residual-to-gap bound. Ratios are medians of bound divided by proximal-candidate gap at the original stopping checkpoint. Reached counts runs in which the bound attained the relative target by that checkpoint.}",
        r"\label{tab:residual-gap-audit}",
        r"\small",
        r"\begin{tabular}{@{}lrrrr@{}}\toprule",
        r"& \multicolumn{2}{c}{Consistent} & \multicolumn{2}{c}{Proximal} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Model & Ratio & Reached & Ratio & Reached \\",
        r"\midrule",
    ]
    for model, label in MODELS.items():
        line = [label]
        for entry in summary:
            if entry["model"] == model:
                ratio = entry["median_terminal_bound_over_gap"]
                mantissa, exponent = f"{ratio:.2e}".split("e")
                line.extend([
                    rf"${mantissa}\times 10^{{{int(exponent)}}}$",
                    f"{entry['target_reached_by_bound']}/{entry['fits']}",
                ])
        rows.append(" & ".join(line) + r" \\")
    rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(rows), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1, help="Independent samples evaluated in parallel")
    args = parser.parse_args()
    tasks = list(product(MODELS, DESIGN["seeds"]))
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.jobs == 1:
        batches = list(map(run_sample, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            batches = list(pool.map(run_sample, tasks))
    fits = [fit for batch in batches for fit in batch]
    observations = [row for fit in fits for row in fit]
    summary = configuration_statistics(fits)

    results = ROOT / "results"
    tables = ROOT / "tables"
    results.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    write_csv(results / "residual_gap_bridge.csv", observations)
    write_csv(results / "residual_gap_bridge_summary.csv", summary)
    write_tex(tables / "residual_gap_bridge.tex", summary)
    max_correction = max(row["proximal_candidate_domain_correction"] for row in observations)
    print(f"Checked {len(fits)} fits, {len(observations)} checkpoints and both outputs.")
    print(f"Largest numerical candidate-domain correction: {max_correction:.3g}.")
    for entry in summary:
        print(entry)


if __name__ == "__main__":
    main()
