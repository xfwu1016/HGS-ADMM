"""Independent layout and weight-policy check on crossed Huber/logistic designs."""
from __future__ import annotations

import csv
import json
from itertools import product
from statistics import median
from time import perf_counter

import numpy as np
from scipy.special import expit

from centralized_benchmark import certificate
from consensus_baseline import consensus_admm
from hybrid_solver import GraphSplitting, make_regression
from pilot_continuation import step
from run_experiments import RESULTS, ar_design

MODELS = ("huber", "logistic")
CORRELATIONS = (0.2, 0.8)
SUPPORTS = ("concentrated", "spread")
LAYOUTS = ((2, 8), (4, 4), (8, 2))
METRICS = ("unit", "budget")
REPLICATES = range(3)
N, P, SPARSITY = 160, 240, 12
PILOT_STEPS, MAIN_TARGET, REFERENCE_TARGET, MAX_STEPS = 30, 1e-6, 1e-11, 30000
RHO_CANDIDATES = (0.25, 1.0, 4.0)
MATCHED_METHODS = ("unit", "budget", "consensus")


def dataset(model, correlation, support, seed):
    rng = np.random.default_rng(seed)
    X = ar_design(rng, N, P, correlation)
    beta = np.zeros(P)
    active = (
        np.arange(SPARSITY)
        if support == "concentrated"
        else np.arange(SPARSITY) * (P // SPARSITY)
    )
    amplitude = 1.0 if model == "huber" else 0.7
    beta[active] = amplitude * np.where(np.arange(SPARSITY) % 2, -1, 1)
    if model == "huber":
        noise = rng.normal(size=N)
        contaminated = rng.random(N) < 0.1
        noise[contaminated] += rng.normal(scale=10, size=contaminated.sum())
        y = X @ beta + noise
    else:
        y = np.where(rng.random(N) < expit(X @ beta), 1, -1)
    args = dict(loss=model, loss_parameter=1.345 if model == "huber" else 1.0,
                l1=0.025 if model == "huber" else 0.02, ridge=0.02)
    return X, y, args


def checkpoints(X, y, args, layout, metric, target, minimum_steps=0, *, rho=1.0):
    solver = GraphSplitting(make_regression(X, y, *layout, **args), rho=rho, weights=metric)
    history = []
    for k in range(1, MAX_STEPS + 1):
        step(solver)
        if k == 1 or k % 10 == 0:
            beta = np.concatenate([solver.eta[index] for index in solver.xs])
            primal, dual, relative_gap = certificate(X, y, beta, args)
            history.append((k, primal, dual, relative_gap))
            if k >= minimum_steps and relative_gap <= target:
                return history, solver.communication_per_iteration
    raise RuntimeError(f"Gap target {target:g} missed at {layout}, {metric}")


def matched_storage_run(X, y, args, method, rho):
    sample_parts = 4
    began = perf_counter()
    if method == "consensus":
        out = consensus_admm(X, y, sample_parts, args, tol=MAIN_TARGET,
                             rho=rho, maxiter=MAX_STEPS, certificate=certificate)
        if not out["converged"]:
            raise RuntimeError("Consensus missed the shared gap target")
        iterations = out["iterations"]
        gap = out["relative_gap"]
        per_update = out["communication_per_iteration"]
        n_checks = out["diagnostic_evaluations"]
        per_check = sample_parts * (X.shape[1] + 3)
    else:
        trace, per_update = checkpoints(X, y, args, (sample_parts, 1), method, MAIN_TARGET, rho=rho)
        iterations, _, _, gap = trace[-1]
        n_checks = len(trace)
        per_check = per_update + 3 * (sample_parts + 1)
    return dict(method=method, rho=rho, iterations=iterations, relative_gap=gap,
                checks=n_checks, per_update=per_update, per_check=per_check,
                payload=iterations * per_update + n_checks * per_check,
                sequential_seconds=perf_counter() - began)


def evaluate_layout(X, y, args, layout, reference):
    result = {}
    for metric in METRICS:
        trace, per_update = checkpoints(X, y, args, layout, metric, MAIN_TARGET, PILOT_STEPS)
        per_check = per_update + 3 * sum(layout)
        pilot_index = next(index for index, point in enumerate(trace) if point[0] == PILOT_STEPS)
        first = next(index for index, point in enumerate(trace) if point[3] <= MAIN_TARGET)
        pilot_cost = PILOT_STEPS * per_update + (pilot_index + 1) * per_check
        # The reference objective and dual bound bracket the true minimizer.
        possible = next(point[0] for point in trace
                        if (point[1] - reference[1]) / max(1, abs(point[1])) <= MAIN_TARGET)
        guaranteed = next(point[0] for point in trace
                          if (point[1] - reference[2]) / max(1, abs(point[1])) <= MAIN_TARGET)
        if not possible <= guaranteed <= trace[first][0]:
            raise ArithmeticError("Inconsistent first-hit bracket")
        result[metric] = dict(iterations=trace[first][0], gap=trace[first][3],
                              fixed_cost=trace[first][0] * per_update + (first + 1) * per_check,
                              pilot_cost=pilot_cost, pilot_gap=trace[pilot_index][3],
                              possible_hit=possible, guaranteed_hit=guaranteed)
    eligible = [metric for metric in METRICS if result[metric]["iterations"] <= PILOT_STEPS]
    if eligible:
        selected = min(eligible, key=lambda metric: result[metric]["fixed_cost"])
    else:
        selected = min(METRICS, key=lambda metric: result[metric]["pilot_gap"])
    other = next(metric for metric in METRICS if metric != selected)
    chosen_cost = max(result[selected]["fixed_cost"], result[selected]["pilot_cost"])
    return dict(sample_blocks=layout[0], feature_blocks=layout[1],
                update_scalars=per_update, check_scalars=per_check,
                selected=selected, race_cost=chosen_cost + result[other]["pilot_cost"],
                **{f"{metric}_{key}": value for metric, record in result.items()
                   for key, value in record.items()})


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, sample_count):
    groups = [
        (model, layout, [row for row in rows if row["model"] == model
                         and (row["sample_blocks"], row["feature_blocks"]) == layout])
        for model, layout in product(MODELS, LAYOUTS)
    ]
    groups.append(("All", None, rows))
    records = []
    for model, layout, group in groups:
        records.append(dict(
            model=model, layout="All" if layout is None else "x".join(map(str, layout)),
            paired_cells=len(group),
            independent_samples=sample_count if layout is None else len(group),
            select_cheaper=sum(row[f'{row["selected"]}_fixed_cost']
                               == min(row["unit_fixed_cost"], row["budget_fixed_cost"]) for row in group),
            race_beats_unit=sum(row["race_cost"] < row["unit_fixed_cost"] for row in group),
            race_beats_budget=sum(row["race_cost"] < row["budget_fixed_cost"] for row in group),
            median_pct_vs_unit=median(100 * (row["race_cost"] / row["unit_fixed_cost"] - 1) for row in group),
            median_pct_vs_budget=median(100 * (row["race_cost"] / row["budget_fixed_cost"] - 1) for row in group),
            median_pct_vs_oracle=median(100 * (row["race_cost"] /
                                               min(row["unit_fixed_cost"], row["budget_fixed_cost"]) - 1)
                                        for row in group),
            exact_objective_hits=sum(row[f"{metric}_possible_hit"] == row[f"{metric}_guaranteed_hit"]
                                     for row in group for metric in METRICS),
        ))
    return records


def write_table(records):
    lines = [
        r"% Generated by code/heldout_metric_race.py.",
        r"\begin{table}[pos=tbp]\centering\small",
        r"\caption{Two-metric pilot on independent designs.}",
        r"\label{tab:heldout-race}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lcrrr@{}}",
        r"\toprule",
        r"Model & Grid & Cheaper & \multicolumn{2}{c}{Median payload change (\%)}\\",
        r" & & chosen & vs. unit & vs. budget\\",
        r"\midrule",
    ]
    for record in records:
        layout = record["layout"].replace("x", r"\times")
        grid = "---" if layout == "All" else "$" + layout + "$"
        picked = f'{record["select_cheaper"]}/{record["paired_cells"]}'
        row = [record["model"].capitalize(), grid, picked,
               f'{record["median_pct_vs_unit"]:+.1f}', f'{record["median_pct_vs_budget"]:+.1f}']
        lines.append(" & ".join(row) + r"\\")
    lines += [
        r"\bottomrule\end{tabular*}\par\vspace{3pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"Each model uses 12 independent datasets. The same datasets are reused across grids.",
        r"The last row pools all 72 paired cells.",
        r"Payload changes are relative to the named fixed rule and include both pilot prefixes.",
        r"Cheaper chosen compares the selected metric's fixed cost with the smaller",
        r"fixed cost on the same dataset and grid, allowing ties.",
        r"\end{minipage}\end{table}",
    ]
    tables = RESULTS.parent / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "heldout_metric_race.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_storage_table(records):
    names = {"unit": "Graph, unit", "budget": "Graph, budget", "consensus": "Sample consensus"}
    lines = [
        r"% Generated by code/heldout_metric_race.py.",
        r"\begin{table}[pos=tbp]\centering\small",
        r"\caption{Matched-storage comparison at a relative gap of $10^{-6}$.}",
        r"\label{tab:heldout-storage}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
        r"\toprule",
        r"Method & $\rho$ & Iterations & \makecell{Total\\scalars} & Time (ms)\\",
        r"\midrule",
    ]
    for model in MODELS:
        lines.append(r"\multicolumn{5}{@{}l}{\textit{" + model.capitalize() + r"}}\\")
        for method in MATCHED_METHODS:
            group = [row for row in records if row["model"] == model and row["method"] == method]
            rho = group[0]["rho"]
            values = [median(row["iterations"] for row in group),
                      median(row["payload"] for row in group) / 1e6,
                      1e3 * median(row["sequential_seconds"] for row in group)]
            lines.append(f"{names[method]} & {rho:g} & {values[0]:g} & {values[1]:.3f} & {values[2]:.0f}"
                         + r"\\")
    lines += [
        r"\bottomrule\end{tabular*}\par\vspace{3pt}",
        r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
        r"Medians over eight held-out datasets per model, using four sample blocks",
        r"and the same loss-gradient certificate. Total scalars include updates and",
        r"certificate bounds, in millions. Single-process time includes setup",
        r"and certificates. Tuning used separate datasets and the common grid $\{0.25,1,4\}$; all selections reached its lower boundary.",
        r"\end{minipage}\end{table}",
    ]
    tables = RESULTS.parent / "tables"
    tables.mkdir(exist_ok=True)
    (tables / "matched_storage_test.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    rows, sample_choices, matched_tuning, matched_test = [], [], [], []
    test_designs = []
    for index, (model, correlation, support, replicate) in enumerate(
            product(MODELS, CORRELATIONS, SUPPORTS, REPLICATES)):
        seed = 20261200 + index
        X, y, args = dataset(model, correlation, support, seed)
        references, _ = checkpoints(X, y, args, (1, 1), "unit", REFERENCE_TARGET)
        reference = references[-1]
        design = dict(model=model, correlation=correlation, support=support, replicate=replicate,
                      seed=seed, reference_primal=reference[1], reference_dual=reference[2])
        if replicate == 0:
            matched_tuning.extend(dict(**design, **matched_storage_run(X, y, args, method, rho))
                                  for method, rho in product(MATCHED_METHODS, RHO_CANDIDATES))
        else:
            test_designs.append(design)
        sample_rows = [dict(**design, **evaluate_layout(X, y, args, layout, reference))
                       for layout in LAYOUTS]
        rows.extend(sample_rows)
        geometry = min(sample_rows, key=lambda row: row["update_scalars"])
        oracle = min(row[f"{metric}_fixed_cost"] for row in sample_rows for metric in METRICS)
        sample_choices.append(dict(**design, selected_layout=f'{geometry["sample_blocks"]}x{geometry["feature_blocks"]}',
                                   unit_cost=geometry["unit_fixed_cost"], budget_cost=geometry["budget_fixed_cost"],
                                   race_cost=geometry["race_cost"], hindsight_cost=oracle))
        print(f"{model} {correlation:.1f} {support} replicate {replicate} finished", flush=True)
    selected_rho = {}
    for model, method in product(MODELS, MATCHED_METHODS):
        def tuning_cost(rho):
            return median(row["payload"] for row in matched_tuning
                          if row["model"] == model and row["method"] == method and row["rho"] == rho)
        selected_rho[model, method] = min(RHO_CANDIDATES, key=tuning_cost)
    for design in test_designs:
        model = design["model"]
        X, y, args = dataset(model, design["correlation"], design["support"], design["seed"])
        matched_test.extend(dict(**design, **matched_storage_run(X, y, args, method, selected_rho[model, method]))
                            for method in MATCHED_METHODS)
    summaries = summarize(rows, len(sample_choices))
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "heldout_metric_race.csv", rows)
    write_csv(RESULTS / "heldout_metric_race_summary.csv", summaries)
    write_csv(RESULTS / "heldout_geometry_choice.csv", sample_choices)
    write_csv(RESULTS / "matched_storage_tuning.csv", matched_tuning)
    write_csv(RESULTS / "matched_storage_test.csv", matched_test)
    write_table(summaries)
    write_storage_table(matched_test)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
