"""Compare layout choices within samples using a common numerical reference.

The fixed-16 grid uses the unit and uniform-budget arms of the pilot experiment.
A single high-accuracy primal/dual pair per sample supplies every distance in
that grid, including the structural multipliers in the fused regression. Other
quantities come from the main runs and the first-hit communication accounting.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid_solver import GraphSplitting, make_regression
from run_experiments import distance_diagnostics, split_theta, synthetic_case

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
KEY = ["case", "row_parts", "column_parts", "metric"]
MODELS = ["fused", "huber", "logistic"]
OUTCOMES = ["iterations", "update_payload", "total_payload_bound"]
SETS = {
    "primary": [(1, 1), (4, 1), (4, 4), (8, 4)],
    "hybrid": [(4, 4), (8, 4)],
    "fixed16": [(2, 8), (4, 4), (8, 2)],
}


def model_name(case):
    if case.startswith("synthetic_"):
        return case.split("_")[1]
    return case.removeprefix("real_")


def fixed16_distances():
    archive = pd.read_csv(RESULTS / "pilot_budget_runs.csv")
    active = archive[archive.metric.isin(["unit", "budget"])]
    records = []
    for case, arms in active.groupby("case"):
        seed = int(arms.seed.iloc[0])
        data = synthetic_case(model_name(case), seed)
        design, response = data["X"], data["y"]

        def problem(m, j):
            return make_regression(design, response, m, j, **data["args"])

        reference = GraphSplitting(problem(1, 1), weights="unit", rho=1)
        fit = reference.solve(tol=1e-10, maxiter=50000)
        if not fit["converged"]:
            raise RuntimeError(f"Reference missed its target for {case}")
        with np.load(RESULTS / "coefficients" / f"{case}_reference.npz") as stored:
            np.testing.assert_allclose(fit["coef"], stored["coef"], atol=1e-9, rtol=1e-9)
        for arm in arms.itertuples():
            solver = GraphSplitting(problem(arm.row_parts, arm.column_parts), weights=arm.metric, rho=1)
            theta = split_theta(reference, arm.row_parts)
            diag = distance_diagnostics(solver, fit["coef"], theta)
            records.append(dict(case=case, row_parts=arm.row_parts, column_parts=arm.column_parts,
                                metric=arm.metric, c0=diag["distance_constant_approx"],
                                iterations=arm.main_iterations, relative_gap=arm.relative_gap))
    return pd.DataFrame(records)


def add_costs(grid, filename):
    costs = pd.read_csv(RESULTS / filename)
    costs = costs[costs.zero_edges.eq(0) & costs.metric.isin(["unit", "budget"])]
    fields = KEY + ["iterations", "update_payload", "total_payload_bound"]
    costs = costs[fields].rename(columns={"iterations": "charged_iterations"})
    merged = grid.merge(costs, on=KEY, how="left", validate="one_to_one")
    if not merged.iterations.eq(merged.charged_iterations).all():
        raise ValueError("First-hit iterations from the solver and accounting differ")
    return merged


def check_overlap(primary, fixed):
    common = primary.merge(fixed, on=KEY, suffixes=("_primary", "_fixed"), validate="one_to_one")
    if len(common) != 30 or not (common.row_parts.eq(4) & common.column_parts.eq(4)).all():
        raise ValueError("Expected two metrics on fifteen shared 4-by-4 cases")
    np.testing.assert_allclose(common.c0_primary, common.c0_fixed, rtol=1e-8)
    for field in OUTCOMES:
        np.testing.assert_array_equal(common[field + "_primary"], common[field + "_fixed"])


def labels(group):
    return "|".join(sorted(f"{r.row_parts}x{r.column_parts}_{r.metric}" for r in group.itertuples()))


def summarize_case(case, arms, candidate_set, metric, choices):
    metrics = ["unit", "budget"] if metric == "both" else [metric]
    expected = {(m, j, w) for m, j in choices for w in metrics}
    if len(arms) != len(expected) or set(zip(arms.row_parts, arms.column_parts, arms.metric)) != expected:
        raise ValueError(f"Incomplete {candidate_set} choice set for {case}, {metric}")
    min_c0 = arms[arms.c0.eq(arms.c0.min())]
    row = dict(case=case, model=model_name(case), candidate_set=candidate_set, metric=metric,
               choices=len(arms), min_c0_arms=labels(min_c0))
    for outcome in OUTCOMES:
        opt = arms[arms[outcome].eq(arms[outcome].min())]
        # Tied scores keep their average ranks. A tied C0 minimum is recorded in full.
        row["spearman_" + outcome] = arms.c0.rank().corr(arms[outcome].rank())
        row["min_" + outcome + "_arms"] = labels(opt)
        row["min_c0_matches_" + outcome] = bool(opt.index.intersection(min_c0.index).size)
        row["best_excess_" + outcome + "_pct"] = 100 * (min_c0[outcome].min() / opt[outcome].min() - 1)
    return row


def summarize(primary, fixed):
    cases = []
    for tag, grid in [("primary", primary), ("hybrid", primary[primary.column_parts.gt(1)]), ("fixed16", fixed)]:
        for case, arms in grid.groupby("case"):
            cases.append(summarize_case(case, arms, tag, "both", SETS[tag]))
            if tag == "fixed16":
                for metric, subset in arms.groupby("metric"):
                    cases.append(summarize_case(case, subset, tag, metric, SETS[tag]))
    cases = pd.DataFrame(cases)
    cases.to_csv(RESULTS / "layout_choice_cases.csv", index=False)
    summary = []
    for (model, group, metric), rows in cases.groupby(["model", "candidate_set", "metric"]):
        record = dict(model=model, candidate_set=group, metric=metric, samples=len(rows))
        for outcome in OUTCOMES:
            rho = rows["spearman_" + outcome]
            for label, value in [("median", rho.median()), ("min", rho.min()), ("max", rho.max())]:
                record[f"{label}_rho_{outcome}"] = value
            record[f"min_c0_hits_{outcome}"] = rows[f"min_c0_matches_{outcome}"].sum()
        summary.append(record)
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "layout_choice_summary.csv", index=False)
    return summary


def weight_pairs(fixed):
    pairs = []
    for (case, m, j), arms in fixed.groupby(["case", "row_parts", "column_parts"]):
        row = summarize_case(case, arms, "fixed_layout", "both", [(m, j)])
        row.update(row_parts=m, column_parts=j)
        pairs.append(row)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(RESULTS / "layout_choice_weight_pairs.csv", index=False)
    return pairs


def write_table(summary, pairs):
    grouped = summary.set_index(["candidate_set", "metric", "model"])
    lines = [
        "% Generated by code/layout_choice.py.",
        r"\begin{table}[pos=tbp]\centering\small",
        r"\caption{Layout and weight selection at a relative gap of $10^{-6}$.}\label{tab:layout-choice}",
        r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr@{}}\toprule",
        r"Comparison & Fused & Huber & Logistic\\\midrule",
        r"\multicolumn{4}{@{}l}{\textit{A. Median rank correlation with iterations}}\\[2pt]",
    ]
    for group, metric, label in [("primary", "both", "Four layouts, both rules (8)"),
                                 ("hybrid", "both", "Hybrid layouts, both (4)"),
                                 ("fixed16", "both", "16 blocks, both rules (6)"),
                                 ("fixed16", "unit", "16 blocks, unit (3)"),
                                 ("fixed16", "budget", "16 blocks, budget (3)")]:
        cells = [label]
        for model in MODELS:
            value = grouped.loc[(group, metric, model), "median_rho_iterations"]
            cells.append(f"${0 if abs(value)<0.005 else value:.2f}$")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\midrule", r"\multicolumn{4}{@{}l}{\textit{B. Minimum-$C_0$ hits at 16 blocks}}\\[2pt]"]
    for metric, label in [("unit", "Choose layout, unit (5)"), ("budget", "Choose layout, budget (5)"),
                          ("fixed_layout", "Choose rule within layout (15)"),
                          ("both", "Choose layout and rule (5)")]:
        cells = [label]
        for model in MODELS:
            outcomes = ["iterations", "total_payload_bound"]
            if metric == "fixed_layout":
                rows = pairs[pairs.model.eq(model)]
                hits = [rows["min_c0_matches_" + outcome].sum() for outcome in outcomes]
            else:
                record = grouped.loc[("fixed16", metric, model)]
                hits = [record["min_c0_hits_" + outcome] for outcome in outcomes]
            cells.append(f"{int(hits[0])}, {int(hits[1])}")
        lines.append(" & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular*}\par\vspace{3pt}",
              r"\begin{minipage}{\linewidth}\footnotesize\raggedright",
              r"In A, parentheses count candidates. Ties receive average ranks. With three or four "
              r"candidates, the Spearman correlation has limited resolution. In B, the first count "
              r"is for iterations and the second for charged traffic (updates plus gap checks). The "
              r"denominator in parentheses counts five samples, or three layouts on each sample. All fits use $\rho=1$ "
              r"and the projected-multiplier certificate.\end{minipage}\end{table}"]
    TABLES.mkdir(exist_ok=True)
    (TABLES / "layout_choice.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Use the stored fixed-16 reference distances to refresh ranks and tables.")
    args = parser.parse_args()
    runs = pd.read_csv(RESULTS / "runs.csv")
    runs = runs[runs.zero_edges.eq(0)].rename(columns={"distance_constant_approx": "c0"})
    primary = add_costs(runs, "fixed_traffic_accounting.csv")
    path = RESULTS / "layout_choice_fixed16.csv"
    if args.summarize_only:
        fixed = pd.read_csv(path, float_precision="round_trip")
    else:
        fixed = fixed16_distances()
        fixed.to_csv(path, index=False)
    fixed = add_costs(fixed, "pilot_traffic_accounting.csv")
    check_overlap(primary, fixed)
    write_table(summarize(primary, fixed), weight_pairs(fixed))
    print(f"Compared {len(primary)} primary configurations and {len(fixed)} fixed-16 configurations.")


if __name__ == "__main__":
    main()
