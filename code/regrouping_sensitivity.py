"""Compare fixed-size row and feature regroupings on the Huber and logistic simulations.

Within each sample, the same prespecified row and column permutations are applied
to both metrics. Reordering the coefficients preserves the regression problem.
Each fit uses the primary stopping schedule. Charged updates per iteration stay
fixed across all four regroupings. Rebuild the derived CSVs with --summarize-only.
"""
import csv
import argparse
from pathlib import Path
import numpy as np

from run_experiments import distance_diagnostics, synthetic_case, split_theta
from hybrid_solver import make_regression, GraphSplitting

RESULTS = Path(__file__).resolve().parents[1] / "results"
GROUPS = 4
ROW_SEED = 61001
COLUMN_SEED = 61002
METRICS = ("unit", "budget")
LAYOUTS = ("original", "rows", "columns", "both")


def write_table(filename, rows):
    with (RESULTS / filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def mean_loss(beta, data):
    y, design, args = data["y"], data["X"], data["args"]
    prediction = design @ beta
    if args["loss"] == "huber":
        residual = np.abs(prediction - y)
        threshold = args["loss_parameter"]
        loss = np.where(residual <= threshold, residual**2 / 2, threshold * (residual - threshold / 2))
    else:
        loss = np.logaddexp(0, -y * prediction)
    return float(np.mean(loss) + args["l1"] * np.abs(beta).sum() + args["ridge"] * np.dot(beta, beta) / 2)


def fixed_permutations(n, p):
    rows = np.random.Generator(np.random.PCG64(ROW_SEED)).permutation(n)
    columns = np.random.Generator(np.random.PCG64(COLUMN_SEED)).permutation(p)
    return rows, columns


def write_permutations(cases):
    with (RESULTS / "runs.csv").open(newline="", encoding="utf-8") as source:
        sizes = sorted({(int(r["n"]), int(r["p"])) for r in csv.DictReader(source)
                        if r["case"] in cases})
    entries = []
    for n, p in sizes:
        rows, columns = fixed_permutations(n, p)
        for axis, order in (("row", rows), ("column", columns)):
            entries.extend(dict(n=n, p=p, axis=axis, new_index=i, original_index=int(j))
                           for i, j in enumerate(order))
    write_table("regrouping_permutations.csv", entries)


def run_case(loss, seed, archived):
    data = synthetic_case(loss, seed)
    x, y, args, name = data["X"], data["y"], data["args"], data["case"]
    n, p = x.shape
    rp, cp = fixed_permutations(n, p)
    one = make_regression(x, y, 1, 1, **args)
    refsolver = GraphSplitting(one, rho=1, weights="unit")
    ref = refsolver.solve(tol=1e-10, maxiter=50000)
    beta, theta = ref["coef"], split_theta(refsolver, 1)[0]
    assert ref["converged"]
    with np.load(RESULTS / "coefficients" / (name + "_reference.npz")) as stored:
        np.testing.assert_allclose(beta, stored["coef"], atol=1e-10, rtol=0)
    original_dual = one.dual_bound([theta])
    rows = []
    for layout in LAYOUTS:
        r = rp if layout in ("rows", "both") else np.arange(n)
        c = cp if layout in ("columns", "both") else np.arange(p)
        problem = make_regression(x[np.ix_(r, c)], y[r], GROUPS, GROUPS, **args)
        mapped_theta = np.split(theta[r], GROUPS)
        mapped_beta = beta[c]
        normalized_x = np.block([[f.maps[j] for j in range(GROUPS)] for f in problem.factors])
        expected_x = x[np.ix_(r, c)] / np.sqrt(n)
        assert np.array_equal(normalized_x, expected_x)
        assert all(f.size == 75 for f in problem.factors) and problem.block_sizes == [150] * GROUPS
        objective_error = max(abs(problem.objective(np.split(v[c], GROUPS)) - mean_loss(v, data))
                              for v in (beta, data["truth"], np.linspace(-.25, .25, p)))
        dual_error = abs(problem.dual_bound(mapped_theta) - original_dual)
        assert max(objective_error, dual_error) < 1e-10
        for metric in METRICS:
            solver = GraphSplitting(problem, rho=1, weights=metric)
            assert len(solver.edges) == 16 and solver.communication_per_iteration == 7200
            diag = distance_diagnostics(solver, mapped_beta, mapped_theta)
            fit = solver.solve(tol=1e-6, check_every=10, maxiter=30000)
            assert fit["converged"]
            final_error = abs(mean_loss(fit["coef"][np.argsort(c)], data) - fit["objective"])
            assert final_error < 1e-10
            if layout == "original":
                match = archived[(name, metric)]
                assert fit["iterations"] == int(match["iterations"])
                for key in ("distance_constant_approx", "prediction_dispersion_H", "gradient_dispersion_G"):
                    np.testing.assert_allclose(diag[key], float(match[key]), rtol=1e-9, atol=1e-11)
            rows.append(dict(case=name, loss=loss, seed=seed, metric=metric, layout=layout,
                             C0=diag["distance_constant_approx"], primal_norm_squared=diag["primal_norm_approx"],
                             dual_norm_squared=diag["dual_norm_approx"], H=diag["prediction_dispersion_H"],
                             G=diag["gradient_dispersion_G"], iterations=fit["iterations"], relative_gap=fit["relative_gap"],
                             ref_relative_gap=ref["relative_gap"], reference_objective_check=objective_error,
                             reference_dual_check=dual_error, final_objective_check=final_error))
    print(name, "checked", flush=True)
    return rows


def check_decomposition(raw):
    for case in sorted({r["case"] for r in raw}):
        values = {(r["layout"], r["metric"]): float(r["C0"]) for r in raw if r["case"] == case}
        for metric in METRICS:
            np.testing.assert_allclose(values["both", metric] + values["original", metric],
                                       values["rows", metric] + values["columns", metric], atol=1e-10, rtol=0)
        for layout in LAYOUTS[1:]:
            unit_change = values[layout, "unit"] - values["original", "unit"]
            budget_change = values[layout, "budget"] - values["original", "budget"]
            np.testing.assert_allclose(budget_change, GROUPS * unit_change, atol=1e-10, rtol=0)


def run_all():
    with (RESULTS / "runs.csv").open(newline="", encoding="utf-8") as file:
        archived = {(r["case"], r["metric"]): r for r in csv.DictReader(file)
                    if r["row_parts"] == r["column_parts"] == "4" and r["zero_edges"] == "0"}
    raw = [row for loss in ("huber", "logistic") for seed in range(5)
           for row in run_case(loss, seed, archived)]
    write_table("regrouping_runs.csv", raw)
    return raw


def summarize(raw):
    paired = []
    original = {(r["case"], r["metric"]): r for r in raw if r["layout"] == "original"}
    for row in raw:
        if row["layout"] == "original":
            continue
        orig = original[row["case"], row["metric"]]
        change = row["C0"] - orig["C0"]
        steps = row["iterations"] - orig["iterations"]
        assert abs(change) > 1e-9 * max(row["C0"], orig["C0"])
        if row["layout"] == "rows":
            np.testing.assert_allclose(row["H"], orig["H"], rtol=1e-11)
        if row["layout"] == "columns":
            np.testing.assert_allclose(row["G"], orig["G"], rtol=1e-11)
        direction = "tied_iterations"
        if steps:
            direction = "same" if np.sign(change) == np.sign(steps) else "opposite"
        paired.append(dict(case=row["case"], loss=row["loss"], seed=row["seed"], metric=row["metric"],
                           layout=row["layout"], delta_C0=change, delta_C0_pct=100 * change / orig["C0"],
                           delta_iterations=steps, delta_iterations_pct=100 * steps / orig["iterations"],
                           direction=direction))
    summary = []
    for loss in ("huber", "logistic"):
        for metric in METRICS:
            for layout in LAYOUTS[1:]:
                group = [r for r in paired if (r["loss"], r["metric"], r["layout"]) == (loss, metric, layout)]
                stat = dict(loss=loss, metric=metric, layout=layout)
                for var in ("delta_C0_pct", "delta_iterations", "delta_iterations_pct"):
                    values = [r[var] for r in group]
                    stat.update({f"{var}_{name}": func(values) for name, func in
                                 (("min", min), ("median", np.median), ("max", max))})
                stat.update({d: sum(r["direction"] == d for r in group) for d in ("same", "opposite", "tied_iterations")})
                summary.append(stat)
    check_decomposition(raw)
    write_permutations({r["case"] for r in raw})
    write_table("regrouping_paired.csv", paired)
    write_table("regrouping_summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true",
                        help="Rebuild paired contrasts and summaries from the saved fits.")
    if parser.parse_args().summarize_only:
        with (RESULTS / "regrouping_runs.csv").open(newline="", encoding="utf-8") as file:
            raw = list(csv.DictReader(file))
        for row in raw:
            for field in ("C0", "H", "G"):
                row[field] = float(row[field])
            row["iterations"] = int(row["iterations"])
    else:
        raw = run_all()
    summarize(raw)


if __name__ == "__main__":
    main()
