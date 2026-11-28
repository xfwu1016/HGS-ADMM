"""Repeat main fixed, consensus and continuation comparisons at relative gap 1e-3.

The loose fixed-weight checkpoints must match the ones stored by the original
strict solves. Times are excluded from these endpoint comparisons.
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from consensus_baseline import consensus_admm
from hybrid_solver import GraphSplitting, make_regression
from pilot_continuation import CAP, evaluate_layout, run_prefix, solve_from, switch_output_metric
from run_experiments import LAYOUTS, WEIGHTS, synthetic_case, test_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOL = .001
GRIDS = ((2, 8), (4, 4), (8, 2))


def load_rows(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_rows(name, rows):
    with (RESULTS / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_case(model, seed):
    if model in ("fused", "huber", "logistic"):
        return synthetic_case(model, seed)
    case_name = "real_" + model
    stored = json.loads((RESULTS / "cases.json").read_text())
    model_args = next(case["args"] for case in stored if case["case"] == case_name)
    with np.load(RESULTS / "datasets" / (model + "_split.npz")) as data:
        return dict(case=case_name, seed=seed, kind="real", args=model_args,
                    X=data["X"], y=data["y"], Xtest=data["Xtest"], ytest=data["ytest"],
                    target_scale=float(data["target_scale"]))


def matched_uniform_switch(problem, learned):
    """Continue the same unit pilot under prespecified uniform budget weights."""
    prefix, history, _, _ = run_prefix(problem, "unit", 100, TOL)
    refresh = 0
    if history[-1][-1] > TOL:
        uniform = dict(a=prefix.a.copy(), c=prefix.c.copy(),
                       b=np.array([prefix.da[a] for a, _, _ in prefix.edges], dtype=float),
                       d=np.array([1./prefix.dj[j] for _, j, _ in prefix.edges]))
        state = switch_output_metric(prefix, uniform)
        refresh = prefix.communication_per_iteration // 2
        history, _ = solve_from(state, history, CAP-100, TOL)
    steps, _, _, _, gap = history[-1]
    c_iter = prefix.communication_per_iteration
    per_check = c_iter + 3*(len(problem.factors) + len(problem.block_sizes))
    payload = steps*c_iter + refresh + len(history)*per_check
    return dict(model=learned["model"], case=learned["case"], data=learned["data"], seed=learned["seed"],
                row_parts=learned["row_parts"], column_parts=learned["column_parts"], tol=TOL,
                uniform_iterations=steps, learned_iterations=learned["total_iterations"],
                uniform_relative_gap=gap, learned_relative_gap=learned["relative_gap"], uniform_checks=len(history),
                uniform_refresh=refresh, uniform_payload=payload, learned_payload=learned["payload_with_certificate_bound"])


def evaluate(task):
    model, seed = task
    case = load_case(model, seed)
    public = case["kind"] == "real"
    fixed, consensus, continuation, matched = [], [], [], []
    for m, j in LAYOUTS:
        for metric in WEIGHTS:
            prob = make_regression(case["X"], case["y"], m, j, **case["args"])
            obj = GraphSplitting(prob, rho=1, weights=metric)
            result = obj.solve(tol=TOL, maxiter=30000)
            if not result["converged"]:
                raise RuntimeError(case["case"])
            c_iter = obj.communication_per_iteration
            c_check = c_iter + 3 * (len(prob.factors) + j)
            fixed.append(dict(model=model, case=case["case"], seed=seed, row_parts=m,
                              column_parts=j, metric=metric, tol=TOL,
                              iterations=result["iterations"], relative_gap=result["relative_gap"],
                              communicated_scalars=result["communicated_scalars"],
                              checks=result["diagnostic_evaluations"],
                              payload_bound=result["communicated_scalars"] + result["diagnostic_evaluations"] * c_check,
                              **test_metrics(result["coef"], case)))
    if not public:
        for m in (4, 8):
            result = consensus_admm(case["X"], case["y"], m, case["args"], tol=TOL, rho=1)
            if not result["converged"]:
                raise RuntimeError(case["case"])
            consensus.append(dict(model=model, case=case["case"], seed=seed, row_parts=m,
                                  tol=TOL, iterations=result["iterations"], relative_gap=result["relative_gap"],
                                  communicated_scalars=result["communicated_scalars"],
                                  checks=result["diagnostic_evaluations"], **test_metrics(result["coef"], case)))
    for m, j in (((4, 4), (8, 4)) if public else GRIDS):
        rows = evaluate_layout(case, model, m, j, 100, tolerance=TOL)
        prob = make_regression(case["X"], case["y"], m, j, **case["args"])
        learned = next(r for r in rows if r["method"] == "unit_continue")
        matched.append(matched_uniform_switch(prob, learned))
        for row in rows:
            continuation.append({key: value for key, value in row.items() if not key.endswith("_seconds")})
    return fixed, consensus, continuation, matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    workers = parser.parse_args().workers
    tasks = [(model, seed) for model in ("fused", "huber", "logistic") for seed in range(5)]
    tasks.extend([("diabetes", 2026), ("breast_cancer", 2026)])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(evaluate, tasks))
    fixed, consensus, continuation, matched = [[row for batch in records for row in batch[index]] for index in range(4)]
    assert (len(fixed), len(consensus), len(continuation)) == (136, 30, 245) and len(matched) == 49
    snapshots = {r["key"]: r for r in load_rows("stopping_certificates.csv") if float(r["threshold"]) == TOL}
    for row in fixed:
        key = "{case}_M{row_parts}_N{column_parts}_{metric}".format(**row)
        ref = snapshots[key]
        if int(ref["iteration"]) != row["iterations"]:
            raise RuntimeError("First checkpoint differs for " + key)
        np.testing.assert_allclose(row["relative_gap"], float(ref["relative_gap"]), rtol=1e-6, atol=1e-12)
    for name, rows in (("stopping_loose_fixed.csv", fixed), ("stopping_loose_consensus.csv", consensus),
                       ("stopping_loose_continuation.csv", continuation), ("stopping_loose_matched.csv", matched)):
        save_rows(name, rows)
    print(f"{len(fixed)} loose fixed-weight fits match archived checkpoints; "
          f"{len(consensus)} consensus and {len(continuation)} continuation/control fits and {len(matched)} matched switches attained {TOL:g}.")


if __name__ == "__main__":
    main()
