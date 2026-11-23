"""Pilot-length sensitivity on all nine seed-zero model/layout pairs."""
from pathlib import Path
import csv

import numpy as np
from hybrid_solver import GraphSplitting, make_regression
from pilot_budget_experiment import DESIGN, make_pilot_weights
from run_experiments import synthetic_case

RESULTS = Path(__file__).resolve().parents[1] / "results"
LENGTHS = (25, 100, 400, 1600)


def evaluate(model, rows, columns):
    case = synthetic_case(model, 0)
    problem = make_regression(case["X"], case["y"], rows, columns, **case["args"])
    settings = dict(tol=DESIGN["main_relative_gap"], maxiter=DESIGN["main_iteration_cap"])
    baseline = GraphSplitting(problem, weights="budget").solve(**settings)
    if not baseline["converged"]:
        raise RuntimeError("Uniform comparator failed to reach its target")
    pilot = GraphSplitting(problem, weights="unit")
    for step in range(1, max(LENGTHS) + 1):
        local = pilot.prox(pilot.eta - pilot.u)
        consistent = pilot.project(local)
        pilot.u += local - consistent
        pilot.eta = consistent
        if step not in LENGTHS:
            continue
        primal, dual, gap = pilot.certificate()
        beta = np.concatenate([pilot.eta[s] for s in pilot.xs])
        metric, diag = make_pilot_weights(problem, dict(coef=beta, gap=gap))
        result = GraphSplitting(problem, weights="pilot", custom_weights=metric).solve(**settings)
        if not result["converged"]:
            raise RuntimeError("Pilot-weighted main solve failed to reach its target")
        per_round = pilot.communication_per_iteration
        payload = (step + result["iterations"]) * per_round + 2 * len(pilot.edges)
        yield dict(model=model, seed=0, M=rows, J=columns, pilot_steps=step,
                   pilot_objective=primal, pilot_dual=dual, pilot_gap=gap,
                   pilot_relative_gap=gap / max(1, abs(primal)),
                   upper=diag["edge_upper_pilot"], lower=diag["edge_lower_best"],
                   regret_bound=diag["regret_bound"], main_iterations=result["iterations"],
                   main_relative_gap=result["relative_gap"], converged=result["converged"],
                   uniform_iterations=baseline["iterations"],
                   per_round=per_round, allocation_payload=2 * len(pilot.edges),
                   payload=payload, uniform_payload=baseline["communicated_scalars"],
                   checks_including_pilot=1 + result["diagnostic_evaluations"])


if __name__ == "__main__":
    records = [record for model in DESIGN["models"] for rows, columns in DESIGN["layouts"]
               for record in evaluate(model, rows, columns)]
    with (RESULTS / "pilot_length.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)
    print(f"Completed {len(records)} pilot-length comparisons")
