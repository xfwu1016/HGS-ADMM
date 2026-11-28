"""Check empty-factor errors and preservation of constant data-loss terms.

These small input checks use independent closed-form ridge solutions. An
optional --reference-code path additionally compares unchanged ordinary
iterations with a previous hybrid_solver.py, excluding elapsed times.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import sys

import numpy as np

from hybrid_solver import Factor, GraphSplitting, make_regression


def load_reference(path):
    spec = importlib.util.spec_from_file_location("input_check_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ridge_reference(X, y, ridge):
    n, p = X.shape
    beta = np.linalg.solve(X.T @ X / n + ridge * np.eye(p), X.T @ y / n)
    objective = np.sum((X @ beta - y) ** 2) / (2 * n)
    objective += ridge * np.dot(beta, beta) / 2
    return beta, float(objective)


def run_checks(reference_code=None):
    records = []
    X = np.array([[0., 0.], [0., 0.], [1., 2.], [3., 4.]])
    y = np.array([1., -2., 3., -1.])
    try:
        problem = make_regression(X, y, row_parts=2, column_parts=2,
                                  l1=0., ridge=.2, include_zero_edges=False)
        GraphSplitting(problem)
    except ValueError as exc:
        message = str(exc)
        assert "data_0" in message and "incident edge" in message
        assert "include_zero_edges=True" in message
        records.append({"test": "all_zero_row_block_rejected_clearly",
                        "exception": type(exc).__name__, "message": message})
    else:
        raise AssertionError("A factor without an incident map was accepted")

    try:
        Factor("squared", {}, response=np.zeros(2), label="empty_test").size
    except ValueError as exc:
        assert "empty_test" in str(exc)
        records.append({"test": "direct_empty_factor_dimension_rejected_clearly",
                        "exception": type(exc).__name__})
    else:
        raise AssertionError("An empty factor supplied a dimension")

    response = np.array([1., -1., 1., -1.])
    unpenalized = make_regression(np.ones((4, 1)), response, 1, 1, l1=0., ridge=0.)
    lower = unpenalized.dual_bound([-response / 2])
    assert lower == unpenalized.objective([np.zeros(1)]) == 0.5
    records.append({"test": "zero_adjoint_dual_preserved", "dual": lower})
    design = np.array([[-1.], [1.]])
    options = dict(row_parts=1, column_parts=1, loss="logistic", ridge=.1, l1=.01)
    fits = [GraphSplitting(make_regression(design, labels, **options)).solve(tol=1e-7)
            for labels in ([0, 1], [-1, 1])]
    assert all(fit["converged"] for fit in fits)
    assert np.array_equal(fits[0]["coef"], fits[1]["coef"])
    assert fits[0]["gap"] == fits[1]["gap"]
    records.append({"test": "binary_encodings_agree", "gap": fits[0]["gap"]})
    try:
        make_regression(design, [0, -1], **options)
    except ValueError as error:
        assert "labels" in str(error)
    else:
        raise AssertionError("Mixed logistic label encodings were accepted")
    records.append({"test": "mixed_binary_encodings_rejected"})

    # With these moderate scales Newton stays inside its bracket but
    # alternates close to its endpoints, so the fallback must finish.
    logistic_response = np.sqrt(24) * np.array([1., -1.])
    logistic = Factor("logistic", {0: np.ones((2, 1))},
                      response=logistic_response, scale=1/24)
    prox_input = np.array([-.94, .94])
    proximal_coefficient = .082
    output = logistic.prox(prox_input, proximal_coefficient)
    residual = proximal_coefficient * (output - prox_input)
    residual -= logistic.scale * logistic_response / (1 + np.exp(logistic_response * output))
    error = float(np.max(np.abs(residual)))
    assert error < 1e-13
    records.append({"test": "logistic_bracket_fallback", "stationarity_error": error})

    reference = load_reference(reference_code) if reference_code else None
    rng = np.random.default_rng(20261122)
    ordinary_X = rng.normal(size=(16, 6))
    ordinary_y = ordinary_X @ np.array([1., -.4, .2, 0., .5, -.8])
    ordinary_y += rng.normal(scale=.2, size=16)
    cases = [("zero_rows_retained", X, y, 2, 2, .2, True),
             ("ordinary_dense_ridge", ordinary_X, ordinary_y, 4, 3, .13, False)]
    for name, design, response, rows, cols, ridge, padded in cases:
        beta_exact, objective_exact = ridge_reference(design, response, ridge)
        options = dict(row_parts=rows, column_parts=cols, l1=0., ridge=ridge,
                       include_zero_edges=padded)
        for metric in ("unit", "budget"):
            problem = make_regression(design, response, **options)
            solver = GraphSplitting(problem, weights=metric)
            result = solver.solve(tol=1e-13, maxiter=20000)
            coefficient_error = float(np.linalg.norm(result["coef"] - beta_exact))
            objective_error = abs(result["objective"] - objective_exact)
            assert result["converged"]
            assert coefficient_error < 2e-6 and objective_error < 1e-10
            expected_dimension = (
                sum(f.size for f in problem.factors) + sum(problem.block_sizes)
                + sum(A.shape[0] + A.shape[1] for _, _, A in solver.edges)
            )
            assert result["state_dimension"] == expected_dimension == solver.eta.size
            record = {"test": name, "metric": metric,
                      "iterations": result["iterations"],
                      "relative_gap": result["relative_gap"],
                      "coefficient_error_to_closed_form": coefficient_error,
                      "objective_error_to_closed_form": objective_error,
                      "state_dimension": result["state_dimension"],
                      "state_dimension_is_one_full_lifted_vector": True}
            if name == "zero_rows_retained":
                constant_loss = np.dot(response[:2], response[:2]) / (2 * len(response))
                other_loss = np.sum((design[2:] @ result["coef"] - response[2:]) ** 2)
                other_loss /= 2 * len(response)
                other_loss += ridge * np.dot(result["coef"], result["coef"]) / 2
                preserved_constant = result["objective"] - other_loss
                assert abs(preserved_constant - constant_loss) < 1e-14
                record["constant_loss_retained"] = float(preserved_constant)
            if reference is not None:
                prior_problem = reference.make_regression(design, response, **options)
                prior_solver = reference.GraphSplitting(prior_problem, weights=metric)
                prior = prior_solver.solve(tol=1e-13, maxiter=20000)
                scalar_keys = ("iterations", "objective", "dual", "gap", "relative_gap",
                               "squared_step", "converged", "communicated_scalars",
                               "communication_per_iteration", "largest_local_matrix",
                               "factorization_storage_scalars", "state_dimension",
                               "diagnostic_evaluations", "initial_objective")
                assert all(result[k] == prior[k] for k in scalar_keys)
                assert np.array_equal(result["coef"], prior["coef"])
                assert np.array_equal(solver.eta, prior_solver.eta)
                assert np.array_equal(solver.u, prior_solver.u)
                record["previous_solver_nontiming_values_identical"] = True
            records.append(record)
    output = {"revision": 9, "checks_passed": len(records), "records": records,
              "implementation_change": "Input, certificate, and logistic bracket fallback checks",
              "existing_numerical_results_written": False}
    if reference_code:
        output["reference_solver_sha256"] = hashlib.sha256(Path(reference_code).read_bytes()).hexdigest()
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-code", type=Path)
    parser.add_argument("--output", type=Path,
                        default=Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results"))
                        / "solver_input_checks_v9.json")
    args = parser.parse_args()
    report = run_checks(args.reference_code)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Passed {report['checks_passed']} input/closed-form checks; wrote {args.output}")
