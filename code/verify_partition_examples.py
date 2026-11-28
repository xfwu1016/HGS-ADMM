"""Verify two quadratic examples with nonzero partition disagreement.

Run from any directory: python code/verify_partition_examples.py

The primal--dual solutions are computed by a direct quadratic KKT solve,
independently of hybrid_solver.py. The distance constant is then evaluated
both in the full lifted metric and through the partition decomposition.
The t-dependent families change the operators and objectives. The additional
same-objective examples instead regroup a fixed matrix into equal-sized
blocks and verify equality of the complete quadratic objective. None of
these algebraic checks is an iteration-performance benchmark.
"""
from pathlib import Path
import os
import csv
import json
import platform

import numpy as np


RESULTS = Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results"))
T_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
MU = 0.2
RHO = 1.0
ATOL = 1e-11


def example(family, t, mu):
    """Return scalar feature blocks, factor targets, and analytic quantities."""
    if family == "P":
        blocks = {
            (0, 0): np.array([[1.0], [0.0]]),
            (0, 1): np.array([[0.0], [1.0]]),
            (1, 0): np.array([[t]]),
            (1, 1): np.array([[-t]]),
        }
        targets = [np.full(2, 1.0 + mu), np.zeros(1)]
        expected = {
            "beta": np.ones(2), "h": np.array([1.0, 1.0, 0.0]),
            "theta": np.array([-mu, -mu, 0.0]),
            "g": np.full(2, -mu),
            "H": np.array([2.0, 4.0 * t**2]),
            "G": np.full(2, mu**2),
            "P": 10.0 + 4.0 * t**2, "D": 10.0 * mu**2,
        }
    elif family == "G":
        blocks = {(0, 0): np.array([[t]]), (1, 0): np.array([[-t]])}
        targets = [np.array([-1.0]), np.array([-1.0])]
        expected = {
            "beta": np.zeros(1), "h": np.zeros(2),
            "theta": np.ones(2), "g": np.zeros(1),
            "H": np.zeros(2), "G": np.array([4.0 * t**2]),
            "P": 0.0, "D": 4.0 + 4.0 * t**2,
        }
    else:
        raise ValueError(f"Unknown family: {family}")
    return blocks, targets, expected


def verify(family, t, mu=MU, rho=RHO):
    blocks, targets, expected = example(family, t, mu)
    edges = sorted(blocks)
    number_factors = len(targets)
    number_features = 1 + max(j for _, j in edges)
    sizes = [len(y) for y in targets]
    offsets = np.cumsum([0] + sizes)
    number_outputs = int(offsets[-1])
    slices = [slice(offsets[a], offsets[a + 1]) for a in range(number_factors)]
    operator = np.zeros((number_outputs, number_features))
    for (a, j), block in blocks.items():
        operator[slices[a], j:j + 1] = block
    target = np.concatenate(targets)

    # L(beta,h,theta) = ||h-y||^2/2 + mu||beta||^2/2
    #                    + theta^T(A beta-h).
    # Thus theta = h-y and mu beta + A^T theta = 0.
    p, m = number_features, number_outputs
    kkt = np.block([
        [mu * np.eye(p), np.zeros((p, m)), operator.T],
        [np.zeros((m, p)), np.eye(m), -np.eye(m)],
        [operator, -np.eye(m), np.zeros((m, m))],
    ])
    rhs = np.concatenate([np.zeros(p), target, np.zeros(m)])
    solution = np.linalg.solve(kkt, rhs)
    beta = solution[:p]
    h = solution[p:p + m]
    theta = solution[p + m:]
    g = operator.T @ theta
    h_edge = {(a, j): block[:, 0] * beta[j] for (a, j), block in blocks.items()}
    g_edge = {(a, j): float((block.T @ theta[slices[a]])[0])
              for (a, j), block in blocks.items()}
    omega = {(a, j): 1.0 / sum(aa == a for aa, _ in edges) for a, j in edges}
    pi = {(a, j): 1.0 / sum(jj == j for _, jj in edges) for a, j in edges}

    H = np.zeros(number_factors)
    G = np.zeros(number_features)
    for a, j in edges:
        prediction_difference = h_edge[a, j] - omega[a, j] * h[slices[a]]
        gradient_difference = g_edge[a, j] - pi[a, j] * g[j]
        H[a] += float(prediction_difference @ prediction_difference) / omega[a, j]
        G[j] += gradient_difference**2 / pi[a, j]
    primal_decomposition = 2.0 * (beta @ beta + h @ h) + H.sum()
    dual_decomposition = 2.0 * (theta @ theta + g @ g) + G.sum()

    # Build every lifted coordinate explicitly, in the manuscript's order:
    # z = (factor anchors, feature anchors, output edges, coefficient edges).
    z = np.concatenate([h, beta, *[h_edge[e] for e in edges],
                        np.array([beta[j] for _, j in edges])])
    multiplier = np.concatenate([-theta, g, *[theta[slices[a]] for a, _ in edges],
                                 np.array([-g_edge[e] for e in edges])])
    weight = np.concatenate([
        np.ones(m), np.ones(p),
        *[np.full(sizes[a], 1.0 / omega[a, j]) for a, j in edges],
        np.array([pi[e] for e in edges]),
    ])
    primal_full = float(z @ (weight * z))
    dual_full = float(multiplier @ (multiplier / weight))
    c0_full = rho * primal_full + dual_full / rho
    c0_decomposition = rho * primal_decomposition + dual_decomposition / rho
    c0_expected = rho * expected["P"] + expected["D"] / rho

    errors = {}
    for key, value in (("beta", beta), ("h", h), ("theta", theta),
                       ("g", g), ("H", H), ("G", G)):
        errors[key] = float(np.max(np.abs(value - expected[key])))
        np.testing.assert_allclose(value, expected[key], rtol=0.0, atol=ATOL)
    for key, value in (("P", primal_decomposition), ("D", dual_decomposition)):
        errors[key] = abs(float(value) - expected[key])
        np.testing.assert_allclose(value, expected[key], rtol=0.0, atol=ATOL)
    errors["KKT"] = float(np.max(np.abs(kkt @ solution - rhs)))
    errors["full_decomposition"] = abs(c0_full - c0_decomposition)
    errors["full_analytic"] = abs(c0_full - c0_expected)
    assert max(errors.values()) < ATOL, errors
    # mu>0 ensures a unique primal solution in both families.
    minimum_curvature = float(np.linalg.eigvalsh(operator.T @ operator + mu * np.eye(p))[0])
    assert minimum_curvature > 0.0
    row = {
        "family": family, "t": t, "mu": mu, "rho": rho,
        "H1": float(H[0]), "H2": float(H[1]),
        "G1": float(G[0]), "G2": float(G[1]) if len(G) > 1 else None,
        "P_full": primal_full, "P_decomposition": float(primal_decomposition),
        "P_analytic": expected["P"],
        "D_full": dual_full, "D_decomposition": float(dual_decomposition),
        "D_analytic": expected["D"],
        "C0_full": c0_full, "C0_decomposition": float(c0_decomposition),
        "C0_analytic": c0_expected,
        "KKT_max_abs_residual": errors["KKT"],
        "solution_max_abs_error": max(errors[k] for k in ("beta", "h", "theta", "g")),
        "C0_max_abs_error": max(errors["full_decomposition"], errors["full_analytic"]),
        "minimum_objective_curvature": minimum_curvature,
    }
    return row, errors


def _verify_same_objective_case(family, grouping, mu, rho):
    """Regroup a fixed objective and check it in original and lifted coordinates."""
    mixed = grouping == "mixed"
    groups = [np.array([0, 2]), np.array([1, 3])] if mixed else [np.array([0, 1]), np.array([2, 3])]
    if family == "P":
        original_operator = np.array([[1., 1., -1., -1.], [1., 1., 1., 1.]])
        original_target = np.array([0., 4. + mu])
        row_groups, column_groups = [np.arange(2)], groups
        beta_expected = np.ones(4)
        theta_expected = np.array([0., -mu])
        H_expected, G_expected = (0. if mixed else 16.), 0.
        P_base, D_base = 40., 10. * mu**2
    elif family == "G":
        original_operator = np.ones((4, 1))
        original_target = np.array([0., 0., 2. + mu / 2., 2. + mu / 2.])
        row_groups, column_groups = groups, [np.array([0])]
        beta_expected = np.ones(1)
        theta_expected = np.array([1., 1., -1. - mu / 2., -1. - mu / 2.])
        H_expected, G_expected = 0., (0. if mixed else (4. + mu)**2)
        P_base, D_base = 10., 8. + 4. * mu + 3. * mu**2
    else:
        raise ValueError(f"Unknown family: {family}")
    h_expected = original_operator @ beta_expected
    g_expected = -mu * beta_expected
    blocks = {(a, j): original_operator[np.ix_(rows, columns)]
              for a, rows in enumerate(row_groups)
              for j, columns in enumerate(column_groups)}
    edges = sorted(blocks)
    operator = np.block([[blocks[a, j] for j in range(len(column_groups))]
                         for a in range(len(row_groups))])
    row_order, column_order = np.concatenate(row_groups), np.concatenate(column_groups)
    target = np.concatenate([original_target[rows] for rows in row_groups])
    m, p = operator.shape
    row_offsets = np.cumsum([0] + [len(rows) for rows in row_groups])
    column_offsets = np.cumsum([0] + [len(columns) for columns in column_groups])
    factor_slices = [slice(row_offsets[a], row_offsets[a + 1]) for a in range(len(row_groups))]
    feature_slices = [slice(column_offsets[j], column_offsets[j + 1]) for j in range(len(column_groups))]

    # Equality of Hessian, linear coefficient, and constant proves equality
    # of the full quadratic objective for every original coefficient vector.
    restored_operator = np.zeros_like(original_operator)
    restored_operator[np.ix_(row_order, column_order)] = operator
    restored_hessian = np.zeros((p, p))
    restored_hessian[np.ix_(column_order, column_order)] = operator.T @ operator + mu * np.eye(p)
    restored_linear = np.zeros(p)
    restored_linear[column_order] = -operator.T @ target
    errors = {
        "operator_reassembly": float(np.max(np.abs(restored_operator - original_operator))),
        "objective_hessian": float(np.max(np.abs(restored_hessian - original_operator.T @ original_operator - mu * np.eye(p)))),
        "objective_linear": float(np.max(np.abs(restored_linear + original_operator.T @ original_target))),
        "objective_constant": abs(float(target @ target - original_target @ original_target)) / 2.,
    }
    kkt = np.block([
        [mu * np.eye(p), np.zeros((p, m)), operator.T],
        [np.zeros((m, p)), np.eye(m), -np.eye(m)],
        [operator, -np.eye(m), np.zeros((m, m))],
    ])
    rhs = np.concatenate([np.zeros(p), target, np.zeros(m)])
    solution = np.linalg.solve(kkt, rhs)
    beta, h, theta = solution[:p], solution[p:p + m], solution[p + m:]
    g = operator.T @ theta
    errors["KKT"] = float(np.max(np.abs(kkt @ solution - rhs)))
    for key, value, expected, order in (
            ("beta", beta, beta_expected, column_order),
            ("h", h, h_expected, row_order),
            ("theta", theta, theta_expected, row_order),
            ("g", g, g_expected, column_order)):
        original_value = np.empty_like(value)
        original_value[order] = value
        errors[key] = float(np.max(np.abs(original_value - expected)))
    h_edge = {(a, j): block @ beta[feature_slices[j]] for (a, j), block in blocks.items()}
    g_edge = {(a, j): block.T @ theta[factor_slices[a]] for (a, j), block in blocks.items()}
    omega = 1. / len(column_groups)
    pi = 1. / len(row_groups)
    H = sum(float(np.sum((h_edge[a, j] - omega * h[factor_slices[a]])**2)) / omega for a, j in edges)
    G = sum(float(np.sum((g_edge[a, j] - pi * g[feature_slices[j]])**2)) / pi for a, j in edges)
    P_decomposition = 2. * (beta @ beta + h @ h) + H
    D_decomposition = 2. * (theta @ theta + g @ g) + G
    z = np.concatenate([h, beta, *[h_edge[e] for e in edges],
                        *[beta[feature_slices[j]] for _, j in edges]])
    multiplier = np.concatenate([-theta, g, *[theta[factor_slices[a]] for a, _ in edges],
                                 *[-g_edge[e] for e in edges]])
    weight = np.concatenate([
        np.ones(m), np.ones(p),
        *[np.full(len(row_groups[a]), 1. / omega) for a, _ in edges],
        *[np.full(len(column_groups[j]), pi) for _, j in edges],
    ])
    P_full = float(z @ (weight * z))
    D_full = float(multiplier @ (multiplier / weight))
    P_expected, D_expected = P_base + H_expected, D_base + G_expected
    C0_full = rho * P_full + D_full / rho
    C0_decomposition = rho * P_decomposition + D_decomposition / rho
    C0_expected = rho * P_expected + D_expected / rho
    errors.update({
        "H": abs(H - H_expected), "G": abs(G - G_expected),
        "P": abs(float(P_decomposition) - P_expected),
        "D": abs(float(D_decomposition) - D_expected),
        "full_decomposition": abs(C0_full - C0_decomposition),
        "full_analytic": abs(C0_full - C0_expected),
    })
    assert max(errors.values()) < ATOL, (family, grouping, errors)
    row = {
        "family": family, "grouping": grouping, "mu": mu, "rho": rho,
        "omega": omega, "pi": pi, "H": H, "G": G,
        "P_base": P_base, "D_base": D_base,
        "P_full": P_full, "P_decomposition": float(P_decomposition), "P_analytic": P_expected,
        "D_full": D_full, "D_decomposition": float(D_decomposition), "D_analytic": D_expected,
        "C0_full": C0_full, "C0_decomposition": float(C0_decomposition), "C0_analytic": C0_expected,
        "objective_coefficient_max_abs_error": max(errors[k] for k in ("objective_hessian", "objective_linear", "objective_constant")),
        "KKT_max_abs_residual": errors["KKT"],
        "solution_max_abs_error": max(errors[k] for k in ("beta", "h", "theta", "g")),
        "C0_max_abs_error": max(errors["full_decomposition"], errors["full_analytic"]),
    }
    return row, errors


def verify_same_objective_partitions(mu=MU, rho=RHO):
    """Check four regroupings of two fixed quadratic objectives and save results."""
    records, max_errors = [], {}
    for family in ("P", "G"):
        for grouping in ("mixed", "homogeneous"):
            row, errors = _verify_same_objective_case(family, grouping, mu, rho)
            records.append(row)
            for key, value in errors.items():
                max_errors[key] = max(max_errors.get(key, 0.), value)
    results = RESULTS
    results.mkdir(exist_ok=True)
    with (results / "partition_same_objective.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "passed", "cases": len(records), "mu": mu, "rho": rho,
        "absolute_tolerance": ATOL, "maximum_absolute_errors": max_errors,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "groupings_zero_based": {"mixed": [[0, 2], [1, 3]], "homogeneous": [[0, 1], [2, 3]]},
        "prediction_example": {
            "operator": [[1, 1, -1, -1], [1, 1, 1, 1]], "target": [0, 4 + mu],
            "beta": [1, 1, 1, 1], "h": [0, 4], "theta": [0, -mu], "g": [-mu] * 4,
            "H_mixed": 0, "H_homogeneous": 16, "G": 0,
            "P": "40+H", "D": "10*mu^2",
        },
        "gradient_example": {
            "operator": [[1], [1], [1], [1]], "target": [0, 0, 2 + mu / 2, 2 + mu / 2],
            "beta": [1], "h": [1, 1, 1, 1],
            "theta": [1, 1, -1 - mu / 2, -1 - mu / 2], "g": [-mu],
            "H": 0, "G_mixed": 0, "G_homogeneous": "(4+mu)^2",
            "P": "10", "D": "8+4*mu+3*mu^2+G",
        },
        "interpretation": [
            "Each pair retains exactly the same original objective and unique primal solution.",
            "Block sizes, graph degrees, and incidence weights are unchanged within each pair.",
            "All unreplicated norm terms stay fixed; only H or G changes.",
            "Objective equality is checked through every quadratic coefficient and operator reassembly.",
            "These geometric constant comparisons do not predict iteration performance.",
        ],
    }
    with (results / "partition_same_objective_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Verified {len(records)} same-objective partition cases; maximum error {max(max_errors.values()):.3e}.")
    for row in records:
        print(f"{row['family']} {row['grouping']:11s}: "
              f"H={row['H']:7.4f}, G={row['G']:7.4f}, C0={row['C0_full']:8.4f}")
    return summary


def main():
    records = []
    max_errors = {}
    for family in ("P", "G"):
        for t in T_VALUES:
            row, errors = verify(family, t)
            records.append(row)
            for key, value in errors.items():
                max_errors[key] = max(max_errors.get(key, 0.0), value)
    results = RESULTS
    results.mkdir(exist_ok=True)
    with (results / "partition_examples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "status": "passed", "cases": len(records), "mu": MU, "rho": RHO,
        "t_values": list(T_VALUES), "absolute_tolerance": ATOL,
        "maximum_absolute_errors": max_errors,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "family_P": {"H": "(2, 4*t^2)", "G": "(mu^2, mu^2)",
                     "P": "10+4*t^2", "D": "10*mu^2",
                     "C0_at_rho_1": "10+4*t^2+10*mu^2"},
        "family_G": {"H": "(0, 0)", "G": "4*t^2", "P": "0",
                     "D": "4+4*t^2", "C0_at_rho_1": "4+4*t^2"},
        "interpretation": [
            "Each family has fixed aggregate optimal beta, h, theta, and g as t varies.",
            "The graph topology and incidence budgets stay fixed within each family.",
            "Family P varies local predictions; Family G varies local gradients.",
            "The operators and quadratic objective change with t.",
            "These are algebraic geometry examples, not iteration-performance predictions.",
        ],
    }
    with (results / "partition_examples_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"Verified {len(records)} quadratic KKT solutions and lifted distance identities.")
    print(f"Maximum absolute check error: {max(max_errors.values()):.3e}")
    for row in records:
        print(f"{row['family']} t={row['t']:4.2f}: "
              f"P={row['P_full']:8.4f}, D={row['D_full']:8.4f}, "
              f"C0={row['C0_full']:8.4f}")
    verify_same_objective_partitions()


if __name__ == "__main__":
    main()
