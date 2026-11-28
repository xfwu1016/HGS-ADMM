"""Range of the zero-start distance over the fused-chain KKT multiplier set.

Use the coefficient fits from the main experiment to identify candidate signs,
solve the restricted quadratic exactly up to rounding, then verify optimality
by constructing feasible multipliers. Function values include all lifted
coordinates, recalculated by the same formulas as the main numerical record.
"""
import csv
from itertools import combinations
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from run_experiments import LAYOUTS, RESULTS, distance_diagnostics, synthetic_case
from hybrid_solver import GraphSplitting, make_regression

TOL = 1e-10
FIXED16 = ((2, 8), (4, 4), (8, 2))
ENDPOINT_MARGIN = 1e-7  # Numerical sensitivity margin in C0 units.


def recover_coefficients(case):
    X, y = case["X"], case["y"]
    n, p = X.shape
    lam, mu, tau = (case["args"][k] for k in ("l1", "ridge", "fusion"))
    b0 = np.load(RESULTS / "coefficients" / (case["case"] + "_reference.npz"))["coef"]
    cuts = np.r_[0, np.flatnonzero(abs(np.diff(b0)) > 1e-6) + 1, p]
    groups = list(zip(cuts[:-1], cuts[1:]))
    S = np.stack([(np.arange(p) >= lo) & (np.arange(p) < hi)
                  for lo, hi in groups if abs(b0[lo:hi].mean()) > 1e-6], axis=1).astype(float)
    signs = np.where(abs(b0) > 1e-6, np.sign(b0), 0)
    jumps = np.where(abs(np.diff(b0)) > 1e-6, np.sign(np.diff(b0)), 0)
    Z = X @ S
    group_gradient = lam * signs + tau * (np.r_[0, jumps] - np.r_[jumps, 0])
    means = np.linalg.solve(Z.T @ Z / n + mu * (S.T @ S), Z.T @ y / n - S.T @ group_gradient)
    beta = S @ means
    np.testing.assert_array_equal(np.sign(beta), signs)
    np.testing.assert_array_equal(np.sign(np.diff(beta)), jumps)
    gradient = X.T @ (X @ beta - y) / n
    # x_i corresponds to contrast beta_(i+1) - beta_i, with virtual x_0=x_p=0.
    anchors = np.r_[0, tau * jumps, 0]
    return beta, gradient, cuts, anchors


def reachable_values(drift, left, right, tau, lam):
    """Every vertex has each coordinate joined by tight increments to a box bound."""
    length = len(drift)
    partial = np.r_[0, drift.cumsum()]
    values = [np.array([left])]
    for i in range(1, length):
        candidates = []
        for j in range(length + 1):
            bounds = [left] if j == 0 else [right] if j == length else [-tau, tau]
            k = abs(i - j)
            offsets = partial[i] - partial[j] + lam * np.arange(-k, k + 1, 2)
            for bound in bounds:
                candidates.extend(bound + offsets)
        values_i = np.unique(np.round(candidates, 13))
        values.append(values_i[(values_i >= -tau - TOL) & (values_i <= tau + TOL)])
    return [*values, np.array([right])]


def stage_cost(left, right, i, g, mass, wleft, wright, shared):
    gap = left - right
    structural = wleft[i] * left**2 + wright[i] * right**2
    if shared[i]:
        structural -= 2 * wleft[i] * left * right
    return (g[i] + gap)**2 + mass[i] * right**2 + structural


def max_path(nodes, lo, drift, lam, coefficients):
    """Maximize the additive convex quadratic over all feasible vertex paths."""
    score = np.array([0.])
    back = []
    for k, (prev, curr) in enumerate(zip(nodes[:-1], nodes[1:])):
        l, r = prev[:, None], curr[None, :]
        valid = abs(r - l - drift[k]) <= lam + TOL
        candidate = np.where(valid, score[:, None] + stage_cost(l, r, lo+k, *coefficients), -np.inf)
        idx = np.argmax(candidate, axis=0)
        score = candidate[idx, np.arange(len(curr))]
        back.append(idx)
    if not np.isfinite(score[0]):
        raise ValueError("Candidate signs do not admit a feasible multiplier.")
    indices = [0]
    for choices in reversed(back):
        indices.append(int(choices[indices[-1]]))
    return np.array([v[i] for v, i in zip(nodes, reversed(indices))])


def min_path(upper_path, lo, drift, lam, tau, coefficients):
    g, mass, wl, wr, shared = (a[lo:lo+len(drift)] for a in coefficients)
    cross = wl * shared
    constant = np.r_[upper_path[0], np.zeros(len(drift)-1), upper_path[-1]]
    M = np.diff(np.eye(len(upper_path)), axis=0)[:,1:-1]
    offsets = np.diff(constant)
    lower, upper = drift - lam - offsets, drift + lam - offsets

    def fun(v):
        path = np.r_[constant[0], v, constant[-1]]
        l, r = path[:-1], path[1:]
        residual = g + l - r
        cost = (residual**2 + (mass + wr)*r**2 + wl*l**2 - 2*cross*l*r).sum()
        left_grad = 2*(residual + wl*l - cross*r)
        right_grad = 2*(-residual + (mass+wr)*r - cross*l)
        return cost, left_grad[1:] + right_grad[:-1]

    optimum = minimize(fun, upper_path[1:-1], method="SLSQP", jac=True,
                       bounds=Bounds(-tau, tau), constraints=[LinearConstraint(M, lower, upper)],
                       options={"ftol": 1e-12, "maxiter": 500})
    v = optimum.x
    # Convexity and the minimizing linearization furnish an independent lower bound.
    grad = fun(v)[1]
    tangent = linprog(grad, A_ub=np.vstack([M, -M]), b_ub=np.r_[upper, -lower],
                      bounds=(-tau,tau), method="highs")
    if not tangent.success:
        raise ValueError(tangent.message)
    err = max(0., grad @ v - tangent.fun)
    if err > 2e-6:
        raise ValueError(f"Convex minimum failed first-order check: {err:g}")
    return np.r_[constant[0], v, constant[-1]], err


def objective_weights(solver, ndata, p, gradient):
    # Coefficients in the chain quadratic. At group boundaries, neighboring
    # edge-dual contributions occur in distinct square norms.
    structural_groups = np.empty(p-1, dtype=int)
    for idx, rr in enumerate(np.array_split(np.arange(p-1), len(solver.problem.factors)-ndata)):
        structural_groups[rr] = ndata + idx
    columns = np.repeat(np.arange(len(solver.problem.block_sizes)), solver.problem.block_sizes)
    edge_w = {(a,j):1/solver.d[e] for e,(a,j,_) in enumerate(solver.edges)}
    q_mass = {a:1/solver.a[a]+sum(1/solver.b[e] for e,(f,_,_) in enumerate(solver.edges) if a==f)
              for a in np.unique(structural_groups)}
    mass = np.r_[[q_mass[a] for a in structural_groups], 0.]
    wleft = np.r_[0., [edge_w[a,j] for a,j in zip(structural_groups, columns[1:])]]
    wright = np.r_[[edge_w[a,j] for a,j in zip(structural_groups, columns[:-1])], 0.]
    shared = np.r_[False, structural_groups[:-1] == structural_groups[1:], False]
    return gradient, mass, wleft, wright, shared


def run_seed(seed, layouts=None):
    case = synthetic_case("fused", seed)
    beta, gradient, cuts, anchors = recover_coefficients(case)
    X,y = case["X"],case["y"]
    n,p = X.shape
    lam, mu, tau = (case["args"][k] for k in ("l1", "ridge", "fusion"))
    drift = gradient + mu * beta
    nodes = {(lo,hi): reachable_values(drift[lo:hi], anchors[lo], anchors[hi], tau, lam)
             for lo, hi in zip(cuts[:-1], cuts[1:]) if beta[lo] == 0 and hi-lo>1}
    cases = []
    for M, J in LAYOUTS if layouts is None else layouts:
        for metric in ("unit","budget"):
            solver = GraphSplitting(make_regression(X, y, M, J, **case["args"]), weights=metric)
            q_min, q_max = anchors.copy(), anchors.copy()
            error = 0.
            c = objective_weights(solver, M, p, gradient)
            for lo,hi in zip(cuts[:-1], cuts[1:]):
                if hi-lo == 1:
                    continue
                if beta[lo] != 0:
                    path = anchors[lo] + np.cumsum(drift[lo:hi-1] + lam*np.sign(beta[lo]))
                    q_min[lo+1:hi] = q_max[lo+1:hi] = path
                else:
                    path = max_path(nodes[lo,hi], lo, drift[lo:hi], lam, c)
                    q_max[lo:hi+1] = path
                    path, err = min_path(path, lo, drift[lo:hi], lam, tau, c)
                    q_min[lo:hi+1] = path
                    error += err
            values = []
            kkt_error = 0.
            for path in (q_min,q_max):
                slack = np.diff(path) - drift
                residual = np.where(beta!=0, abs(slack-lam*np.sign(beta)), np.maximum(abs(slack)-lam, 0))
                kkt_error = max(kkt_error, residual.max(), (abs(path)-tau).max())
                jump = np.flatnonzero(np.diff(beta))+1
                np.testing.assert_allclose(path[jump],tau*np.sign(np.diff(beta))[jump-1],atol=TOL)
                thetas = [*np.array_split((X @ beta-y)/np.sqrt(n), M),
                          *np.array_split(path[1:-1],case["args"]["fusion_parts"])]
                values.append(distance_diagnostics(solver, beta, thetas)["distance_constant_approx"])
            s_min, s_max = [sum(stage_cost(q[i],q[i+1],i,*c) for i in range(p)) for q in (q_min,q_max)]
            np.testing.assert_allclose(values[1]-values[0],s_max-s_min,atol=1e-8)
            if kkt_error > 1e-8 or error>1e-5 or values[1]<values[0]:
                raise ValueError((kkt_error, error, values))
            cases.append(dict(seed=seed, row_parts=M, column_parts=J, metric=metric,
                              min_C0=values[0],max_C0=values[1],
                              min_linearization_error=error,max_kkt_violation=kkt_error))
    return cases


def compare_ranges(records, layouts):
    """Compare each unordered pair of selected layout/metric ranges within a seed."""
    selected = [r for r in records if (r["row_parts"], r["column_parts"]) in layouts]
    expected = {(m, j, metric) for m, j in layouts for metric in ("unit", "budget")}
    pairs = []
    for seed in range(5):
        group = [r for r in selected if r["seed"] == seed]
        if len(group) != len(expected) or {
            (r["row_parts"], r["column_parts"], r["metric"]) for r in group
        } != expected:
            raise ValueError(f"Incomplete multiplier ranges for layouts {layouts}, seed {seed}")
        intervals = []
        for r in group:
            lower = r["min_C0"] - r["min_linearization_error"] - ENDPOINT_MARGIN
            upper = r["max_C0"] + ENDPOINT_MARGIN
            intervals.append((lower, upper, r))
        intervals.sort(key=lambda item: item[0])
        for (lo1, hi1, a), (lo2, hi2, b) in combinations(intervals, 2):
            pairs.append(dict(seed=seed, first_rows=a["row_parts"],
                              first_columns=a["column_parts"], first_metric=a["metric"],
                              second_rows=b["row_parts"], second_columns=b["column_parts"],
                              second_metric=b["metric"], first_lower=lo1, first_upper=hi1,
                              second_lower=lo2, second_upper=hi2,
                              separation=lo2-hi1, overlaps=(lo2<=hi1)))
    return pairs


def compare_fixed16_ranges(records):
    return compare_ranges(records, FIXED16)


def save_records(name, records):
    with (RESULTS / name).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    layouts = (*LAYOUTS, *(layout for layout in FIXED16 if layout not in LAYOUTS))
    records = []
    for seed in range(5):
        records.extend(run_seed(seed, layouts))
        print(f"Checked fused seed {seed}", flush=True)
    primary = [r for r in records if (r["row_parts"], r["column_parts"]) in LAYOUTS]
    additional = [r for r in records if (r["row_parts"], r["column_parts"]) not in LAYOUTS]
    primary_pairs = compare_ranges(primary, LAYOUTS)
    pairs = compare_fixed16_ranges(records)
    save_records("fused_dual_ranges.csv", primary)
    save_records("fused_dual_fixed16_extra.csv", additional)
    save_records("fused_dual_primary_pairs.csv", primary_pairs)
    save_records("fused_dual_fixed16_pairs.csv", pairs)
    print(f"Primary layouts: {sum(p['overlaps'] for p in primary_pairs)}/{len(primary_pairs)} pairs overlap.")
    print(f"Fixed 16 blocks: {sum(p['overlaps'] for p in pairs)}/{len(pairs)} pairs overlap; "
          f"smallest separation {min(p['separation'] for p in pairs):.9g}.")
