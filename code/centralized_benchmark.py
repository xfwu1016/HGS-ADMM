"""Single-thread FISTA vs sequential 4x4 budget-graph splitting.

Both optimizers start at zero; both stop using the same loss-gradient dual
candidate evaluated at their coefficient vector. Timing includes setup and
checks but excludes data generation, post-run verification and file I/O.
"""
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.special import expit, xlogy
from threadpoolctl import threadpool_info

from hybrid_solver import GraphSplitting, make_regression, soft
from pilot_continuation import step
from stopping_sensitivity import RESULTS, save_rows
from run_experiments import synthetic_case


def gradient_coordinates(X, y, beta, args):
    z = X @ beta
    if args["loss"] == "huber":
        return np.clip(z-y, -args["loss_parameter"], args["loss_parameter"])
    return -y * expit(-y*z)


def certificate(X, y, beta, args):
    n = len(y)
    z = X @ beta
    if args["loss"] == "huber":
        residual = z - y
        derivative = np.clip(residual, -args["loss_parameter"], args["loss_parameter"])
        loss = np.mean(derivative*residual - .5*derivative**2)
        conjugate = np.mean(derivative*y + .5*derivative**2)
    else:
        u = expit(-y*z)
        derivative = -y*u
        loss = np.logaddexp(0., -y*z).mean()
        conjugate = np.mean(xlogy(u,u) + xlogy(1.-u,1.-u))
    primal = loss + args["l1"]*np.linalg.norm(beta,1) + .5*args["ridge"]*(beta@beta)
    excess = soft(X.T @ derivative / n, args["l1"])
    dual = -conjugate - .5*(excess@excess) / args["ridge"]
    gap = primal - dual
    if gap < -1e-12:
        raise ArithmeticError("Invalid dense certificate")
    return primal, dual, max(gap, 0.) / max(1., abs(primal))


def solve(case, method):
    X, y, args = case["X"], case["y"], case["args"]
    n, p = X.shape
    start = perf_counter()
    if method == "fista":
        gram = X @ X.T if n <= p else X.T @ X
        curvature = 1. if args["loss"] == "huber" else .25
        L = (1 + 1e-12) * curvature * np.linalg.eigvalsh(gram)[-1] / n
        beta = np.zeros(p)
        z, t = beta.copy(), 1.
    else:
        solver = GraphSplitting(make_regression(X, y, 4, 4, **args), rho=1, weights="budget")
    setup_seconds = perf_counter() - start
    for k in range(1, 30001):
        if method == "fista":
            gradient = X.T @ gradient_coordinates(X, y, z, args) / n
            new_beta = soft(z-gradient/L, args["l1"]/L) / (1+args["ridge"]/L)
            new_t = .5 * (1 + np.sqrt(1 + 4*t*t))
            z = new_beta + ((t-1)/new_t)*(new_beta-beta)
            beta, t = new_beta, new_t
        else:
            step(solver)
        if k == 1 or k % 10 == 0:
            if method == "graph":
                beta = np.concatenate([solver.eta[s] for s in solver.xs])
            primal, dual, relative = certificate(X, y, beta, args)
            if relative <= 1e-6:
                elapsed = perf_counter() - start
                break
    else:
        raise RuntimeError("Missed benchmark gap")
    # Verify the shared dense certificate against the graph objective/conjugates.
    graph_problem = make_regression(X, y, 1, 1, **args)
    theta = gradient_coordinates(X, y, beta, args) / np.sqrt(n)
    np.testing.assert_allclose([primal, dual],
                               [graph_problem.objective([beta]), graph_problem.dual_bound([theta])],
                               rtol=1e-12, atol=1e-12)
    return dict(iterations=k, setup_seconds=setup_seconds, total_seconds=elapsed,
                primal=primal, dual=dual, relative_gap=relative, gap_target=1e-6)


def main():
    rows = []
    for model in ("huber", "logistic"):
        for seed in range(5):
            case = synthetic_case(model, seed)
            for rep in range(3):
                order = ("fista", "graph") if (rep+seed) % 2 == 0 else ("graph", "fista")
                pair = {}
                for method in order:
                    result = solve(case, method)
                    rows.append(dict(model=model, seed=seed, repetition=rep, method=method, **result))
                    pair[method] = result
                if max(x["dual"] for x in pair.values()) > min(x["primal"] for x in pair.values()) + 1e-12:
                    raise ArithmeticError("Disjoint primal-dual intervals for the shared objective")
    assert len(rows) == 60
    save_rows("centralized_benchmark.csv", rows)
    cpuinfo = Path("/proc/cpuinfo")
    cpu = platform.processor() or platform.machine()
    if cpuinfo.is_file():
        cpu = next((line.partition(":")[2].strip() for line in cpuinfo.read_text().splitlines()
                    if line.startswith("model name")), cpu)
    environment = {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "software": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "blas_libraries": [
            {name: pool.get(name) for name in ("internal_api", "version", "num_threads", "architecture")}
            for pool in threadpool_info() if pool["user_api"] == "blas"
        ],
        "csv_sha256": hashlib.sha256((RESULTS / "centralized_benchmark.csv").read_bytes()).hexdigest(),
    }
    (RESULTS / "centralized_benchmark_environment.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print("60 matched single-thread timings, including setup and certification.")


if __name__ == "__main__":
    main()
