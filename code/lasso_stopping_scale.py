"""Compare a known absolute statistical scale and a strict endpoint on pure lasso."""
import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np

from hybrid_solver import GraphSplitting, make_regression, soft
from pilot_continuation import step
from stopping_sensitivity import save_rows


def run():
    n, p, sparsity = 256, 64, 4
    sigma, delta, kappa = .5, .05, 1.
    rng = np.random.default_rng(1107)
    q, _ = np.linalg.qr(rng.standard_normal((n, p)), mode="reduced")
    X = np.sqrt(n) * q
    np.testing.assert_allclose(X.T @ X / n, np.eye(p), atol=2e-14)
    truth = np.zeros(p)
    truth[:sparsity] = [1., -1., 1., -1.]
    lam = 2 * sigma * np.sqrt(2 * np.log(2*p/delta) / n)
    eps = .5 * lam**2 * sparsity / kappa
    rows = []
    for seed in range(5):
        noise = np.random.default_rng(33071 + seed).normal(scale=sigma, size=n)
        score = np.max(np.abs(X.T @ noise / n))
        if 2*score > lam:
            raise RuntimeError(f"Score assumption failed at seed {seed}")
        y = X @ truth + noise
        exact = soft(X.T @ y / n, lam)
        p_star = .5*np.mean((X@exact-y)**2) + lam*np.linalg.norm(exact, 1)
        for m, j in ((4,4), (8,4)):
            prob = make_regression(X, y, m, j, l1=lam, ridge=0., loss="squared")
            for metric in ("unit", "budget"):
                solver = GraphSplitting(prob, rho=1., weights=metric)
                targets = {"statistical": eps, "strict": 1e-7}
                for k in range(1, 30001):
                    step(solver)
                    if k != 1 and k % 10:
                        continue
                    primal, dual, gap = solver.certificate()
                    if not dual <= p_star + 1e-12 <= primal + 2e-12:
                        raise ArithmeticError("Analytic optimum outside the gap interval")
                    for endpoint, threshold in list(targets.items()):
                        if gap > threshold:
                            continue
                        beta = np.concatenate([solver.eta[idx] for idx in solver.xs])
                        err = np.linalg.norm(X @ (beta-truth)) / np.sqrt(n)
                        opt_err = np.linalg.norm(X @ (beta-exact)) / np.sqrt(n)
                        bound = 1.5 * lam * np.sqrt(sparsity / kappa) + np.sqrt(2 * gap)
                        if err > bound + 1e-12 or opt_err > np.sqrt(2*gap) + 1e-12:
                            raise ArithmeticError("Certificate violation")
                        rows.append(dict(seed=seed, row_parts=m, column_parts=j, metric=metric,
                                         endpoint=endpoint, threshold=threshold, iterations=k, gap=gap,
                                         score=score, lambda1=lam, kappa=kappa, sparsity=sparsity,
                                         prediction_error=err, prediction_bound=bound,
                                         prediction_distance_to_solution=opt_err, primal=primal, dual=dual))
                        del targets[endpoint]
                    if not targets:
                        break
                if targets:
                    raise RuntimeError("Missing lasso endpoint")
    assert len(rows) == 40
    save_rows("lasso_stopping_scale.csv", rows)
    print(f"20 pairs, absolute statistical target {eps:.8g} and strict target 1e-7.")


if __name__ == "__main__":
    run()
