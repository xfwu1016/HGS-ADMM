"""Exact ridge example for the partition constant; no estimated optimum is used.

Run from any directory: python code/analytic_partition.py
This script intentionally uses a direct vector implementation independent of
the general solver in hybrid_solver.py.
"""
from pathlib import Path
import os
import csv
import numpy as np

RESULTS = Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results"))


def run(rows=8, metric="budget", n=64, mu=0.2, rho=1.0, tolerance=1e-10):
    groups = np.array_split(np.arange(n), rows)
    a = 1.0 / np.sqrt(n)
    target = np.ones(n) / np.sqrt(n)
    d = np.full(rows, 1.0 / rows if metric == "budget" else 1.0)
    h = np.zeros(n); beta = 0.0; he = np.zeros(n); be = np.zeros(rows)
    uh = np.zeros(n); ub = 0.0; uhe = np.zeros(n); ube = np.zeros(rows)
    optimum = 1.0 / (1.0 + mu)
    theta_star = (optimum - 1.0) * np.ones(n) / np.sqrt(n)
    primal_constant = (4.0 if metric == "budget" else rows + 3.0) * optimum**2
    dual_constant = (4.0 if metric == "budget" else 3.0 + 1.0/rows) * mu**2 * optimum**2
    c0 = rho * primal_constant + dual_constant / rho
    residuals = []
    records = []
    for k in range(1, 200001):
        hp = (target + rho*(h-uh))/(1.0+rho)
        bp = rho*(beta-ub)/(rho+mu)
        bep = np.empty(rows); hep = np.empty(n)
        for i, idx in enumerate(groups):
            bep[i] = (d[i]*(beta-ube[i]) + a*np.sum(he[idx]-uhe[idx])) / (d[i]+len(idx)/n)
            hep[idx] = a*bep[i]
        beta_new = (bp + np.dot(d,bep))/(1.0+d.sum())
        delta = (hp-hep)/2.0
        h_new = hp-delta; he_new = hep+delta
        # Full-space weighted residual, including all consistent coefficient copies.
        residual2 = rho*((1+d.sum())*(beta_new-beta)**2
                         + np.sum((h_new-h)**2) + np.sum((he_new-he)**2))
        residual2 += rho*(np.sum((hp-h_new)**2) + (bp-beta_new)**2
                         + np.sum((hep-he_new)**2) + np.dot(d,(bep-beta_new)**2))
        uh += hp-h_new; ub += bp-beta_new
        uhe += hep-he_new; ube += bep-beta_new
        h, beta, he = h_new, beta_new, he_new
        primal = .5*(beta-1.0)**2+.5*mu*beta**2
        # theta from factor proximal optimality belongs to the squared-loss dual domain.
        theta = hp-target
        gradient = a*np.sum(theta)
        dual = -.5*np.dot(theta,theta)-np.dot(target,theta)-gradient**2/(2.0*mu)
        gap = primal-dual
        residuals.append(residual2)
        records.append({"rows":rows,"metric":metric,"iteration":k,"gap":gap,
                        "residual_squared":residual2,"constant_over_k":c0/k})
        if gap <= tolerance:
            break
    rs = np.asarray(residuals)
    assert np.max(np.diff(rs)) < 1e-12, "Fixed-metric residual is not monotone"
    assert np.max(np.arange(1,len(rs)+1)*rs-c0) < 1e-11, "Theoretical bound failed"
    assert abs(beta-optimum) <= np.sqrt(2*max(gap,0)/mu)+1e-10
    assert np.max(abs(uhe+uh)) < 1e-12
    assert abs(ub+np.dot(d,ube)) < 1e-12
    result = {"rows": rows, "metric":metric, "primal_constant":primal_constant,
              "dual_constant":dual_constant,"C0":c0,"iterations":k,
              "gap":gap,"coefficient":beta,"exact_coefficient":optimum,
              "max_k_residual2_over_C0":max(np.arange(1,len(rs)+1)*rs/c0)}
    return result, records


def main():
    RESULTS.mkdir(exist_ok=True)
    summary=[]; traces=[]
    for rows in (1,2,4,8,16,32):
        for metric in ("unit","budget"):
            result, trace=run(rows,metric)
            summary.append(result);traces.extend(trace)
    for name,data in (("analytic_partition.csv",summary),("analytic_partition_traces.csv",traces)):
        with (RESULTS/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    print("Exact ridge partition study completed; all invariance and rate checks passed.")
    for row in summary:
        print(f"M={row['rows']:2d} {row['metric']:6s}: C0={row['C0']:.6f}, iterations={row['iterations']}, gap={row['gap']:.3e}")


if __name__ == "__main__":
    main()
