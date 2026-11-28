"""Verify an example using only exact rational arithmetic (Python 3).

Use one two-dimensional factor f(h) = ||h-y||^2/2, y=(1,1/10), and
two scalar feature penalties r_j(beta_j)=beta_j^2/2. The two nonzero
edge operators are e_1 and e_2, the columns of I_2. Each edge has its
own two-dimensional output. Set rho=a=c_j=d_j=1, b_j=1/omega_j.

The exact optimum is beta*=y/2, h*=beta*, theta*=h*-y=-beta*,
g*=theta*. Each feature is incident on one factor, so pi_j=1.
In the zero-start squared distance the primal part is
    3||beta*||^2 + sum_j beta*_j^2/omega_j.
The dual part is 4||beta*||^2. Cauchy--Schwarz minimizes the sum
at omega_j=|beta*_j|/sum_i |beta*_i|, giving (10/11,1/11).
Uniform weights yield C0=909/400 and the oracle yields C0=207/100.

The dual for this representation is D(theta)=-y.theta-||theta||^2.
The loss conjugate has full domain, so the manuscript's projected
factor-anchor multiplier is simply theta=-u_h at rho=a=1.
For the loss-gradient candidate theta=beta-y the gap is twice the primal
objective error. Both are also checked on the same exact iterates.
All tolerances, updates, comparisons, and residual checks use fractions.
Only the stored decimal gap uses floating point.
"""

import csv
import os
from fractions import Fraction as F
from pathlib import Path

RESULTS = Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results"))


Y = (F(1), F(1, 10))
TARGET = F(1, 10**6)
H = (0, 1)
BETA = (2, 3)
EDGE_H = ((4, 5), (6, 7))
EDGE_BETA = (8, 9)
DIMENSION = 10


def constant(weights):
    beta_star = tuple(y / 2 for y in Y)
    return 7 * sum(b*b for b in beta_star) + sum(
        b*b/w for b, w in zip(beta_star, weights)
    )


def step(eta, dual, weights):
    inputs = [e - u for e, u in zip(eta, dual)]
    local = [F(0)] * DIMENSION
    for j, w in enumerate(weights):
        local[H[j]] = (inputs[H[j]] + Y[j]) / 2
        local[BETA[j]] = inputs[BETA[j]] / 2
        # Project (h_e,b_e) onto h_e=e_j*b_e. The other output is zero.
        diagonal = EDGE_H[j][j]
        coefficient = (inputs[diagonal] + w*inputs[EDGE_BETA[j]]) / (1+w)
        local[EDGE_BETA[j]] = local[diagonal] = coefficient

    consistent = local.copy()
    for i in range(2):
        discrepancy = local[H[i]] - sum(local[row[i]] for row in EDGE_H)
        # The exchange denominator is 1+omega_1+omega_2=2.
        consistent[H[i]] -= discrepancy / 2
        for row, w in zip(EDGE_H, weights):
            consistent[row[i]] += w * discrepancy / 2
        consensus = (local[BETA[i]] + local[EDGE_BETA[i]]) / 2
        consistent[BETA[i]] = consistent[EDGE_BETA[i]] = consensus

    next_dual = [u + x - e for u, x, e in zip(dual, local, consistent)]
    for i in range(2):
        assert consistent[H[i]] == sum(consistent[row[i]] for row in EDGE_H)
        assert next_dual[BETA[i]] == -next_dual[EDGE_BETA[i]]
        for row, w in zip(EDGE_H, weights):
            assert next_dual[row[i]] == -w*next_dual[H[i]]
    return consistent, next_dual


def first_hits(errors):
    first = next(k for k, error in errors.items() if error <= TARGET)
    scheduled = next(k for k, error in errors.items()
                     if (k == 1 or k % 10 == 0) and error <= TARGET)
    return first, scheduled


def evaluate(name, weights, anchor_hits, gradient_hits, primal_hits):
    c0 = constant(weights)
    metric = [F(1)] * DIMENSION
    for row, w in zip(EDGE_H, weights):
        for slot in row:
            metric[slot] = 1/w
    eta = [F(0)] * DIMENSION
    dual = eta.copy()
    gaps = {}
    gradient_gaps = {}
    primal_errors = {}
    primal_optimum = sum(y*y / 4 for y in Y)
    for k in range(1, 41):
        next_eta, next_dual = step(eta, dual, weights)
        residual_squared = sum(
            w*((e1-e0)**2 + (u1-u0)**2)
            for w, e1, e0, u1, u0 in zip(metric, next_eta, eta, next_dual, dual)
        )
        assert k*residual_squared <= c0
        eta, dual = next_eta, next_dual
        beta = [eta[j] for j in BETA]
        theta = [-dual[i] for i in H]
        primal = sum(((b-y)**2 + b**2)/2 for b, y in zip(beta, Y))
        dual_value = -sum(y*t + t**2 for y, t in zip(Y, theta))
        assert 0 <= primal <= 1  # Absolute and manuscript relative gaps coincide.
        gaps[k] = primal - dual_value
        assert gaps[k] >= 0
        gradient = [b-y for b, y in zip(beta, Y)]
        gradient_dual = -sum(y*t + t*t for y, t in zip(Y, gradient))
        gradient_gaps[k] = primal - gradient_dual
        primal_errors[k] = primal - primal_optimum
        assert gradient_gaps[k] == 2 * primal_errors[k]

    first, first_checked = first_hits(gaps)
    checkpoints = [k for k in gaps if k == 1 or k % 10 == 0]
    assert (first, first_checked) == anchor_hits
    assert first_hits(gradient_gaps) == gradient_hits
    assert first_hits(primal_errors) == primal_hits
    print(f"{name}: C0={c0}, first hit={first}, first scheduled hit={first_checked}")
    print(f"  (every step, scheduled) loss gradient={gradient_hits}, primal error={primal_hits}")
    return [
        {
            "allocation": name,
            "omega_1": str(weights[0]),
            "omega_2": str(weights[1]),
            "C0": str(c0),
            "iteration": k,
            "scheduled_check": int(k in checkpoints),
            "relative_gap": float(gap),
            "meets_target": int(gap <= TARGET),
            "first_hit_every_iteration": first,
            "first_hit_scheduled": first_checked,
        }
        for k, gap in gaps.items()
    ]


if __name__ == "__main__":
    uniform = (F(1, 2), F(1, 2))
    oracle = (F(10, 11), F(1, 11))
    assert constant(uniform) == F(909, 400)
    assert constant(oracle) == F(207, 100)
    rows = evaluate("uniform", uniform, (14, 20), (15, 20), (14, 20))
    rows += evaluate("oracle", oracle, (27, 30), (27, 30), (25, 30))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / "oracle_stopping_example.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
