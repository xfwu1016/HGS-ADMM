"""Compare three output allocations after the same unit-weight pilot.

Run with the archived solver and results (Python 3.12, original requirements):
python reproduce.py --step matched-control --jobs 3

All three arms keep the primal and unscaled dual state at a single weight change.
The prescribed weights follow from graph degrees. The zero-start and
retained-state rules both pay the same allocation cost. Every switched arm
pays for one refresh of the edge vectors. High-accuracy coefficients are
read only for retrospective evaluation of the residual-bound numerator.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from statistics import median

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np


def load_solver(code_root):
    sys.path.insert(0, str(code_root / "code"))


@lru_cache(maxsize=2)
def get_case(model, seed, public):
    from run_experiments import prepare_real_case, synthetic_case
    if public:
        return prepare_real_case(model, split_seed=seed, save_split=False)
    return synthetic_case(model, seed)


def retained_weights(pilot, diagnostics, learned):
    from pilot_budget_experiment import water_fill

    shares = np.zeros(len(pilot.edges))
    for star in diagnostics["stars"]:
        indices = pilot.edge_by_factor[star["factor"]]
        deviations = []
        for e in indices:
            a, j, operator = pilot.edges[e]
            deviations.append(np.linalg.norm(operator @ pilot.eta[pilot.xs[j]]
                                             - pilot.eta[pilot.yes[e]]))
        upper = np.asarray(deviations) + diagnostics["radius"] * np.asarray(star["operator_norm"])
        shares[indices] = water_fill(upper, diagnostics["floor_fraction"])
    return dict(learned, b=1 / shares)


def reference_terms(pilot, reference, arms, floor_fraction):
    """Evaluate only the part of the suffix numerator that varies with output weights."""
    from pilot_budget_experiment import water_fill

    blocks = np.split(reference, np.cumsum(pilot.problem.block_sizes)[:-1])
    deviations = np.array([np.linalg.norm(operator @ blocks[j] - pilot.eta[pilot.yes[e]])
                           for e, (_, j, operator) in enumerate(pilot.edges)])
    best = np.zeros_like(deviations)
    for indices in pilot.edge_by_factor:
        best[indices] = water_fill(deviations[indices], floor_fraction)
    squared = pilot.rho * deviations ** 2
    terms = {name + "_reference_term": float(np.dot(squared, weights["b"]))
             for name, weights, _ in arms}
    minimum = float(np.sum(squared / best))
    if any(v < minimum - 2e-12 * max(1, minimum) for v in terms.values()):
        raise ArithmeticError("Retrospective minimum exceeds an allocation's value")
    return dict(terms, optimal_reference_term=minimum)


def run_pair(task):
    row, results_dir = task
    from hybrid_solver import make_regression
    from pilot_budget_experiment import make_pilot_weights
    from pilot_continuation import CAP, run_prefix, solve_from, switch_output_metric

    model, seed = row["model"], int(row["seed"])
    m, j = int(row["row_parts"]), int(row["column_parts"])
    case = get_case(model, seed, row["data"] == "real")
    problem = make_regression(case["X"], case["y"], m, j, **case["args"])
    length, tol = int(row["requested_prefix_iterations"]), float(row["gap_target"])
    pilot, history, _, _ = run_prefix(problem, "unit", length, tol)
    pilot_finished = history[-1][-1] <= tol
    if pilot_finished:
        arms = (("learned", None, 0), ("uniform", None, 0), ("centered", None, 0))
    else:
        estimate = np.concatenate([pilot.eta[s] for s in pilot.xs])
        learned, diagnostics = make_pilot_weights(problem, {"coef": estimate, "gap": history[-1][3]})
        centered = retained_weights(pilot, diagnostics, learned)
        uniform = dict(a=np.ones_like(pilot.a), c=np.ones_like(pilot.c),
                       b=np.array([pilot.da[a] for a, j, A in pilot.edges], dtype=float),
                       d=np.array([1.0 / pilot.dj[j] for a, j, A in pilot.edges]))
        for name in ("a", "c", "d"):
            np.testing.assert_array_equal(learned[name], uniform[name])
        allocation = 3 * len(pilot.edges)
        arms = (("learned", learned, allocation), ("uniform", uniform, 0),
                ("centered", centered, allocation))
    c_iter = pilot.communication_per_iteration
    refresh = 0 if pilot_finished else c_iter // 2
    per_check = c_iter + 3 * (len(problem.factors) + len(problem.block_sizes))
    out = dict(data=row["data"], model=model, seed=seed, sample_blocks=m,
               feature_blocks=j, pilot_iterations=history[-1][0], gap_target=tol,
               scalars_per_iteration=c_iter, refresh_scalars=refresh,
               scalars_per_gap_check_bound=per_check, learned_allocation_scalars=arms[0][2])
    for arm, weights, allocation in arms:
        trace = history
        if weights is not None:
            state = switch_output_metric(pilot, weights)
            trace, _ = solve_from(state, history, CAP - length, tol)
        steps, primal, dual, gap, relative_gap = trace[-1]
        if not 0 <= relative_gap <= tol or any(t[-1] <= tol for t in trace[:-1]):
            raise RuntimeError("Invalid first stopping checkpoint")
        total = steps*c_iter + allocation + refresh + len(trace)*per_check
        out.update({arm+"_iterations": steps, arm+"_checks": len(trace),
                    arm+"_relative_gap": relative_gap, arm+"_scalars": total})
        if arm == "learned":
            for name, value in (("total_iterations",steps), ("checkpoints",len(trace)),
                                ("allocation_payload",allocation), ("transition_payload",refresh),
                                ("certificate_payload_bound",len(trace)*per_check),
                                ("payload_with_certificate_bound",total)):
                if int(row[name]) != value:
                    raise AssertionError(f"Archived {name} differs in {row['case']}, {m} x {j}")
            np.testing.assert_allclose([primal,dual,gap], [float(row[name]) for name in
                                       ("objective","dual","absolute_gap")], rtol=1e-8, atol=2e-12)
    out["paired_percent_change"] = 100*(out["learned_scalars"]/out["uniform_scalars"]-1)
    # Keep the previously published two-arm record and its schema intact.
    original = {key: value for key, value in out.items() if not key.startswith("centered_")}
    terms = dict.fromkeys([name + "_reference_term" for name in
                           ("learned", "uniform", "centered", "optimal")])
    if not pilot_finished:
        path = results_dir / "coefficients" / (row["case"] + "_reference.npz")
        with np.load(path) as stored:
            terms = reference_terms(pilot, stored["coef"], arms, diagnostics["floor_fraction"])
    retained = dict(out, **terms, centered_allocation_scalars=arms[2][2])
    return original, retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--archive-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("matched_output_control.csv"))
    parser.add_argument("--retained-output", type=Path, default=Path("retained_output_control.csv"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    load_solver(code_root)
    from pilot_continuation import DESIGN, PUBLIC_CASES, PUBLIC_LAYOUTS
    records = []
    for part in ("runs", "public"):
        path = args.archive_results / f"pilot_continuation_{part}.csv"
        with path.open(newline="") as stream:
            records.extend(r for r in csv.DictReader(stream) if r["method"] == "unit_continue")
    keys = [(r["model"], int(r["seed"]), int(r["row_parts"]), int(r["column_parts"])) for r in records]
    expected = {(k,s,m,j) for k in DESIGN["models"] for s in DESIGN["seeds"] for m,j in DESIGN["layouts"]}
    expected |= {(k,2026,m,j) for k in PUBLIC_CASES for m,j in PUBLIC_LAYOUTS}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Archived configurations differ from the complete manuscript design")
    with ProcessPoolExecutor(args.workers, initializer=load_solver, initargs=(code_root,)) as pool:
        outputs = list(pool.map(run_pair, ((r, args.archive_results) for r in records)))
    rows, retained = zip(*outputs)
    for path, data in ((args.output, rows), (args.retained_output, retained)):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
    for label in ("synthetic","real"):
        ratios = [x["paired_percent_change"] for x in rows if x["data"] == label]
        print(f"{label}: learned weights save scalars in {sum(x<0 for x in ratios)}/{len(ratios)} pairs; "
              f"median paired percentage change {median(ratios):+.3f}%")
        group = [x for x in retained if x["data"] == label]
        for comparator in ("uniform", "learned"):
            changes = [100 * (x["centered_scalars"]/x[comparator+"_scalars"]-1) for x in group]
            switched = [x for x in group if x["centered_reference_term"] is not None]
            lower = sum(x["centered_reference_term"] < x[comparator+"_reference_term"] for x in switched)
            tied = sum(x["centered_iterations"] == x[comparator+"_iterations"] for x in group)
            print(f"{label}: retained-state allocation vs {comparator} saves scalars "
                  f"in {sum(x<0 for x in changes)}/{len(changes)}, "
                  f"ties {sum(x==0 for x in changes)}, median {median(changes):+.4f}%; "
                  f"lower reference term in {lower}/{len(switched)}, "
                  f"equal iteration count in {tied}/{len(group)}")

if __name__ == "__main__":
    main()
