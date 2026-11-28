"""Generate manuscript timing text and supplementary fusion ranges from numerical results."""
import hashlib
import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"


def timing_text():
    csv_path = RESULTS / "centralized_benchmark.csv"
    environment = json.loads((RESULTS / "centralized_benchmark_environment.json").read_text())
    if environment["csv_sha256"] != hashlib.sha256(csv_path.read_bytes()).hexdigest():
        raise ValueError("Timing data and their environment record are from different runs.")
    timing = pd.read_csv(csv_path)
    groups = ["model", "method", "seed"]
    if (len(timing) != 60 or timing.duplicated(groups + ["repetition"]).any()
            or not timing.groupby(groups).size().eq(3).all()
            or not timing.groupby(groups).iterations.nunique().eq(1).all()
            or not timing.gap_target.eq(1e-6).all()):
        raise ValueError("Expected three consistent timings per method, model and seed at gap 1e-6.")
    metrics = ["iterations", "total_seconds"]
    by_sample = timing.groupby(groups)[metrics].median()
    reference = by_sample.groupby(["model", "method"]).median()
    source = json.loads((RESULTS / "stopping_comparisons.json").read_text())["benchmark"]
    summary = pd.DataFrame(source).set_index(["model", "method"])[metrics].sort_index()
    pd.testing.assert_frame_equal(reference, summary, check_exact=False, rtol=1e-12, atol=1e-12)

    original = pd.read_csv(RESULTS / "runs.csv")
    original = original.loc[(original.kind == "synthetic") & (original.row_parts == 4)
                            & (original.column_parts == 4) & (original.metric == "budget")
                            & (original.rho == 1) & (original.zero_edges == 0)
                            & original.loss.isin(["huber", "logistic"])]
    projected = original.rename(columns={"loss": "model"}).set_index(["model", "seed"])
    gradient_graph = by_sample.xs("graph", level="method")
    gradient_fista = by_sample.xs("fista", level="method")
    expected = pd.MultiIndex.from_product([["huber", "logistic"], range(5)], names=["model", "seed"])
    for data in (projected, gradient_graph, gradient_fista):
        pd.testing.assert_index_equal(data.index.sort_values(), expected)

    change = gradient_graph.iterations - projected.iterations
    earlier = int((change.loc["huber"] < 0).sum())
    ties = int((change.loc["logistic"] == 0).sum())
    later = int((change.loc["logistic"] > 0).sum())
    paired_text = (
        f"the loss gradient gives an earlier certificate in {earlier}/5 Huber fits. "
        f"For logistic regression, it gives the same stopping iteration as the anchor multiplier "
        f"in {ties}/5 fits and a later one in {later}/5."
    )

    def sample_counts(data):
        cells = []
        for model in ("huber", "logistic"):
            counts = data.loc[model, "iterations"]
            cells.append(f"{counts.mean():,.0f} ({counts.median():,.0f})")
        return " & ".join(cells)

    rows = [f"Graph & Projected anchor & {sample_counts(projected)}",
            f"Graph & Loss gradient & {sample_counts(gradient_graph)}",
            f"FISTA & Loss gradient & {sample_counts(gradient_fista)}"]

    def elapsed_pair(model):
        fista = 1000 * reference.loc[(model, "fista"), "total_seconds"]
        graph = 1000 * reference.loc[(model, "graph"), "total_seconds"]
        return f"{fista:.0f} and {graph:.0f} ms"

    return r"""% Generated from matched stopping records by code/publication_excerpts.py.
\begin{table}[pos=tbp]
\centering\small
\caption{First checkpoint meeting a relative gap of $10^{-6}$. Entries are
mean (median) iterations over the same five samples. Both graph rows use
the same $4\times4$ budget-weighted iterates at $\rho=1$.}
\label{tab:benchmark-candidates}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrr@{}}
\toprule
Solver & Dual candidate & Huber & Logistic \\
\midrule
""" + "".join(row + r" \\" + "\n" for row in rows) + r"""\bottomrule
\end{tabular*}
\end{table}

On the same graph iterates (Table~\ref{tab:benchmark-candidates}),
""" + paired_text + r"""
The single-machine benchmark pairs the budget graph with centralized
accelerated proximal gradient (FISTA, \citealp{BeckTeboulle2009}).
A FISTA step applies $\bm X$ and $\bm X^{\mathsf T}$ once each and evaluates a
coordinatewise elastic-net proximal map. A graph sweep calls 16 local
projections in sequence, each using its cached Cholesky factor, and
updates the factor and coefficient variables. These implementations use
vectorized matrix operations for FISTA and a Python loop over graph edges.

With the loss-gradient certificate, median elapsed times for FISTA and
graph splitting were """ + elapsed_pair("huber") + " for Huber and " + elapsed_pair("logistic") + r"""
for logistic regression. The timings include setup and gap checks on one
BLAS thread. The reported times are medians over five samples, each
represented by the median of three repetitions. These elapsed times depend on the
implementation and hardware and exclude network communication.
"""


def scientific_tex(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def read_fusion_ranges(filename, layouts):
    ranges = pd.read_csv(RESULTS / filename, float_precision="round_trip")
    keys = ["seed", "row_parts", "column_parts", "metric"]
    expected = {(seed, m, j, weight) for seed in range(5)
                for m, j in layouts for weight in ("unit", "budget")}
    if len(ranges) != len(expected) or set(ranges[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("Missing or duplicate layout/weight combinations in the fused multiplier study.")
    columns = ["min_C0", "max_C0", "min_linearization_error", "max_kkt_violation"]
    if (not all(map(math.isfinite, ranges[columns].to_numpy().flat))
            or (ranges.min_C0 <= 0).any() or (ranges.max_C0 < ranges.min_C0).any()
            or (ranges[["min_linearization_error", "max_kkt_violation"]] < 0).any().any()):
        raise ValueError("Invalid fused distance range or verification residual.")
    return ranges


def fusion_table():
    from fused_dual_ranges import ENDPOINT_MARGIN, FIXED16, LAYOUTS, compare_ranges

    margin_tex = scientific_tex(ENDPOINT_MARGIN).removeprefix(r"1.00\times ")
    ranges = read_fusion_ranges("fused_dual_ranges.csv", LAYOUTS)
    extra = read_fusion_ranges("fused_dual_fixed16_extra.csv", set(FIXED16) - set(LAYOUTS))
    all_ranges = pd.concat([ranges, extra], ignore_index=True)
    records = all_ranges.to_dict("records")
    comparisons = {}
    for study, layouts in (("primary", LAYOUTS), ("fixed16", FIXED16)):
        pairs = pd.DataFrame(compare_ranges(records, layouts))
        published = pd.read_csv(RESULTS / f"fused_dual_{study}_pairs.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(pairs, published, rtol=1e-13, atol=1e-11)
        comparisons[study] = pairs
    primary, fixed = comparisons["primary"], comparisons["fixed16"]
    overlaps = primary[primary.overlaps]
    overlap_detail = ""
    if len(overlaps) == 1:
        pair = overlaps.iloc[0]
        metrics = {"unit": "unit weights", "budget": "uniform budgets"}
        def describe(side):
            return (rf"${pair[side+'_rows']}\times{pair[side+'_columns']}$ with {metrics[pair[side+'_metric']]} "
                    rf"($[{pair[side+'_lower']:.3f},{pair[side+'_upper']:.3f}]$)")
        overlap_detail = f" At seed {pair.seed}, {describe('first')} overlapped {describe('second')}."
    width = 100 * (ranges.max_C0 - ranges.min_C0) / ranges.min_C0
    bounds = width.groupby(ranges.seed).agg(["min", "max"])
    tabular_rows = [
        " & ".join([label, *(f"{value:.3f}" for value in bounds[col])]) + r" \\"
        for col, label in (("min", "Smallest width"), ("max", "Largest width"))
    ]
    return r"""% Generated from the fused multiplier ranges and paired comparisons by code/publication_excerpts.py.
\begin{table}[H]
\centering
\caption{Relative width of the fused-regression distance constant,
$100(C_{0,\max}-C_{0,\min})/C_{0,\min}$, in percent.
Each seed covers four primary layouts and two metrics.}
\label{tab:fused-widths}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Seed & 0 & 1 & 2 & 3 & 4 \\ \midrule
""" + "\n".join(tabular_rows) + r"""
\bottomrule\end{tabular}
\end{table}
""" + (rf"The largest gap between a numerical minimum and its linearization lower bound was "
       rf"${scientific_tex(ranges.min_linearization_error.max())}$ for the {len(ranges)} primary ranges and "
       rf"${scientific_tex(extra.min_linearization_error.max())}$ for the {len(extra)} additional ranges "
       rf"($2\times8$ and $8\times2$). Across all {len(all_ranges)} ranges the largest recorded KKT residual "
       rf"was ${scientific_tex(all_ranges.max_kkt_violation.max())}$." + "\n\n"
       rf"For each pair, the comparison uses the linearization lower bound and the computed maximum, "
       rf"moving both outward by ${margin_tex}$ in $C_0$ units. "
       rf"Intervals overlapped in {len(overlaps)} of {len(primary)} within-seed primary pairs."
       + overlap_detail + " "
       rf"At 16 blocks, {fixed.overlaps.sum()} of {len(fixed)} pairs overlapped "
       rf"(smallest separation ${fixed.separation.min():.4f}$)." + "\n")


if __name__ == "__main__":
    TABLES.mkdir(exist_ok=True)
    for filename, content in (("experiment_timing_text.tex", timing_text()),
                              ("supp_fused_widths.tex", fusion_table())):
        (TABLES / filename).write_text(content, encoding="utf-8")
        print(f"Generated tables/{filename}.")
