"""Summaries and manuscript table from the endpoint and timing experiments."""
import json

import pandas as pd

from stopping_sensitivity import RESULTS, ROOT


MODELS = {"fused": "Squared fused", "huber": "Huber", "logistic": "Logistic",
          "diabetes": "Diabetes", "breast_cancer": "Breast cancer"}


def read(name):
    return pd.read_csv(RESULTS / (name+".csv"))


def fixed_ratios(rows):
    means = rows.groupby(["model", "row_parts", "column_parts", "metric"]).iterations.mean().unstack()
    return means.budget / means.unit


def consensus_traffic(fixed, consensus):
    graph = fixed[(fixed.column_parts == 4) & (fixed.metric == "budget")]
    keys = ["model", "row_parts"]
    return (graph.groupby(keys).communicated_scalars.mean() /
            consensus.groupby(keys).communicated_scalars.mean()).dropna()


def continuation_summary(df):
    out = {}
    for data_kind, group in df.groupby("data"):
        costs = group.pivot(index=["model", "seed", "row_parts", "column_parts"],
                            columns="method", values="payload_with_certificate_bound")
        learned = costs["unit_continue"]
        out[data_kind] = {}
        for name, comparator in (("restart", costs["unit_restart"]), ("best_fixed", costs[["unit","budget"]].min(axis=1))):
            out[data_kind][name] = comparison(learned, comparator)
        out[data_kind]["early_unit_pilots"] = int(((group.method == "unit_continue") & (group.transition_payload == 0)).sum())
    return out


def comparison(learned, comparator):
    return {"n": len(learned), "lower": int((learned < comparator).sum()),
            "equal": int((learned == comparator).sum()), "higher": int((learned > comparator).sum()),
            "median_pct_change": float(100 * (learned / comparator - 1).median())}


def ratio_table(loose, strict):
    header = r"""% Generated from the fixed-weight stopping records by stopping_summaries.py.
\begin{table}[pos=tbp]
\centering
\small
\caption{Budget-to-unit iteration ratios at two gap targets.}
\label{tab:tolerance-rankings}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}
\toprule
& \multicolumn{2}{c}{$4\times4$} & \multicolumn{2}{c}{$8\times4$}\\
\cmidrule(lr){2-3}\cmidrule(l){4-5}
Model & $10^{-3}$ & Strict & $10^{-3}$ & Strict \\
\midrule
"""
    lines = []
    for model, label in MODELS.items():
        if model == "diabetes":
            lines.append(r"\addlinespace[2pt]")
        ratios = [series[model, m, 4] for m in (4,8) for series in (loose, strict)]
        lines.append(label + " & " + " & ".join(f"{r:.2f}" for r in ratios) + r" \\")
    footer = r"""
\bottomrule
\end{tabular*}
\par\vspace{3pt}
\begin{minipage}{\linewidth}
\footnotesize\raggedright Projected anchor multipliers certify each relative gap. Strict targets are
$10^{-6}$ for simulations and $10^{-7}$ for public data. Ratios use unrounded
five-sample mean iterations or counts from the single public-data split.
Values below one favor budget weights.
\end{minipage}
\end{table}
"""
    (ROOT / "tables" / "tolerance_rankings.tex").write_text(header + "\n".join(lines) + footer)


def main():
    fixed = read("runs")
    fixed = fixed[fixed.zero_edges == 0].copy()
    fixed["model"] = fixed.case.str.replace(r"^synthetic_|^real_|_\d+$", "", regex=True)
    loose = read("stopping_loose_fixed")
    ratios, initial = fixed_ratios(loose), fixed_ratios(fixed)
    ratio_table(ratios, initial)
    contrasts = [{"model": model, "row_parts": m, "column_parts": j, "loose": r,
                  "strict": initial[model, m, j]} for (model,m,j),r in ratios.items()]
    consensus = read("consensus_baseline")
    consensus["model"] = consensus.case.str.replace(r"^synthetic_|_\d+$", "", regex=True)
    c_loose = consensus_traffic(loose, read("stopping_loose_consensus"))
    c_strict = consensus_traffic(fixed, consensus)
    c_ratios = [dict(model=model, row_parts=m, graph_to_consensus_at_loose=ratio,
                     graph_to_consensus_at_strict=c_strict[model,m]) for (model,m),ratio in c_loose.items()]
    c_long = pd.concat([read("pilot_continuation_runs"), read("pilot_continuation_public")])
    cont = dict(strict=continuation_summary(c_long), loose=continuation_summary(read("stopping_loose_continuation")))
    ctrl = {}
    for label, frame, left, right in (("loose",read("stopping_loose_matched"),"learned_payload","uniform_payload"),
                                      ("strict",read("matched_output_control"),"learned_scalars","uniform_scalars")):
        ctrl[label] = {name: comparison(g[left],g[right]) for name,g in frame.groupby("data")}
    lasso = read("lasso_stopping_scale")
    lasso_summary = []
    for ep,g in lasso.groupby("endpoint"):
        lasso_summary.append(dict(endpoint=ep, target=float(g.threshold.iloc[0]),
                                  it_min=int(g.iterations.min()), it_max=int(g.iterations.max()),
                                  pred_error_min=float(g.prediction_error.min()), pred_error_max=float(g.prediction_error.max())))
    benchmark = read("centralized_benchmark")
    seed_medians = benchmark.groupby(["model","method","seed"])[["iterations","total_seconds"]].median()
    timings = seed_medians.groupby(["model","method"]).median().reset_index().to_dict(orient="records")
    output = dict(fixed_iteration_ratios=contrasts, consensus_optimization_ratios=c_ratios,
                  continuation=cont, matched_output_controls=ctrl, lasso=lasso_summary, benchmark=timings)
    (RESULTS / "stopping_comparisons.json").write_text(json.dumps(output, indent=2) + "\n")
    print("Regenerated tolerance_rankings.tex and stopping_comparisons.json.")


if __name__ == "__main__":
    main()
