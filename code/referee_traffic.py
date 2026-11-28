"""Construct update and certificate payload tables from recorded stopping checkpoints.

Run from any directory with Python >= 3.9. The script reads the original run CSVs
and creates derived results/tables. All costs describe point-to-point scalars,
not measurements on a physical network. Every first-hit checkpoint is charged once.
No solving, external dependencies, or modification of inputs is needed.
"""

import csv
import json
from pathlib import Path
from statistics import mean, stdev

from stopping_checks import checkpoint_count
from table_values import iteration_mean, sig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "tables"
MODELS = {"squared": "Squared fused", "huber": "Huber", "logistic": "Logistic"}


def load_rows(filename):
    with (RESULTS / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(filename, rows):
    with (RESULTS / filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def account(record, method, cases):
    graph = method != "consensus"
    pilot_run = "main_iterations" in record
    case = record["case"]
    m = int(record["row_parts"])
    j = int(record["column_parts"]) if graph else 1
    n, p = int(record["n"]), int(record["p"])
    round_cost = int(record["communication_per_iteration"])
    k = int(record["main_iterations"] if pilot_run else record["iterations"])
    prefix = int(record.get("pilot_iterations", 0))
    checks = checkpoint_count(k)
    recorded_checks = int(record["main_diagnostic_evaluations"] if pilot_run
                          else record["diagnostic_evaluations"])
    if recorded_checks != checks:
        raise ValueError(f"Recorded and scheduled check counts differ for {case}: {recorded_checks} vs {checks}")
    checks += int(record.get("pilot_diagnostic_evaluations", 0))
    args = cases[case]["args"]
    groups = args.get("fusion_parts", 0) if args["fusion"] > 0 else 0
    data_per_round = 2 * (j * n + m * p) if graph else 2 * m * p
    structure_per_round = round_cost - data_per_round
    assert structure_per_round >= 0
    if not (graph and groups):
        assert structure_per_round == 0
    # In the sample-consensus update beta is broadcast before each local solve.
    # Each sample site can reuse that beta to form its loss and dual adjoint.
    check_cost = round_cost + 3 * (m + groups + j) if graph else m * (p + 3)
    updates = (k + prefix) * round_cost
    allocation = int(record.get("allocation_payload", 0))
    if pilot_run:
        assert updates + allocation == int(record["total_optimization_allocation_payload"])
        assert check_cost == int(record["certificate_payload_bound_per_check"])
    else:
        assert updates == int(record["communicated_scalars"])
    return dict(case=case, loss=record["loss"], row_parts=m, column_parts=j,
                metric=method, zero_edges=int(record.get("zero_edges", 0)),
                iterations=k, pilot_iterations=prefix, charged_checkpoints=checks,
                update_per_round=round_cost, data_per_round=data_per_round,
                structural_per_round=structure_per_round, update_payload=updates,
                allocation_payload=allocation, certificate_per_check_bound=check_cost,
                certificate_payload_bound=checks * check_cost,
                total_payload_bound=updates + allocation + checks * check_cost)


def grouped(records, loss, m, metric):
    rows = [r for r in records if r["case"].startswith("synthetic_") and
            r["loss"] == loss and r["row_parts"] == m and r["metric"] == metric and
            (metric == "consensus" or r["column_parts"] == 4) and not r["zero_edges"]]
    if len(rows) != 5:
        raise ValueError(f"Expected five runs for {loss}, {m}, {metric}")
    return rows


def iterations(rows):
    values = [r["iterations"] for r in rows]
    return f"{iteration_mean(mean(values))} ({sig(stdev(values))})"


def millions(rows, column):
    return f"{mean(r[column] for r in rows) / 1e6:.2f}"


def comparison_table(records, comparison):
    if comparison == "hybrid":
        names = ("unit", "budget")
        headers = ("Unit", "Budget")
        first_header = "Layout"
        caption = "Iterations and payload on hybrid data layouts."
        notes = (r"Unit and budget are fixed incidence metrics. Projected anchor multipliers certify "
                 r"the relative gap target $10^{-6}$. Percentage changes use unrounded means.")
    else:
        names = ("consensus", "budget")
        headers = ("Consensus", "Graph")
        first_header = "$M$"
        caption = "Sample consensus and budget-weighted graph splitting."
        notes = ("Consensus sites each store all features. The graph uses $M"+r"\times4$ blocks. "
                 r"Both target relative gap $10^{-6}$. The graph uses projected anchor multipliers. "
                 "Consensus projects its prediction and, where applicable, fusion multipliers.")
    header = [r"\begin{table*}[pos=tbp]", r"\centering\small",
              rf"\caption{{{caption}}}\label{{tab:{comparison}}}",
              r"\renewcommand{\arraystretch}{1.10}",
              r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcrrrrrr@{}}",
              r"\toprule", r" & & \multicolumn{2}{c}{Iterations} & "
              r"\multicolumn{2}{c}{Update scalars} & "
              r"\multicolumn{2}{c}{Updates + certificate bound}\\",
              r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
              " & ".join(["Model", first_header, *headers, *headers, *headers]) + r"\\",
              r"\midrule"]
    lines = header
    for loss, label in MODELS.items():
        if loss != "squared":
            lines.append(r"\addlinespace[2pt]")
        for m in (4, 8):
            groups = [grouped(records, loss, m, metric) for metric in names]
            layout = f"${m}"+r"\times4$" if comparison == "hybrid" else str(m)
            cells = [label if m == 4 else "", layout, *map(iterations, groups)]
            for column in ("update_payload", "total_payload_bound"):
                cells.extend(millions(g, column) for g in groups)
            lines.append(" & ".join(cells) + r"\\")
    lines.extend([r"\bottomrule\end{tabular*}", r"\par\vspace{3pt}",
                  r"\begin{minipage}{\textwidth}\footnotesize\raggedright "
                  + notes + " Iteration means over five samples are rounded to the nearest integer (sample SD in parentheses). "
                  "Payload is mean logical traffic in millions of scalars. "
                  "The last two columns add the protocol bound at each stopping check.",
                  r"\end{minipage}\end{table*}"])
    return "\n".join(lines) + "\n"


def model_settings(cases):
    lines = [r"\begin{table}[pos=htbp]", r"\centering\small",
             r"\caption{Parameters of the five regression models.}\label{tab:model-settings}",
             r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr@{}}",
             r"\toprule Model & $\lambda_1$ & $\mu$ & $\lambda_{\rm fuse}$ & $\delta$\\\midrule"]
    for case, title in (("synthetic_fused_0", "Squared fused"),
                        ("synthetic_huber_0", "Huber"),
                        ("synthetic_logistic_0", "Logistic"),
                        ("real_diabetes", "Diabetes"),
                        ("real_breast_cancer", "Breast cancer")):
        args = cases[case]["args"]
        if case.startswith("synthetic_"):
            assert all(cases[case[:-1] + str(s)]["args"] == args for s in range(5))
        values = [f'{args[key]:g}' for key in ("l1", "ridge", "fusion")]
        delta = f'{args["loss_parameter"]:g}' if args["loss"] == "huber" else "--"
        lines.append(" & ".join([title, *values, delta]) + r"\\")
    lines.extend([r"\bottomrule\end{tabular*}", r"\par\vspace{3pt}",
                  r"\begin{minipage}{\linewidth}\footnotesize\raggedright "
                  r"The ridge term is $\mu\|\bm\beta\|_2^2/2$. The Huber cutoff "
                  r"$\delta$ uses the response scale supplied to the loss. The public-data "
                  r"$\lambda_1$ values were selected on validation data using fits certified to relative gap $10^{-8}$ with projected anchor multipliers.",
                  r"\end{minipage}\end{table}"])
    return "\n".join(lines) + "\n"


def main():
    with (RESULTS / "cases.json").open(encoding="utf-8") as handle:
        cases = {r["case"]: r for r in json.load(handle)}
    records = [account(r, r["metric"], cases) for r in load_rows("runs.csv")]
    records.extend(account(r, "consensus", cases) for r in load_rows("consensus_baseline.csv"))
    pilot = [account(r, r["metric"], cases) for r in load_rows("pilot_budget_runs.csv")]
    write_csv("fixed_traffic_accounting.csv", records)
    write_csv("pilot_traffic_accounting.csv", pilot)
    TABLES.mkdir(exist_ok=True)
    for comparison in ("hybrid", "consensus"):
        (TABLES / f"{comparison}_accounting.tex").write_text(
            comparison_table(records, comparison), encoding="utf-8")
    (TABLES / "model_settings.tex").write_text(model_settings(cases), encoding="utf-8")
    print(f"Accounted for {len(records)} fixed-weight and {len(pilot)} pilot-grid records.")


if __name__ == "__main__":
    main()
