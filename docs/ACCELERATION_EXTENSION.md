# Joint metric extension of 28 November 2026

The new code changes the graph metric once after a common pilot. It estimates the remaining primal and dual error magnitudes from consistent-state and unscaled-multiplier increments. It learns output shares, coefficient-copy shares and a scalar penalty. The floored shares and clipped square-root balance exactly minimize the stated increment surrogate when both weighted increment norms are positive. That fact is not a general speed guarantee. The selected suffix still uses the original GraphSplitting solver.

## Reproduction

Run from 04_Reproducibility with the original dependencies. These commands are separate from the archived 38-stage pipeline:

```bash
python code/check_joint_metric.py
python code/acceleration_diagnostics.py --mode original --jobs 3
python code/acceleration_diagnostics.py --mode sweep --jobs 3
python code/acceleration_diagnostics.py --mode hetero --jobs 3
python code/joint_acceleration.py --mode original --jobs 3
python code/joint_acceleration.py --mode hetero --jobs 3
python code/summarize_acceleration.py
python code/summarize_rho_diagnostics.py
```

The `--jobs` option parallelizes independent experiments, not workers inside one distributed regression fit. It does not establish parallel speedup. Full-grid runs can be substantially slower than the automatic joint-rule runs, especially at small penalties. Candidates have a common suffix cap of 30,000 updates; retain capped records and their final gaps.

## Files and samples

- `results/acceleration/original.csv`: 49 original configurations, four output-allocation arms at rho=1, including an actual reference-oracle continuation.
- `results/acceleration/sweep.csv`: 17 original 4x4 cases, six shared rho candidates and four output-allocation arms. Selection within a case is retrospective diagnostic information.
- `results/acceleration/hetero.csv`: additional signal-heterogeneity grid diagnostics; 72 configurations, four arms and six penalties. The specified initial three seeds per model/level are available for tuning; five further seeds are reserved for evaluation. These records do not select the joint rule's parameters.
- `results/acceleration/joint_original.csv`: six paired controls on 49 original configurations.
- `results/acceleration/joint_hetero.csv`: six controls on 144 condition cells. The primary new test subset contains 90 cells from 30 base draws (two regimes, three models and five seeds); levels reuse the random base. Initial diagnostic cases contain 54 cells from 18 base draws.
- `summary.json`, `rho_summary.json`: summaries generated from these records.
- `design.json`, `joint_design.json`: fixed settings. The joint rule was developed after exploratory smoke cases at seed 8100 and the original data; none of seeds 8200--8204 selected its rule, lag, floor or penalty interval.
- `joint_validation.json`: independent constrained-allocation checks and state-transfer invariants.
- `smoke.csv`, `joint_smoke.csv`: exploratory execution checks only; excluded from publication summaries. The latter precedes the added same-rho control and final conservative penalty-broadcast ledger.

## Communication and timing

Every primary joint arm charges the same 100-step unit pilot, all suffix updates, certificate payload bounds at the same checkpoints, and one edge-state refresh. Output allocation requires one scalar norm upload and one scalar weight return per edge. Joint allocation does this for both output and coefficient-copy shares (4E scalars). A conservative all-coordinator reduction for the two weighted sums costs 2(E+A+J)(A+J), followed by E scalar penalty broadcasts to edge workers. Shared-penalty controls charge the same selection overhead as the joint rule.

The ledger was tightened after simulation by adding the final E-scalar penalty broadcast to every automatic arm. This alters selection and total scalar counts only, not iterates, gaps or timings. The delivered generator includes it. All selected-penalty timings are sequential single-host simulations; they are not repeated multi-host timing measurements.

## Interpretation

Oracle reference-fit cost is excluded; the oracle is not a feasible performance comparator. The uniform automatic-penalty control is essential: most aggregate savings over rho=1 come from penalty selection. Joint graph weights add conditional gains under unequal sample-block design scales. Signal heterogeneity alone does not reliably help and can hurt logistic fits. The common-penalty control verifies that the favourable sample-scale gains persist when rho is held equal.

The original solver, raw results, old tuning grids and earlier TCP records are unchanged. The earlier run manifests document their own versions; the new extension has separate code, results and validation records.
