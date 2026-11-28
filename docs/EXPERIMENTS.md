# Experiment and output map

Run commands from the directory containing `reproduce.py`, using the virtual-environment interpreter shown in the [README](../README.md). The `FULL`, `VALIDATE` and `POSTPROCESS` sequences in [reproduce.py](../reproduce.py) are the executable source of truth for ordering and expected outputs.

The entry point covers the original 38-stage study. The current joint-metric extension must additionally be run using [ACCELERATION_EXTENSION.md](ACCELERATION_EXTENSION.md); it is not part of `--all` or `--postprocess`.

## Entry-point modes

| Mode | Purpose | Prior results required? |
|---|---|---|
| `--validate-minimal` | Five algebraic/reduced-state/example checks using NumPy | No |
| `--validate` | Seven numerical-check stages with the complete environment | No |
| `--all --jobs 3` | All 38 stages, then result-count/convergence/output checks | No; stages create their inputs in order |
| `--postprocess` | Rebuild summaries, diagnostics, numerical tables and four figures | Yes, a complete full run including traces and coefficients |
| `--step NAME` | Run one named stage | Its prerequisites, if any |

Numerical-check modes write under `results_check/<timestamp>/`. Other modes write to the standard `results/`, `tables/`, `figures/` and `logs/` directories. To run only one stage, inspect `python reproduce.py --help` for the available names. A successful individual stage does not establish completion of the full study.

## Full study components

| Component | Principal scripts | Selected generated records |
|---|---|---|
| Solver identities, input checks, exact examples | `check_reduced_state.py`, `validate_solver.py`, `check_solver_inputs.py`, `check_pilot_budget_theory.py`, `analytic_partition.py`, `verify_partition_examples.py`, `oracle_stopping_example.py` | Validation reports and exact-example records |
| Main graph splitting and comparison methods | `run_experiments.py`, `consensus_baseline.py`, `penalty_sensitivity.py` | `runs.csv` (138), `consensus_baseline.csv` (30), `penalty_sensitivity.csv` (27) |
| Pilot weights and retained-state controls | `pilot_budget_experiment.py`, `pilot_length.py`, `pilot_box.py`, `pilot_continuation.py`, `matched_output_control.py` | Pilot, continuation and matched-control records, with allocation/transition accounting |
| Stopping and centralized comparison | `stopping_sensitivity.py`, `lasso_stopping_scale.py`, `centralized_benchmark.py`, `replicate_calibration.py`, `residual_gap_bridge.py` | Tolerance/calibration studies and `centralized_benchmark.csv` (60) |
| Partition and dual diagnostics | `fused_dual_ranges.py`, `regrouping_sensitivity.py`, `layout_choice.py`, `fixed16_bound_audit.py`, `heldout_metric_race.py` | Regrouping, fixed-block, multiplier-range and held-out metric comparisons |
| Repeated public-data splits | `repeated_real_splits.py` | `repeated_real_splits.csv` (40 records across five seeds per dataset) |
| Complete two-process TCP optimization | `tcp_distributed_solver.py`, `merge_tcp_distributed_results.py` | `tcp_distributed_solver.csv` (14 runs) and validation summary |
| Reporting and figure generation | `referee_diagnostics.py`, `referee_traffic.py`, `stopping_summaries.py`, `publication_excerpts.py`, `make_tables.py`, `extension_outputs.py`, `make_figures.py`, `make_workflow.py` | Summaries, numerical LaTeX outputs and four figure stems |

The reporting scripts with `referee_` in their filenames are executable rank/traffic calculations used by the current pipeline; they are not private reviewer correspondence. Numerical table templates needed by the generators are included under `code/table_templates/`.

## TCP configurations and timing scope

| Dataset/profile | Split seeds | Weight arms | Runs |
|---|---|---|---:|
| Diabetes / LAN | 2026–2030 | unit, budget | 10 |
| Diabetes / WAN | 2026 | unit, budget | 2 |
| Breast cancer / LAN | 2026 | unit, budget | 2 |

The `4×1` optimization is split into two process roles. Worker holds all four sample blocks; Coordinator handles the feature-anchor state. The processes exchange the optimization messages over loopback TCP and compare the final result with an in-process reference using the same stopping target.

The LAN profile adds 0.1 ms nominal one-way latency and a 1000 Mbps serialization-delay parameter; the WAN profile uses 5 ms and 20 Mbps. The implementation adds these delays in the application layer. It neither caps the socket's actual bandwidth nor establishes a physical multi-host deployment. The byte ledger counts serialized application messages, including frame headers, rather than TCP/IP/link-layer packets. Runtime output records retain the process and accounting checks.

The optional corrected `tcp_transport_replay.py` is not in `FULL`. It only replays a simplified synthetic message schedule and is not manuscript evidence. `compare_replay.py` is an optional comparison utility, not a substitute for executing the full stage sequence.

## Figures and regenerated text

| Current manuscript location | Output stem | Generator |
|---|---|---|
| Figure 1, factor structure | `fig1_factor_structure` | `make_figures.py` |
| Figure 2, algorithm workflow | `fig_alg1_workflow` | `make_workflow.py` |
| Figure 3, convergence traces | `fig2_convergence` | `make_figures.py` |
| Figure 4, communication comparison | `fig3_communication` | `make_figures.py` |
| Figure 5, joint-weight heterogeneity | `fig_joint_heterogeneity` | `summarize_acceleration.py` (separate extension) |

The four original stems are generated in PDF, SVG and PNG formats. The workflow is a mathematical schematic, not a plot of estimated statistical quantities. Its feedback labels represent a single combined vector per edge in each direction.

Tables and numerical manuscript excerpts are rebuilt from the run outputs. In particular, the centralized timing excerpt is recomputed from that run's measured times, and the real-data/TCP summaries use their raw records. The end-of-run verification checks 29 required LaTeX table/excerpt files and all 12 required figure files, as well as detailed row counts and convergence conditions. Run manifests distinguish a completed full study from an individual successful command.

## Repeated within-split TCP timing audit

The optional command `python code/tcp_timing_repeat_audit.py` executes three paired Unit/Budget timing repetitions on the Diabetes split with seed 2026 under the LAN-labelled application-delay profile. The order is randomized within each pair using a fixed order seed. Each arm is run in a fresh parent Python process to avoid accumulating multiprocessing resource-tracker state. The script validates convergence and agreement with the single-process reference on every run. It writes `results/tcp_timing_repeats_focus.csv`; the archived run summary is `results/tcp_timing_repeats_focus_summary.json`. This is a within-split timing diagnostic only. It does not add independent data splits, change the canonical 14-run TCP study, or establish physical multi-host performance.

### Pilot uncertainty diagnostic added in the editorial revision

`python code/pilot_uncertainty_summary.py` reads the saved 100-step pilot records and writes `pilot_uncertainty_diagnostics.csv`, `pilot_uncertainty_summary.json`, `pilot_uncertainty_table.tex` and `pilot_uncertainty_text.tex`. Data factors only; 15 configurations per model reuse five independent samples on three layouts. The existing table stage invokes this diagnostic. No new optimization runs are needed.

## Adaptive comparison and new-draw robustness

The current revision adds seven-arm comparisons, new high-dimensional draws and pilot/lag sensitivity. These are separate from the original 38 stages and joint extension. See [ADAPTIVE_COMPARISON.md](ADAPTIVE_COMPARISON.md) for commands, safeguarded SRA/RB definitions, failure-aware summaries and complete cost accounting.
