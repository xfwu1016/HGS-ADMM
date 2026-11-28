# Weighted graph splitting: reproducibility code

Code for **Weighted graph splitting for structured regression with sample and feature partitions**, by Xiaofei Wu.

[中文说明](README_ZH.md) · [Data and preprocessing](docs/DATA.md) · [Experiment and output map](docs/EXPERIMENTS.md) · [Environment details](docs/ENVIRONMENT.md)

The main entry point, [reproduce.py](reproduce.py), runs the original 38-stage study and its four figures. The current manuscript additionally requires the separate joint-metric experiments and Figure 5; run the commands in [ACCELERATION_EXTENSION.md](docs/ACCELERATION_EXTENSION.md) after the original study, followed by the adaptive controls and new-draw sensitivity commands in [ADAPTIVE_COMPARISON.md](docs/ADAPTIVE_COMPARISON.md). The `--all` flag alone does not reproduce either extension. A fresh run needs no supplied result files or manuscript source. The public data are included in the installed scikit-learn distribution; the experiments do not download data.

[RUN_VERIFICATION.json](RUN_VERIFICATION.json) records the completed clean run, its environment and checks. It is a compact verification record, not an input dataset or a substitute for rerunning the code.

## Install and run

Run these commands from the directory containing `reproduce.py`. The recorded environment uses **Python 3.12.14**. The full runtime dependency snapshot is in [requirements-lock.txt](requirements-lock.txt); [requirements.txt](requirements.txt) retains the five main scientific package versions.

Linux/macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py --validate
.venv/bin/python reproduce.py --all --jobs 3
```

Windows PowerShell, with Python 3.12 installed:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe reproduce.py --validate
.venv\Scripts\python.exe reproduce.py --all --jobs 3
```

The validated platform is Linux. The commands use Python's cross-platform `spawn` process mode for TCP experiments, but Windows/macOS timings and rendering have not been separately certified. No LaTeX installation or GPU is needed to run the studies or generate the figure and table sources.

`--validate` runs seven numerical-check stages; it is **not the full experimental study**. `--all` runs 38 stages, including 138 main records, 40 repeated-split records, 14 complete two-process TCP runs, pilot/continuation studies, stopping and partition diagnostics, and the final output checks. All stages must finish successfully before the run manifest reports `status: passed`. The manifest includes row-count checks and script hashes.

The archived pre-revision clean Linux run completed all 38 stages in **888.52 seconds (about 14.8 minutes)** with `--jobs 3` and one BLAS thread per solver. This is a measured reference for that environment, not a guaranteed runtime on another machine. The full study contains many optimization fits, strict stopping targets and deliberate waits in the TCP delay profiles. Use the actual elapsed time recorded in `logs/run_*.json` for your hardware. `--jobs 3` parallelizes selected independent studies; use `--jobs 1` when memory or CPU availability is limited. Wall times depend on hardware, operating-system scheduling and load.

## Fresh execution and regeneration

The GitHub-oriented code archive contains no generated experiment results. Extract it into a new directory for an independent run. `--all` starts the stage sequence from the beginning and overwrites its named outputs; it does not implement automatic resume. A submission archive may additionally contain generated outputs for inspection.

After a **completed original 38-stage run**, regenerate its summaries, tables and four figures without repeating the original optimization studies. The joint-extension summary commands remain separate:

```bash
.venv/bin/python reproduce.py --postprocess
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Individual stages are available through `--step`; their earlier dependencies must already exist. See `python reproduce.py --help` and the [experiment map](docs/EXPERIMENTS.md). Validation results go to timestamped `results_check/` directories and do not replace the study tables.

Generated files are placed in:

| Directory | Contents |
|---|---|
| `results/` | Raw records, saved coefficients and traces, diagnostic summaries and processed split data |
| `tables/` | Numerical LaTeX tables and manuscript text excerpts |
| `figures/` | Four original figures (PDF/SVG/PNG) and joint-heterogeneity Figure 5 (PDF/PNG) |
| `logs/` | Per-stage stdout and JSON execution manifests |
| `results_check/` | Timestamped numerical-validation outputs |

The figure filenames are stable identifiers and do not all match the current manuscript numbering:

| Manuscript figure | File stem | Script |
|---|---|---|
| Figure 1: factor structure | `fig1_factor_structure` | `code/make_figures.py` |
| Figure 2: algorithm workflow | `fig_alg1_workflow` | `code/make_workflow.py` |
| Figure 3: convergence | `fig2_convergence` | `code/make_figures.py` |
| Figure 4: communication | `fig3_communication` | `code/make_figures.py` |
| Figure 5: joint-weight heterogeneity | `fig_joint_heterogeneity` | `code/summarize_acceleration.py` |

## Interpretation and reproducibility limits

The main graph-splitting experiments execute logical workers within one process. The TCP study executes Worker and Coordinator roles in separate operating-system processes on one host using a loopback socket. One Worker aggregates four sample blocks. Its LAN/WAN profiles add application-level latency and serialization delays; they are not measurements of a physical multi-host network or bandwidth-controlled link. Local loopback sockets must be permitted.

The repeated public-data study uses five split seeds per dataset. Diabetes retunes the penalty within each training split. The breast-cancer study freezes the penalty selected on the original validation split; later test partitions can overlap that selection split. Its predictive scores are descriptive checks of optimizer agreement at a fixed penalty, not an independent evaluation of a new tuning procedure. See [DATA.md](docs/DATA.md).

The optional `--step tcp-replay` command is a separate synthetic transport diagnostic and is excluded from `--all` and from manuscript evidence. It requires completed repeated-split results. The complete distributed optimization evidence comes from `tcp_distributed_solver.py` and its merged 14-run output.

A separate within-split timing audit can be rerun with `python code/tcp_timing_repeat_audit.py`. It performs three paired runs on the Diabetes seed-2026 LAN split, randomizes the Unit/Budget execution order within each pair, and writes `results/tcp_timing_repeats_focus.csv`. This is a timing-variability diagnostic, not an additional independent split and not a physical-network benchmark. The archived summary is in `results/tcp_timing_repeats_focus_summary.json`; timings will vary across machines and system load.

Seeds and stopping rules are fixed in the code. Numerical rounding, BLAS implementations and font availability can affect exact bytes or rendering; timings are not expected to repeat exactly. The run manifests and benchmark environment record make these differences inspectable. The version snapshot does not include operating-system libraries or a wheel cache; [ENVIRONMENT.md](docs/ENVIRONMENT.md) explains offline installation.

## Availability and permissions

The code is available at https://github.com/xfwu1016/HGS-ADMM. The code does not require credentials or a private data service.

No separate code license was supplied with the source. No new license or grant of rights has been added. For permissions, contact the author at `xfwu1016@ynu.edu.cn`. Third-party software and data remain subject to their respective terms.

## Editorial revision of 27 November 2026

The optimization algorithms, tuning grids, seeds and archived numerical fits are unchanged. This revision adds `code/pilot_uncertainty_summary.py`, which derives envelope and allocation diagnostics from `results/pilot_budget_weights.json` without fitting a model. It is also called by the existing table-generation stage, so the pipeline still has 38 stages. Run `python code/pilot_uncertainty_summary.py` to regenerate its CSV, JSON and two LaTeX outputs. The new diagnostic checks the stored envelope formulas and simplex normalization.

Some detailed tables have moved from the manuscript to the supplement; column headings now distinguish update scalars, update-plus-certificate bounds and TCP application frames. The old clean-run manifest records the pre-revision implementation; the separate `EDITORIAL_REVISION_CHECKS.json` records checks for this revision. The complete optimization pipeline was not rerun for these editorial changes.


## Joint metric extension (28 November 2026)

The current manuscript adds a one-switch joint allocation and penalty rule with separate oracle and heterogeneity diagnostics. See [ACCELERATION_EXTENSION.md](docs/ACCELERATION_EXTENSION.md) for reproduction commands, ledger, sample structure and limitations. The original 38-stage pipeline remains unchanged; new experiments are separate.

## Adaptive comparison and new-draw robustness

The current revision adds seven-arm comparisons, new high-dimensional draws and pilot/lag sensitivity. These are separate from the original 38 stages and joint extension. See [ADAPTIVE_COMPARISON.md](docs/ADAPTIVE_COMPARISON.md) for commands, safeguarded SRA/RB definitions, failure-aware summaries and complete cost accounting.
