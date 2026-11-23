# Graph splitting for structured regression with hybrid data partitions

Reproduction code accompanying the manuscript by Xiaofei Wu.

[中文运行说明](README_ZH.md)

This code-only package contains the solvers, data generators, complete experiment settings, validation scripts, and figure/table generators. It contains no manuscript, saved result arrays, precomputed figures, or virtual environment. Extract the ZIP and upload the contents of this directory to your GitHub repository.

## Requirements

The manuscript environment uses **Python 3.12.14** and the exact dependency versions in `requirements.txt`: NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, Matplotlib 3.10.8, and pandas 2.2.3. Python 3.12 is recommended. No LaTeX installation is needed to run the experiments or export the figures and numerical table source.

Synthetic data are generated from the seeds encoded in `code/run_experiments.py`. The diabetes and Wisconsin Diagnostic Breast Cancer datasets are included with scikit-learn. The complete pipeline needs no separately supplied data files or network data download after dependencies have been installed.

## Windows: run from PowerShell

Install Python 3.12 first. Open PowerShell in this extracted directory, where `reproduce.py` is visible, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe reproduce.py --validate
.\.venv\Scripts\python.exe reproduce.py --all
```

Calling the environment's Python directly avoids the need to activate a PowerShell script. `--validate` checks the implementation and exact examples; **it does not reproduce the full numerical study**. `--all` performs those checks again, executes every full-size experiment, and generates all numerical tables and figures.

## Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python reproduce.py --validate
.venv/bin/python reproduce.py --all
```

## Execution modes

| Command | Work performed |
|---|---|
| `python reproduce.py --validate` | Independent solver checks, input checks, budget-certificate checks, analytic ridge and exact regrouping examples |
| `python reproduce.py --all` | Complete study: validation, 138 main graph runs, 30 sample-consensus runs, 27 penalty-sensitivity runs, 135 pilot-comparison main solves and 45 pilot phases, then tables and figures |
| `python reproduce.py --postprocess` | Regenerate pilot diagnostics, tables and figures from an already completed full run |
| `python reproduce.py --step figures` | Regenerate only the three figures from complete main-run outputs |
| `python reproduce.py --step NAME` | Run one stage shown by `python reproduce.py --help`; dependencies from earlier stages must already exist |

The entry point creates output directories, sets a noninteractive plotting backend and one BLAS thread, uses the current Python environment for every script, and stops on a failing stage. Each stage replaces its own generated outputs; use a fresh extracted directory for an independent rerun.

`--jobs 1` is the default. `--jobs 2` or `--jobs 3` can run independent pilot jobs concurrently. Every optimization solve remains a sequential simulation; this option does not change data, iterations, tolerances or the interpretation of solver timings.

## Full pipeline and outputs

The full sequence is:

1. `validate_solver.py`, `check_solver_inputs.py`, `check_pilot_budget_theory.py`.
2. `analytic_partition.py`, `verify_partition_examples.py`.
3. `run_experiments.py` with all five seeds and both public-data examples.
4. `consensus_baseline.py`, then `penalty_sensitivity.py`.
5. `pilot_budget_experiment.py` with the prescribed 100-step pilot and all configurations.
6. `make_tables.py`, then `make_figures.py`.

| Output directory | Contents generated when run |
|---|---|
| `results/` | CSV summaries, JSON configuration/check records, coefficients, data splits and convergence traces |
| `tables/` | Seven LaTeX files covering numerical Tables 2–11 |
| `figures/` | Figures 1–3 in PDF, PNG and editable SVG; Figure 1 PNG is 1200 dpi, Figures 2–3 are 900 dpi |
| `logs/` | Stage logs and a JSON run record with return codes, dependency versions and elapsed times |

Table 1 is the manuscript's conceptual model taxonomy, not an empirical output. Its content is not needed to run the code. The full entry point checks expected record counts, convergence flags and the existence of every figure/table file at completion.

See [the manuscript-to-code map](docs/REPRODUCIBILITY_MAP.md) and [the detailed experiment protocol](code/README.txt) for model scaling, hyperparameters, stopping criteria, saved fields and individual script commands.

## Numerical interpretation

All comparison configurations use the same prescribed data and objective within a sample. Main synthetic comparisons use relative Fenchel gap `1e-6`; public-data runs use `1e-7`; numerical references use `1e-10`. The fixed sixteen-data-block experiment uses `(2,8)`, `(4,4)` and `(8,2)`, five seeds, `rho=1`, a 100-step unit-metric pilot and budget floor `0.05`.

The scripts retain all specified outcomes, including configurations where a budget has no work advantage. Iterative payload counts transmitted scalars. Pilot/allocation payload and certificate evaluations are reported separately. Timing is for the sequential implementation and varies by computer; it is not a measured distributed wall time.

Fixed random seeds and pinned packages define the reproduction. Different BLAS implementations or platforms can cause small floating-point changes or a nearby stopping checkpoint; elapsed-time columns and PDF metadata are not expected to match byte-for-byte. The convergence certificates, problem definitions and full result tables are retained for comparison.

## Additional finite-pilot checks
The complete run now also evaluates the corner bound on all 45 original pilots and runs all 36 seed-zero combinations of model, layout and pilot length. The outputs are results/pilot_box.csv and results/pilot_length.csv. They are generated by the pilot-box and pilot-length stages. The input checks cover zero-adjoint unpenalized dual candidates, equivalent binary logistic label encodings and rejection of invalid mixed labels. With an unpenalized feature objective and nonzero adjoint, radial scaling gives a feasible zero candidate and can leave the stopping gap positive. The code estimates communication as logical scalar messages on one process.

## Source layout

- `reproduce.py`: portable pipeline entry point; no model or solver updates.
- `code/hybrid_solver.py`: weighted factor-graph splitting and original-objective certificate.
- `code/run_experiments.py`: main data generation, estimation, tuning and saved traces.
- `code/consensus_baseline.py`: sample-consensus comparator.
- `code/pilot_budget_experiment.py`: certified output allocation and fixed-block comparison.
- Remaining scripts: independent numerical checks, exact examples, sensitivity and output generation.
- `requirements.txt`: fixed dependency versions.
- `.gitignore`: keeps generated outputs, caches and local environments out of an ordinary Git commit.
