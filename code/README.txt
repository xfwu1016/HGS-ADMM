Recommended entry point: run python reproduce.py --all from the repository root.
The manual commands below assume the code/ directory and existing output
folders. The entry point creates those folders and supplies the dependency order.

REPRODUCING THE CSDA EXPERIMENTS

The code simulates weighted graph-projection splitting sequentially in one
Python process. No elapsed time in this package is a distributed wall time.
No measured distributed speedup, privacy guarantee, or network benchmark is
claimed. Communication is counted from the algorithm's transmitted vectors.

Files
  hybrid_solver.py      Generic factor-graph solver and Fenchel certificate.
  validate_solver.py    Independent numerical checks using scikit-learn,
                        SciPy HiGHS linear programming, and a dual optimizer.
  run_experiments.py    Five-seed simulations and two public-data examples.
  consensus_baseline.py  Sample-consensus ADMM comparator, 30 runs.
  analytic_partition.py  Independent exact ridge partition example, 12 runs.
  verify_partition_examples.py  Direct quadratic KKT and full lifted-norm checks
                        for four same-objective and twelve supplementary cases.
  penalty_sensitivity.py  Separate 27-run supplementary rho sensitivity.
  pilot_budget_experiment.py  Fixed sixteen-data-block comparison and pilot budgets.
  check_solver_inputs.py  Input errors and independent ridge checks.
  check_pilot_budget_theory.py  Independent budget and certificate checks.
  make_tables.py       All numerical LaTeX tables generated from final CSVs.
  make_figures.py       Regenerate vector PDF figures and PNG previews;
                        All three figures also have editable SVG exports.

Environment and execution
  Python 3.12; verified with 3.12.14. Install ../requirements.txt.
  Set OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1 before starting Python.
  From this directory:
    python validate_solver.py
    python check_solver_inputs.py
    python check_pilot_budget_theory.py
    python analytic_partition.py
    python verify_partition_examples.py
    python run_experiments.py
    python consensus_baseline.py
    python penalty_sensitivity.py
    python pilot_budget_experiment.py --workers 1
    python make_tables.py
    python make_figures.py
  These commands write ../results, ../figures, and ../tables. The main
  run_experiments.py command runs five replicates by default.
  --seeds 1 --skip-real provides a shorter run, overwriting the result files.
  The shorter command is exploratory only and cannot reproduce the full
  manuscript. Use a separate directory to retain any complete results.

Supplementary penalty-parameter sensitivity
  Run: python penalty_sensitivity.py
  This runs seed 0 of the three synthetic models at rho=.1, 1, and 10.
  Graph splitting uses layout (8,4), both unit and budget metrics; the
  sample-consensus comparator uses eight sample holders. Every method uses
  relative gap 1e-6 and an iteration cap of 50,000. All outcomes are saved in
  ../results/penalty_sensitivity.csv, including runs that hit the cap.
  This separate analysis does not select or modify the main experiment rho.

Exact cancellation examples
  Run: python verify_partition_examples.py
  The four same-objective cases in Appendix B compare two feature groupings
  and two sample groupings. The script independently solves the full quadratic
  KKT systems and checks every quadratic coefficient after regrouping, the
  operator reassembly, the analytic solution, H/G, and the full lifted norms.
  Results are in partition_same_objective.csv and its summary JSON.
  Twelve supplementary t-varying cases are retained in partition_examples.csv
  and its summary JSON. Those families vary the operator while holding the
  aggregate solution quantities fixed; they are not partitions of one fixed
  objective and are not used to claim an iteration-count improvement.
  The theoretical proofs are collected in Appendix A.

Pilot budgets and sixteen-data-block comparison
  Run: python pilot_budget_experiment.py --workers 3
  Use --workers 1 for one independent job at a time. Every solve remains
  a sequential simulation with one BLAS thread. Three concurrent jobs reduce
  experiment duration; their times are not distributed solver speedups.
  The fixed design contains the same three synthetic models, seeds 0--4,
  n=300, p=600, rho=1 and layouts (2,8), (4,4), (8,2). Thus 16 DATA blocks
  are fixed. Fused-penalty graph incidences depend on feature grouping.
  The comparison executes 135 main solves and 45 pilots. A pilot takes
  exactly 100 unit-metric steps, computes one finite absolute-gap certificate,
  and selects the floored output budget in Proposition 5.5 with nu=0.05.
  Feature budgets are uniform. The main run restarts from zero and keeps
  its selected metric fixed; all main targets are relative gap 1e-6.

  The radius uses the actual statistical ridge coefficient. Local spectral
  norms and the pilot coefficient form the upper/lower contribution bounds.
  Clipped proportional weights minimize the upper-envelope objective;
  Vplus-Vminus certifies distance-constant suboptimality over output budgets
  with the same floor and fixed feature budgets and rho.
  The selected budget reduces the computable upper bound in every case.
  After charging the pilot and allocation, only 1/45 individual settings
  and 0/9 group means use less payload than uniform budgets. All outcomes,
  including this cost, are retained and discussed.

  The script writes only results/pilot_budget_* and
  tables/pilot_budget_results.tex. These include 135 rows, 45 pilot records,
  180 coefficient arrays, 135 full trajectories, all weights, Vplus/Vminus,
  reference-oracle diagnostics, source hashes, methodology and validation.
  The 30 overlapping 4x4 unit/budget executions reproduce the original
  coefficient arrays bitwise and their non-timing outputs exactly.
  To rebuild summaries and the table without solving again:
    python pilot_budget_experiment.py --rebuild-summaries

  Additional weight payload is 2*|E| scalars: each worker uploads one
  operator norm and receives one weight. Factor coordinators reuse edge
  predictions and the gap from the pilot certificate. For the mu>0 examples,
  one certificate has the protocol bound C_iter+3*(F+J); the actual code
  evaluates these diagnostics centrally and stores their counts separately.

Independent validation checks
  python check_solver_inputs.py
  python check_pilot_budget_theory.py
  The first verifies explicit empty-factor errors and padded/ordinary ridge
  examples against independent closed-form solutions. A fully zero data row
  block requires include_zero_edges=True to retain its constant objective
  contribution; otherwise the helper raises an explanatory ValueError.
  The second independently enumerates active budget floors in 644 allocation
  cases and checks 250 certified quadratic models, including 50 exact pilots.
  It writes results/pilot_budget_theory_checks.json.
  state_dimension in solve() is the coordinate dimension of ONE full lifted
  variable eta. It is not the sum of all persistent/temporary arrays or the
  memory occupied by design blocks and cached factorizations.

Figures 1, 2 and 3
  All figures share a restrained blue/ochre palette and thin axes. Figures
  2 and 3 use 9-point headings, axis labels and legends, with 8-point ticks.
  PDF text uses embedded TrueType fonts. Figure 1 is an 83.74 x 46 mm
  factor-operator block matrix: its first two rows represent a 2 x 3 data
  partition, and the final two rows represent structural factors. Indexed
  nonzero maps and explicit zeros show the factor-feature dependencies.
  Mathematical labels are 12 pt and row-group labels 9 pt. Figure 1 is
  exported at 1200 dpi (3956 x 2173 pixels), with vector PDF and SVG versions.

  Figure 2 uses the full cas-dc width, 173.8 x 62 mm, with three equal
  panels and a shared logarithmic gap scale. At layout (M,J)=(8,4), all
  30 original trajectories and 3,779 checkpoints are plotted without
  smoothing, omission, extrapolation or an aggregate trajectory. Path
  simplification is disabled. Symbols mark actual final checkpoints;
  the dotted line marks the 1e-6 stopping threshold.

  Figure 3 also uses 173.8 x 62 mm. It includes all 120 individual
  observations, 60 paired links and 24 means across three models and
  four layouts. The x-axis title is shared; vertical scales vary by model.

  Every figure has embedded-font vector PDF and editable SVG. Figure 1
  has a 1200 dpi PNG; Figures 2 and 3 retain their 900 dpi PNG exports.
  results/figure*_qa.json records dimensions, encoding, data counts,
  source hashes and programmatic data/layout checks. Visual inspection of
  the exports remains separate from those programmatic checks.

Numerical tables
  Run: python make_tables.py
  This reads the completed outputs of run_experiments.py,
  consensus_baseline.py, analytic_partition.py, and penalty_sensitivity.py.
  It regenerates all empirical and exact-example tables in ../tables.
  Exact integer counts are retained; means, standard deviations, and payload
  summaries use three significant figures. Stopping-table test scores use
  three decimal places for RMSE and five decimal places for log loss.
  Tables follow the Elsevier cas-dc layout: the analytic Table 7 occupies
  one column, while wide tables in the main text span both columns.
  Tables 8--11 use the native 8-point table font and retain source order.
  Table 8 groups its 24 metric/layout rows under three model headings;
  Tables 9--11 use compact headers. All original numerical cells are
  preserved. The separate pilot script generates Table 4.

Model and exact scaling convention
  Every simulation solves
    mean_i loss(y_i, X_i beta) + lambda1 ||beta||_1
      + mu ||beta||_2^2 / 2 + lambdaTV ||D beta||_1,
  with the displayed positive mu included in the statistical model itself.
  D has -1 on the diagonal and +1 on the superdiagonal. The sign choice does
  not affect ||D beta||_1. D is not divided by sqrt(n).

  All data maps use A_i = X_i/sqrt(n), where n is the total training sample
  size, independent of the number of sample partitions. The scalar loss
  factors are changed exactly, so this never changes the statistical model:
  - Squared: f_i(z)=||z-y_i/sqrt(n)||^2/2.
  - Huber: f_i(z)=sum Huber_(delta/sqrt(n))(z-y_i/sqrt(n)).
  - Logistic: f_i(z)=sum log(1+exp(-sqrt(n)*label_i*z))/n,
    where label_i is -1 or +1.
  - Quantile (validation only): f_i(z)=sum check_tau(y_i/sqrt(n)-z)/sqrt(n).
  Structured factors use f_D(z)=lambdaTV ||z||_1. Their difference rows are
  divided into four fixed consecutive groups in every fused simulation.
  This grouping stays fixed as the data row/column layout changes.

  Parameters are fixed prospectively in run_experiments.py for the simulation
  runs. They are not selected separately for metrics or layouts. rho=1 for
  all main solves; the separate sensitivity study varies rho explicitly.
  Initialization is zero, with weights and rho fixed during each run.

Metrics
  The four scalar weights are a (factor anchor), c (coefficient anchor),
  b (edge output), and d (edge coefficient copy).
  unit:   a=c=b=d=1.
  budget: a=c=1, b=factor degree, d=1/(coefficient-node degree).
  The budget metric is evaluated without claiming that it improves every
  model or layout. All completed runs, including slower ones, are retained.

Primal-dual stopping
  A feasible objective is evaluated at the consensus coefficient vector.
  The loss dual variables are -rho*a*u_anchor, projected onto each conjugate
  domain. For mu>0, the elastic-net conjugate is
      ||soft(-sum A^T theta,lambda1)||^2/(2*mu).
  This gives a finite, valid Fenchel lower bound at every diagnostic check.
  With mu=0, the dual variables are jointly rescaled when necessary to satisfy
  the lasso dual constraint. A materially negative gap causes an exception.
  Relative gap is gap/max(1,abs(primal objective)). Thresholds are 1e-6 for
  simulations, 1e-7 for public examples, and 1e-10 for numerical references.
  The checks occur at iteration 1 and every 10 iterations. The final bound is
  evaluated once more for the returned result. diagnostic_evaluations records
  their total. Logistic proximal problems are solved by scalar Newton steps
  with a safeguarded Newton/bisection procedure to floating-point precision,
  while the other local proximal maps are closed form.

Communication and timing
  Optimization traffic per iteration is exactly
      2 sum_edges (factor dimension + coefficient-block dimension)
  floating-point scalars under one upload and one download per edge direction.
  Multiply by 8 for float64 bytes. This is an algorithmic payload count, with
  no headers, transport protocol, latency, bandwidth, or synchronization model.
  The implementation evaluates gaps centrally offline. Such diagnostics are
  not included in optimization traffic; the number of evaluations is reported
  separately. elapsed_seconds includes periodic gap evaluations, but excludes
  setup and the final audit checks after the iterative loop. setup_seconds covers
  graph construction and cached Cholesky factorizations. Matrix factors use
  the smaller of observation-space and coefficient-space systems.

Synthetic data
  n=300, p=600, 1000 independent test observations, five seeds (0,...,4).
  Gaussian covariates follow AR(1) correlation 0.4. No intercept is used; the
  data-generating intercept is zero. All exact coefficients and random seeds
  are given by synthetic_case().
  - Squared fused elastic net: lambda1=.02, mu=.01, lambdaTV=.08.
    Piecewise-constant signals cross feature partition boundaries.
  - Huber elastic net: delta=1.345, lambda1=.025, mu=.02.
    Each training observation independently receives an additive N(0,100)
    contamination term with probability 0.1. Test responses have N(0,1) noise.
  - Logistic elastic net: lambda1=.02, mu=.02; sparse logistic signal.
  Layouts are (M,N)=(1,1),(4,1),(4,4),(8,4). The (4,1) runs give a
  sample-only graph-splitting comparator. This comparator is not CPADMM.
  For fused seed 0, two additional (4,4) runs deliberately retain zero
  structural blocks, to compare sparse incidence with rectangular padding.

Public examples
  Diabetes is loaded with scaled=False. Degree-2 polynomial features without
  a bias term give 65 columns. The 70 percent training / 30 percent test split
  has seed 2026. One quarter of the training observations is held out for
  validation with seed 2027. Feature scaling and response standardization are
  fitted on the fitting subset during selection. The lasso grid is
  {.005,.015,.05}; mu=.01 and delta=1.345 are fixed. Validation MSE selects
  lambda1. Final preprocessing is refitted using the complete training set.
  Test RMSE is reported in the original response units.

  Breast cancer uses the built-in diagnostic dataset, with the same split
  seeds and proportions and stratification by outcome. Features are scaled
  using training data only. The lambda1 grid is {.001,.003,.01,.03}, mu=.01;
  validation log loss chooses lambda1. The positive class is benign (the
  original scikit-learn target 1). The logistic model has no separate
  intercept; its predictor uses standardized features. Test log loss, AUC,
  and accuracy are reported. Test observations never choose parameters.

  The generated results/datasets/*_split.npz files contain final transformed arrays,
  exact indices for training/test/fitting/validation, and preprocessing values.
  Both public examples simulate storage partitioning of centrally available
  public data; they are not deployments across independent institutions.

Results
  runs.csv                    Every completed optimization run.
  summary.csv                 Simulation means and sample standard deviations.
  traces.jsonl.gz             Objective, dual bound, gap, metric step, time,
                              and optimization traffic along each run.
  stopping_certificates.csv   First attainment of several gap thresholds,
                              prediction metrics and coefficient differences
                              from a high-accuracy numerical reference.
  coefficients/*.npz          Final and reference coefficient vectors.
  methods.json               Runtime versions, scaling, metric and tolerance.
  cases.json                 Model parameters and public-data tuning details.
  validation.json            Independent verification errors.
  completion.json            Completed-run count and convergence status.

  distance_constant_approx and the H/G dispersion columns substitute a
  high-accuracy numerical KKT pair into the theoretical zero-start distance.
  They are diagnostic approximations, not certified exact constants. The
  budget decomposition identity is checked numerically. The strong-convexity
  stopping bound concerns distance to the exact estimator; the stored
  comparison is to a reference whose nonzero gap is separately documented.

Current extension: code/pilot_box.py and code/pilot_length.py generate results/pilot_box.csv and results/pilot_length.csv (Section 7.4).
