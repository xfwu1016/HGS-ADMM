# Adaptive controls and new-draw sensitivity (28 November 2026)

This third experiment group supplements the original 38-stage study and the joint-metric extension. It does not replace their archived data. Rules, sample sizes, stopping targets and sensitivity settings were written before the nonsmoke outcomes. The common safeguard was adopted after an exploratory unbounded SRA smoke run overflowed; this history is retained in `results/adaptive_comparison/uncapped_smoke_note.txt`. Smoke results do not enter publication tables.

## Reproduction

From the reproduction root, after installing `requirements.txt`:

```bash
python code/adaptive_comparators.py --mode core --jobs 4
python code/adaptive_comparators.py --mode robust --jobs 5
python code/summarize_adaptive.py
python code/calibration_cost_audit.py
python code/verify_adaptive.py
```

Use fewer jobs on a smaller machine. Each BLAS instance uses one thread. These are separate commands, not stages of `reproduce.py --all`. The last three commands require the archived or regenerated input records. `--resume` skips only whole completed cases already written to the corresponding CSV, checking the expected number of arms per case; it does not resume an individual solver. Do not resume after changing the scientific settings. Sequential timings include selection, pilot, local refactorization and solving, but are not network timings and are not used to claim speedup.

## Comparators

Seven arms share the same problem, accuracy target and certificate schedule:

| Record name | Rule |
|---|---|
| uniform_auto | 100 unit-weight pilot steps, uniform budgets, one increment-based scalar selection |
| joint_auto | Same pilot, two constrained incidence allocations and one scalar selection |
| norm_auto | Same pilot, floored shares proportional to each local design block's Frobenius norm in both stars, same scalar-selection formula |
| uniform_sra | uniform_auto followed by a scalar SRA update every five iterations |
| joint_sra | joint_auto followed by the same SRA updates |
| sra_zero | Uniform budgets and scalar SRA from zero; no pilot charge |
| rb_zero | Uniform budgets and conventional residual balancing from zero; no pilot charge |

The block-norm arm is a simple problem-based heuristic, not iterative Ruiz equilibration. SRA follows the scalar spectral-radius approximation in Lozenski, McCann and Wohlberg (2026), IEEE OJSP 7:410–427, DOI 10.1109/OJSP.2026.3664275, building on McCann and Wohlberg (2024), DOI 10.1109/OJSP.2023.3349115. In fixed metric W, its ratio is ||delta lambda||_(W^-1) / ||delta eta||_W. It uses **consecutive** states at every fifth iteration. It does not use the one-switch rule's 20-step lag. Zero numerator or denominator respectively divides or multiplies the old penalty by ten; if both vanish it retains the penalty.

Residual balancing uses r=||delta lambda||_(W^-1)/rho and s=rho||delta eta||_W. It doubles rho when r>10s, halves rho when s>10r and otherwise retains it, every five iterations. All scalar selections and updates use the common interval [1/256,4]. Thus the published comparison is explicitly a **safeguarded SRA implementation**, not an unconstrained reproduction of all SRA variants or multiparameter adaptation. The unscaled multiplier is preserved by u_new=(rho_old/rho_new)u_old. Scalar changes take place after any certificate and before the next feedback; fixed W permits reuse of the local factorization. The paper's fixed-suffix theorem applies to one-switch arms, not continuously adaptive SRA or RB. A compact penalty interval alone does not establish their convergence.

## Samples and sensitivity

The core CSV contains 973 rows: 49 original configurations and 90 earlier heterogeneity cells, each with seven arms. This is a retrospective comparator extension, not a newly held-out test.

The robustness CSV contains 1,320 rows. For each of squared, Huber and logistic losses, ten new base draws (seeds 9300–9309 with model offsets) use n=240, p=480 and Gaussian AR(1) correlation 0.4. Four fixed sample quarters receive geometric scale factors from 1 to 1 or 16, followed by global column standardization. Four fixed predictor groups each contain four alternating-sign active coefficients; the full coefficient norm is three. Response laws, penalties and Huber cutoff follow the main heterogeneity study. Each base draw is reused across two scale ratios and two layouts (4x4, 8x2), making 120 default cells from 30 independent base draws. These are not 120 independent datasets.

At ratio 16 the uniform and joint one-switch arms additionally use pilot/lag pairs (50,20), (200,20), (100,10), (100,40), supplementing default (100,20). The share floor remains 0.05. No outcome selects a preferred setting; all settings are reported.

## Certificates, failures and accounting

The original synthetic and new heterogeneity target is relative primal-dual gap 1e-6; the two public datasets use 1e-7. All arms check iteration 1 and multiples of 10, with a total cap of 30,000 steps including any pilot. Rows distinguish converged, iteration_cap and numerical_failure. Failed runs are retained. Their partial costs are never ranked against certified solutions. Cost ratios require both paired arms to certify. A gap on a failed row is only the last attempted or available diagnostic, not a certified endpoint.

The common point-to-point ledger charges C_iter per update and C_iter+3(F+J) per certificate. Each two-star allocation charges 4E scalars. Each scalar selection/update charges the conservative reduction-and-broadcast bound 2(E+F+J)(F+J)+E. A retained-state metric transfer charges C_iter/2. These are specified logical protocols, not claims of minimum necessary implementation traffic. Update-only comparisons isolate iteration effects from this reduction bound. No scalar update requires an additional vector refresh because it occurs before the next feedback.

Stable Euclidean norms and nonfinite/overflow checks are used. After the formal runs began, the exception handler was broadened from floating-point/overflow exceptions to their ArithmeticError parent so that a possible dual-certificate inconsistency would also be logged rather than aborting a batch. This does not change iterates, stopping rules or any completed record. The design wording was also clarified to exclude both continuous SRA and RB from the fixed-suffix guarantee. Runtime changes added completed-case resume and changed process counts only.

`calibration_cost_audit.py` separately charges both 200-step replicate pilots, all their certificates, scalar uploads and the calibration broadcast. The terminal partial predictions already held by factor coordinators are reused. It reports full calibration charge per downstream fit and one charge shared across six prespecified fits. Different stopping targets prevent interpreting these ratios as same-accuracy acceleration.
