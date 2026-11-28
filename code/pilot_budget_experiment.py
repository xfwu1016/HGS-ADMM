"""Fixed-worker comparison of unit, uniform and certificate-based budgets.

The prospective design is contained in DESIGN below.  A 100-step unit pilot
produces an absolute Fenchel gap, hence a strong-convexity coefficient radius.
Output budgets minimize a separable upper bound over a floored simplex.  Each
main solve then restarts at zero and uses its fixed metric throughout.

This script writes only pilot_budget_* outputs and pilot_budget_results.tex.
It preserves all pre-existing numerical results.  Individual optimization jobs
are sequential simulations; optional process parallelism expedites independent
jobs and is not a measurement of distributed optimization speed.
"""
from __future__ import annotations

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize

from hybrid_solver import GraphSplitting, make_regression
from run_experiments import synthetic_case, test_metrics
from make_tables import write_pilot_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
DESIGN = dict(
    models=['fused', 'huber', 'logistic'], seeds=list(range(5)), n=300, p=600,
    data_workers=16, layouts=[[2, 8], [4, 4], [8, 2]],
    metrics=['unit', 'budget', 'pilot'], rho=1.0,
    main_relative_gap=1e-6, main_iteration_cap=50000, main_check_every=10,
    pilot_iterations=100, pilot_metric='unit', pilot_certificate_evaluations=1,
    floor_fraction=0.05, initialization='zero for every pilot and every main solve',
    coefficient_radius='sqrt(2 * absolute_pilot_Fenchel_gap / ridge)',
    upper_contribution='norm(A_aj @ beta_pilot_j) + norm(A_aj, 2) * radius',
    lower_contribution='max(0, norm(A_aj @ beta_pilot_j) - norm(A_aj, 2) * radius)',
    output_budget='argmin sum(c_aj**2 / omega_aj), sum omega=1, omega>=0.05/degree',
    coefficient_budget='pi_aj = 1 / feature_degree',
    metric='a=c=1, b_aj=1/omega_aj, d_aj=pi_aj; kept fixed in each main solve',
    structural_factors='Four fixed consecutive difference-row groups for fused regression',
    parameter_selection='All settings fixed before the extension; no selection by test outcomes',
    optimization_payload='2 sum_edges(m_a+p_j) scalars per optimization iteration',
    pilot_weight_payload='2*number_of_edges: one operator-norm scalar uploaded and one chosen weight returned per edge; factor coordinators reuse the edge predictions and global gap from the pilot certificate',
    certificate_payload_bound='C_iter + 3*(number_of_factors + number_of_features) per checkpoint',
    certificate_payload_interpretation='A protocol bound, not measured communication; diagnostics executed centrally',
    upper_P='2*(norm(beta_pilot)+R)^2 + sum_a(norm(A_a beta_pilot)+R*sqrt(sum_j||A_aj||op^2))^2 + sum_edges(c_aj^2/omega_aj)',
    regret_bound='sum_a(Vplus_a-Vminus_a), against the best floored output budget at fixed pi',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def water_fill(values, floor_fraction=0.05):
    """Minimize sum(values**2 / w) over sum(w)=1, w>=floor_fraction/d.

    The active-set construction uses the KKT equality w_i proportional to c_i
    off the floor.  All-zero inputs have every feasible allocation optimal;
    return uniform.  Positive homogeneity keeps the calculation well scaled.
    """
    c = np.asarray(values, dtype=float)
    if c.ndim != 1 or len(c) == 0 or np.any(c < 0) or not np.isfinite(c).all():
        raise ValueError('values must be a finite nonnegative nonempty vector')
    if not 0 < floor_fraction < 1:
        raise ValueError('floor_fraction must lie strictly between zero and one')
    d = len(c)
    lower = floor_fraction / d
    if not np.any(c):
        return np.full(d, 1.0 / d)
    c = c / c.max()
    free = np.ones(d, dtype=bool)
    w = np.full(d, lower)
    for _ in range(d + 1):
        mass = 1.0 - lower * np.count_nonzero(~free)
        total = c[free].sum()
        candidate = mass * c[free] / total if total else np.full(free.sum(), mass/free.sum())
        too_small = candidate < lower
        if not np.any(too_small):
            w[free] = candidate
            break
        indices = np.flatnonzero(free)
        free[indices[too_small]] = False
    else:
        raise ArithmeticError('water-filling active set did not terminate')
    if abs(w.sum() - 1) > 2e-14 or np.any(w < lower - 2e-14):
        raise ArithmeticError('invalid water-filling allocation')
    return w


def validate_water_fill():
    """Independent constrained numerical minimizations and KKT checks."""
    cases = [[0, 0, 0], [1, 2, 4], [1e-5, .4, 10], [0, 0, 1], [1, 1, 1]]
    rng = np.random.default_rng(20261122)
    cases += [rng.lognormal(size=d).tolist() for d in (2, 4, 7, 10)]
    records = []
    for c0 in cases:
        c = np.asarray(c0, float)
        w = water_fill(c)
        lower = DESIGN['floor_fraction']/len(c)
        scaled = c/max(1., c.max())
        objective = lambda x: np.sum(scaled**2/x)
        numerical = minimize(objective, np.full(len(c), 1/len(c)),
                             jac=lambda x: -scaled**2/x**2,
                             constraints={'type': 'eq', 'fun': lambda x: x.sum()-1,
                                          'jac': lambda x: np.ones_like(x)},
                             bounds=[(lower, 1)]*len(c), method='SLSQP',
                             options={'ftol': 1e-12, 'maxiter': 1000})
        error = abs(objective(w)-numerical.fun)
        if not numerical.success or error > 2e-8*max(1., objective(w)):
            raise ArithmeticError(f'water-filling numerical check failed: {numerical.message}')
        free = w > lower + 1e-10
        ratios = c[free]/w[free]
        spread = np.ptp(ratios) if len(ratios) else 0.
        if spread > 1e-10*max(1., c.max()):
            raise ArithmeticError('water-filling KKT equality failed')
        if len(ratios) and np.any(c[~free]/w[~free] > ratios[0] + 1e-9*max(1., c.max())):
            raise ArithmeticError('water-filling KKT inequality failed')
        records.append(dict(values=c.tolist(), weights=w.tolist(),
                            independent_objective_error=float(error),
                            numerical_optimizer_success=bool(numerical.success),
                            kkt_free_ratio_spread=float(spread)))
    return records


def pilot(solver):
    start = perf_counter()
    for _ in range(DESIGN['pilot_iterations']):
        xi = solver.prox(solver.eta-solver.u)
        eta = solver.project(xi)
        solver.u += xi-eta
        solver.eta = eta
    primal, dual, gap = solver.certificate()
    return dict(coef=np.concatenate([solver.eta[s] for s in solver.xs]),
                iterations=DESIGN['pilot_iterations'], objective=primal, dual=dual,
                gap=gap, relative_gap=gap/max(1., abs(primal)),
                elapsed_seconds=perf_counter()-start,
                invariant_error=solver.invariant_error(), diagnostic_evaluations=1)


def reference_P(solver, beta):
    blocks = np.split(beta, np.cumsum(solver.problem.block_sizes)[:-1])
    total = sum(c*np.dot(v, v) for c, v in zip(solver.c, blocks))
    for a, f in enumerate(solver.problem.factors):
        value = sum(A@blocks[j] for j, A in f.maps.items())
        total += solver.a[a]*np.dot(value, value)
    for e, (a, j, A) in enumerate(solver.edges):
        value = A@blocks[j]
        total += solver.b[e]*np.dot(value, value)+solver.d[e]*np.dot(blocks[j], blocks[j])
    return float(total)


def make_pilot_weights(problem, pilot_result):
    start = perf_counter()
    beta = pilot_result['coef']
    radius = float(np.sqrt(2*pilot_result['gap']/problem.ridge))
    blocks = np.split(beta, np.cumsum(problem.block_sizes)[:-1])
    edges = [(a, j, A) for a, f in enumerate(problem.factors) for j, A in f.maps.items()]
    dj = np.bincount([j for a, j, A in edges], minlength=len(blocks))
    normA = np.array([np.linalg.norm(A, 2) for a, j, A in edges])
    prediction = np.array([np.linalg.norm(A@blocks[j]) for a, j, A in edges])
    upper = prediction + normA*radius
    lower = np.maximum(0., prediction-normA*radius)
    omega = np.zeros(len(edges)); uniform = np.zeros(len(edges)); low_weights = np.zeros(len(edges))
    star_records = []
    anchor_upper = 2*(np.linalg.norm(beta)+radius)**2
    for a, f in enumerate(problem.factors):
        es = np.array([i for i, (aa, j, A) in enumerate(edges) if aa == a])
        omega[es] = water_fill(upper[es], DESIGN['floor_fraction'])
        uniform[es] = 1/len(es)
        low_weights[es] = water_fill(lower[es], DESIGN['floor_fraction'])
        h = sum(A@blocks[j] for j, A in f.maps.items())
        anchor_upper += (np.linalg.norm(h)+radius*np.linalg.norm(normA[es]))**2
        vplus = float(np.sum(upper[es]**2/omega[es]))
        vminus = float(np.sum(lower[es]**2/low_weights[es]))
        vuniform = float(np.sum(upper[es]**2/uniform[es]))
        if vplus > vuniform+1e-10*max(1., vuniform) or vminus > vplus+1e-10*max(1., vplus):
            raise ArithmeticError('pilot bound optimization check failed')
        star_records.append(dict(factor=a, kind=f.kind, degree=len(es),
                                 omega=omega[es].tolist(), lower_bound_omega=low_weights[es].tolist(),
                                 c_upper=upper[es].tolist(), c_lower=lower[es].tolist(),
                                 operator_norm=normA[es].tolist(), prediction_norm=prediction[es].tolist(),
                                 Vplus=vplus, Vminus=vminus, Vuniform=vuniform,
                                 regret_bound=vplus-vminus))
    weights = dict(a=np.ones(len(problem.factors)), c=np.ones(len(blocks)),
                   b=1/omega, d=np.array([1/dj[j] for a, j, A in edges]))
    vplus = sum(s['Vplus'] for s in star_records)
    vminus = sum(s['Vminus'] for s in star_records)
    vuniform = sum(s['Vuniform'] for s in star_records)
    record = dict(radius=radius, floor_fraction=DESIGN['floor_fraction'],
                  anchor_upper=float(anchor_upper), edge_upper_pilot=vplus,
                  edge_upper_uniform=vuniform, edge_lower_best=vminus,
                  upper_P_pilot=float(anchor_upper+vplus),
                  upper_P_uniform=float(anchor_upper+vuniform),
                  upper_P_ratio=(float((anchor_upper+vplus)/(anchor_upper+vuniform))
                                 if anchor_upper+vuniform else 1.0),
                  edge_upper_ratio=float(vplus/vuniform) if vuniform else 1.,
                  regret_bound=vplus-vminus, minimum_omega=float(omega.min()),
                  allocation_seconds=perf_counter()-start,
                  edge_order=[[a, j] for a, j, A in edges], stars=star_records)
    return weights, record


def run_case(task):
    kind, seed = task
    case = synthetic_case(kind, seed)
    reference = np.load(RESULTS/'coefficients'/(case['case']+'_reference.npz'))
    beta_ref = reference['coef'].copy()
    objective_ref = float(reference['objective']); gap_ref = float(reference['gap'])
    records=[]; pilots=[]; allocations=[]; arrays={}; traces=[]
    for M, J in DESIGN['layouts']:
        key=f"{case['case']}_M{M}_N{J}"
        start=perf_counter()
        problem=make_regression(case['X'], case['y'], M, J, **case['args'])
        pilot_solver=GraphSplitting(problem, rho=DESIGN['rho'], weights='unit')
        pilot_setup=perf_counter()-start
        p_result=pilot(pilot_solver)
        weights, allocation=make_pilot_weights(problem, p_result)
        allocations.append(dict(key=key, **allocation))
        arrays[key+'_pilot_phase']=p_result['coef']
        certificate_cost=pilot_solver.communication_per_iteration+3*(len(problem.factors)+len(problem.block_sizes))
        pilot_payload=DESIGN['pilot_iterations']*pilot_solver.communication_per_iteration
        allocation_payload=2*len(pilot_solver.edges)
        pilots.append(dict(key=key, case=case['case'], seed=seed, loss=case['args']['loss'],
                           row_parts=M, column_parts=J, iterations=p_result['iterations'],
                           objective=p_result['objective'], dual=p_result['dual'],
                           absolute_gap=p_result['gap'], relative_gap=p_result['relative_gap'],
                           radius=allocation['radius'], setup_seconds=pilot_setup,
                           elapsed_seconds=p_result['elapsed_seconds'],
                           allocation_seconds=allocation['allocation_seconds'],
                           optimization_payload=pilot_payload, allocation_payload=allocation_payload,
                           diagnostic_evaluations=1, certificate_payload_bound=certificate_cost,
                           upper_P_pilot=allocation['upper_P_pilot'],
                           upper_P_uniform=allocation['upper_P_uniform'],
                           upper_P_ratio=allocation['upper_P_ratio'], regret_bound=allocation['regret_bound'],
                           edge_upper_pilot=allocation['edge_upper_pilot'],
                           edge_upper_uniform=allocation['edge_upper_uniform'],
                           edge_lower_best=allocation['edge_lower_best'],
                           edge_upper_ratio=allocation['edge_upper_ratio'],
                           primal_invariant_error=p_result['invariant_error'][0],
                           dual_invariant_error=p_result['invariant_error'][1]))
        for metric in DESIGN['metrics']:
            start=perf_counter()
            solver=GraphSplitting(problem, rho=DESIGN['rho'], weights=metric,
                                  custom_weights=weights if metric=='pilot' else None)
            setup=perf_counter()-start
            result=solver.solve(tol=DESIGN['main_relative_gap'], maxiter=DESIGN['main_iteration_cap'],
                                check_every=DESIGN['main_check_every'])
            is_pilot=metric=='pilot'
            paid_pilot=pilot_payload if is_pilot else 0
            paid_allocation=allocation_payload if is_pilot else 0
            paid_checks=1 if is_pilot else 0
            common=dict(case=case['case'], seed=seed, loss=case['args']['loss'],
                        row_parts=M, column_parts=J, data_workers=M*J, metric=metric,
                        rho=DESIGN['rho'], n=len(case['y']), p=len(beta_ref),
                        l1=problem.l1, ridge=problem.ridge, fusion=case['args'].get('fusion',0),
                        main_iterations=result['iterations'],
                        pilot_iterations=DESIGN['pilot_iterations'] if is_pilot else 0,
                        total_iterations=result['iterations']+(DESIGN['pilot_iterations'] if is_pilot else 0),
                        converged=result['converged'], objective=result['objective'], dual=result['dual'],
                        absolute_gap=result['gap'], relative_gap=result['relative_gap'],
                        objective_difference_reference=result['objective']-objective_ref,
                        coefficient_difference_reference=float(np.linalg.norm(result['coef']-beta_ref)),
                        reference_gap=gap_ref,
                        coefficient_radius=float(np.sqrt(2*result['gap']/problem.ridge)),
                        communication_per_iteration=solver.communication_per_iteration,
                        main_optimization_payload=result['communicated_scalars'],
                        pilot_optimization_payload=paid_pilot, allocation_payload=paid_allocation,
                        total_optimization_allocation_payload=result['communicated_scalars']+paid_pilot+paid_allocation,
                        main_diagnostic_evaluations=result['diagnostic_evaluations'],
                        pilot_diagnostic_evaluations=paid_checks,
                        total_diagnostic_evaluations=result['diagnostic_evaluations']+paid_checks,
                        certificate_payload_bound_per_check=certificate_cost,
                        certificate_payload_bound=(result['diagnostic_evaluations']+paid_checks)*certificate_cost,
                        edges=len(solver.edges), factors=len(problem.factors), features=len(problem.block_sizes),
                        state_dimension=result['state_dimension'],
                        largest_local_matrix=result['largest_local_matrix'],
                        max_local_factorization_storage=result['largest_local_matrix']**2,
                        factorization_storage_scalars=result['factorization_storage_scalars'],
                        main_setup_seconds=setup, main_elapsed_seconds=result['elapsed_seconds'],
                        pilot_setup_seconds=pilot_setup if is_pilot else 0.,
                        pilot_elapsed_seconds=p_result['elapsed_seconds'] if is_pilot else 0.,
                        allocation_seconds=allocation['allocation_seconds'] if is_pilot else 0.,
                        primal_invariant_error=result['invariant_error'][0],
                        dual_invariant_error=result['invariant_error'][1],
                        P_at_numerical_reference=reference_P(solver,beta_ref),
                        upper_P_uniform=allocation['upper_P_uniform'] if metric!='unit' else '',
                        upper_P_pilot=allocation['upper_P_pilot'] if metric!='unit' else '',
                        upper_P_ratio=allocation['upper_P_ratio'] if metric!='unit' else '',
                        output_budget_regret_bound=allocation['regret_bound'] if metric!='unit' else '',
                        source='new solve')
            common.update(test_metrics(result['coef'],case))
            records.append(common)
            arrays[key+'_'+metric]=result['coef']
            traces.append(dict(key=key+'_'+metric, history=result['history']))
    return dict(case=case['case'], records=records, pilots=pilots, allocations=allocations,
                arrays=arrays, traces=traces)


def write_csv(path, records):
    if not records:
        return
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def annotate_reference_regret(records, allocations):
    """Check the floored oracle at a stored finite-precision reference.

    The exact theorem concerns the unknown optimum.  These diagnostics use
    the saved numerical reference and explicitly enlarge the pilot ball by
    sqrt(2*G_reference/mu), so the check does not silently treat it as exact.
    """
    diagnostics=[]
    for allocation in allocations:
        key=allocation['key']
        selected=[r for r in records if f"{r['case']}_M{r['row_parts']}_N{r['column_parts']}"==key]
        assert len(selected)==3
        first=selected[0]
        kind='fused' if first['loss']=='squared' else first['loss']
        case=synthetic_case(kind,int(first['seed']))
        M,J=int(first['row_parts']),int(first['column_parts'])
        problem=make_regression(case['X'],case['y'],M,J,**case['args'])
        ref=np.load(RESULTS/'coefficients'/(case['case']+'_reference.npz'))
        beta=ref['coef'];radius_ref=float(np.sqrt(2*float(ref['gap'])/problem.ridge))
        blocks=np.split(beta,np.cumsum(problem.block_sizes)[:-1])
        anchor=2*np.dot(beta,beta)
        oracle=0.;adjusted_upper=0.;adjusted_lower=0.
        for a,(factor,star) in enumerate(zip(problem.factors,allocation['stars'])):
            h=sum(A@blocks[j] for j,A in factor.maps.items())
            anchor+=np.dot(h,h)
            q=np.array([np.linalg.norm(A@blocks[j]) for j,A in factor.maps.items()])
            w=water_fill(q,DESIGN['floor_fraction'])
            value=float(np.sum(q*q/w));oracle+=value
            star['reference_oracle_omega']=w.tolist()
            star['reference_oracle_edge_value']=value
            plus=np.asarray(star['c_upper'])+radius_ref*np.asarray(star['operator_norm'])
            minus=np.maximum(0.,np.asarray(star['c_lower'])-radius_ref*np.asarray(star['operator_norm']))
            adjusted_upper+=float(np.sum(plus*plus/np.asarray(star['omega'])))
            adjusted_lower+=float(np.sum(minus*minus/water_fill(minus,DESIGN['floor_fraction'])))
        oracle+=anchor
        adjusted_regret=adjusted_upper-adjusted_lower
        allocation['P_floor_oracle_reference']=float(oracle)
        allocation['reference_error_radius']=radius_ref
        allocation['regret_bound_reference_adjusted']=adjusted_regret
        for r in selected:
            r['P_floor_oracle_reference']=float(oracle) if r['metric']!='unit' else ''
            r['output_budget_regret_reference']=r['P_at_numerical_reference']-oracle if r['metric']!='unit' else ''
            r['reference_error_radius']=radius_ref
            r['regret_bound_reference_adjusted']=adjusted_regret if r['metric']=='pilot' else ''
            r['edge_upper_pilot']=allocation['edge_upper_pilot'] if r['metric']!='unit' else ''
            r['edge_upper_uniform']=allocation['edge_upper_uniform'] if r['metric']!='unit' else ''
            r['edge_upper_ratio']=allocation['edge_upper_pilot']/allocation['edge_upper_uniform'] if r['metric']!='unit' else ''
        p=next(r for r in selected if r['metric']=='pilot')
        actual=p['output_budget_regret_reference']
        if actual < -1e-8*max(1.,oracle) or actual > adjusted_regret+1e-8*max(1.,adjusted_regret):
            raise ArithmeticError('reference-adjusted regret certificate check failed')
        diagnostics.append(dict(key=key,reference_error_radius=radius_ref,
                                floor_oracle_P_reference=float(oracle),
                                pilot_P_reference=p['P_at_numerical_reference'],
                                actual_regret_reference=float(actual),
                                regret_bound=allocation['regret_bound'],
                                reference_adjusted_regret_bound=adjusted_regret,
                                original_bound_also_contains_diagnostic=bool(actual<=allocation['regret_bound']+1e-8),
                                relative_bound_utilization=float(actual/max(adjusted_regret,np.finfo(float).tiny))))
    return diagnostics


def summaries(records):
    output=[]
    fields=['main_iterations','total_iterations','total_optimization_allocation_payload',
            'main_optimization_payload','pilot_optimization_payload','allocation_payload',
            'relative_gap','coefficient_difference_reference','max_local_factorization_storage',
            'factorization_storage_scalars','P_at_numerical_reference','upper_P_ratio',
            'output_budget_regret_bound','main_elapsed_seconds',
            'P_floor_oracle_reference','output_budget_regret_reference','edge_upper_ratio']
    for kind in DESIGN['models']:
        loss='squared' if kind=='fused' else kind
        for M,J in DESIGN['layouts']:
            for metric in DESIGN['metrics']:
                group=[r for r in records if r['loss']==loss and r['row_parts']==M and r['column_parts']==J and r['metric']==metric]
                assert len(group)==5
                row=dict(loss=loss,row_parts=M,column_parts=J,metric=metric,replicates=len(group),
                         converged=sum(r['converged'] for r in group))
                for field in fields:
                    values=np.array([float(r[field]) for r in group if r[field]!=''])
                    row[field+'_mean']=float(values.mean()) if len(values) else ''
                    row[field+'_sd']=float(values.std(ddof=1)) if len(values)>1 else ''
                output.append(row)
    return output


def check_saved_outputs(records, arrays, traces):
    """Check saved endpoints and exact overlap with independently saved runs."""
    with (RESULTS/'runs.csv').open() as stream:
        previous=list(csv.DictReader(stream))
    matches=0
    for r in records:
        if r['row_parts']!=4 or r['column_parts']!=4 or r['metric'] not in ('unit','budget'):
            continue
        old=next(o for o in previous if o['case']==r['case'] and o['row_parts']=='4'
                 and o['column_parts']=='4' and o['metric']==r['metric'] and o['zero_edges']=='0')
        key=f"{r['case']}_M4_N4_{r['metric']}"
        original=np.load(RESULTS/'coefficients'/(key+'.npz'))['coef']
        if not np.array_equal(arrays[key],original) or r['main_iterations']!=int(old['iterations']):
            raise ArithmeticError('existing 4x4 coefficient/iteration check failed')
        for current,saved in [('objective','objective'),('dual','dual'),('absolute_gap','gap'),('relative_gap','relative_gap')]:
            if r[current]!=float(old[saved]):
                raise ArithmeticError('existing 4x4 certificate check failed')
        matches+=1
    indexed={f"{r['case']}_M{r['row_parts']}_N{r['column_parts']}_{r['metric']}":r for r in records}
    assert len(traces)==135 and len(arrays)==180 and matches==30
    for trace in traces:
        row=indexed[trace['key']];history=trace['history']
        assert history[-1]['iteration']==row['main_iterations']
        assert len(history)==row['main_diagnostic_evaluations']
        if row['converged']:
            assert history[-1]['relative_gap']<=DESIGN['main_relative_gap']
            assert all(h['relative_gap']>DESIGN['main_relative_gap'] for h in history[:-1])
    pilot_rows=[r for r in records if r['metric']=='pilot']
    budgets={(r['case'],r['row_parts'],r['column_parts']):r for r in records if r['metric']=='budget'}
    ratios=[r['P_at_numerical_reference']/budgets[(r['case'],r['row_parts'],r['column_parts'])]['P_at_numerical_reference'] for r in pilot_rows]
    work_better=sum(r['total_optimization_allocation_payload']<budgets[(r['case'],r['row_parts'],r['column_parts'])]['total_optimization_allocation_payload'] for r in pilot_rows)
    return dict(existing_4x4_non_timing_matches=matches,existing_4x4_coefficients_bitwise_identical=True,
                trace_count=len(traces),coefficient_arrays=len(arrays),all_first_certified_checkpoints_verified=True,
                reference_P_pilot_lower_than_uniform_count=sum(x<1 for x in ratios),
                reference_P_ratio_min=min(ratios),reference_P_ratio_max=max(ratios),
                individual_pilot_payload_improvements_over_uniform=int(work_better))



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--rebuild-summaries',action='store_true',
                        help='Recompute reference-regret diagnostics and tables from completed new runs')
    args=parser.parse_args()
    if args.rebuild_summaries:
        with (RESULTS/'pilot_budget_runs.csv').open() as stream:
            records=list(csv.DictReader(stream))
        for r in records:
            for k,v in r.items():
                if v in ('True','False'):
                    r[k]=v=='True'
                else:
                    try:
                        x=float(v);r[k]=int(x) if x.is_integer() else x
                    except ValueError:
                        pass
        allocations=json.loads((RESULTS/'pilot_budget_weights.json').read_text())
        diagnostics=annotate_reference_regret(records,allocations)
        with (RESULTS/'pilot_budget_pilots.csv').open() as stream:
            pilot_records=list(csv.DictReader(stream))
        indexed={a['key']:a for a in allocations}
        for r in pilot_records:
            a=indexed[r['key']]
            for field in ('edge_upper_pilot','edge_upper_uniform','edge_lower_best'):
                r[field]=a[field]
            r['edge_upper_ratio']=a['edge_upper_pilot']/a['edge_upper_uniform']
        write_csv(RESULTS/'pilot_budget_pilots.csv',pilot_records)
        summary=summaries(records)
        write_csv(RESULTS/'pilot_budget_runs.csv',records)
        write_csv(RESULTS/'pilot_budget_summary.csv',summary)
        (RESULTS/'pilot_budget_weights.json').write_text(json.dumps(allocations,indent=2)+'\n')
        (RESULTS/'pilot_budget_reference_regret.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
        methods=json.loads((RESULTS/'pilot_budget_methods.json').read_text())
        methods.update(DESIGN)
        methods.setdefault('script_sha256_at_execution',methods['script_sha256'])
        methods['script_sha256']=sha256(__file__)
        methods['postprocessing_note']='Reference-oracle diagnostics, output-contribution bound ratios and table regenerated from complete stored runs; no parameter selection or numerical result modification.'
        (RESULTS/'pilot_budget_methods.json').write_text(json.dumps(methods,indent=2)+'\n')
        validation=json.loads((RESULTS/'pilot_budget_validation.json').read_text())
        validation['reference_regret_checks']=len(diagnostics)
        validation['all_reference_regrets_in_adjusted_bound']=all(r['actual_regret_reference']<=r['reference_adjusted_regret_bound']+1e-8 for r in diagnostics)
        validation['all_reference_regrets_also_in_original_bound']=all(r['original_bound_also_contains_diagnostic'] for r in diagnostics)
        validation['maximum_reference_regret_bound_utilization']=max(r['relative_bound_utilization'] for r in diagnostics)
        saved=np.load(RESULTS/'pilot_budget_coefficients.npz')
        with gzip.open(RESULTS/'pilot_budget_traces.jsonl.gz','rt') as stream:
            traces=[json.loads(line) for line in stream]
        validation.update(check_saved_outputs(records,{k:saved[k] for k in saved.files},traces))
        (RESULTS/'pilot_budget_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        write_pilot_table(summary)
        print('Rebuilt summaries and reference-regret certificates for',len(diagnostics),'pilots')
        return
    original={str(f.relative_to(ROOT)):sha256(f) for f in RESULTS.rglob('*')
              if f.is_file() and f.suffix in {'.csv','.gz','.npz'} and not f.name.startswith('pilot_budget_')}
    checks=validate_water_fill()
    methods=dict(**DESIGN,process_workers=args.workers,python=platform.python_version(),
                 numpy=np.__version__,scipy=scipy.__version__,blas_threads=1,
                 timing='Independent jobs may run concurrently; every solve is a sequential simulation. No timings are interpreted as network or distributed speedups.',
                 existing_results_reused=False,script_sha256=sha256(__file__),
                 script_sha256_at_execution=sha256(__file__),
                 solver_sha256_at_start=sha256(ROOT/'code/hybrid_solver.py'))
    (RESULTS/'pilot_budget_methods.json').write_text(json.dumps(methods,indent=2)+'\n')
    tasks=[(kind,seed) for kind in DESIGN['models'] for seed in DESIGN['seeds']]
    start=perf_counter();answers=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_case,task):task for task in tasks}
        for future in as_completed(futures):
            result=future.result();answers.append(result)
            print(f'{result["case"]}: {len(result["records"])} main solves, '
                  f'{sum(r["converged"] for r in result["records"])} converged; '
                  f'elapsed batch {perf_counter()-start:.1f}s',flush=True)
    answers.sort(key=lambda r:r['case'])
    records=[r for a in answers for r in a['records']]
    pilots=[r for a in answers for r in a['pilots']]
    allocation=[r for a in answers for r in a['allocations']]
    arrays={k:v for a in answers for k,v in a['arrays'].items()}
    assert len(records)==135 and len(pilots)==45 and len(arrays)==180
    reference_regret=annotate_reference_regret(records,allocation)
    summary=summaries(records)
    write_csv(RESULTS/'pilot_budget_runs.csv',records)
    write_csv(RESULTS/'pilot_budget_pilots.csv',pilots)
    write_csv(RESULTS/'pilot_budget_summary.csv',summary)
    np.savez_compressed(RESULTS/'pilot_budget_coefficients.npz',**arrays)
    (RESULTS/'pilot_budget_weights.json').write_text(json.dumps(allocation,indent=2)+'\n')
    (RESULTS/'pilot_budget_reference_regret.json').write_text(json.dumps(reference_regret,indent=2)+'\n')
    with gzip.open(RESULTS/'pilot_budget_traces.jsonl.gz','wt') as stream:
        for a in answers:
            for trace in a['traces']:stream.write(json.dumps(trace)+'\n')
    write_pilot_table(summary)
    changed=[name for name,digest in original.items() if sha256(ROOT/name)!=digest]
    if changed:
        raise RuntimeError(f'Pre-existing numerical files changed: {changed}')
    radius_check=[]
    for r in records:
        allowance=r['coefficient_radius']+np.sqrt(2*r['reference_gap']/r['ridge'])
        radius_check.append(r['coefficient_difference_reference']<=allowance+1e-8)
    validation=dict(main_solves=len(records),pilot_solves=len(pilots),
                    converged_main_solves=sum(r['converged'] for r in records),
                    main_relative_gap_max=max(r['relative_gap'] for r in records),
                    finite_pilot_certificates=all(np.isfinite(r['absolute_gap']) and r['absolute_gap']>=0 for r in pilots),
                    all_coefficient_reference_differences_within_two_certificate_radii=all(radius_check),
                    all_pilot_upper_bounds_no_larger_than_uniform=all(r['upper_P_ratio']<=1+1e-12 for r in pilots),
                    maximum_upper_bound_ratio=max(r['upper_P_ratio'] for r in pilots),
                    minimum_upper_bound_ratio=min(r['upper_P_ratio'] for r in pilots),
                    maximum_main_invariant_error=max(max(r['primal_invariant_error'],r['dual_invariant_error']) for r in records),
                    original_numerical_files_checked=len(original),original_numerical_files_changed=changed,
                    water_filling_independent_checks=checks,batch_elapsed_seconds=perf_counter()-start,
                    reference_regret_checks=len(reference_regret),
                    all_reference_regrets_in_adjusted_bound=all(r['actual_regret_reference']<=r['reference_adjusted_regret_bound']+1e-8 for r in reference_regret),
                    all_reference_regrets_also_in_original_bound=all(r['original_bound_also_contains_diagnostic'] for r in reference_regret),
                    maximum_reference_regret_bound_utilization=max(r['relative_bound_utilization'] for r in reference_regret),
                    solver_sha256_at_end=sha256(ROOT/'code/hybrid_solver.py'))
    validation.update(check_saved_outputs(records,arrays,[t for a in answers for t in a['traces']]))
    (RESULTS/'pilot_budget_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (RESULTS/'pilot_budget_original_hashes.json').write_text(json.dumps(original,indent=2)+'\n')
    print(json.dumps({k:v for k,v in validation.items() if k!='water_filling_independent_checks'},indent=2),flush=True)


if __name__=='__main__':
    main()
