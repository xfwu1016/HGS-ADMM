"""Finite, supplementary rho sensitivity analysis; never retunes main runs.

Three synthetic seed-0 cases, graph layout (8,4), two metrics, and the
sample-consensus comparator with eight sample holders. All methods use
rho in {0.1, 1, 10}, relative Fenchel gap 1e-6, and at most 50,000 iterations.
Every result is retained, including any failure to reach the requested gap
before the iteration cap.
"""
from pathlib import Path
from time import perf_counter
import csv,json
from hybrid_solver import GraphSplitting,make_regression
from run_experiments import synthetic_case
from consensus_baseline import consensus_admm

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['case','loss','seed','n','p','method','metric','row_parts','column_parts','rho',
        'maxiter','tolerance','iterations','objective','dual','gap','relative_gap','converged',
        'communication_per_iteration','communicated_scalars','diagnostic_evaluations',
        'setup_seconds','elapsed_seconds']


def main():
    destination=ROOT/'results'/'penalty_sensitivity.csv'
    records=[];began=perf_counter()
    with destination.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for kind in ['fused','huber','logistic']:
            case=synthetic_case(kind,0)
            for method,metric,N in [('graph','unit',4),('graph','budget',4),('sample_consensus','none',1)]:
                for rho in [.1,1.,10.]:
                    if method=='graph':
                        started=perf_counter()
                        problem=make_regression(case['X'],case['y'],8,4,**case['args'])
                        solver=GraphSplitting(problem,rho=rho,weights=metric)
                        setup=perf_counter()-started
                        out=solver.solve(tol=1e-6,maxiter=50000)
                        out['setup_seconds']=setup
                    else:
                        out=consensus_admm(case['X'],case['y'],8,case['args'],rho=rho,tol=1e-6,maxiter=50000)
                    row={k:v for k,v in out.items() if k in FIELDS}
                    row.update(case=case['case'],loss=case['args']['loss'],seed=0,n=case['X'].shape[0],
                               p=case['X'].shape[1],method=method,metric=metric,row_parts=8,column_parts=N,
                               rho=rho,maxiter=50000,tolerance=1e-6)
                    writer.writerow(row);f.flush();records.append(row)
                    print(f"{kind} {method} {metric} rho={rho:g}: iterations={out['iterations']} "
                          f"gap={out['relative_gap']:.3e} converged={out['converged']}",flush=True)
    metadata=dict(completed_runs=len(records),seed=0,rho_grid=[.1,1.,10.],graph_layout=[8,4],
                  sample_consensus_holders=8,relative_gap_target=1e-6,maxiter=50000,
                  capped_runs=sum(not r['converged'] for r in records),elapsed_seconds=perf_counter()-began,
                  purpose='Independent supplementary sensitivity analysis; not used to select or alter the main experiment parameters',
                  retention='All grid runs retained, including iteration-capped runs',
                  communication='Optimization payload only; periodic gap diagnostics are excluded from payload counts',
                  timing='Sequential simulation includes periodic gap evaluations, excludes setup and final graph audit checks')
    (ROOT/'results'/'penalty_sensitivity.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()
