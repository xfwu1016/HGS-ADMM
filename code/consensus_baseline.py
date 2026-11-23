"""Sample-consensus ADMM comparator based on Wu et al. (CSDA 203, 108081).

Uses their two-block ordering and, for fusion, the central proximal
linearization with M+4.01. A ridge term is handled in the central proximal
map for the fused model, so the objective matches the graph solver exactly.
For elastic net the ridge auxiliary variable is retained as in their paper.
This is a Python implementation, not a timing benchmark of the R package.
"""
from pathlib import Path
from time import perf_counter
import csv,json
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from hybrid_solver import make_regression,soft
from run_experiments import synthetic_case,test_metrics

ROOT=Path(__file__).resolve().parents[1]


def dt(v,p):
    out=np.zeros(p);out[:-1]-=v;out[1:]+=v
    return out


def consensus_admm(X,y,row_parts,args,tol=1e-6,rho=1.0,maxiter=50000):
    began=perf_counter()
    problem=make_regression(X,y,row_parts,1,**args)
    fs=problem.factors[:row_parts];m=row_parts;p=X.shape[1]
    matrices=[f.maps[0] for f in fs]
    chols=[cho_factor(np.eye(A.shape[0])+A@A.T,check_finite=False) for A in matrices]
    local=np.zeros((m,p));uc=np.zeros((m,p))
    ut=[np.zeros(f.size) for f in fs]
    pred=[np.zeros(f.size) for f in fs]
    beta=np.zeros(p);fusion=args.get('fusion',0);ridge=args['ridge'];l1=args['l1']
    aux=np.zeros(p-1 if fusion else p);ug=aux.copy()
    setup=perf_counter()-began;started=perf_counter();history=[]
    for k in range(1,maxiter+1):
        aggregate=(local+uc).sum(axis=0)
        if fusion:
            eta=m+4.01
            grad=m*beta-aggregate+dt(np.diff(beta)-aux+ug,p)
            beta=soft(beta-grad/eta,l1/(rho*eta))/(1+ridge/(rho*eta))
            aux=soft(np.diff(beta)+ug,fusion/rho)
            ug+=np.diff(beta)-aux
        else:
            beta=soft((aggregate+aux-ug)/(m+1),l1/(rho*(m+1)))
            aux=rho*(beta+ug)/(rho+ridge)
            ug+=beta-aux
        for i,(A,f,chol) in enumerate(zip(matrices,fs,chols)):
            pred[i]=f.prox(A@local[i]+ut[i],rho)
            center=beta-uc[i]
            # Woodbury, equal to (I+A'A)^(-1)(A'(t-u)+beta-u_c).
            local[i]=center+A.T@cho_solve(chol,pred[i]-ut[i]-A@center,check_finite=False)
            ut[i]+=A@local[i]-pred[i]
            uc[i]+=local[i]-beta
        if k==1 or k%10==0 or k==maxiter:
            theta=[rho*v for v in ut]
            if fusion:
                lengths=[f.size for f in problem.factors[m:]]
                theta+=list(np.split(rho*ug,np.cumsum(lengths)[:-1]))
            primal=problem.objective([beta]);dual=problem.dual_bound(theta);gap=primal-dual
            if gap < -1e-9*max(1,abs(primal)):raise RuntimeError('invalid lower bound')
            gap=max(gap,0);relative=gap/max(1,abs(primal))
            history.append(dict(iteration=k,gap=gap,relative_gap=relative))
            if relative<=tol:break
    return dict(coef=beta,iterations=k,objective=primal,dual=dual,gap=gap,relative_gap=relative,
                converged=relative<=tol,setup_seconds=setup,elapsed_seconds=perf_counter()-started,
                communication_per_iteration=2*m*p,communicated_scalars=2*m*p*k,
                diagnostic_evaluations=len(history),history=history)


def validate():
    from hybrid_solver import GraphSplitting
    rng=np.random.default_rng(819)
    for loss,fusion in [('squared',0),('squared',.05),('huber',0),('logistic',0)]:
        X=rng.normal(size=(70,35));b=rng.normal(size=35);y=X@b+rng.normal(size=70)
        if loss=='logistic':y=np.where(y>0,1,-1)
        args=dict(loss=loss,loss_parameter=1.345,l1=.04,ridge=.03,fusion=fusion,fusion_parts=4)
        prob=make_regression(X,y,2,2,**args)
        ref=GraphSplitting(prob,weights='budget').solve(tol=1e-10,maxiter=50000)
        out=consensus_admm(X,y,4,args,tol=1e-9)
        assert out['converged']
        assert abs(out['objective']-ref['objective']) < 2e-8
        print(loss,fusion,'verified difference',out['objective']-ref['objective'],flush=True)


def main():
    validate()
    output=ROOT/'results'/'consensus_baseline.csv'
    fields=['case','loss','seed','n','p','row_parts','iterations','objective','dual','gap','relative_gap',
            'converged','setup_seconds','elapsed_seconds','communication_per_iteration',
            'communicated_scalars','diagnostic_evaluations','test_rmse','test_logloss','test_auc','test_accuracy',
            'objective_difference_reference','coefficient_difference_reference']
    allrows=[]
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for kind in ['fused','huber','logistic']:
            for seed in range(5):
                case=synthetic_case(kind,seed)
                reference=np.load(ROOT/'results'/'coefficients'/(case['case']+'_reference.npz'))
                for m in (4,8):
                    out=consensus_admm(case['X'],case['y'],m,case['args'])
                    row={k:v for k,v in out.items() if k in fields}
                    row.update(case=case['case'],loss=case['args']['loss'],seed=seed,n=case['X'].shape[0],
                               p=case['X'].shape[1],row_parts=m,**test_metrics(out['coef'],case),
                               objective_difference_reference=out['objective']-float(reference['objective']),
                               coefficient_difference_reference=np.linalg.norm(out['coef']-reference['coef']))
                    writer.writerow(row);f.flush();allrows.append(row)
                    print(case['case'],m,'iterations',out['iterations'],'gap',out['relative_gap'],flush=True)
                    if not out['converged']:raise RuntimeError('comparison did not reach tolerance')
    (ROOT/'results'/'consensus_baseline_summary.json').write_text(json.dumps(dict(
        completed=len(allrows),all_converged=all(r['converged'] for r in allrows),
        implementation='Python reimplementation of sample-consensus ADMM; not original R package timing',
        fusion_update='proximal central linearization with eta=M+4.01, ridge included in central proximal term',
        traffic='2 M p scalars per iteration, aggregating beta_m+u_m before transmission; diagnostics excluded',
        rho=1,gap_target=1e-6),indent=2))


if __name__=='__main__':main()
