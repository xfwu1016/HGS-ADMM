"""Reproduce the manuscript's simulations and public-data examples.

Run with one BLAS thread, e.g.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python run_experiments.py
All elapsed times are sequential simulation times, including periodic gap
evaluations and excluding setup and the final state-invariant audit.
"""
from __future__ import annotations
import argparse,csv,gzip,json,os,platform,time
from pathlib import Path
import numpy as np
import scipy,sklearn
from scipy.special import expit
from sklearn.datasets import load_diabetes,load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from hybrid_solver import GraphSplitting,make_regression

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
LAYOUTS=((1,1),(4,1),(4,4),(8,4))
WEIGHTS=("unit","budget")
FIELDS=["case","kind","seed","loss","n","p","row_parts","column_parts","metric","zero_edges",
        "l1","ridge","fusion","rho","iterations","objective","dual","gap","relative_gap","converged",
        "objective_difference_reference","coefficient_difference_reference","coefficient_error_truth",
        "test_rmse","test_logloss","test_auc","test_accuracy","setup_seconds","elapsed_seconds",
        "communication_per_iteration","communicated_scalars","diagnostic_evaluations","state_dimension",
        "largest_local_matrix","factorization_storage_scalars","edges","primal_invariant_error",
        "dual_invariant_error","distance_constant_approx","primal_norm_approx","dual_norm_approx",
        "prediction_dispersion_H","gradient_dispersion_G","budget_identity_error"]


def ar_design(rng,n,p,corr=.4):
    X=rng.normal(size=(n,p))
    for j in range(1,p):X[:,j]=corr*X[:,j-1]+np.sqrt(1-corr*corr)*X[:,j]
    return X


def synthetic_case(kind,seed,n=300,p=600):
    rng=np.random.default_rng(202600+1000*{"fused":0,"huber":1,"logistic":2}[kind]+seed)
    X=ar_design(rng,n,p);Xt=ar_design(rng,1000,p)
    truth=np.zeros(p)
    if kind=="fused":
        for lo,hi,value in [(20,45,1),(145,170,-1),(290,320,.8),(440,460,-.8)]:truth[lo:hi]=value
        y=X@truth+rng.normal(size=n);yt=Xt@truth+rng.normal(size=1000)
        args=dict(loss="squared",loss_parameter=1,l1=.02,ridge=.01,fusion=.08,fusion_parts=4)
    elif kind=="huber":
        truth[:15]=np.where(np.arange(15)%2, -1.0, 1.0)
        noise=rng.normal(size=n);contamination=rng.random(n)<.1
        noise[contamination]+=rng.normal(scale=10,size=int(contamination.sum()))
        y=X@truth+noise;yt=Xt@truth+rng.normal(size=1000)
        args=dict(loss="huber",loss_parameter=1.345,l1=.025,ridge=.02,fusion=0)
    else:
        truth[:12]=np.where(np.arange(12)%2,-.7,.7)
        y=2*(rng.random(n)<expit(X@truth))-1
        yt=2*(rng.random(1000)<expit(Xt@truth))-1
        args=dict(loss="logistic",loss_parameter=1,l1=.02,ridge=.02,fusion=0)
    return dict(case=f"synthetic_{kind}_{seed}",kind="synthetic",seed=seed,X=X,y=y,Xtest=Xt,ytest=yt,
                truth=truth,args=args,target_scale=1.0,metadata=dict(generator="AR(1)",correlation=.4,
                test_n=1000,noise="10 percent additive N(0,100) contamination" if kind=="huber" else "standard"))


def test_metrics(beta,case):
    pred=case['Xtest']@beta;y=case['ytest']
    if case['args']['loss']=="logistic":
        probs=expit(pred)
        return dict(test_rmse="",test_logloss=float(log_loss((y+1)/2,probs)),
                    test_auc=float(roc_auc_score((y+1)/2,probs)),
                    test_accuracy=float(accuracy_score(y,np.where(pred>=0,1,-1))))
    return dict(test_rmse=float(np.sqrt(np.mean((pred-y)**2))*case.get('target_scale',1)),
                test_logloss="",test_auc="",test_accuracy="")


def distance_diagnostics(solver,coef,theta):
    """Evaluate the zero-start distance formula at a high-accuracy numerical KKT pair.

    These are approximate diagnostic constants, not certified unknown optima.
    """
    blocks=np.split(coef,np.cumsum(solver.problem.block_sizes)[:-1])
    z=[sum(A@blocks[j] for j,A in f.maps.items()) for f in solver.problem.factors]
    g=[np.zeros(p) for p in solver.problem.block_sizes]
    for f,t in zip(solver.problem.factors,theta):
        for j,A in f.maps.items():g[j]+=A.T@t
    pn=sum(a*np.dot(v,v) for a,v in zip(solver.a,z))+sum(c*np.dot(v,v) for c,v in zip(solver.c,blocks))
    dn=sum(np.dot(t,t)/a for a,t in zip(solver.a,theta))+sum(np.dot(v,v)/c for c,v in zip(solver.c,g))
    H=0.;G=0.
    for e,(a,j,A) in enumerate(solver.edges):
        contrib=A@blocks[j];grad=A.T@theta[a]
        pn+=solver.b[e]*np.dot(contrib,contrib)+solver.d[e]*np.dot(blocks[j],blocks[j])
        dn+=np.dot(theta[a],theta[a])/solver.b[e]+np.dot(grad,grad)/solver.d[e]
        omega=1/solver.da[a];pi=1/solver.dj[j]
        dz=contrib-omega*z[a];dg=grad-pi*g[j]
        H+=np.dot(dz,dz)/omega;G+=np.dot(dg,dg)/pi
    identity=""
    if solver.weights=="budget":
        rhs_p=2*sum(np.dot(v,v) for v in blocks)+2*sum(np.dot(v,v) for v in z)+H
        rhs_d=2*sum(np.dot(v,v) for v in theta)+2*sum(np.dot(v,v) for v in g)+G
        identity=abs(pn-rhs_p)+abs(dn-rhs_d)
        assert identity<1e-7*max(1,pn+dn)
    return dict(distance_constant_approx=solver.rho*pn+dn/solver.rho,
                primal_norm_approx=pn,dual_norm_approx=dn,
                prediction_dispersion_H=H,gradient_dispersion_G=G,budget_identity_error=identity)


def split_theta(reference_solver,M):
    s=[f.project_dual(-reference_solver.rho*reference_solver.a[a]*reference_solver.u[sl])
       for a,(f,sl) in enumerate(zip(reference_solver.problem.factors,reference_solver.ys))]
    return list(np.array_split(s[0],M))+s[1:]


def run_case(case,writer,traces,stopping,records,extra_padded=False):
    X,y=case['X'],case['y'];args=case['args'];n,p=X.shape
    refprob=make_regression(X,y,1,1,**args)
    refsolver=GraphSplitting(refprob,weights="unit",rho=1)
    reference=refsolver.solve(tol=1e-10,maxiter=50000)
    if not reference['converged']:raise RuntimeError(f"reference failed {case['case']}")
    np.savez_compressed(RESULTS/"coefficients"/(case['case']+"_reference.npz"),coef=reference['coef'],
                        objective=reference['objective'],gap=reference['gap'],truth=case.get('truth',[]))
    configs=[(M,N,w,False) for M,N in LAYOUTS for w in WEIGHTS]
    if extra_padded:configs.extend((4,4,w,True) for w in WEIGHTS)
    for M,N,metric,padded in configs:
        start=time.perf_counter()
        problem=make_regression(X,y,M,N,**args,include_zero_edges=padded)
        solver=GraphSplitting(problem,rho=1,weights=metric)
        setup=time.perf_counter()-start
        result=solver.solve(tol=1e-6 if case['kind']=="synthetic" else 1e-7,maxiter=30000,
                            snapshot_tolerances=(1e-3,1e-4,1e-5,1e-6,1e-7))
        row=dict(case=case['case'],kind=case['kind'],seed=case['seed'],loss=args['loss'],n=n,p=p,
                 row_parts=M,column_parts=N,metric=metric,zero_edges=int(padded),l1=args['l1'],
                 ridge=args['ridge'],fusion=args.get('fusion',0),rho=1,
                 objective_difference_reference=result['objective']-reference['objective'],
                 coefficient_difference_reference=float(np.linalg.norm(result['coef']-reference['coef'])),
                 coefficient_error_truth=float(np.linalg.norm(result['coef']-case['truth'])) if 'truth' in case else "",
                 setup_seconds=setup,edges=len(solver.edges),
                 primal_invariant_error=result['invariant_error'][0],dual_invariant_error=result['invariant_error'][1])
        for key in FIELDS:
            if key in result:row[key]=result[key]
        row.update(test_metrics(result['coef'],case))
        row.update(distance_diagnostics(solver,reference['coef'],split_theta(refsolver,M)))
        writer.writerow(row);records.append(row)
        key=f"{case['case']}_M{M}_N{N}_{metric}"+("_padded" if padded else "")
        traces.write(json.dumps(dict(key=key,history=result['history']))+"\n")
        np.savez_compressed(RESULTS/"coefficients"/(key+".npz"),coef=result['coef'])
        for threshold,snapshot in result['snapshots'].items():
            d2=float(np.sum((snapshot['coef']-reference['coef'])**2))
            stopping.append(dict(key=key,case=case['case'],threshold=threshold,iteration=snapshot['iteration'],
                                 gap=snapshot['gap'],relative_gap=snapshot['relative_gap'],
                                 squared_distance_reference=d2,strong_convexity_bound=2*snapshot['gap']/args['ridge'],
                                 **test_metrics(snapshot['coef'],case)))
        print(f"{key}: k={result['iterations']} gap={result['relative_gap']:.2e} sec={result['elapsed_seconds']:.2f}",flush=True)
        if not result['converged']:raise RuntimeError(f"failed target {key}")
    return dict(case=case['case'],reference_objective=reference['objective'],reference_gap=reference['gap'],
                args=args,metadata=case.get('metadata',{}))


def prepare_real_case(which,split_seed=2026,*,save_split=True):
    if which=="diabetes":
        data=load_diabetes(scaled=False);Xraw=data.data;yraw=data.target
        baseidx=np.arange(len(yraw))
        train,test=train_test_split(baseidx,test_size=.3,random_state=split_seed)
        fit,val=train_test_split(train,test_size=.25,random_state=split_seed+1)
        poly=PolynomialFeatures(degree=2,include_bias=False)
        Xall=poly.fit_transform(Xraw)
        scale=StandardScaler().fit(Xall[fit]);X=scale.transform(Xall[fit]);Xv=scale.transform(Xall[val])
        ym=yraw[fit].mean();ys=yraw[fit].std();y=(yraw[fit]-ym)/ys;yv=(yraw[val]-ym)/ys
        grid=[.005,.015,.05];args=dict(loss="huber",loss_parameter=1.345,ridge=.01,fusion=0)
    else:
        data=load_breast_cancer();Xall=data.data;yraw=2*data.target-1;baseidx=np.arange(len(yraw))
        train,test=train_test_split(baseidx,test_size=.3,stratify=yraw,random_state=split_seed)
        fit,val=train_test_split(train,test_size=.25,stratify=yraw[train],random_state=split_seed+1)
        scale=StandardScaler().fit(Xall[fit]);X=scale.transform(Xall[fit]);Xv=scale.transform(Xall[val])
        y=yraw[fit];yv=yraw[val];grid=[.001,.003,.01,.03]
        args=dict(loss="logistic",loss_parameter=1,ridge=.01,fusion=0)
    tuning=[]
    for lam in grid:
        problem=make_regression(X,y,1,1,l1=lam,**args)
        out=GraphSplitting(problem).solve(tol=1e-8,maxiter=30000)
        if not out['converged']:raise RuntimeError("tuning solve did not converge")
        pred=Xv@out['coef']
        score=float(np.mean((pred-yv)**2)) if which=="diabetes" else float(log_loss((yv+1)/2,expit(pred)))
        tuning.append(dict(l1=lam,validation_score=score,relative_gap=out['relative_gap']))
    selected=min(tuning,key=lambda a:a['validation_score'])['l1'];args['l1']=selected
    scale=StandardScaler().fit(Xall[train]);X=scale.transform(Xall[train]);Xt=scale.transform(Xall[test])
    if which=="diabetes":
        ym=yraw[train].mean();ys=yraw[train].std();y=(yraw[train]-ym)/ys;yt=(yraw[test]-ym)/ys
    else:
        y=yraw[train];yt=yraw[test];ys=1;ym=0
    if save_split:
        np.savez_compressed(RESULTS/"datasets"/(which+"_split.npz"),X=X,y=y,Xtest=Xt,ytest=yt,
                            train_indices=train,test_indices=test,fit_indices=fit,validation_indices=val,
                            feature_means=scale.mean_,feature_scales=scale.scale_,target_mean=ym,target_scale=ys)
    meta=dict(dataset=which,split_seed=split_seed,train_n=len(train),test_n=len(test),fit_n=len(fit),
              validation_n=len(val),tuning=tuning,selected_l1=selected,ridge=args['ridge'],
              validation_score="standardized MSE" if which=="diabetes" else "log loss",
              polynomial_degree=2 if which=="diabetes" else 1,intercept="training response mean" if which=="diabetes" else "fixed zero on standardized features",
              preprocessing="Feature scaling fitted on fit subset for tuning, then on all training data for final estimates; test data never used for tuning or scaling")
    if which=="breast_cancer":meta['positive_class']='benign (original scikit-learn target 1)'
    return dict(case="real_"+which,kind="real",seed=split_seed,X=X,y=y,Xtest=Xt,ytest=yt,
                args=args,target_scale=ys,metadata=meta)


def write_summary(records):
    keys=["loss","row_parts","column_parts","metric","zero_edges"]
    numeric=["iterations","communicated_scalars","elapsed_seconds","relative_gap","coefficient_difference_reference",
             "test_rmse","test_logloss","test_auc","distance_constant_approx","prediction_dispersion_H","gradient_dispersion_G"]
    groups={}
    for row in records:
        if row['kind']!="synthetic":continue
        key=tuple(row[k] for k in keys);groups.setdefault(key,[]).append(row)
    with (RESULTS/"summary.csv").open('w',newline='') as f:
        fields=keys+["replicates"]+[f"{col}_{stat}" for col in numeric for stat in ("mean","sd")]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for key,rows in groups.items():
            out=dict(zip(keys,key));out['replicates']=len(rows)
            for col in numeric:
                v=[float(r[col]) for r in rows if r[col] != ""]
                out[col+"_mean"]=np.mean(v) if v else ""
                out[col+"_sd"]=np.std(v,ddof=1) if len(v)>1 else (0 if v else "")
            writer.writerow(out)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--skip-real',action='store_true')
    args=ap.parse_args()
    for directory in (RESULTS,RESULTS/"coefficients",RESULTS/"datasets"):directory.mkdir(parents=True,exist_ok=True)
    methods=dict(synthetic_n=300,synthetic_p=600,seeds=list(range(args.seeds)),layouts=LAYOUTS,metrics=WEIGHTS,
                 rho=1,gap_definition="(P(beta)-D(theta))/max(1,abs(P(beta)))",synthetic_gap_tolerance=1e-6,
                 real_gap_tolerance=1e-7,reference_gap_tolerance=1e-10,check_every=10,
                 communication="2 sum_edges(m_factor+p_feature) float64 scalars per optimization iteration; diagnostics are evaluated centrally offline and are excluded, with their number recorded",
                 design_scaling="Data operators X/sqrt(n). Squared f=||z-y/sqrt(n)||^2/2. Huber f=sum Huber_(delta/sqrt(n))(z-y/sqrt(n)). Logistic f=sum log(1+exp(-sqrt(n)*label*z))/n. Structural D maps unscaled, f_D=lambda_TV||z_D||_1.",
                 fusion_factor_blocks=4,weights="unit: a=c=b=d=1; budget: a=c=1, b_e=factor degree, d_e=1/feature degree",
                 timing="sequential simulation on one Python process; includes periodic gap evaluations, excludes setup and the final state-invariant audit; no distributed speedup claim",
                 python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                 platform=platform.platform(),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),omp_threads=os.environ.get('OMP_NUM_THREADS'))
    cpuinfo=Path('/proc/cpuinfo')
    methods['cpu_model']=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
                               if line.startswith('model name')), 'not reported') if cpuinfo.exists() else platform.processor()
    (RESULTS/"methods.json").write_text(json.dumps(methods,indent=2))
    records=[];stopping=[];cases=[];start=time.perf_counter()
    with (RESULTS/"runs.csv").open('w',newline='') as f,gzip.open(RESULTS/"traces.jsonl.gz",'wt') as traces:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader()
        for kind in ("fused","huber","logistic"):
            for seed in range(args.seeds):
                case=synthetic_case(kind,seed)
                cases.append(run_case(case,writer,traces,stopping,records,extra_padded=(kind=="fused" and seed==0)))
                f.flush();traces.flush()
        if not args.skip_real:
            for name in ("diabetes","breast_cancer"):
                case=prepare_real_case(name)
                cases.append(run_case(case,writer,traces,stopping,records));f.flush();traces.flush()
    write_summary(records)
    with (RESULTS/"stopping_certificates.csv").open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stopping[0]));writer.writeheader();writer.writerows(stopping)
    (RESULTS/"cases.json").write_text(json.dumps(cases,indent=2))
    (RESULTS/"completion.json").write_text(json.dumps(dict(completed_runs=len(records),elapsed_seconds=time.perf_counter()-start,
                                                         all_converged=all(r['converged'] for r in records)),indent=2))
    print(f"FINISHED {len(records)} runs in {time.perf_counter()-start:.1f}s",flush=True)


if __name__=="__main__":main()
