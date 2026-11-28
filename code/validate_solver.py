"""Independent numerical validation, not a performance benchmark."""
import json
from pathlib import Path
import os
import numpy as np
from scipy.optimize import brentq, linprog, minimize
from scipy.special import expit
from sklearn.linear_model import ElasticNet
from hybrid_solver import Factor, GraphSplitting, make_regression, soft


def run_validation():
    rng=np.random.default_rng(917)
    X=rng.normal(size=(100,42));truth=np.r_[np.ones(8),np.zeros(34)]
    y=X@truth+rng.normal(size=100)
    records=[]
    # Proximal first-order conditions, including conjugate-domain feasibility.
    for kind in ("squared","huber","quantile","logistic","l1","l2"):
        response=(np.array([-1,1,-1,1])*np.sqrt(100) if kind=="logistic" else np.array([.4,-.7,.2,1.1]))
        f=Factor(kind,{0:np.eye(4)},response,scale=.01 if kind=="logistic" else .3,
                 parameter=.4 if kind=="quantile" else .7)
        v=np.array([-1.4,.5,1.8,-.1]);coefficient=.8;z=f.prox(v,coefficient);s=coefficient*(v-z)
        domain_error=np.linalg.norm(s-f.project_dual(s))
        fenchel_error=abs(f.value(z)+f.conjugate(s)-np.dot(s,z))
        assert domain_error<1e-11 and fenchel_error<1e-11
        records.append(dict(test="proximal_fenchel_identity",loss=kind,domain_error=domain_error,fenchel_error=fenchel_error))
    # Small positive rho can make unrestricted logistic Newton steps cycle.
    # Compare both label signs and scaled labels with an independent root solve.
    for response,scale in ((1.,1.),(-1.,1.),(10.,.01),(-10.,.01)):
        f=Factor("logistic",{0:np.ones((1,1))},np.array([response]),scale=scale)
        v=-10./response
        for coefficient in (1.,.01,.001):
            shift=scale*response/coefficient
            equation=lambda z:z-v-shift*expit(-response*z)
            reference=brentq(equation,min(v,v+shift),max(v,v+shift),xtol=1e-14)
            z=float(f.prox(np.array([v]),coefficient)[0])
            error=abs(z-reference);residual=abs(equation(z))
            assert error<1e-11 and residual<1e-10
            records.append(dict(test="logistic_prox_safeguarded_root",response=response,scale=scale,
                                coefficient=coefficient,root_error=error,stationarity_error=residual))
    ref=ElasticNet(alpha=.06,l1_ratio=2/3,fit_intercept=False,tol=1e-13,max_iter=100000).fit(X,y).coef_
    for weights in ("unit","budget"):
        problem=make_regression(X,y,4,3,l1=.04,ridge=.02)
        solver=GraphSplitting(problem,weights=weights)
        v=rng.normal(size=solver.eta.size);pv=solver.project(v)
        idempotence=np.linalg.norm(solver.project(pv)-pv)
        orthogonality=abs(np.dot(solver.W*(v-pv),pv))
        result=solver.solve(tol=1e-10,maxiter=30000)
        err=np.linalg.norm(result['coef']-ref)
        assert err < 2e-5 and result['relative_gap'] < 1e-10
        assert idempotence<1e-11 and orthogonality<1e-10
        records.append(dict(test="squared_enet_sklearn",weights=weights,coefficient_error=err,
                            relative_gap=result['relative_gap'],projector_idempotence=idempotence,
                            projector_orthogonality=orthogonality))
    # Quantile fused lasso has an independent linear-programming formulation.
    rng=np.random.default_rng(7)
    n,p=30,9;X=rng.normal(size=(n,p));y=X@np.r_[np.ones(3),np.zeros(p-3)]+rng.normal(size=n)
    D=np.diff(np.eye(p),axis=0);tau=.4;lam=.1;tv=.08
    # x=(beta, abs(beta), positive_residual, negative_residual, abs(D beta)).
    size=2*p+2*n+(p-1)
    c=np.r_[np.zeros(p),lam*np.ones(p),tau/n*np.ones(n),(1-tau)/n*np.ones(n),tv*np.ones(p-1)]
    Aeq=np.zeros((n,size));Aeq[:,:p]=X;Aeq[:,2*p:2*p+n]=np.eye(n);Aeq[:,2*p+n:2*p+2*n]=-np.eye(n)
    Aub=np.zeros((2*p+2*(p-1),size))
    Aub[:p,:p]=np.eye(p);Aub[:p,p:2*p]=-np.eye(p)
    Aub[p:2*p,:p]=-np.eye(p);Aub[p:2*p,p:2*p]=-np.eye(p)
    Aub[2*p:2*p+p-1,:p]=D;Aub[2*p:2*p+p-1,2*p+2*n:]=-np.eye(p-1)
    Aub[2*p+p-1:,:p]=-D;Aub[2*p+p-1:,2*p+2*n:]=-np.eye(p-1)
    lp=linprog(c,A_ub=Aub,b_ub=np.zeros(Aub.shape[0]),A_eq=Aeq,b_eq=y,
               bounds=[(None,None)]*p+[(0,None)]*(size-p),method="highs")
    assert lp.success
    for weights in ("unit","budget"):
        problem=make_regression(X,y,2,3,l1=lam,ridge=0,loss="quantile",loss_parameter=tau,fusion=tv,fusion_parts=3)
        result=GraphSplitting(problem,weights=weights).solve(tol=1e-5,maxiter=40000)
        err=abs(result['objective']-lp.fun)
        assert err < 5e-5 and result['relative_gap']<1e-5
        records.append(dict(test="quantile_fused_highs_lp",weights=weights,objective_error=err,
                            relative_gap=result['relative_gap']))
    # Fused ridge/elastic net: independent smooth box-constrained dual solve.
    n,p=80,35;X=rng.normal(size=(n,p));y=X@np.r_[np.ones(10),np.zeros(p-10)]+rng.normal(size=n)
    D=np.diff(np.eye(p),axis=0);lam=.04;tv=.07;mu=.03
    def dual_fg(v):
        theta,h=v[:n],v[n:]
        b=soft(-X.T@theta-D.T@h,lam)/mu
        val=.5*n*np.dot(theta,theta)+np.dot(y,theta)+.5*mu*np.dot(b,b)
        grad=np.r_[n*theta+y-X@b,-D@b]
        return val,grad
    opt=minimize(dual_fg,np.zeros(n+p-1),method="L-BFGS-B",jac=True,
                 bounds=[(None,None)]*n+[(-tv,tv)]*(p-1),
                 options={"ftol":1e-14,"gtol":1e-10,"maxiter":30000,"maxls":50})
    for weights in ("unit","budget"):
        problem=make_regression(X,y,3,3,l1=lam,ridge=mu,fusion=tv,fusion_parts=3)
        result=GraphSplitting(problem,weights=weights).solve(tol=1e-10,maxiter=50000)
        err=result['objective']+opt.fun
        assert abs(err)<1e-7 and result['relative_gap']<1e-10
        records.append(dict(test="squared_fused_independent_dual",weights=weights,objective_minus_dual=err,
                            independent_optimizer_success=bool(opt.success),relative_gap=result['relative_gap']))
    return records


if __name__=="__main__":
    output=Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results")) / "validation.json"
    records=run_validation();output.write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))
