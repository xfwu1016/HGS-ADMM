"""Independent finite-dimensional checks of the pilot-budget inequalities.

Small simplex optimizers are obtained by enumerating active floor constraints,
independently of the water-filling implementation used by the experiment.
The test model is P(beta)=mu ||beta-beta_star||^2/2, with arbitrary zero-valued
convex factors on the supplied maps. Its exact dual solution is zero, so the
computed pilot objective is an actual Fenchel gap, not a guessed radius.
"""
from pathlib import Path
import os
from itertools import combinations
import json
import numpy as np


def exact_by_active_sets(c,nu):
    c=np.asarray(c,float);d=len(c);floor=nu/d
    if np.max(c)==0:return np.full(d,1/d),0.0
    best=None
    for count in range(d):
        for chosen in combinations(range(d),count):
            fixed=set(chosen);free=[i for i in range(d) if i not in fixed]
            total=c[free].sum()
            if total==0:continue
            w=np.full(d,floor)
            w[free]=(1-count*floor)*c[free]/total
            if np.any(w[free]<floor-1e-13):continue
            val=float(np.sum(c*c/w))
            if best is None or val<best[1]:best=(w,val)
    assert best is not None
    return best


def proposed_by_bisection(c,nu):
    c=np.asarray(c,float);d=len(c)
    if np.max(c)==0:return np.full(d,1/d)
    scaled=c/np.max(c);lo=np.finfo(float).tiny;hi=d/(1-nu)
    for _ in range(150):
        mid=(lo+hi)/2
        if np.maximum(nu/d,scaled/mid).sum()>1:lo=mid
        else:hi=mid
    return np.maximum(nu/d,scaled/hi)


def main():
    rng=np.random.default_rng(20261122)
    max_weight_error=0.0;max_regret_excess=0.0
    min_slack=float('inf');allocations=0;graph_tests=0
    for d in range(1,8):
        for nu in (.01,.05,.5,.95):
            samples=[np.zeros(d),np.ones(d),np.arange(d,dtype=float)]
            samples += [rng.lognormal(0,4,d)*(rng.random(d)>.3) for _ in range(20)]
            for c in samples:
                ref,val=exact_by_active_sets(c,nu)
                actual=proposed_by_bisection(c,nu)
                max_weight_error=max(max_weight_error,float(np.max(np.abs(ref-actual))))
                assert np.all(actual>=nu/d-1e-13)
                assert abs(actual.sum()-1)<1e-12
                assert np.isclose(np.sum(c*c/actual),val,rtol=1e-10,atol=1e-10)
                allocations+=1
    for trial in range(250):
        dims=rng.integers(1,6,4);p=int(sum(dims));cuts=np.r_[0,np.cumsum(dims)]
        beta_star=rng.normal(size=p);mu=float(rng.uniform(.01,2))
        # Include exact pilot points and near-zero radius cases.
        scale=[0.0,1e-7,.01,.2,2.0][trial%5]
        beta_tilde=beta_star+scale*rng.normal(size=p)
        G=mu*np.sum((beta_tilde-beta_star)**2)/2
        R=float(np.sqrt(2*G/mu));nu=.05
        P_anchor=2*np.dot(beta_star,beta_star)
        B=2*(np.linalg.norm(beta_tilde)+R)**2
        selected=oracle=upper=lower=upper_uniform=0.0
        for a in range(3):
            n=int(rng.integers(1,7))
            maps=[rng.normal(size=(n,int(d))) for d in dims]
            true=[];hat=[];alpha=[]
            h_star=np.zeros(n);h_tilde=np.zeros(n)
            for j,A in enumerate(maps):
                bstar=beta_star[cuts[j]:cuts[j+1]]
                bt=beta_tilde[cuts[j]:cuts[j+1]]
                qs=A@bstar;qt=A@bt
                true.append(np.linalg.norm(qs));hat.append(np.linalg.norm(qt))
                alpha.append(np.linalg.norm(A,2));h_star+=qs;h_tilde+=qt
            true=np.asarray(true);hat=np.asarray(hat);alpha=np.asarray(alpha)
            cplus=hat+alpha*R;cminus=np.maximum(0,hat-alpha*R)
            assert np.all(cminus<=true+1e-12)
            assert np.all(true<=cplus+1e-12)
            wp,vp=exact_by_active_sets(cplus,nu)
            wm,vm=exact_by_active_sets(cminus,nu)
            wo,vo=exact_by_active_sets(true,nu)
            selected+=float(np.sum(true**2/wp));oracle+=vo;upper+=vp;lower+=vm
            upper_uniform+=len(true)*float(np.sum(cplus**2))
            P_anchor+=float(h_star@h_star)
            B+=(np.linalg.norm(h_tilde)+np.linalg.norm(alpha)*R)**2
        regret=selected-oracle;bound=upper-lower
        scale_tol=1e-10*max(1,selected,oracle,upper,lower)
        assert regret>=-scale_tol
        assert regret<=bound+scale_tol
        assert upper<=upper_uniform+scale_tol
        assert P_anchor+selected<=B+upper+scale_tol
        max_regret_excess=max(max_regret_excess,regret-bound)
        min_slack=min(min_slack,bound-regret)
        if R==0:
            assert abs(regret)<=scale_tol
            assert abs(bound)<=scale_tol
        graph_tests+=1
    report={'status':'passed','allocation_cases':allocations,
        'graph_certificate_cases':graph_tests,
        'exact_radius_zero_cases':graph_tests//5,
        'optimizer_reference':'exhaustive active-floor-set enumeration',
        'formula_solver':'normalizing scalar bisection',
        'maximum_weight_difference':max_weight_error,
        'maximum_regret_bound_excess':max_regret_excess,
        'minimum_regret_bound_slack':min_slack,
        'claims_checked':['simplex positivity and normalization',
            'water-filling optimality including mixed and all zeros',
            'certificate norm envelopes','selected upper bound <= uniform upper bound',
            'true P regret <= computable Vplus-Vminus',
            'full computable P upper bound','zero-radius floored oracle identity']}
    output=Path(os.environ.get("CSDA_CHECK_OUTPUT", Path(__file__).resolve().parents[1] / "results")) / "pilot_budget_theory_checks.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
