"""Independent surrogate minimization and transfer checks on actual pilot states."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
import numpy as np
from scipy.optimize import minimize
from acceleration_diagnostics import ROOT,hetero
from hybrid_solver import GraphSplitting,make_regression
from pilot_continuation import step
from joint_metric import select_metric,apply_metric
case=hetero('huber',4,99001);s=GraphSplitting(make_regression(case['X'],case['y'],4,4,**case['args']))
for k in range(100):
 step(s)
 if k==79:pe=s.eta.copy();pl=s.rho*s.W*s.u
w,r,diag=select_metric(s,pe,pl);z=apply_metric(s,w,r);de=s.eta-pe;dl=s.rho*s.W*s.u-pl
best=r*np.dot(z.W,de**2)+np.dot(1/z.W,dl**2)/r
rng=np.random.default_rng(7001);ratios=[]
for trial in range(50):
 wo=dict(w);b=np.zeros(len(s.edges));d=np.zeros(len(s.edges))
 for ix in s.edge_by_factor:b[ix]=1/(.05/len(ix)+.95*rng.dirichlet(np.ones(len(ix))))
 for ix in s.edge_by_feature:d[ix]=.05/len(ix)+.95*rng.dirichlet(np.ones(len(ix)))
 wo.update(b=b,d=d);rr=float(np.exp(rng.uniform(np.log(1/256),np.log(4))));zz=apply_metric(s,wo,rr)
 val=rr*np.dot(zz.W,de**2)+np.dot(1/zz.W,dl**2)/rr
 assert best<=val+1e-10;ratios.append(float(val/best))
errors=[]
for slices,groups,vals in [(s.yes,s.edge_by_factor,de),(s.xes,s.edge_by_feature,dl)]:
 for ix in groups:
  c=np.array([np.linalg.norm(vals[slices[e]]) for e in ix]);c=c/max(c.max(),1e-15)
  lo=.05/len(ix);res=minimize(lambda x:np.sum(c*c/x),np.ones(len(ix))/len(ix),method='SLSQP',bounds=[(lo,1)]*len(ix),constraints={'type':'eq','fun':lambda x:x.sum()-1},options={'ftol':1e-12,'maxiter':200})
  chosen=1/w['b'][ix] if slices is s.yes else w['d'][ix];err=abs(np.sum(c*c/chosen)-res.fun);assert res.success and err<1e-8;errors.append(float(err))
report=dict(random_candidates=50,min_random_to_selected=min(ratios),max_constrained_objective_error=max(errors),rho=r,primal_dual_invariants=z.invariant_error(),status='passed')
(ROOT/'results/acceleration/joint_validation.json').write_text(json.dumps(report,indent=2));print(report)
