"""Paired oracle, rho and heterogeneity diagnostics. No archived fits overwritten.
Oracle reference cost is excluded: it is a diagnostic, never a feasible method.
Increment shares use only edge-state changes over the last 20 pilot updates.
"""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from hybrid_solver import GraphSplitting,make_regression
from run_experiments import synthetic_case,prepare_real_case,ar_design
from pilot_budget_experiment import water_fill,make_pilot_weights
from matched_output_control import retained_weights
from pilot_continuation import step
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/acceleration';OUT.mkdir(exist_ok=True)
RHOS=(1/256,1/64,1/16,1/4,1.,4.)
DESIGN=dict(pilot_steps=100,increment_lag=20,pilot_rho=1.,rho_grid=RHOS,floor_fraction=.05,check_every=10,iteration_cap=30000,
 original_oracle='All 49 original configurations at rho=1',original_rho_sweep='4x4: all 15 synthetic samples and both public datasets',
 heterogeneity='n240,p120;4x4;standardized AR(.4);block signal ratio 1,4,16;beta norm3;squared/Huber/logistic',
 train_seeds=[8100,8101,8102],test_seeds=[8200,8201,8202,8203,8204],arms=['uniform','centered','increment','oracle'],
 tuning='Select rho per arm/model/heterogeneity by training median total scalars, freeze for test; account training separately',
 accounting='Pilot+suffix updates+scheduled certificates+one refresh+3 scalars/edge for nonuniform shares',
 interpretation='Oracle reads reference; increment heuristic targets remaining errors without certified allocation regret.')
def write(path,rows):
 if rows:
  with Path(path).open('w',newline='') as f:w=csv.DictWriter(f,rows[0].keys());w.writeheader();w.writerows(rows)
def hetero(model,level,seed):
 rng=np.random.default_rng(seed+{'squared':0,'huber':10000,'logistic':20000}[model]);n,p=240,120
 X=ar_design(rng,n,p,.4);X=(X-X.mean(0))/X.std(0);b=np.zeros(p)
 for j,s in enumerate(np.geomspace(1,level,4)):b[j*30+np.arange(4)]=s*np.array([1,-1,1,-1])
 b*=3/np.linalg.norm(b)
 if model=='logistic':y=np.where(rng.random(n)<expit(X@b),1.,-1.)
 else:
  noise=rng.normal(size=n)
  if model=='huber':noise+=(rng.random(n)<.1)*rng.normal(scale=6,size=n)
  y=X@b+noise
 return dict(X=X,y=y,args=dict(loss=model,loss_parameter=1.345 if model=='huber' else 1,l1=.02,ridge=.02),case=f'hetero_{model}_{level}_{seed}')
def weights(s,values):
 omega=np.zeros(len(s.edges))
 for ix in s.edge_by_factor:omega[ix]=water_fill(values[ix],.05)
 return dict(a=np.ones_like(s.a),c=np.ones_like(s.c),b=1/omega,d=np.array([1/s.dj[j] for a,j,A in s.edges]))
def transfer(s,w,rho):
 z=GraphSplitting(s.problem,rho=rho,custom_weights=w);z.eta[:]=s.eta;z.u[:]=s.rho*s.W*s.u/(rho*z.W)
 assert max(z.invariant_error())<1e-8
 np.testing.assert_allclose(rho*z.W*z.u,s.rho*s.W*s.u,rtol=1e-13,atol=1e-13)
 return z
def execute(task):
 start=time.perf_counter();kind,model,seed,level,m,j,rhos=task
 if kind=='hetero':case=hetero(model,level,seed)
 elif model in ('diabetes','breast_cancer'):case=prepare_real_case(model,split_seed=seed,save_split=False)
 else:case=synthetic_case(model,seed)
 tol=1e-7 if model in ('diabetes','breast_cancer') else 1e-6
 problem=make_regression(case['X'],case['y'],m,j,**case['args']);s=GraphSplitting(problem,rho=1,weights='unit');setup=time.perf_counter()-start;trace=[];previous=None
 work=time.perf_counter()
 for k in range(1,101):
  step(s)
  if k==80:previous=s.eta.copy()
  if k==1 or k%10==0:
   p,d,g=s.certificate();trace.append((k,p,d,g,g/max(1,abs(p))))
   if trace[-1][-1]<=tol:break
 prefix_seconds=time.perf_counter()-work;needs_switch=trace[-1][-1]>tol
 if kind=='hetero':
  rs=GraphSplitting(make_regression(case['X'],case['y'],1,1,**case['args']),rho=.25);ref=rs.solve(tol=1e-11,maxiter=50000)
  assert ref['converged'];beta=ref['coef'];reference_gap=ref['relative_gap']
 else:
  with np.load(ROOT/'results/coefficients'/f"{case['case']}_reference.npz") as f:beta=f['coef'];reference_gap=float(f['gap'])
 blocks=np.split(beta,np.cumsum(problem.block_sizes)[:-1]);dev=np.array([np.linalg.norm(A@blocks[b]-s.eta[s.yes[e]]) for e,(_,b,A) in enumerate(s.edges)])
 uniform=weights(s,np.ones(len(s.edges)));oracle=weights(s,dev);arms={'uniform':uniform,'oracle':oracle};alloc_begin=time.perf_counter()
 if k==100 and trace[-1][-1]>tol:
  learned,diag=make_pilot_weights(problem,{'coef':np.concatenate([s.eta[ix] for ix in s.xs]),'gap':trace[-1][3]})
  arms['centered']=retained_weights(s,diag,learned);inc_begin=time.perf_counter();inc=np.array([np.linalg.norm(s.eta[ix]-previous[ix]) for ix in s.yes]);arms['increment']=weights(s,inc);inc_seconds=time.perf_counter()-inc_begin
 else:arms.update(centered=uniform,increment=uniform);inc_seconds=0
 allocation_seconds=time.perf_counter()-alloc_begin;c=s.communication_per_iteration;check=c+3*(len(problem.factors)+len(problem.block_sizes));rows=[]
 for rho in rhos:
  for arm in ('uniform','centered','increment','oracle'):
   begin=time.perf_counter();z=transfer(s,arms[arm],rho);switch_seconds=time.perf_counter()-begin;checks=len(trace);steps=k;done=trace[-1][-1]<=tol;p,d,g,rg=trace[-1][1:];begin=time.perf_counter()
   if not done:
    for it in range(1,30001):
     step(z);steps=k+it
     if steps%10==0:
      p,d,g=z.certificate();rg=g/max(1,abs(p));checks+=1
      if rg<=tol:done=True;break
   suffix_seconds=time.perf_counter()-begin;alloc=0 if arm=='uniform' or not needs_switch else 3*len(s.edges);refresh=c//2 if needs_switch else 0
   term=float(np.dot(dev**2,arms[arm]['b']));base=float(np.dot(dev**2,uniform['b']))
   rows.append(dict(study=kind,model=model,seed=seed,heterogeneity=level,M=m,J=j,rho=rho,arm=arm,converged=done,iterations=steps,pilot_iterations=k,relative_gap=rg,objective=p,dual=d,total_scalars=steps*c+checks*check+alloc+refresh,checks=checks,per_update=c,per_check=check,allocation_scalars=alloc,refresh_scalars=refresh,reference_term_ratio=term/base if base else 1,oracle_term_ratio=float(np.dot(dev**2,oracle['b']))/base if base else 1,setup_seconds=setup,switch_seconds=switch_seconds,prefix_seconds=prefix_seconds,suffix_seconds=suffix_seconds,allocation_seconds=(allocation_seconds-inc_seconds) if arm=='centered' else inc_seconds if arm=='increment' else 0,reference_gap=reference_gap,diagnostic_oracle=(arm=='oracle')))
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','original','sweep','hetero'],required=True);ap.add_argument('--jobs',type=int,default=3);a=ap.parse_args();(OUT/'design.json').write_text(json.dumps(DESIGN,indent=2))
 if a.mode=='smoke':tasks=[('original','huber',0,0,4,4,(.25,1.)),('hetero','huber',8100,16,4,4,(.25,1.))]
 elif a.mode=='original':tasks=[('original',v,s,0,m,j,(1.,)) for v in ('fused','huber','logistic') for s in range(5) for m,j in ((2,8),(4,4),(8,2))]+[('original',v,2026,0,m,j,(1.,)) for v in ('diabetes','breast_cancer') for m,j in ((4,4),(8,4))]
 elif a.mode=='sweep':tasks=[('sweep',v,s,0,4,4,RHOS) for v in ('fused','huber','logistic') for s in range(5)]+[('sweep',v,2026,0,4,4,RHOS) for v in ('diabetes','breast_cancer')]
 else:tasks=[('hetero',v,s,l,4,4,RHOS) for v in ('squared','huber','logistic') for l in (1,4,16) for s in (8100,8101,8102,8200,8201,8202,8203,8204)]
 rows=[];began=time.perf_counter()
 with ProcessPoolExecutor(a.jobs) as pool:
  fs={pool.submit(execute,t):t for t in tasks}
  for i,f in enumerate(as_completed(fs)):
   rows+=f.result();write(OUT/(a.mode+'.csv'),rows);print(a.mode,i+1,len(tasks),fs[f][:6],round(time.perf_counter()-began,1),flush=True)
 print('completed',len(rows),flush=True)
if __name__=='__main__':main()
