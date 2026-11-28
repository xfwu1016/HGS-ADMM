"""Fixed one-switch rule and controls; held-out cases are never used for tuning."""
import os
for v in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[v]='1'
import argparse,json,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from acceleration_diagnostics import ROOT,OUT,write,hetero
from hybrid_solver import GraphSplitting,make_regression
from run_experiments import synthetic_case,prepare_real_case,ar_design
from pilot_continuation import step
from joint_metric import select_metric,apply_metric
DESIGN=dict(pilot=100,lag=20,floor=.05,rho_bounds=[1/256,4],target=1e-6,
 rule='Output shares proportional to eta-edge increments; copy shares proportional to unscaled-dual edge increments; rho=sqrt(D_increment/P_increment)',
 arms=['uniform_fixed','uniform_auto','output_auto','joint_fixed','joint_auto','uniform_joint_rho'],
 signal_levels=[1,4,16],row_scale_levels=[1,4,16],train=[8100,8101,8102],test=[8200,8201,8202,8203,8204],
 common_design='n240,p120;4x4;AR(.4), column-standardized; beta norm3; l1=.02 ridge=.02',
 row_scale_design='Multiply consecutive sample quarters by geometrically spaced 1..level before global column standardization; four coefficients per feature block, equal amplitudes',
 controls='Every arm shares the same unit pilot, state transfer, target and checkpoint schedule. Uniform_auto uses the identical automatic rho formula in the uniform metric.',
 choice='All settings fixed before this joint-rule experiment; all heterogeneity levels reported. Train cases are diagnostic; no parameter fitted to held-out outcomes.',
 accounting='Full pilot+suffix+certificates+refresh; Collect one norm and return one weight for each allocation: 4E for joint; output=2E; automatic rho=2 local sums per role to all coordinators, bounded by 2(E+A+J)(A+J)+E, including the scalar rho broadcast to edge workers; allocation selection messages counted once',
 timing='Sequential simulated execution, repeated timing not claimed')
def makecase(model,level,seed,regime):
 if regime=='signal':return hetero(model,level,seed)
 rng=np.random.default_rng(seed+{'squared':30000,'huber':40000,'logistic':50000}[model]);n,p=240,120
 X=ar_design(rng,n,p,.4)*np.repeat(np.geomspace(1,level,4),60)[:,None];X=(X-X.mean(0))/X.std(0)
 beta=np.zeros(p)
 for j in range(4):beta[30*j+np.arange(4)]=np.array([1,-1,1,-1])
 beta*=3/np.linalg.norm(beta)
 if model=='logistic':y=np.where(rng.random(n)<expit(X@beta),1.,-1.)
 else:
  noise=rng.normal(size=n)
  if model=='huber':noise+=(rng.random(n)<.1)*rng.normal(scale=6,size=n)
  y=X@beta+noise
 return dict(X=X,y=y,args=dict(loss=model,loss_parameter=1.345 if model=='huber' else 1,l1=.02,ridge=.02))
def evaluate(task):
 regime,model,seed,level,m,j=task;began=time.perf_counter()
 if regime=='original':case=prepare_real_case(model,split_seed=seed,save_split=False) if model in ('diabetes','breast_cancer') else synthetic_case(model,seed)
 else:case=makecase(model,level,seed,regime)
 tol=1e-7 if model in ('diabetes','breast_cancer') else 1e-6
 s=GraphSplitting(make_regression(case['X'],case['y'],m,j,**case['args']),rho=1,weights='unit');setup=time.perf_counter()-began;checks=0;start=time.perf_counter()
 for k in range(1,101):
  step(s)
  if k==80:pe=s.eta.copy();pl=s.rho*s.W*s.u
  if k==1 or k%10==0:
   p,d,g=s.certificate();rg=g/max(1,abs(p));checks+=1
   if rg<=tol:break
 prefix=time.perf_counter()-start;switch=rg>tol
 specs={};selection={}
 if switch:
  for name,kw in [('uniform',dict(uniform=True)),('output',dict(joint=False)),('joint',dict(joint=True))]:
   b=time.perf_counter();w,r,diag=select_metric(s,pe,pl,**kw);selection[name]=time.perf_counter()-b;specs[name]=(w,r,diag)
 E=len(s.edges);A=len(s.ys);J=len(s.xs);ci=s.communication_per_iteration;cc=ci+3*(A+J);rows=[]
 # A conservative ledger broadcasts local sums to every coordinator; no free global centre.
 reduce_cost=2*(E+A+J)*(A+J)+E
 for arm in DESIGN['arms']:
  name,mode=('uniform','joint_rho') if arm=='uniform_joint_rho' else arm.split('_');begin=time.perf_counter()
  if switch:
   w,r,diag=specs[name];rho=specs['joint'][1] if mode=='joint_rho' else 1. if mode=='fixed' else r;z=apply_metric(s,w,rho)
   weight_cost=0 if name=='uniform' else 2*E if name=='output' else 4*E
   allocation=(4*E+reduce_cost) if mode=='joint_rho' else weight_cost+(reduce_cost if mode=='auto' else 0);refresh=ci//2;selection_time=selection['joint'] if mode=='joint_rho' else selection[name]
  else:z=s;rho=1.;allocation=refresh=0;selection_time=0
  refactor=time.perf_counter()-begin;steps=k;nchecks=checks;rel=rg;objective=p;dual=d;done=not switch;start=time.perf_counter()
  if switch:
   for it in range(1,30001):
    step(z);steps=k+it
    if steps%10==0:
     objective,dual,gap=z.certificate();rel=gap/max(1,abs(objective));nchecks+=1
     if rel<=tol:done=True;break
  elapsed=time.perf_counter()-start
  rows.append(dict(regime=regime,model=model,seed=seed,heterogeneity=level,M=m,J=j,split='test' if seed>=8200 and regime!='original' else 'diagnostic',arm=arm,rho=rho,converged=done,iterations=steps,pilot_iterations=k,relative_gap=rel,objective=objective,dual=dual,total_scalars=steps*ci+nchecks*cc+allocation+refresh,edges=E,per_update=ci,per_check=cc,checks=nchecks,selection_scalars=allocation,refresh_scalars=refresh,setup_seconds=setup,prefix_seconds=prefix,selection_seconds=selection_time,refactor_seconds=refactor,suffix_seconds=elapsed,total_seconds=setup+prefix+selection_time+refactor+elapsed))
 return rows
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['original','hetero','smoke'],required=True);ap.add_argument('--jobs',type=int,default=3);a=ap.parse_args();(OUT/'joint_design.json').write_text(json.dumps(DESIGN,indent=2))
 if a.mode=='original':tasks=[('original',v,s,0,m,j) for v in ('fused','huber','logistic') for s in range(5) for m,j in ((2,8),(4,4),(8,2))]+[('original',v,2026,0,m,j) for v in ('diabetes','breast_cancer') for m,j in ((4,4),(8,4))]
 elif a.mode=='smoke':tasks=[('row_scale','huber',8100,16,4,4),('row_scale','logistic',8100,16,4,4),('original','huber',0,0,4,4)]
 else:tasks=[(reg,v,s,l,4,4) for reg in ('signal','row_scale') for v in ('squared','huber','logistic') for l in (1,4,16) for s in (8100,8101,8102,8200,8201,8202,8203,8204)]
 rows=[];began=time.perf_counter()
 with ProcessPoolExecutor(a.jobs) as pool:
  fs={pool.submit(evaluate,t):t for t in tasks}
  for i,f in enumerate(as_completed(fs)):
   rows+=f.result();write(OUT/('joint_'+a.mode+'.csv'),rows);print(a.mode,i+1,len(tasks),fs[f],round(time.perf_counter()-began,1),flush=True)
 print('completed',len(rows),flush=True)
if __name__=='__main__':main()
