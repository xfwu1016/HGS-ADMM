"""Prespecified SRA and block-norm controls plus new-draw sensitivity checks.
SRA: consecutive differences, update period 5, zero-norm multiplier10;
Lozenski et al. (2026), III.C. SRA uses the same [1/256,4] safeguards as the joint rule; no test tuning.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.special import expit
from scipy.linalg import norm as stable_norm
from hybrid_solver import GraphSplitting,make_regression
from pilot_continuation import step
from joint_metric import select_metric,apply_metric
from joint_acceleration import makecase
from pilot_budget_experiment import water_fill
from run_experiments import synthetic_case,prepare_real_case,ar_design
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/adaptive_comparison'
ARMS=['uniform_auto','joint_auto','norm_auto','uniform_sra','joint_sra','sra_zero','rb_zero']
SETTINGS=[(100,20),(50,20),(200,20),(100,10),(100,40)]
DESIGN=dict(arms=ARMS,settings=SETTINGS,cap=30000,checkpoint=10,sra_period=5,sra_zero_multiplier=10,sra_bounds=[1/256,4],rb_mu=10,rb_tau=2,
 scalar_accounting='2(E+F+J)(F+J)+E per scalar selection; 4E per two-star allocation; C_iter/2 per retained-state metric refresh',
 core='49 original configurations and all90 previously held-out heterogeneity cells; retrospective extension, not new holdout claim',
 robustness='New seeds9300..9309, n240 p480, AR(.4), four fixed data-scale strata levels1,16, layouts4x4/8x2, three losses. Seven arms at100/20; uniform_auto/joint_auto at four further settings at level16.',
 norm_rule='Frobenius block norms and floored magnitude shares in both stars, same one-switch increment penalty; not iterative Ruiz equilibration.',
 freeze='Written before nonsmoke outcomes; outcomes do not select rules, floor, update period or settings.',
 limits='SRA and RB are continuously adaptive; fixed-suffix convergence theorem not asserted for either. Sequential timings, not network speedup.')
def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def high_case(model,seed,level):
 rng=np.random.default_rng(seed+{'squared':60000,'huber':70000,'logistic':80000}[model]);n,p=240,480
 X=ar_design(rng,n,p,.4)*np.repeat(np.geomspace(1,level,4),n//4)[:,None];X=(X-X.mean(0))/X.std(0)
 beta=np.zeros(p)
 for j in range(4):beta[p//4*j+np.arange(4)]=[1,-1,1,-1]
 beta*=3/np.linalg.norm(beta)
 if model=='logistic':y=np.where(rng.random(n)<expit(X@beta),1.,-1.)
 else:
  noise=rng.normal(size=n)
  if model=='huber':noise+=(rng.random(n)<.1)*rng.normal(scale=6,size=n)
  y=X@beta+noise
 return dict(X=X,y=y,args=dict(loss=model,loss_parameter=1.345 if model=='huber' else 1,l1=.02,ridge=.02))
def block_norm_metric(s,pe,pl):
 norms=np.array([np.linalg.norm(A,'fro') for a,j,A in s.edges]);omega=np.zeros(len(norms));pi=omega.copy()
 for ix in s.edge_by_factor:omega[ix]=water_fill(norms[ix],.05)
 for ix in s.edge_by_feature:pi[ix]=water_fill(norms[ix],.05)
 w=dict(a=np.ones_like(s.a),c=np.ones_like(s.c),b=1/omega,d=pi);W=np.ones_like(s.W)
 for e in range(len(norms)):W[s.yes[e]]=w['b'][e];W[s.xes[e]]=w['d'][e]
 de=s.eta-pe;dl=s.rho*s.W*s.u-pl;P=np.dot(W,de*de);D=np.dot(1/W,dl*dl)
 return w,float(np.clip(np.sqrt(D/P),1/256,4)) if P>0 and D>0 else s.rho
def sra_value(s,pe,pl):
 de=s.eta-pe;dl=s.rho*s.W*s.u-pl;den=stable_norm(np.sqrt(s.W)*de);num=stable_norm(dl/np.sqrt(s.W))
 if num>0 and den>0:r=num/den
 elif num==0 and den>0:r=s.rho/10
 elif num>0 and den==0:r=s.rho*10
 else:r=s.rho
 if not np.isfinite(r) or r<=0:raise FloatingPointError('Invalid SRA penalty')
 return float(r)
def rb_value(s,pe,pl):
 de=s.eta-pe;dl=s.rho*s.W*s.u-pl
 primal=stable_norm(dl/np.sqrt(s.W))/s.rho
 dual=s.rho*stable_norm(np.sqrt(s.W)*de)
 if primal>10*dual:return 2*s.rho
 if dual>10*primal:return s.rho/2
 return s.rho
def change_rho(s,rho):s.u*=s.rho/rho;s.rho=float(rho)
def evaluate(task):
 regime,model,seed,level,m,j=task
 if regime=='original':case=prepare_real_case(model,split_seed=seed,save_split=False) if model in ('diabetes','breast_cancer') else synthetic_case(model,seed)
 elif regime=='highdim':case=high_case(model,seed,level)
 else:case=makecase(model,level,seed,regime)
 problem=make_regression(case['X'],case['y'],m,j,**case['args']);tol=1e-7 if model in ('diabetes','breast_cancer') else 1e-6;rows=[]
 for pilot,lag in SETTINGS if regime=='highdim' and level==16 else SETTINGS[:1]:
  start=time.perf_counter();s=GraphSplitting(problem,rho=1,weights='unit');prefix_checks=0;pe=pl=None
  for k in range(1,pilot+1):
   step(s)
   if k==pilot-lag:pe=s.eta.copy();pl=(s.rho*s.W*s.u).copy()
   if k==1 or k%10==0:
    primal,dual,gap=s.certificate();rel=gap/max(1,abs(primal));prefix_checks+=1
    if rel<=tol:break
  prefix_seconds=time.perf_counter()-start;switch=rel>tol;selection_times={};E=len(s.edges);F=len(s.ys);J=len(s.xs);ci=s.communication_per_iteration;cc=ci+3*(F+J);reduce=2*(E+F+J)*(F+J)+E;specs={}
  if switch:
   for name in ['uniform','joint']:
    selection_start=time.perf_counter();w,r,d=select_metric(s,pe,pl,uniform=name=='uniform');specs[name]=(w,r);selection_times[name]=time.perf_counter()-selection_start
   selection_start=time.perf_counter();specs['norm']=block_norm_metric(s,pe,pl);selection_times['norm']=time.perf_counter()-selection_start
  for arm in ARMS if (pilot,lag)==SETTINGS[0] else ARMS[:2]:
   start=time.perf_counter();adaptations=selection=refresh=clippings=0;selection_seconds=0.;failure=""
   if arm in ('sra_zero','rb_zero'):
    z=GraphSplitting(problem,rho=1,weights='budget');count=checks=0;done=False;objective=dual_value=relative=float('nan');used_prefix=0;initial_rho=1.
   else:
    count=k;checks=prefix_checks;done=not switch;objective=primal;dual_value=dual;relative=rel;used_prefix=k
    if switch:
     name=arm.split('_')[0];selection_seconds=selection_times[name];w,r=specs[name];z=apply_metric(s,w,r);initial_rho=r;selection=reduce+(0 if name=='uniform' else 4*E);refresh=ci//2
    else:z=s;initial_rho=1.
   adaptive=arm.endswith('sra') or arm in ('sra_zero','rb_zero');rlo=rhi=z.rho
   if not done:
    try:
     with np.errstate(over='raise',invalid='raise',divide='raise'):
      for it in range(count+1,30001):
       if adaptive and it%5==0:previous_eta=z.eta.copy();previous_lambda=(z.rho*z.W*z.u).copy()
       count=it;step(z)
       if not np.all(np.isfinite(z.eta)) or not np.all(np.isfinite(z.u)):raise FloatingPointError('Nonfinite state')
       if it==1 or it%10==0:
        checks+=1;objective,dual_value,g=z.certificate();relative=g/max(1,abs(objective))
        if not np.isfinite(relative):raise FloatingPointError('Nonfinite certificate')
        if relative<=tol:done=True;break
       if adaptive and it%5==0:
        adaptations+=1;selection+=reduce
        raw=rb_value(z,previous_eta,previous_lambda) if arm=="rb_zero" else sra_value(z,previous_eta,previous_lambda);new=float(np.clip(raw,1/256,4));clippings+=new!=raw;change_rho(z,new);rlo=min(rlo,new);rhi=max(rhi,new)
    except (ArithmeticError,ValueError) as error:
     failure=type(error).__name__+': '+str(error)
   rows.append(dict(regime=regime,model=model,seed=seed,heterogeneity=level,M=m,J=j,n=case['X'].shape[0],p=case['X'].shape[1],pilot=pilot,lag=lag,arm=arm,converged=done,status="converged" if done else "numerical_failure" if failure else "iteration_cap",failure=failure,iterations=count,actual_pilot=used_prefix,relative_gap=relative,objective=objective,dual=dual_value,initial_rho=initial_rho,final_rho=z.rho,min_rho=rlo,max_rho=rhi,adaptations=adaptations,clippings=clippings,selection_seconds=selection_seconds,edges=E,factors=F,features=J,per_update=ci,per_check=cc,checks=checks,selection_scalars=selection,refresh_scalars=refresh,total_scalars=count*ci+checks*cc+selection+refresh,seconds=time.perf_counter()-start+selection_seconds+(prefix_seconds if used_prefix else 0)))
 return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','core','robust'],required=True);ap.add_argument('--jobs',type=int,default=3);ap.add_argument('--resume',action='store_true');a=ap.parse_args();OUT.mkdir(exist_ok=True)
 if not (OUT/'design.json').exists():(OUT/'design.json').write_text(json.dumps(DESIGN,indent=2))
 if a.mode=='smoke':tasks=[('row_scale','huber',8100,16,4,4),('original','fused',0,0,8,2),('highdim','logistic',9200,16,4,4)]
 elif a.mode=='core':tasks=[('original',v,s,0,m,j) for v in ('fused','huber','logistic') for s in range(5) for m,j in [(2,8),(4,4),(8,2)]]+[('original',v,2026,0,m,j) for v in ['diabetes','breast_cancer'] for m,j in [(4,4),(8,4)]]+[(reg,v,s,l,4,4) for reg in ['signal','row_scale'] for v in ['squared','huber','logistic'] for l in [1,4,16] for s in range(8200,8205)]
 else:tasks=[('highdim',v,s,l,m,j) for v in ['squared','huber','logistic'] for s in range(9300,9310) for l in [1,16] for m,j in [(4,4),(8,2)]]
 rows=[]
 if a.resume and (OUT/(a.mode+'.csv')).exists():
  rows=list(csv.DictReader((OUT/(a.mode+'.csv')).open()))
  keys={}
  for row in rows:
   key=(row['regime'],row['model'],int(row['seed']),int(row['heterogeneity']),int(row['M']),int(row['J']))
   keys[key]=keys.get(key,0)+1
  for key,n in keys.items():assert n==(15 if key[0]=='highdim' and key[3]==16 else 7),(key,n)
  tasks=[task for task in tasks if task not in keys]
  print('resuming',len(rows),'rows;',len(tasks),'remaining cases',flush=True)
 start=time.perf_counter()
 with ProcessPoolExecutor(a.jobs) as pool:
  fs={pool.submit(evaluate,t):t for t in tasks}
  for k,f in enumerate(as_completed(fs)):
   rows+=f.result();write(OUT/(a.mode+'.csv'),rows);print(a.mode,k+1,len(tasks),fs[f],round(time.perf_counter()-start,1),flush=True)
 print('rows',len(rows),'capped',sum(x['status']!='converged' for x in rows),flush=True)
if __name__=='__main__':main()
