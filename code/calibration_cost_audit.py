"""Deterministic communication ledger from archived replicate-fit checkpoints."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,json
from collections import defaultdict
import numpy as np
from hybrid_solver import GraphSplitting,make_regression
from run_experiments import synthetic_case
R=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((R/'results/replicate_calibration_runs.csv').open()));out=[]
for seed in range(5):
 c=synthetic_case('fused',seed);pilot=GraphSplitting(make_regression(c['X'],c['y'],4,4,**c['args']))
 ci=pilot.communication_per_iteration;cc=ci+3*(len(pilot.ys)+len(pilot.xs))
 # Reuse partial predictions from the two terminal certificates; four scalar
 # squared norms and one tolerance broadcast to every coordinator.
 comparison=4+len(pilot.ys)+len(pilot.xs);cost=2*(200*ci+21*cc)+comparison
 selected=[x for x in rows if int(x['seed'])==seed]
 for x in selected:
  m,j=int(x['row_parts']),int(x['column_parts']);s=GraphSplitting(make_regression(c['X'],c['y'],m,j,**c['args']))
  u=s.communication_per_iteration;check=u+3*(len(s.ys)+len(s.xs));a,b=int(x['calibrated_step']),int(x['strict_step'])
  ac=a*u+(1+a//10)*check;bc=b*u+(1+b//10)*check
  out.append(dict(seed=seed,M=m,J=j,metric=x['metric'],calibration_scalars=cost,comparison_scalars=comparison,calibrated_scalars=ac,strict_scalars=bc,single_fit_cost_ratio=(cost+ac)/bc))
summary=[]
for seed in range(5):
 xs=[x for x in out if x['seed']==seed];cost=xs[0]['calibration_scalars']
 summary.append(dict(seed=seed,calibration_scalars=cost,single_fit_median_ratio=float(np.median([x['single_fit_cost_ratio'] for x in xs])),six_fit_total_ratio=(cost+sum(x['calibrated_scalars'] for x in xs))/sum(x['strict_scalars'] for x in xs)))
path=R/'results/adaptive_comparison';path.mkdir(exist_ok=True)
with (path/'calibration_cost.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(path/'calibration_cost_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
