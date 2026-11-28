"""Repeated paired public-data splits with a common 1e-7 gap target.

The penalty is validation-tuned separately on diabetes splits. For breast cancer,
we freeze lambda=0.001, the value selected on the manuscript's original split,
to isolate partition/metric robustness at a fixed penalty. Later test partitions
can overlap the original validation set; these repeats are not an independent
assessment of predictive generalization after tuning.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from run_experiments import prepare_real_case
from hybrid_solver import GraphSplitting,make_regression
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'/'repeated_real_splits.csv'
SEEDS=list(range(2026,2031)); FIELDS=['dataset','split_seed','n_train','n_test','p','loss','lambda','M','J','weight','iterations','relative_gap','objective','test_rmse','test_logloss','test_auc','test_accuracy','communicated_scalars','elapsed_seconds','objective_difference_1x1','coefficient_distance_1x1','converged','tuning_protocol']
def get_case(ds,seed):
    if ds=='diabetes':
        c=prepare_real_case(ds,split_seed=seed,save_split=False)
        return c['X'],c['y'],c['Xtest'],c['ytest'],c['args'],c.get('target_scale',1.),'split-specific validation tuning'
    if ds!='breast_cancer': raise ValueError(f'Unsupported dataset: {ds}')
    raw=load_breast_cancer(); idx=np.arange(len(raw.target)); yraw=2*raw.target-1
    tr,te=train_test_split(idx,test_size=.3,stratify=yraw,random_state=seed)
    sc=StandardScaler().fit(raw.data[tr]); X=sc.transform(raw.data[tr]); Xt=sc.transform(raw.data[te])
    args=dict(loss='logistic',loss_parameter=1,ridge=.01,fusion=0,l1=.001)
    return X,yraw[tr],Xt,yraw[te],args,1.,'lambda frozen at 0.001 selected on the original 2026 validation split'
def paired_values(rows,dataset,M,J,key):
    """Group by split seed and reject duplicate or missing paired arms."""
    pairs={}
    for row in rows:
        if row['dataset']!=dataset or int(row['M'])!=M or int(row['J'])!=J: continue
        seed=int(row['split_seed']); arm=row['weight']
        if arm in pairs.setdefault(seed,{}):
            raise ValueError(f'Duplicate arm: {(dataset,seed,M,J,arm)}')
        pairs[seed][arm]=float(row[key])
    for seed,arms in pairs.items():
        if set(arms)!={'unit','budget'}:
            raise ValueError(f'Incomplete pair: {(dataset,seed,M,J)}')
    return {seed:pairs[seed] for seed in sorted(pairs)}

def summarize(rows):
    rows=[dict(row,M=int(row['M']),J=int(row['J']),split_seed=int(row['split_seed'])) for row in rows]
    keys=[(r['dataset'],r['split_seed'],r['M'],r['J'],r['weight']) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError('Duplicate experimental configuration')
    if any(str(r['converged']).lower()!='true' or not np.isfinite(float(r['relative_gap']))
           or not 0<=float(r['relative_gap'])<=1e-7 for r in rows):
        raise ValueError('Invalid convergence flag or stopping gap')
    summary={}; paired={}
    for ds in sorted({r['dataset'] for r in rows}):
      for M,J,w in sorted({(r['M'],r['J'],r['weight']) for r in rows if r['dataset']==ds}):
        z=[r for r in rows if r['dataset']==ds and (r['M'],r['J'],r['weight'])==(M,J,w)]
        keys=['iterations','communicated_scalars','elapsed_seconds']+(['test_rmse'] if ds=='diabetes' else ['test_logloss','test_auc','test_accuracy'])
        summary[f'{ds}_{M}x{J}_{w}']={k:{'mean':float(np.mean([float(r[k]) for r in z])),'sd':float(np.std([float(r[k]) for r in z],ddof=1)) if len(z)>1 else None} for k in keys if all(r[k] not in ('',None) and np.isfinite(float(r[k])) for r in z)}
      traffic=paired_values(rows,ds,4,1,'communicated_scalars')
      iterations=paired_values(rows,ds,4,1,'iterations')
      savings=[100*(v['unit']-v['budget'])/v['unit'] for v in traffic.values()]
      it_savings=[100*(v['unit']-v['budget'])/v['unit'] for v in iterations.values()]
      paired[ds]={'paired_splits':len(savings),'split_seeds':list(traffic),'traffic_saving_percent_mean':float(np.mean(savings)),'traffic_saving_percent_sd':float(np.std(savings,ddof=1)) if len(savings)>1 else None,'traffic_saving_percent_by_split':savings,'iteration_saving_percent_mean':float(np.mean(it_savings)),'budget_wins':sum(v['budget']<v['unit'] for v in traffic.values())}
      if ds=='diabetes':
        v4=paired_values(rows,ds,4,4,'communicated_scalars')
        s4=[100*(v['unit']-v['budget'])/v['unit'] for v in v4.values()]
        paired[ds]['4x4_traffic_saving_percent_mean']=float(np.mean(s4)); paired[ds]['4x4_budget_wins']=sum(v['budget']<v['unit'] for v in v4.values())
    return {'seeds':sorted({r['split_seed'] for r in rows}),'rows':len(rows),'summary':summary,'paired_comparisons':paired,
            'missing_reference_coefficient_comparisons':sum(r.get('coefficient_distance_1x1') in ('',None) for r in rows),
            'scope':'Fixed-penalty optimizer comparison; breast-cancer repeats can overlap the original penalty-selection validation set.'}

def main():
    global OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=OUT.parent)
    ap.add_argument('--summarize-only',action='store_true')
    args_cli=ap.parse_args();args_cli.output_dir.mkdir(parents=True,exist_ok=True)
    OUT=args_cli.output_dir/'repeated_real_splits.csv'
    if args_cli.summarize_only:
      with OUT.open(newline='') as f: rows=list(csv.DictReader(f))
      (OUT.parent/'repeated_real_splits_summary.json').write_text(json.dumps(summarize(rows),indent=2,allow_nan=False)+'\n')
      print('SUMMARIZED',len(rows),'rows');return
    rows=[]; start=time.perf_counter()
    for ds in ('diabetes','breast_cancer'):
      for seed in SEEDS:
        X,y,Xt,yt,args,ys,tuning=get_case(ds,seed); n,p=X.shape
        configs=[(1,1,'unit'),(4,1,'unit'),(4,1,'budget')]
        if ds=='diabetes': configs += [(4,4,'unit'),(4,4,'budget')]
        base=None
        for M,J,w in configs:
          t=time.perf_counter(); solver=GraphSplitting(make_regression(X,y,M,J,**args),weights=w,rho=1)
          out=solver.solve(tol=1e-7,maxiter=30000); elapsed=time.perf_counter()-t
          if not out['converged']: raise RuntimeError(f'not converged: {ds} {seed} {M}x{J} {w}')
          if (M,J,w)==(1,1,'unit'): base=out
          pred=Xt@out['coef']
          if args['loss']=='logistic':
            yy=(yt+1)/2; pp=expit(pred); met=dict(test_rmse='',test_logloss=float(log_loss(yy,pp)),test_auc=float(roc_auc_score(yy,pp)),test_accuracy=float(np.mean((pred>=0)==(yt==1))))
          else: met=dict(test_rmse=float(np.sqrt(np.mean((pred-yt)**2))*ys),test_logloss='',test_auc='',test_accuracy='')
          rows.append(dict(dataset=ds,split_seed=seed,n_train=n,n_test=len(yt),p=p,loss=args['loss'],**{'lambda':args['l1']},M=M,J=J,weight=w,iterations=out['iterations'],relative_gap=out['relative_gap'],objective=out['objective'],**met,communicated_scalars=out['communicated_scalars'],elapsed_seconds=elapsed,objective_difference_1x1='' if base is None else out['objective']-base['objective'],coefficient_distance_1x1='' if base is None else float(np.linalg.norm(out['coef']-base['coef'])),converged=out['converged'],tuning_protocol=tuning))
          print(f'{ds} {seed} {M}x{J} {w}: iter={out["iterations"]} traffic={out["communicated_scalars"]} gap={out["relative_gap"]:.1e} sec={elapsed:.2f}',flush=True)
          if base is not None:
            for r in rows:
              if r['dataset']==ds and r['split_seed']==seed and r['objective_difference_1x1']=='':
                r['objective_difference_1x1']=r['objective']-base['objective']; r['coefficient_distance_1x1']=float(np.linalg.norm(out['coef']-base['coef']))
          with OUT.open('w',newline='') as f: wri=csv.DictWriter(f,fieldnames=FIELDS);wri.writeheader();wri.writerows(rows)
    summary=summarize(rows);summary['elapsed_seconds']=time.perf_counter()-start
    (OUT.parent/'repeated_real_splits_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',len(rows),'rows',time.perf_counter()-start)
if __name__=='__main__':main()
