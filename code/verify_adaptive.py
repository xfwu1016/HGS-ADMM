"""Independent identities and old/new paired-record reconciliation, no new full fits."""
import csv,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from adaptive_comparators import sra_value,rb_value,change_rho
ROOT=Path(__file__).resolve().parents[1]
def main():
 W=np.array([.5,2.,3.,.25]);rho=.7;eta=np.array([.1,2.,-1.,4.]);u=np.array([1.,-.5,2.,-3.]);pe=np.array([0.,1.,2.,3.]);pl=np.array([.2,.1,-.1,.3])
 z=SimpleNamespace(W=W,rho=rho,eta=eta.copy(),u=u.copy());de=np.sqrt(W)*(eta-pe);dl=(rho*W*u-pl)/np.sqrt(W)
 assert np.isclose(sra_value(z,pe,pl),np.linalg.norm(dl)/np.linalg.norm(de))
 lam=rho*W*u;change_rho(z,3.);assert np.allclose(z.rho*z.W*z.u,lam);assert np.array_equal(z.eta,eta)
 assert sra_value(z,z.eta,lam)==3.
 assert sra_value(z,pe,lam)==.3
 assert sra_value(z,z.eta,pl)==30.
 z=SimpleNamespace(W=np.ones(2),rho=1.,eta=np.array([1.,0.]),u=np.array([20.,0.]));assert rb_value(z,np.zeros(2),np.zeros(2))==2.
 z.u=np.array([.01,0.]);assert rb_value(z,np.zeros(2),np.zeros(2))==.5
 z.u=np.array([1.,0.]);assert rb_value(z,np.zeros(2),np.zeros(2))==1.
 old=[]
 for n in ['joint_original.csv','joint_hetero.csv']:old+=list(csv.DictReader((ROOT/'results/acceleration'/n).open()))
 keys=['regime','model','seed','heterogeneity','M','J','arm'];index={tuple(x[k] for k in keys):x for x in old};n=0;max_gap=0.
 for r in csv.DictReader((ROOT/'results/adaptive_comparison/core.csv').open()):
  if r['arm'] not in ['uniform_auto','joint_auto']:continue
  o=index[tuple(r[k] for k in keys)]
  for k in ['iterations','checks','total_scalars','per_update','per_check','selection_scalars','refresh_scalars']:assert int(r[k])==int(o[k]),(keys,r,k,r[k],o[k])
  assert np.isclose(float(r['objective']),float(o['objective']),rtol=1e-11,atol=1e-12)
  max_gap=max(max_gap,abs(float(r['relative_gap'])-float(o['relative_gap'])));n+=1
 assert n==278,n
 result=dict(status='passed',paired_old_new_records=n,max_absolute_gap_difference=max_gap,checks=['transformed-coordinate SRA ratio','unscaled multiplier preservation','zero-norm fallbacks','residual-balance branches','old/new endpoint and full ledger identity'])
 (ROOT/'results/adaptive_comparison/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
