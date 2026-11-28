"""Summarize complete common-grid diagnostics, retaining capped candidates."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/acceleration';T=ROOT/'tables'
def main():
 d=pd.read_csv(D/'sweep.csv');assert len(d)==17*6*4,'Common-grid diagnostic is incomplete'
 rows=[];summary=[]
 for model,g in d.groupby('model',sort=False):
  u=g[g.arm=='uniform'];best=u[u.converged].sort_values(['total_scalars','rho']).groupby('seed').first()
  vals=best.rho.value_counts().sort_index();choices=', '.join(f'{r:g} ({n})' for r,n in vals.items())
  count=int((~g.converged).sum());summary.append(dict(model=model,cells=len(best),uniform_best_rho_counts={str(k):int(v) for k,v in vals.items()},capped_candidates=count,total_candidates=len(g)))
  name={'fused':'Squared fused','huber':'Huber','logistic':'Logistic','diabetes':'Diabetes','breast_cancer':'Breast cancer'}[model]
  rows.append(f'{name} & {len(best)} & {choices} & {count}/{len(g)} \\\\')
 text=r'''\section*{Extended penalty and oracle diagnostics}
The common-grid diagnostic uses the $4\times4$ layout, all five original samples for each simulated model, and the original split of each public dataset. All four output-allocation rules (uniform, certified centered, increment-based and reference-oracle) share the same 100-step pilot, state transfer, target and candidate penalties $\{1/256,1/64,1/16,1/4,1,4\}$. Each suffix is capped at 30,000 updates. A capped run is not treated as a successful cheaper fit. The complete candidate-level CSV retains its final gap and cap status. The oracle excludes reference-fit cost and is diagnostic only.
\begin{table}[H]\centering\small
\caption{Common-grid penalty diagnostic. The best penalty is chosen retrospectively within each uniform-weight case, not prospectively on test data. Parentheses count cases selecting each value.}
\label{tab:extended-rho}
\begin{tabular}{lrlr}\toprule
Model & Cases & Best uniform $\rho$ (count) & Capped/all candidates\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table}
All five fused cases select the upper grid endpoint for uniform shares, so even this expanded grid does not bracket their optimal penalty. This sensitivity analysis supplies equal candidate grids, not a fully optimized method ranking. It is separate from the automatic increment rule, which computes its penalty from a single pilot without trying this grid. The original fixed-penalty output-only oracle comparison uses all 49 configurations and saves charged traffic in only six, despite excluding the cost of obtaining its reference coefficients. Neither an oracle allocation nor a smaller state-distance term guarantees earlier certification.
'''
 (T/'supp_acceleration_diagnostics.tex').write_text(text);(D/'rho_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
