"""Rebuild failure-aware comparator and sensitivity tables; no model fitting."""
import csv,json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/adaptive_comparison';TABLES=ROOT/'tables'
KEY=['regime','model','seed','heterogeneity','M','J','pilot','lag']
ARM_LABEL={'uniform_auto':'Uniform one-switch','joint_auto':'Joint one-switch','norm_auto':'Block-norm one-switch','uniform_sra':'Uniform + SRA','joint_sra':'Joint + SRA','sra_zero':'SRA from zero','rb_zero':'RB from zero'}
def read(name,n):
 rows=list(csv.DictReader((OUT/(name+'.csv')).open()));assert len(rows)==n,(name,len(rows))
 assert len({tuple(r[k] for k in KEY+['arm']) for r in rows})==n
 for r in rows:
  for k in ['iterations','checks','selection_scalars','refresh_scalars','per_update','per_check','total_scalars']:r[k]=int(r[k])
  assert r['total_scalars']==r['iterations']*r['per_update']+r['checks']*r['per_check']+r['selection_scalars']+r['refresh_scalars']
  if r['status']=='converged':assert float(r['relative_gap']) <= (1e-7 if r['model'] in ['diabetes','breast_cancer'] else 1e-6)
 return rows
def group(rows):
 d=defaultdict(dict)
 for r in rows:d[tuple(r[k] for k in KEY)][r['arm']]=r
 return d

def compare(rows,arm,ref='joint_auto'):
 vals=[];updates=[];wins=0;total=0;statuses=Counter()
 for key,d in group(rows).items():
  if arm not in d or ref not in d:continue
  total+=1;a,b=d[ref],d[arm];statuses[b['status']]+=1
  if a['status']!='converged' or b['status']!='converged':continue
  v=100*(1-a['total_scalars']/b['total_scalars']);vals.append(v);updates.append(100*(1-a['iterations']/b['iterations']));wins+=v>0
 return dict(cells=total,paired=len(vals),reference_status=dict(statuses),median=float(np.median(vals)) if vals else None,min=float(min(vals)) if vals else None,max=float(max(vals)) if vals else None,wins=wins,update_median=float(np.median(updates)) if vals else None)
def fmt(v):return '--' if v is None else f'{0. if round(v,1)==0 else v:.1f}'
def main():
 core=read('core',973);robust=read('robust',1320);default=[r for r in robust if r['pilot']=='100' and r['lag']=='20'];summ={}
 for name,rows in [('Original',[r for r in core if r['regime']=='original']),('Earlier heterogeneity',[r for r in core if r['regime']!='original']),('High-dimensional',default)]:
  summ[name]={arm:compare(rows,arm) for arm in ARM_LABEL if arm!='joint_auto'}
 lines=[r'\begin{table*}[pos=tbp]\centering\small',r'\caption{Joint one-switch communication relative to external controls. Percentages are median paired savings among pairs in which both arms certify; failed runs are never ranked by their partial cost.}',r'\label{tab:adaptive-controls}',r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llrrr@{}}\toprule',r'Study & Comparator & Certified pairs & Total saving (\%) & Update-only saving (\%)\\\midrule']
 for name in summ:
  for arm in ['norm_auto','sra_zero','rb_zero']:
   d=summ[name][arm];lines.append(f"{name} & {ARM_LABEL[arm]} & {d['paired']}/{d['cells']} & {fmt(d['median'])} & {fmt(d['update_median'])} "+r'\\')
 lines +=[r'\bottomrule\end{tabular*}',r'\par\vspace{3pt}\begin{minipage}{\textwidth}\footnotesize\raggedright SRA denotes the safeguarded spectral-radius-approximation update, RB conventional residual balancing. Both zero-start controls use uniform graph budgets. The total includes update payloads, certificates and parameter-selection messages; update-only uses identical per-round payloads and therefore compares iteration counts. Original cells reuse samples across layouts. New high-dimensional cells reuse ten draws per model across two levels and two layouts.\end{minipage}',r'\end{table*}']
 (TABLES/'table_adaptive_controls.tex').write_text('\n'.join(lines)+'\n')
 # Complete statuses and all comparisons in supplement.
 lines=[r'\section{Adaptive controls and failure-aware comparisons}',r'\label{sec:adaptive-details}',r'The SRA controls use the norm-ratio update of \citet{McCannWohlberg2024,LozenskiEtAl2026} with a common scalar safeguard; implementation and accounting follow the protocol in the main text. The new comparison reuses the 49 original cells and 90 earlier heterogeneity cells. It adds 120 high-dimensional cells from 30 new base draws. Shared-pilot SRA continuations start at each corresponding one-switch metric and penalty; zero-start SRA and residual balancing use uniform budgets from the start. All candidates have a total cap of 30,000 iterations. Table~\ref{tab:adaptive-status} retains certification failures separately from successful paired costs.',r'\begin{table}[htbp]\centering\small',r'\caption{Certification status at the default pilot and lag. Cap and failure columns distinguish iteration limits from numerical failure; no failed run contributes a cost ratio.}',r'\label{tab:adaptive-status}',r'\begin{tabular}{llrrr}\toprule Study & Arm & Certified & Cap & Numerical\\\midrule']
 for name,rows in [('Original',[r for r in core if r['regime']=='original']),('Earlier heterogeneity',[r for r in core if r['regime']!='original']),('High-dimensional',default)]:
  for arm in ARM_LABEL:
   c=Counter(r['status'] for r in rows if r['arm']==arm);lines.append(f"{name} & {ARM_LABEL[arm]} & {c['converged']} & {c['iteration_cap']} & {c['numerical_failure']} "+r'\\')
 lines +=[r'\bottomrule\end{tabular}\end{table}',r'\clearpage',r'\begin{table}[htbp]\centering\small',r'\caption{All comparator savings and ranges. Positive values favor the joint one-switch rule. Medians and ranges use only certified pairs.}',r'\label{tab:adaptive-all}',r'\begin{tabular}{llrrr}\toprule Study & Comparator & Pairs & Median (\%) & Range (\%)\\\midrule']
 for name in summ:
  for arm,d in summ[name].items():lines.append(f"{name} & {ARM_LABEL[arm]} & {d['paired']}/{d['cells']} & {fmt(d['median'])} & {('--' if d['min'] is None else '['+fmt(d['min'])+', '+fmt(d['max'])+']')} "+r'\\')
 lines +=[r'\bottomrule\end{tabular}\end{table}',r'Failure gaps in the raw records are diagnostic values from the last attempted or available certificate, not certified endpoints. The numerical-failure status takes precedence over those values. Bounds on the scalar parameter do not imply convergence for continuously adaptive SRA or RB. Poor SRA outcomes here concern this indicator-subspace splitting, update schedule and safeguard, not all applications of SRA.']
 (TABLES/'supp_adaptive_comparison.tex').write_text('\n'.join(lines)+'\n')
 # New-draw sensitivity, stratified with original random bases reused.
 sens=[]
 for m in ['squared','huber','logistic']:
  for M,J in [('4','4'),('8','2')]:
   for p,l in [('50','20'),('100','10'),('100','20'),('100','40'),('200','20')]:
    xs=[r for r in robust if r['model']==m and r['M']==M and r['J']==J and r['pilot']==p and r['lag']==l and r['heterogeneity']=='16'];d=compare(xs,'uniform_auto');sens.append(dict(model=m,M=M,J=J,pilot=p,lag=l,**d))
 lines=[r'\section{New-draw sensitivity of the one-switch allocation}',r'\label{sec:joint-sensitivity}',r'For each loss, ten new Gaussian AR(1) designs have $n=240$ and $p=480$. Four consecutive sample strata receive geometric scales from 1 to the stated ratio before global column standardization. The coefficient vector has four alternating-sign nonzero entries in each of four fixed predictor groups and norm 3. The response laws and penalties match the main heterogeneity study. Each base draw is reused across scale ratios 1 and 16 and layouts $4\times4$ and $8\times2$. At ratio 16, the pilot and lag are changed one at a time; the share floor remains 0.05. No new outcomes select a setting.',r'\begin{table}[htbp]\centering\small',r'\caption{Joint versus uniform one-switch allocation at sample-scale ratio 16. Both arms use their own increment penalty and identical pilot/lag. All percentages include selection cost.}',r'\label{tab:joint-sensitivity}',r'\begin{tabular}{lrrrrrr}\toprule Model & $M\times J$ & Pilot & Lag & Pairs & Median (\%) & Range (\%)\\\midrule']
 for d in sens:lines.append(f"{d['model'].capitalize()} & ${d['M']}\\times{d['J']}$ & {d['pilot']} & {d['lag']} & {d['paired']}/{d['cells']} & {fmt(d['median'])} & {('--' if d['min'] is None else '['+fmt(d['min'])+', '+fmt(d['max'])+']')} "+r'\\')
 lines +=[r'\bottomrule\end{tabular}\end{table}']
 (TABLES/'supp_robustness.tex').write_text('\n'.join(lines)+'\n')
 # Editorial rounding of the earlier table; raw results and signs remain in CSV.
 old_table=TABLES/'table_joint_original.tex'
 if old_table.exists():
  t=old_table.read_text().replace('-0.0','0.0')
  note=' Magnitudes below 0.05\\% round to 0.0.'
  anchor='Entries are medians of paired cell-level savings.'
  if note not in t:t=t.replace(anchor,anchor+note)
  old_table.write_text(t)
 (OUT/'summary.json').write_text(json.dumps(dict(comparisons=summ,sensitivity=sens,rows=dict(core=len(core),robust=len(robust)),statuses=dict(Counter(r['status'] for r in core+robust))),indent=2))
 print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
