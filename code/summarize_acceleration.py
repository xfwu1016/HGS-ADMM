"""Generate all new summaries, tables and the conditional-gain figure from CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/acceleration';T=ROOT/'tables';F=ROOT/'figures'
LABEL={'squared':'Squared','fused':'Squared fused','huber':'Huber','logistic':'Logistic','diabetes':'Diabetes','breast_cancer':'Breast cancer'}
def pairs(d):return d.pivot(index=['regime','model','seed','heterogeneity','M','J'],columns='arm',values='total_scalars')
def stats(t):
 out={'cells':len(t)}
 for name in ['uniform_fixed','uniform_auto','uniform_joint_rho']:
  s=100*(1-t.joint_auto/t[name]);out[name]={'median_saving':float(s.median()),'min_saving':float(s.min()),'max_saving':float(s.max()),'wins':int((s>1e-10).sum()),'ties':int((s.abs()<1e-10).sum())}
 return out
def main():
 orig=pd.read_csv(D/'joint_original.csv');new=pd.read_csv(D/'joint_hetero.csv');test=new[new.split=='test'];ot=pairs(orig);nt=pairs(test)
 report={'original':stats(ot),'new_test':stats(nt),'new_base_draws':30,'new_condition_cells':90,'groups':[]}
 rows=[]
 for model,g in ot.groupby(level='model',sort=False):
  v=stats(g);rows.append(f"{LABEL[model]} & {len(g)} & {v['uniform_fixed']['median_saving']:.1f} & {v['uniform_auto']['median_saving']:.1f} & {v['uniform_auto']['wins']}/{len(g)} & {v['uniform_joint_rho']['median_saving']:.1f} \\\\")
 head=r'''\begin{table*}[pos=tbp]\centering\small
\caption{Joint increment rule on the original configurations. Positive paired percentages denote less total charged communication. All arms retain the same pilot.}
\label{tab:joint-original}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lrrrrr@{}}\toprule
 & & \multicolumn{4}{c}{Joint rule versus uniform shares}\\
\cmidrule(l){3-6}
Model & Cells & Fixed $\rho=1$ (\%) & Automatic $\rho$ (\%) & Wins & Same joint $\rho$ (\%)\\\midrule
'''
 foot=r'''\bottomrule\end{tabular*}\par\vspace{3pt}
\begin{minipage}{\textwidth}\footnotesize\raggedright
Entries are medians of paired cell-level savings. Each simulated model uses five samples on three layouts; each public dataset uses two layouts of one fixed split. Wins compare the joint rule with uniform shares using their own increment-based penalty. The last column holds the selected joint penalty fixed in both arms and charges identical selection overhead. Costs include pilot updates, suffix updates, certificate bounds, allocation messages, penalty broadcasts and state refresh.\end{minipage}\end{table*}
'''
 (T/'table_joint_original.tex').write_text(head+'\n'.join(rows)+'\n'+foot)
 rows=[]
 for (reg,model,level),g in nt.groupby(level=['regime','model','heterogeneity']):
  v=stats(g);report['groups'].append(dict(regime=reg,model=model,level=int(level),**v))
  rows.append(f"{'Row scale' if reg=='row_scale' else 'Signal'} & {LABEL[model]} & {level} & {v['uniform_fixed']['median_saving']:.2f} & {v['uniform_auto']['median_saving']:.2f} & {v['uniform_auto']['wins']}/5 & {v['uniform_joint_rho']['median_saving']:.2f} \\\\")
 (T/'supp_joint_results.tex').write_text(r'''\section*{Joint weights and penalty: all new test conditions}
\begin{table}[H]\centering\small
\caption{Paired communication savings of the joint rule on the new test conditions. Positive values favour the joint rule.}
\label{tab:joint-heterogeneity}
\begin{tabular}{llrrrrr}\toprule
Regime & Model & Ratio & Fixed (\%) & Auto (\%) & Wins & Same $\rho$ (\%)\\\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\par\vspace{4pt}
\begin{minipage}{\linewidth}\footnotesize\raggedright
Each row uses five paired samples. Within a regime and model, the three ratios reuse the same five random bases. Thus the table contains 90 condition cells from 30 base draws. Fixed, Auto and Same $\rho$ refer to uniform shares with $\rho=1$, their own automatic penalty, or the joint rule's penalty, respectively. Wins refer to Auto. The two regimes vary block signal amplitudes or sample-block predictor scales, with columns standardized globally and the coefficient norm fixed.\end{minipage}\end{table}
''')
 # Every cell shown, not only favourable regimes.
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
 fig,axs=plt.subplots(1,3,figsize=(9,2.65),sharey=True)
 for ax,model in zip(axs,['squared','huber','logistic']):
  for reg,color,marker,offset in [('row_scale','#0072B2','o',-.035),('signal','#D55E00','s',.035)]:
   vals=[]
   for x,level in enumerate([1,4,16]):
    g=nt.xs((reg,model,level),level=['regime','model','heterogeneity']);s=100*(1-g.joint_auto/g.uniform_auto);vals.append(s.median());ax.scatter(x+offset+np.linspace(-.055,.055,len(s)),s,color=color,alpha=.45,s=13,marker=marker,zorder=2)
   ax.plot(np.arange(3)+offset,vals,color=color,marker=marker,label='Sample-block scale' if reg=='row_scale' else 'Feature-block signal',lw=1.4,ms=4)
  ax.axhline(0,color='#555',lw=.7);ax.set_xticks([0,1,2],['1','4','16']);ax.set_xlabel('Largest/smallest block ratio');ax.set_title(LABEL[model]);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylabel('Communication saving versus\nuniform automatic penalty (%)');axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.63),ncol=2,frameon=False)
 fig.subplots_adjust(bottom=.25,wspace=.12);fig.savefig(F/'fig_joint_heterogeneity.pdf');fig.savefig(F/'fig_joint_heterogeneity.png',dpi=220);plt.close(fig)
 # Original output-only oracle diagnostic.
 d=pd.read_csv(D/'original.csv');p=d.pivot(index=['model','seed','M','J'],columns='arm',values='total_scalars');ss=100*(1-p.oracle/p.uniform)
 report['output_oracle']={'cells':len(p),'wins':int((ss>0).sum()),'median_saving':float(ss.median()),'min_saving':float(ss.min()),'max_saving':float(ss.max())}
 (D/'summary.json').write_text(json.dumps(report,indent=2))
 print(json.dumps({k:v for k,v in report.items() if k!='groups'},indent=2))
if __name__=='__main__':main()
