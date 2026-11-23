# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) factor/operator block schematic -> cross-type inherit -> param inherit
# (b) convergence lines -> LineTrend/plot_sweep.py -> param inherit
# (c) paired method points -> PairedBoxScatter/plot_PairedBoxScatter1.R -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.
# The LineTrend asset contains two aggregate fixed-step sweeps. Our multi-seed
# adaptive-stopping histories have a different structure; param inheritance
# retains unframed axes and alpha=0.8, with thinner strokes for ten raw traces.
# Both companion asset PNGs were inspected. The paired-plot asset's boxplots
# and test annotations are not inherited: all five observations and exact
# arithmetic means are shown, without inferential claims. Revision 5 uses
# native CAS widths and a restrained operator-block schematic. Revision 6
# increases Figure 1 print-scale clarity and exports it at 1200 dpi;
# Figures 2 and 3 retain their 900 dpi exports and unchanged plotting code.
# Academic Figure Skill Typography Baseline — COPY VERBATIM, place at TOP of script
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 8,
    "figure.titlesize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
})

# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING   = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL  = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED  = "#B2182B"
GREY        = "#999999"
BLACK       = "#222222"

# Academic Figure Skill Export Baseline — COPY VERBATIM
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Standard Academic Figure Skill export: vector PDF + 300dpi PNG preview."""
    fig.savefig(f"{filename}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(f"{filename}.png", bbox_inches="tight", dpi=300)

# One publication-scale style governs all three figures. The physical canvas
# matches the supplied CAS double-column class rather than a generic journal.
mpl.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "axes.edgecolor": "#59616A",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#59616A",
    "ytick.color": "#59616A",
    "mathtext.fontset": "cm",
    "savefig.bbox": None,
    "path.simplify": False,
})

from pathlib import Path
import gzip, json
from hashlib import sha256
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
COLORS={'unit':CATEGORICAL[0],'budget':CATEGORICAL_EXTENDED[9]}
LOSS_NAMES={'squared':'Squared + fusion','huber':'Huber','logistic':'Logistic'}
PNG_DPI=900
PANEL_HEIGHT_MM=62


def export_figure(fig, stem, axes, *, png_dpi=PNG_DPI):
    """Keep the declared canvas and verify labels before vector/raster export."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();canvas=fig.get_window_extent()
    for ax in axes:
        box=ax.get_tightbbox(renderer)
        assert canvas.contains(*box.min) and canvas.contains(*box.max), (stem,box)
    for artist in list(fig.texts)+list(fig.legends):
        box=artist.get_window_extent(renderer)
        assert canvas.contains(*box.min) and canvas.contains(*box.max), (stem,box)
    for extension in ['pdf','png','svg']:
        fig.savefig(FIG/f'{stem}.{extension}',dpi=png_dpi,
                    bbox_inches=None,pad_inches=0,facecolor='white')
    plt.close(fig)


def panel_heading(ax, index, title):
    ax.text(0,1.080,f'({chr(97+index)})',transform=ax.transAxes,
            fontsize=9,fontweight='bold',va='baseline',color=BLACK)
    ax.text(.165,1.080,title,transform=ax.transAxes,
            fontsize=9,va='baseline',color=BLACK)


def metric_legend(fig, points=False):
    handles=[]
    for metric,marker in [('unit','o'),('budget','D')]:
        handles.append(Line2D([0],[0],color=COLORS[metric],
                       ls='' if points else ((0,(3.2,1.8)) if metric=='unit' else '-'),
                       lw=1.0,marker=marker,ms=3.8,
                       markerfacecolor='white' if metric=='unit' else COLORS[metric],
                       markeredgewidth=.75,label=metric.capitalize()))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.994),
               ncol=2,columnspacing=2.6,handlelength=2.4,handletextpad=.65,
               borderaxespad=.1)


def schematic():
    """An operator-block matrix at the exact CAS single-column width.

    The first two factor rows are a dense 2 x 3 data partition. Two further
    structural factors connect only adjacent feature blocks. Symbols and
    Explicit A_aj labels expose the linear maps, rather than only their
    support. Neutral zero cells remain literal zero maps, not missing data.
    """
    width_mm=83.74;height_mm=46
    mask=np.array([[1,1,1],[1,1,1],[1,1,0],[0,1,1]],dtype=bool)
    with mpl.rc_context({'font.size':8,'savefig.bbox':None}):
        fig=plt.figure(figsize=(width_mm/25.4,height_mm/25.4))
        ax=fig.add_axes([0,0,1,1])
        ax.set_xlim(0,width_mm);ax.set_ylim(height_mm,0);ax.axis('off')
        # Coordinates are millimetres: typography and gaps are chosen at the
        # final publication size, rather than enlarged and shrunk afterwards.
        col_x=np.array([35.7,53.7,71.7])
        row_y=np.array([12.2,20.4,32.0,40.2])
        cell_w,cell_h=16.0,6.8
        for label,y,color in [('Data',16.3,COLORS['unit']),
                               ('Structure',36.1,COLORS['budget'])]:
            ax.text(7.65,y,label,ha='center',va='center',
                    fontsize=9,color=color)
            ax.plot([15.3,15.3],[y-7.65,y+7.65],color=color,lw=1.2,
                    solid_capstyle='butt')
        for j,x in enumerate(col_x):
            ax.text(x,4.3,rf'$\beta_{j+1}$',ha='center',va='center',fontsize=12)
        for i in range(4):
            color=COLORS['unit'] if i<2 else COLORS['budget']
            fill='#F0F5FA' if i<2 else '#FAF3EB'
            rule='#ABC2D8' if i<2 else '#D5B18C'
            y=row_y[i]
            ax.text(21.2,y,rf'$f_{i+1}$',ha='center',va='center',fontsize=12)
            for j,x in enumerate(col_x):
                ax.add_patch(Rectangle((x-cell_w/2,y-cell_h/2),cell_w,cell_h,
                             facecolor=fill if mask[i,j] else 'white',
                             edgecolor=rule if mask[i,j] else '#B8C0C8',
                             linewidth=.65,zorder=0))
                if mask[i,j]:
                    ax.text(x,y,rf'$\mathbf{{A}}_{{{i+1}{j+1}}}$',
                            ha='center',va='center',fontsize=12,color=color)
                else:
                    ax.text(x,y,r'$0$',ha='center',va='center',
                            fontsize=12,color='#59616A')
        # One bracketed augmented operator; the white gap separates the two
        # factor classes without introducing an extra row or an extra map.
        ax.plot([27.0,25.5,25.5,27.0],[8.3,8.3,44.1,44.1],
                color='#59616A',lw=.85,solid_capstyle='butt')
        ax.plot([80.4,81.9,81.9,80.4],[8.3,8.3,44.1,44.1],
                color='#59616A',lw=.85,solid_capstyle='butt')
        # Verify every label at its final physical size, including pairwise
        # separation. The plot is exact line art rather than a raster inset.
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer();canvas=fig.get_window_extent()
        text_boxes=[artist.get_window_extent(renderer) for artist in ax.texts]
        for box in text_boxes:
            assert canvas.contains(*box.min) and canvas.contains(*box.max)
        for index,box in enumerate(text_boxes):
            assert not any(box.overlaps(other) for other in text_boxes[index+1:])
        export_figure(fig,'fig1_factor_structure',[ax],png_dpi=1200)
    qa={
        'archetype':'factor-operator block schematic','target':'Elsevier cas-dc single column',
        'native_width_mm':width_mm,'native_height_mm':height_mm,
        'claim':'A 2-by-3 data partition and cross-block penalties share one sparse factor-feature representation.',
        'incidence_mask':mask.astype(int).tolist(),'data_factor_rows':[1,2],
        'structural_factor_rows':[3,4],'nonzero_incidences':int(mask.sum()),
        'encoding':'Blue operator blocks=data factors; ochre operator blocks=structural factors; neutral 0=known zero map. Group location and labels supplement color.',
        'labels':'beta1-beta3, f1-f4, ten nonzero A_aj blocks, two exact zeros; Data and Structure identify factor classes.',
        'font_pt':{'column_labels':12,'row_and_operator_labels':12,'zero_entries':12,'factor_group_labels':9},
        'export':['vector PDF','1200 dpi PNG','editable SVG'],
        'data_status':'Exact conceptual example, not estimated experimental outcomes',
        'layout_checks':'All 21 text bounds lie within the native-size canvas and are pairwise disjoint',
        'statistics':'Not applicable to a mathematical incidence example',
        'design_review':{
            'style':'Larger 12 pt mathematical labels and 9 pt factor-group names; stronger 0.65 pt cell rules and 0.85 pt operator brackets, with blue/ochre fills, dark neutral zeros, aligned labels and unchanged matrix geometry.',
            'code':'Native CAS column size; embedded TrueType PDF; editable SVG; 1200 dpi PNG; label bounds and pairwise separation asserted.',
            'integrity':'The 4-by-3 mask retains all 10 nonzero incidences of the conceptual example.',
            'visual':'Revision 6 PDF inspected at 150 and 200 dpi and in a rendered manuscript-page preview.',
        },
    }
    (ROOT/'results'/'figure1_qa.json').write_text(json.dumps(qa,indent=2)+'\n')


def load_data():
    rows=pd.read_csv(ROOT/'results'/'runs.csv')
    assert len(rows)==138 and rows.converged.all()
    traces={}
    with gzip.open(ROOT/'results'/'traces.jsonl.gz','rt') as f:
        for line in f:
            a=json.loads(line);traces[a['key']]=a['history']
    return rows,traces


def convergence(rows,traces):
    """A compact full-width convergence figure for the supplied cas-dc class.

    Each line is one full recorded history. The numerical summary remains in
    the manuscript table, keeping the figure focused on the convergence paths.
    """
    width_mm=173.8;height_mm=PANEL_HEIGHT_MM
    style={
        'font.size':9,'axes.labelsize':9,'xtick.labelsize':8,
        'ytick.labelsize':8,'legend.fontsize':9,'axes.edgecolor':'#59616A',
        'text.color':'#222222','axes.labelcolor':'#222222',
        'xtick.color':'#59616A','ytick.color':'#59616A','savefig.bbox':None,
    }
    summaries={};point_count=0;curves=0;point_ranges={}
    with mpl.rc_context(style):
        fig,axes=plt.subplots(1,3,sharey=True,figsize=(width_mm/25.4,height_mm/25.4))
        fig.subplots_adjust(left=.083,right=.988,bottom=.195,top=.785,wspace=.25)
        for index,(kind,loss,title,xmax,xticks) in enumerate([
            ('fused','squared','Squared + fusion',2000,[0,1000,2000]),
            ('huber','huber','Huber',2400,[0,1000,2000]),
            ('logistic','logistic','Logistic',1200,[0,500,1000]),
        ]):
            ax=axes[index];summaries[kind]={};point_ranges[kind]={}
            for metric in ['unit','budget']:
                ends=[];all_y=[]
                for seed in range(5):
                    key=f'synthetic_{kind}_{seed}_M8_N4_{metric}'
                    h=traces[key]
                    x=np.array([v['iteration'] for v in h])
                    y=np.array([v['relative_gap'] for v in h])
                    assert np.isfinite(y).all() and (y>5e-7).all() and (y<500).all()
                    assert np.all(np.diff(x)>0) and x[-1]<xmax and y[-1]<=1e-6
                    matched=rows[(rows.kind=='synthetic')&(rows.loss==loss)&
                                 (rows.seed==seed)&(rows.row_parts==8)&
                                 (rows.column_parts==4)&(rows.metric==metric)&
                                 (rows.zero_edges==0)]
                    assert len(matched)==1 and x[-1]==int(matched.iloc[0].iterations)
                    # All saved checkpoints are connected in their original order.
                    ax.plot(x,y,color=COLORS[metric],
                            ls=(0,(3.2,1.8)) if metric=='unit' else '-',
                            lw=.72,alpha=.8,zorder=3,
                            solid_capstyle='round',dash_capstyle='butt')
                    ax.plot(x[-1],y[-1],ls='',marker='o' if metric=='unit' else 'D',
                            ms=2.6,mfc='white' if metric=='unit' else COLORS[metric],
                            mec=COLORS[metric],mew=.65,zorder=4)
                    ends.append(int(x[-1]));all_y.extend(y.tolist())
                    point_count+=len(x);curves+=1
                summaries[kind][metric]={
                    'n':len(ends),'endpoints':ends,
                    'mean':float(np.mean(ends)),
                    'sample_sd':float(np.std(ends,ddof=1)),
                }
                point_ranges[kind][metric]={'min':min(all_y),'max':max(all_y)}
            ax.set_yscale('log');ax.set_ylim(5e-7,500)
            ax.set_yticks([1e-6,1e-4,1e-2,1,100])
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            # Quiet horizontal rules support log-scale comparison; no
            # vertical grid is needed for continuous trajectories.
            for level in [1e-4,1e-2,1,100]:
                ax.axhline(level,color='#E5E9ED',lw=.35,zorder=0)
            ax.axhline(1e-6,color='#8F98A1',ls=(0,(1.2,2.1)),lw=.6,zorder=1)
            ax.set_xlim(0,xmax);ax.set_xticks(xticks)
            ax.tick_params(axis='both',length=2.8,pad=3.2)
            panel_heading(ax,index,title)
            if index==0:ax.set_ylabel('Relative gap',labelpad=5)
        metric_legend(fig)
        fig.text(.544,.040,'Iteration',ha='center',va='center',fontsize=9)
        export_figure(fig,'fig2_convergence',axes)
    assert curves==30 and point_count==3779
    qa={
        'archetype':'quantitative_grid','target':'Elsevier cas-dc full text width',
        'native_width_mm':width_mm,'native_height_mm':height_mm,
        'claim':'The effect of fixed incidence weighting on convergence differs across regression criteria.',
        'source':'results/traces.jsonl.gz','source_sha256':sha256((ROOT/'results/traces.jsonl.gz').read_bytes()).hexdigest(),
        'endpoint_source':'results/runs.csv','endpoint_source_sha256':sha256((ROOT/'results/runs.csv').read_bytes()).hexdigest(),
        'curves':curves,'checkpoint_points':point_count,'point_ranges':point_ranges,
        'omitted_runs':0,'omitted_checkpoint_points':0,'modified_values':0,
        'smoothing':'none','aggregation_curve':'none','matplotlib_path_simplification':False,
        'termination':'Each original trajectory ends at its actual last checkpoint.',
        'endpoint_symbols':'Open blue circles for unit; filled ochre diamonds for budget; all 30 recorded endpoints shown.',
        'summaries_not_drawn':summaries,
        'replicate_definition':'Five independent generated training/test samples per model, seeds 0-4; shared samples across metrics.',
        'sample_sizes':{'training':300,'test':1000},
        'algorithm_settings':{'sample_blocks':8,'feature_blocks':4,'rho':1},
        'metric_definition':'(primal objective - finite Fenchel dual bound) / max(1, abs(primal objective))',
        'baseline_definition':'Unit metric versus fixed uniform incidence-budget metric on the same criterion and sample.',
        'uncertainty':'All five seed trajectories shown; no aggregate, confidence interval or error bar.',
        'statistical_tests':'None; descriptive comparison','multiple_comparison_correction':'Not applicable',
        'target_relative_gap':1e-6,'export':['vector PDF','900 dpi PNG','editable SVG'],
        'font_pt':{'axis_labels':9,'tick_labels':8,'legend':9,'panel_labels':9},
        'layout_checks':'All axes, labels and legend fit within the native-size canvas.',
        'removed_annotations':'Iteration-summary footer, verbose panel titles and tolerance legend entry; definitions retained in caption/table.',
        'design_review':{
            'style':'Three equal panels; 9 pt headings and axis titles with 8 pt ticks; shared log scale and fine horizontal reference rules; blue/ochre with dashed/solid curves and matching endpoint symbols in one compact legend.',
            'code':'Native CAS text width; embedded TrueType PDF; editable SVG; 900 dpi PNG; explicit bounds assertions.',
            'integrity':'All 30 original histories and 3779 checkpoints retained; final iterations match runs.csv.',
            'visual':'Generation checks data and layout bounds; visually inspect the exported figures.',
        },
    }
    (ROOT/'results'/'figure2_qa.json').write_text(json.dumps(qa,indent=2)+'\n')


def communication(rows):
    """Display every paired seed at fixed categorical positions, with means.

    PairedBoxScatter parameters inherited: unframed axes, neutral paired
    segments (alpha=0.75, width=0.45), and visible individual observations.
    Five samples do not require a boxplot or a significance annotation.
    """
    width_mm=173.8;height_mm=PANEL_HEIGHT_MM
    fig,axes=plt.subplots(1,3,figsize=(width_mm/25.4,height_mm/25.4))
    fig.subplots_adjust(left=.083,right=.988,bottom=.195,top=.785,wspace=.25)
    layouts=[(1,1),(4,1),(4,4),(8,4)]
    endpoints=[];summaries={};pair_count=0;mean_count=0
    for index,loss in enumerate(['squared','huber','logistic']):
        ax=axes[index];summaries[loss]={}
        subset=rows[(rows.kind=='synthetic')&(rows.loss==loss)&(rows.zero_edges==0)]
        for pos,(M,N) in enumerate(layouts):
            g=subset[(subset.row_parts==M)&(subset.column_parts==N)]
            assert len(g)==10
            summaries[loss][f'{M}x{N}']={}
            for seed in range(5):
                u=g[(g.seed==seed)&(g.metric=='unit')].iloc[0].communicated_scalars/1e6
                b=g[(g.seed==seed)&(g.metric=='budget')].iloc[0].communicated_scalars/1e6
                jitter=(seed-2)*.035
                ax.plot([pos-.16+jitter,pos+.16+jitter],[u,b],
                        color='#AAB0B6',lw=.45,alpha=.75,zorder=1)
                pair_count+=1
            for metric,offset,marker in [('unit',-.16,'o'),('budget',.16,'D')]:
                values=g[g.metric==metric].sort_values('seed').communicated_scalars.to_numpy()/1e6
                assert len(values)==5
                x=pos+offset+(np.arange(5)-2)*.035
                ax.scatter(x,values,s=13,marker=marker,zorder=3,
                           facecolors='white' if metric=='unit' else COLORS[metric],
                           edgecolors=COLORS[metric] if metric=='unit' else 'white',
                           linewidth=.75 if metric=='unit' else .35)
                # The mean bar remains readable when nearly equal raw points
                # overlap; its short overprint leaves the marker outlines.
                ax.plot([pos+offset-.13,pos+offset+.13],[np.mean(values)]*2,
                        color=BLACK,lw=.85,zorder=4,solid_capstyle='butt')
                mean_count+=1
                summaries[loss][f'{M}x{N}'][metric]={
                    'n':5,'mean':float(np.mean(values)),
                    'sample_sd_not_drawn':float(np.std(values,ddof=1)),
                }
                endpoints.extend([{'loss':loss,'sample_blocks':M,'feature_blocks':N,
                                   'metric':metric,'seed':int(seed),'x':float(x[seed]),
                                   'million_scalars':float(value)}
                                  for seed,value in enumerate(values)])
        ax.set_xticks(range(4),[f'{m} × {n}' for m,n in layouts])
        ax.set_xlim(-.42,3.42)
        # Panel-specific linear ranges retain within-model resolution; units
        # and zero origin are identical. Every observed value is inside range.
        upper={'squared':34,'huber':29,'logistic':14}[loss]
        ax.set_ylim(0,upper)
        ax.set_yticks([0,10,20,30] if loss=='squared' else
                     ([0,10,20] if loss=='huber' else [0,5,10]))
        for level in ax.get_yticks()[1:]:
            ax.axhline(level,color='#E5E9ED',lw=.35,zorder=0)
        assert subset.communicated_scalars.max()/1e6<upper
        ax.tick_params(axis='both',length=2.8,pad=3.2)
        panel_heading(ax,index,LOSS_NAMES[loss])
        if index==0:ax.set_ylabel('Traffic (million scalars)',labelpad=5)
    metric_legend(fig,points=True)
    fig.text(.544,.040,'Sample × feature blocks',ha='center',va='center',fontsize=9)
    export_figure(fig,'fig3_communication',axes)
    assert len(endpoints)==120 and pair_count==60 and mean_count==24
    qa={
        'archetype':'paired_points_with_mean_marks','target':'Elsevier cas-dc full text width',
        'native_width_mm':width_mm,'native_height_mm':height_mm,
        'claim':'Communication depends on both the block layout and model; paired seeds expose between-sample variation.',
        'source':'results/runs.csv','source_sha256':sha256((ROOT/'results/runs.csv').read_bytes()).hexdigest(),
        'individual_points':len(endpoints),'paired_segments':pair_count,'mean_marks':mean_count,
        'omitted_runs':0,'modified_values':0,'smoothing':'none',
        'value_transformation':'communicated_scalars / 1e6; exact unit conversion only',
        'replicate_definition':'Five independent generated training/test samples per model, seeds 0-4; matched seed pairs share the sample.',
        'algorithm_settings':{'rho':1,'layouts':layouts,'relative_gap_target':1e-6},
        'encoding':'Open blue circles=unit; filled ochre diamonds=budget; gray segment=matched seed; black tick=arithmetic mean.',
        'horizontal_offset':'Fixed categorical jitter (seed-2)*0.035 separates observations; no quantitative x values are altered.',
        'axis_scales':'Linear, zero-based; panel limits 34, 29 and 14 million scalars.',
        'uncertainty':'All five observations shown; no confidence intervals or significance tests.',
        'statistical_tests':'None; descriptive comparison','multiple_comparison_correction':'Not applicable',
        'summaries':summaries,'plotted_values':endpoints,
        'export':['vector PDF','900 dpi PNG','editable SVG'],
        'font_pt':{'axis_labels':9,'tick_labels':8,'legend':9,'panel_labels':9},
        'layout_checks':'All axes, labels and legend fit within the native-size canvas.',
        'design_review':{
            'style':'Same native width, 9/8 pt typography, panel alignment, fine reference rules and blue/ochre palette as Figure 2; enlarged individual symbols and visible mean marks; one shared x label.',
            'code':'Fixed native dimensions; TrueType PDF embedding; editable SVG; 900 dpi PNG; explicit label and data-bound checks.',
            'integrity':'All 120 original endpoint values and 60 matched pairs are retained; all 24 means computed from five points.',
            'visual':'Generation checks data and layout bounds; visually inspect the exported figures.',
        },
    }
    (ROOT/'results'/'figure3_qa.json').write_text(json.dumps(qa,indent=2)+'\n')


def main():
    rows,traces=load_data();schematic();convergence(rows,traces);communication(rows)
    qa=dict(figures=3,all_runs=138,simulation_replicates=5,trace_source='results/traces.jsonl.gz',
            endpoint_source='results/runs.csv',figure1='Single-column augmented operator-block matrix: ten explicit nonzero A_aj blocks, two exact zeros, two data-factor rows and two cross-block structural-factor rows; exact conceptual example.',
            figure2='Compact full-width three-panel convergence plot at M=8,J=4: all 30 original trajectories and 3779 recorded checkpoints. Each line ends at actual termination. Summaries remain in tables, with no aggregate or smoothed curves.',
            figure3='All five seed-paired endpoint traffic values at each layout. Points are individual runs; short black lines are means. No sampling or omitted runs.',
            statistical_tests='None: descriptive numerical comparisons, no significance claims',
            uncertainty='Individual independent simulation seeds shown, no confidence-interval claim',
            export='Vector PDF with embedded TrueType fonts and editable SVG; Figure 1 PNG 1200 dpi, Figures 2 and 3 PNG 900 dpi',
            source_integrity='No modification, subsampling or selection of favorable numerical outcomes')
    (ROOT/'results'/'figure_qa.json').write_text(json.dumps(qa,indent=2))
    print('Generated three vector figures and previews from verified raw runs.')


if __name__=='__main__':main()
