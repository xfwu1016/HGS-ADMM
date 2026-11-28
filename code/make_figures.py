"""Generate the factor diagram, convergence traces and paired communication plots."""
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

# Colors shared across the convergence and communication plots.
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

# Vector and raster export settings.
mpl.rcParams.update({
    "pdf.fonttype": 42,         # TrueType font embedding
    "svg.fonttype": "none",     # editable text in SVG
    "savefig.bbox": "tight",    # trim whitespace
    "savefig.dpi": 300,
})

def save_cns_figure(fig, filename):
    """Save a vector PDF and a 300 dpi PNG preview."""
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
from matplotlib.patches import Polygon, Rectangle

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
    """Draw the sample-by-feature partition and its structural operators."""
    width_mm, height_mm = 173.8, 66.0
    mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=bool)
    expected_labels = [f'A_{i + 1}{j + 1}' for i, j in np.argwhere(mask)]
    zeros = [[int(i) + 1, int(j) + 1] for i, j in np.argwhere(~mask)]
    assert len(expected_labels) == 10 and zeros == [[3, 3], [4, 1]]

    data_ink, structure_ink, neutral = '#23567C', '#814C18', '#526474'
    col_x = [72.0, 108.9, 145.8]
    row_y = [21.6, 34.7, 49.3, 60.3]
    cell_w, cell_h = 31.0, 8.6
    rise_x, rise_y = 3.9, 2.8  # Projection of a common, illustrative depth.

    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width_mm)
    ax.set_ylim(height_mm, 0)
    ax.axis('off')
    label_artists, entry_artists, rendered_labels = [], [], []

    def label(x, y, text, *, size=14, color='#243847', align='center', weight='normal'):
        artist = ax.text(x, y, text, ha=align, va='center', fontsize=size,
                         color=color, fontweight=weight, zorder=5)
        label_artists.append(artist)
        return artist

    label(108.9, 4.6, 'Feature blocks', size=10.5, color=neutral, weight='bold')
    for j, x in enumerate(col_x, start=1):
        label(x, 10.8, rf'$\boldsymbol{{\beta}}_{j}$')

    label(29.3, 25.9, 'Data factors', size=11, color=data_ink, align='right', weight='bold')
    label(29.3, 30.0, 'Sample blocks', size=9.3, color=neutral, align='right')
    label(29.3, 54.7, 'Structural\nfactors', size=11, color=structure_ink,
          align='right', weight='bold')
    ax.plot([33.3, 33.3], [17.5, 38.6], color='#5487AD', lw=1.35)
    ax.plot([33.3, 33.3], [45.2, 64.4], color='#AD7841', lw=1.35)
    ax.plot([55.8, 166.6], [42.3, 42.3], color='#CCD4D9', lw=.65)

    for i, y in enumerate(row_y):
        data_row = i < 2
        ink = data_ink if data_row else structure_ink
        label(41.1, y, rf'$f_{i + 1}$', color=ink)
        if data_row:
            label(46.1, y, rf'$(I_{i + 1})$', size=10.5, color=neutral, align='left')
        for j, x in enumerate(col_x):
            left, top = x - cell_w / 2, y - cell_h / 2
            right, bottom = left + cell_w, top + cell_h
            if data_row:
                top_face = [(left, top), (left + rise_x, top - rise_y),
                            (right + rise_x, top - rise_y), (right, top)]
                right_face = [(right, top), (right + rise_x, top - rise_y),
                              (right + rise_x, bottom - rise_y), (right, bottom)]
                for points, fill in [(top_face, '#F3F9FD'), (right_face, '#AACBE0')]:
                    ax.add_patch(Polygon(points, closed=True, facecolor=fill,
                                         edgecolor='#6C9ABA', linewidth=.7, joinstyle='round'))
                front, edge = '#E2F0FA', '#6C9ABA'
            elif mask[i, j]:
                front, edge = '#FBEDD9', '#BC966D'
            else:
                front, edge = '#F9FAFB', '#A6B0B8'
            ax.add_patch(Rectangle((left, top), cell_w, cell_h, facecolor=front,
                                   edgecolor=edge, linewidth=.75,
                                   linestyle='-' if mask[i, j] else (0, (2.5, 2.2))))
            if mask[i, j]:
                text = label(x, y, rf'$\mathbf{{A}}_{{{i + 1}{j + 1}}}$', size=15, color=ink)
                rendered_labels.append(f'A_{i + 1}{j + 1}')
            else:
                text = label(x, y, r'$0$', size=15, color=neutral)
            entry_artists.append((text, left, top, right, bottom))

    assert rendered_labels == expected_labels and len(entry_artists) == 12
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.get_window_extent()
    boxes = [artist.get_window_extent(renderer) for artist in label_artists]
    minimum_gap_px = float('inf')
    for index, box in enumerate(boxes):
        assert canvas.contains(*box.min) and canvas.contains(*box.max), label_artists[index].get_text()
        for other in boxes[index + 1:]:
            assert not box.overlaps(other), label_artists[index].get_text()
            dx = max(box.x0 - other.x1, other.x0 - box.x1, 0)
            dy = max(box.y0 - other.y1, other.y0 - box.y1, 0)
            minimum_gap_px = min(minimum_gap_px, np.hypot(dx, dy))
    for artist, left, top, right, bottom in entry_artists:
        cell = ax.transData.transform([[left, bottom], [right, top]])
        box = artist.get_window_extent(renderer)
        assert cell[0, 0] < box.x0 < box.x1 < cell[1, 0]
        assert cell[0, 1] < box.y0 < box.y1 < cell[1, 1]

    export_figure(fig, 'fig1_factor_structure', [], png_dpi=1200)
    qa = {
        'archetype': 'operator-block schematic with oblique data tiles',
        'target': 'CSDA, Elsevier cas-dc double column',
        'native_width_mm': width_mm,
        'native_height_mm': height_mm,
        'incidence_mask': mask.astype(int).tolist(),
        'data_factor_rows': [1, 2],
        'structural_factor_rows': [3, 4],
        'nonzero_incidences': len(expected_labels),
        'nonzero_labels': expected_labels,
        'zero_cells': zeros,
        'encoding': ('Six blue tiles show sample-by-feature data blocks. Flat ochre cells '
                     'show structural operators on the same feature blocks. Dashed neutral '
                     'cells display the two exact zero maps. The projected depth is schematic.'),
        'layout_checks': {'text_count': len(boxes), 'all_text_within_canvas': True,
                          'all_text_pairwise_disjoint': True,
                          'all_entries_within_own_cells': True,
                          'minimum_text_gap_mm': round(float(minimum_gap_px * 25.4 / fig.dpi), 3)},
        'export': ['vector PDF', '1200 dpi PNG', 'editable SVG'],
        'data_status': 'Exact mathematical schematic, without estimated quantities.',
    }
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results' / 'figure1_qa.json').write_text(json.dumps(qa, indent=2) + '\n')


def load_data():
    rows=pd.read_csv(ROOT/'results'/'runs.csv')
    assert len(rows)==138 and rows.converged.all()
    traces={}
    with gzip.open(ROOT/'results'/'traces.jsonl.gz','rt') as f:
        for line in f:
            record = json.loads(line)
            key = record['key']
            if key in traces or not record['history']:
                raise ValueError(f'Duplicate or empty trajectory: {key}')
            traces[key] = record['history']
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
    settings = json.loads((ROOT / 'results' / 'methods.json').read_text())
    check_every = settings['check_every']
    target = settings['synthetic_gap_tolerance']
    y_low, y_high = 5e-7, 500
    with mpl.rc_context(style):
        fig,axes=plt.subplots(1,3,sharey=True,figsize=(width_mm/25.4,height_mm/25.4))
        fig.subplots_adjust(left=.083,right=.988,bottom=.195,top=.785,wspace=.25)
        for index,(kind,loss,title,xmax,xticks) in enumerate([
            ('fused','squared','Squared + fusion',2000,[0,1000,2000]),
            ('huber','huber','Huber',2400,[0,1000,2000]),
            ('logistic','logistic','Logistic',1200,[0,500,1000]),
        ]):
            ax=axes[index];summaries[kind]={};point_ranges[kind]={}
            largest_endpoint = 0
            for metric in ['unit','budget']:
                ends=[];all_y=[]
                for seed in range(5):
                    key=f'synthetic_{kind}_{seed}_M8_N4_{metric}'
                    h=traces[key]
                    x=np.array([v['iteration'] for v in h])
                    y=np.array([v['relative_gap'] for v in h])
                    matched=rows[(rows.kind=='synthetic')&(rows.loss==loss)&
                                 (rows.seed==seed)&(rows.row_parts==8)&
                                 (rows.column_parts==4)&(rows.metric==metric)&
                                 (rows.zero_edges==0)]
                    if len(matched) != 1:
                        raise ValueError(f'Expected one endpoint record for {key}')
                    endpoint = matched.iloc[0]
                    last = int(endpoint.iterations)
                    checkpoints = [1, *range(check_every, last + 1, check_every)]
                    if checkpoints[-1] != last:
                        checkpoints.append(last)  # The solver also checks its iteration cap.
                    if not np.array_equal(x, checkpoints):
                        raise ValueError(f'Incomplete or out-of-order checkpoint history: {key}')
                    if (not np.isfinite(y).all() or not (y > 0).all() or
                        not (y[:-1] > target).all() or y[-1] > target or
                        not np.isclose(y[-1], endpoint.relative_gap, rtol=1e-8, atol=1e-15)):
                        raise ValueError(f'Invalid log-gap history or mismatched first crossing: {key}')
                    largest_endpoint = max(largest_endpoint, last)
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
            y_low = min(y_low, .95 * min(values['min'] for values in point_ranges[kind].values()))
            y_high = max(y_high, 1.05 * max(values['max'] for values in point_ranges[kind].values()))
            ax.set_yscale('log')
            ax.set_yticks([1e-6,1e-4,1e-2,1,100])
            ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            # Quiet horizontal rules support log-scale comparison; no
            # vertical grid is needed for continuous trajectories.
            for level in [1e-4,1e-2,1,100]:
                ax.axhline(level,color='#E5E9ED',lw=.35,zorder=0)
            ax.axhline(target,color='#8F98A1',ls=(0,(1.2,2.1)),lw=.6,zorder=1)
            ax.set_xlim(0, max(xmax, 1.02 * largest_endpoint))
            if largest_endpoint <= xmax:
                ax.set_xticks(xticks)
            else:
                ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, integer=True))
            ax.tick_params(axis='both',length=2.8,pad=3.2)
            panel_heading(ax,index,title)
            if index==0:ax.set_ylabel('Relative gap',labelpad=5)
        axes[0].set_ylim(y_low, y_high)
        metric_legend(fig)
        fig.text(.544,.040,'Iteration',ha='center',va='center',fontsize=9)
        export_figure(fig,'fig2_convergence',axes)
    assert curves == 30
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
        'target_relative_gap':target,'export':['vector PDF','900 dpi PNG','editable SVG'],
        'font_pt':{'axis_labels':9,'tick_labels':8,'legend':9,'panel_labels':9},
        'layout_checks':'All axes, labels and legend fit within the native-size canvas.',
        'removed_annotations':'Iteration-summary footer, verbose panel titles and tolerance legend entry; definitions retained in caption/table.',
        'design_review':{
            'style':'Three equal panels; 9 pt headings and axis titles with 8 pt ticks; shared log scale and fine horizontal reference rules; blue/ochre with dashed/solid curves and matching endpoint symbols in one compact legend.',
            'code':'Native CAS text width; embedded TrueType PDF; editable SVG; 900 dpi PNG; explicit bounds assertions.',
            'integrity':f'All {curves} histories and {point_count} checkpoints retained; first crossings match runs.csv.',
            'visual':'Generation checks data and layout bounds; visually inspect the exported figures.',
        },
    }
    (ROOT/'results'/'figure2_qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    return point_count


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
    rows,traces=load_data();schematic();checkpoint_count=convergence(rows,traces);communication(rows)
    qa=dict(figures=3,all_runs=138,simulation_replicates=5,trace_source='results/traces.jsonl.gz',
            endpoint_source='results/runs.csv',figure1='Double-column schematic (173.8 mm wide): six oblique data tiles in a 2-by-3 partition, four nonzero structural operator blocks, and two exact zero maps.',
            figure2=f'Compact full-width three-panel convergence plot at M=8,J=4: all 30 original trajectories and {checkpoint_count} recorded checkpoints. Each line ends at actual termination. Summaries remain in tables, with no aggregate or smoothed curves.',
            figure3='All five seed-paired endpoint traffic values at each layout. Points are individual runs; short black lines are means. No sampling or omitted runs.',
            statistical_tests='None: descriptive numerical comparisons, no significance claims',
            uncertainty='Individual independent simulation seeds shown, no confidence-interval claim',
            export='Vector PDF with embedded TrueType fonts and editable SVG; Figure 1 PNG 1200 dpi, Figures 2 and 3 PNG 900 dpi',
            source_integrity='No modification, subsampling or selection of favorable numerical outcomes')
    (ROOT/'results'/'figure_qa.json').write_text(json.dumps(qa,indent=2))
    print('Generated three vector figures and previews from verified raw runs.')


if __name__=='__main__':main()

