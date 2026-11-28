# Academic Figure Skill Asset Confirmation (verified against assets/figures/)
# (a) star projections -> cross-type inherit: supplied original vector schematic -> param inherit
# (b) message sequence -> cross-type inherit: supplied original vector schematic -> param inherit
# RULE: "native run" = load pre-rendered PNG via Image.open().ax.imshow().
#       "param inherit" = drawing function below that copies Class A/B/C values.
#       If a panel says "native run" and you write a drawing function, you broke the contract.

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

# The supplied figure is a mathematical schematic, not a quantitative chart.
# There is no compatible production asset in assets/figures/other/. A Sankey
# diagram would encode quantities absent here, so its plotting logic is not used.
# Geometry, role colours, star structure and message ordering are inherited from
# the supplied fig_alg1_workflow.pdf, with exact single-vector feedback labels.
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

mpl.rcParams.update({
    "mathtext.fontset": "stix",
    "savefig.pad_inches": 0,
})

BLUE = "#2166AC"       # Original factor-coordinator colour.
ORANGE = "#B35806"     # Original feature-coordinator colour.
INK = "#1B2430"
MID = "#55616E"
PALE_BLUE = "#F6F9FC"
PALE_ORANGE = "#FCF9F4"
ROW_FILL = "#F4F6F8"
LINE_GREY = "#DCE2E8"


def text(ax, x, y, label, *, size=8, color=INK, ha="center", weight="normal"):
    return ax.text(x, y, label, fontsize=size, color=color, ha=ha,
                   va="center", fontweight=weight, zorder=5)


def segment(ax, xy0, xy1, color, *, lw=0.85, ls="-", alpha=1):
    return ax.plot([xy0[0], xy1[0]], [xy0[1], xy1[1]], color=color,
                   lw=lw, ls=ls, alpha=alpha, zorder=2)[0]


def arrow(ax, xy0, xy1, color, *, dashed=False):
    patch = FancyArrowPatch(xy0, xy1, arrowstyle="-|>", mutation_scale=8,
                            linewidth=0.9, color=color,
                            linestyle=(0, (2.4, 1.6)) if dashed else "-",
                            shrinkA=0, shrinkB=0, zorder=4)
    ax.add_patch(patch)


def star(ax, cx, color, backdrop, factor):
    # Geometry uses millimetres; the two stars intentionally have the same size.
    cy = 21.0
    ax.add_patch(Circle((cx, 27.5), 16.9, facecolor=backdrop,
                        edgecolor="none", zorder=0))
    leaves = [(cx - 13.5, 36.5), (cx, 39.5), (cx + 13.5, 36.5)]
    for i, xy in enumerate(leaves, 1):
        segment(ax, (cx, cy), xy, color, alpha=0.62)
        ax.add_patch(Circle(xy, 2.9, facecolor="white", edgecolor=color,
                            linewidth=0.65, zorder=3))
        label = rf"$h_{{a{i}}}$" if factor else rf"$\beta_{{{i}j}}$"
        text(ax, *xy, label, size=7.5, color=color)
    if factor:
        ax.add_patch(FancyBboxPatch((cx - 3.75, cy - 3.75), 7.5, 7.5,
                     boxstyle="round,pad=0,rounding_size=0.7",
                     linewidth=0.85, facecolor="#EFF6FC", edgecolor=color, zorder=3))
        text(ax, cx, cy, r"$h_a$", size=10, color=color)
        text(ax, cx - 10.4, cy + 7.5, r"$b_{a1}$", size=7.2, color=MID)
        text(ax, cx + 7.3, cy - 1.0, r"$a_a$", size=7.2, color=MID)
        text(ax, cx, 8.7, r"$\sum_j h_{aj}=h_a$", size=9)
    else:
        ax.add_patch(Circle((cx, cy), 4.2, facecolor="#FFF4E9",
                            edgecolor=color, linewidth=0.85, zorder=3))
        text(ax, cx, cy, r"$\beta_j$", size=10, color=color)
        text(ax, cx + 10.5, cy + 7.5, r"$d_{3j}$", size=7.2, color=MID)
        text(ax, cx + 7.4, cy - 1.0, r"$c_j$", size=7.2, color=MID)
        text(ax, cx, 8.7, r"$\beta_{aj}=\beta_j$", size=9)


def actor_node(ax, x, y, color, *, hollow=False):
    ax.add_patch(Circle((x, y), 1.8 if not hollow else 1.2,
                       facecolor="white", edgecolor=color,
                       linewidth=0.7, zorder=3))
    if not hollow:
        ax.add_patch(Circle((x, y), 1.0, facecolor=color, edgecolor="none", zorder=4))


def build_figure():
    fig = plt.figure(figsize=(183 / 25.4, 52 / 25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 183), ylim=(0, 52))
    ax.set_aspect("equal")
    ax.axis("off")

    text(ax, 2.2, 48.7, "(a)", size=9, ha="left", weight="bold")
    text(ax, 9.6, 48.7, "Star projections", size=9, ha="left")
    star(ax, 21.0, BLUE, PALE_BLUE, True)
    star(ax, 61.0, ORANGE, PALE_ORANGE, False)
    segment(ax, (83.0, 3.0), (83.0, 44.8), LINE_GREY, lw=0.6)

    text(ax, 89.0, 48.7, "(b)", size=9, ha="left", weight="bold")
    text(ax, 96.7, 48.7, "One iteration", size=9, ha="left")
    xf, xe, xb = 108.0, 135.2, 162.5
    rows = [33.4, 26.3, 19.1, 12.0, 5.2]
    for k, y in enumerate(rows, 1):
        if k in (2, 4):
            ax.add_patch(FancyBboxPatch((88.2, y - 3.3), 91.4, 6.6,
                         boxstyle="round,pad=0,rounding_size=0.7",
                         facecolor=ROW_FILL, edgecolor="none", zorder=0))
        text(ax, 90.0, y, f"({k})", size=7.2, color=MID, ha="left", weight="bold")

    for x, c in [(xf, BLUE), (xe, MID), (xb, ORANGE)]:
        segment(ax, (x, rows[0]), (x, 2.3), c, lw=0.5, ls=(0, (2.3, 2.3)), alpha=0.65)
        actor_node(ax, x, rows[0], c)

    text(ax, xf, 41.2, r"$f_a$", size=10, color=BLUE)
    text(ax, xe, 41.2, r"$(a,j)$", size=10, color=INK)
    text(ax, xb, 41.2, r"$r_j$", size=10, color=ORANGE)
    for x, c in [(xf, BLUE), (xe, MID), (xb, ORANGE)]:
        segment(ax, (x - 4.3, 38.6), (x + 4.3, 38.6), c, lw=0.55, alpha=0.75)
    text(ax, xf, 37.1, r"$\mathrm{prox}$", size=7.5, color=BLUE)
    text(ax, xe, 37.1, r"$\mathbf{M}_{aj}^{-1}$", size=7.5, color=INK)
    text(ax, xb, 37.1, r"$\mathrm{prox}$", size=7.5, color=ORANGE)

    arrow(ax, (xe - 1.7, rows[1]), (xf + 1.8, rows[1]), BLUE)
    arrow(ax, (xe + 1.7, rows[1]), (xb - 1.8, rows[1]), ORANGE)
    text(ax, (xf + xe) / 2, rows[1] + 2.5, r"$\xi_{h_{aj}}$", size=10.3, color=BLUE)
    text(ax, (xe + xb) / 2, rows[1] + 2.5, r"$\xi_{\beta_{aj}}$", size=10.3, color=ORANGE)

    for x, c in [(xf, BLUE), (xb, ORANGE)]:
        actor_node(ax, x, rows[2], c)
        text(ax, x, rows[2] + 3.2, r"$P_{\mathcal{S}}^W,\,u$", size=7.2, color=c)
    segment(ax, (xf, rows[3]), (xb, rows[3]), MID, lw=0.65, ls=(0, (2.5, 1.8)))
    actor_node(ax, xf, rows[3], MID, hollow=True)
    actor_node(ax, xb, rows[3], MID, hollow=True)
    ax.add_patch(FancyBboxPatch((xe - 8.6, rows[3] - 2.4), 17.2, 4.8,
                 boxstyle="round,pad=0,rounding_size=0.7", linewidth=0.6,
                 edgecolor="#9AA8B5", facecolor="white", zorder=3))
    text(ax, xe, rows[3], r"$G\leq\varepsilon?$", size=8, color=MID)

    # Each feedback arrow is exactly ONE combined vector, as in Algorithm 1.
    arrow(ax, (xf + 1.8, rows[4]), (xe - 1.8, rows[4]), BLUE, dashed=True)
    arrow(ax, (xb - 1.8, rows[4]), (xe + 1.8, rows[4]), ORANGE, dashed=True)
    text(ax, (xf + xe) / 2, rows[4] - 2.3, r"$q_{h_{aj}}$", size=10.3, color=BLUE)
    text(ax, (xe + xb) / 2, rows[4] - 2.3, r"$q_{\beta_{aj}}$", size=10.3, color=ORANGE)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Render Algorithm 1 workflow schematic.")
    parser.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    stem = args.outdir / "fig_alg1_workflow"
    save_cns_figure(fig, stem)
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)
    audit = {
        "kind": "conceptual mathematical schematic; no statistical data",
        "width_mm": 183, "height_mm": 52, "preview_dpi": 300,
        "panels": ["local weighted stars", "chronological message sequence"],
        "feedback_vectors_per_edge": {"factor_to_worker": 1, "feature_to_worker": 1},
        "factor_feedback": "q_haj = eta_haj + (a_a / b_aj) u_ha",
        "feature_feedback": "q_betaaj = eta_betaj - u_betaaj",
        "algorithm_iteration": "all states in feedback are at iteration k+1",
        "statistics": "not applicable: no observations, estimates, tests or uncertainty intervals",
        "original_role_colours_preserved": {"factor": BLUE, "feature": ORANGE},
        "source": "section_algorithm.tex, Algorithm 1, step 5; Proposition prop:invariant compressed dual identities",
    }
    (args.outdir / "workflow_semantic_checks.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
