"""Paper figures from the results files.

Both figures use the same profile layout: for each measure, every proposal's
percentile rank among the proposals shown, averaged per outcome group, with 95%
bootstrap intervals. Rank-based positions match the Mann-Whitney comparisons.
  Figure 1 (outcome profile): all proposals, whole discussions.
  Figure 2 (early window): proposals still undecided 30 days after submission,
                           measured over their first 30 days of comments.

Usage:  python3 analysis/make_figures.py <results_dir> <out_dir>
Requires: matplotlib. Writes fig_outcome_profile.pdf/.png and fig_early_window.pdf/.png.
"""
import csv
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Colors: validated categorical slots 1-3 (CVD-safe all-pairs on white); shapes add a
# second cue for grayscale print. Ink and hairlines are text/chrome tokens.
GROUPS = [('accepted', '#2a78d6', 'o'), ('rejected', '#eb6834', 's'), ('postponed', '#1baf7a', '^')]
INK2, MUTED, GRID, AXIS = '#52514e', '#898781', '#e1e0d9', '#c3c2b7'

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7.5, 'axes.edgecolor': AXIS, 'axes.linewidth': 0.6, 'axes.labelcolor': INK2,
    'xtick.color': MUTED, 'ytick.color': '#0b0b0b', 'xtick.major.size': 0, 'ytick.major.size': 0,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
})


def percentile_ranks(values):
    """Mid-rank percentile (0-100) of each value among all values."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = ((i + j) / 2 + 0.5) / len(values) * 100
        i = j + 1
    return ranks


def profile(rows, sections, legend_labels, xlabel, path, height):
    """Dot plot of mean percentile rank per outcome group, one row per measure.

    sections: [(section title, [(row label, column), ...]), ...]
    """
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(3.5, height))
    y, ticks, labels, heads = 0.0, [], [], []
    for title, measures in reversed(sections):
        for label, _ in reversed(measures):
            y += 1
            ticks.append(y)
            labels.append(label)
        heads.append((title, y))
        y += 0.45
    ticks, labels = ticks[::-1], labels[::-1]
    columns = [col for _, measures in sections for _, col in measures]
    dodge = {'accepted': 0.2, 'rejected': 0.0, 'postponed': -0.2}
    for col, yy in zip(columns, ticks):
        present = [r for r in rows if r.get(col) not in ('', None)]
        pct = percentile_ranks([float(r[col]) for r in present])
        for key, color, marker in GROUPS:
            vals = [p for p, r in zip(pct, present) if r['outcome'] == key]
            mean = sum(vals) / len(vals)
            boots = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(2000))
            ax.plot([boots[49], boots[1949]], [yy + dodge[key]] * 2, color=color, linewidth=1.0,
                    solid_capstyle='round', zorder=2)
            ax.plot(mean, yy + dodge[key], marker=marker, markersize=5, color=color,
                    markeredgecolor='white', markeredgewidth=0.8, linestyle='none', zorder=3)
    ax.axvline(50, color=AXIS, linewidth=0.8, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_xlim(25, 75)
    ax.set_xticks([30, 40, 50, 60, 70])
    ax.set_ylim(0.5, y + 0.15)
    ax.set_xlabel(xlabel)
    for title, top in heads:
        ax.text(25.3, top + 0.42, title, color=MUTED, fontsize=6.5, style='italic', va='bottom')
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.grid(axis='x', color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=m, color=c, markeredgecolor='white', markeredgewidth=0.8,
                      markersize=5, linewidth=1.0, label=lab)
               for (_, c, m), lab in zip(GROUPS, legend_labels)]
    fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.6, 1.0), handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.17 / height))
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', bbox_inches='tight')
    plt.close(fig)


def main(results, out):
    os.makedirs(out, exist_ok=True)
    rq1 = list(csv.DictReader(open(os.path.join(results, 'rq1', 'rq1_rfc_features.csv'))))
    profile(rq1,
            [('Engagement', [('Unique Commenters', 'unique_commenters'), ('Influence Score', 'influence_score'),
                             ('Echo Score', 'echo_score'), ('First Commenter Degree', 'first_commenter_degree')]),
             ('Affect', [('GTT (tone)', 'tone'), ('Excitement', 'excitement'), ('Confusion', 'confusion')])],
            ['Accepted', 'Rejected', 'Postponed'],
            'Mean percentile rank among all proposals (50 = middle)',
            os.path.join(out, 'fig_outcome_profile'), height=2.75)

    rq2 = [r for r in csv.DictReader(open(os.path.join(results, 'rq2', 'rq2_rfc_features.csv')))
           if r['window_days'] == '30']
    profile(rq2,
            [('Engagement', [('Comments', 'n_comments'), ('Influence Score', 'influence_score')]),
             ('Affect', [('GTT (tone)', 'tone'), ('Confusion', 'confusion')])],
            ['Later accepted', 'Later rejected', 'Later postponed'],
            'Mean percentile rank after 30 days (50 = middle)',
            os.path.join(out, 'fig_early_window'), height=1.95)
    print('figures written to', out)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
