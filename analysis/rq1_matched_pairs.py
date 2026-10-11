"""RQ1 robustness: within-author matched pairs (author identity held constant).

Authors who submitted proposals with both outcomes in a comparison are paired
with themselves; a Wilcoxon signed-rank test is run on the per-author
differences. Two designs:
  author_median   the author's median over all proposals with each outcome
                  (the design of the original study)
  time_matched    one pair per author: the two proposals, one per outcome,
                  submitted closest in time, so that the author's standing
                  and experience are as similar as possible
Measures use the whole discussion of each proposal.
Comparisons: accepted vs rejected (primary); accepted vs postponed and
rejected vs postponed (exploratory, fewer authors).

Usage:  python3 analysis/rq1_matched_pairs.py <data_dir> <out_dir>
Writes: rq1_matched_pairs.csv, rq1_matched_pairs_summary.txt
"""
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import Corpus, PAIRS, benjamini_hochberg, star, wilcoxon_signed_rank, write_csv

ENGAGEMENT = ['n_comments', 'unique_commenters', 'influence_score', 'echo_score',
              'first_commenter_degree']
AFFECT = ['confusion', 'excitement']


def author_pairs(c, x, y, design):
    """Return [(author, [prs with outcome x], [prs with outcome y])] per design."""
    by_author = collections.defaultdict(lambda: collections.defaultdict(list))
    for pr, r in c.rfc.items():
        if r['author']:
            by_author[r['author']][r['outcome']].append(pr)
    pairs = []
    for author in sorted(by_author):
        xs, ys = by_author[author][x], by_author[author][y]
        if not (xs and ys):
            continue
        if design == 'time_matched':
            gap = lambda pq: abs((c.rfc[pq[0]]['created'] - c.rfc[pq[1]]['created']).total_seconds())
            px, py = min(((p, q) for p in xs for q in ys), key=lambda pq: (gap(pq), pq))
            xs, ys = [px], [py]
        pairs.append((author, xs, ys))
    return pairs


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(data_dir)
    metrics = ENGAGEMENT + c.tone_features + AFFECT
    feats = {p: c.features(p) for p in c.rfc}

    rows, out = [], ['RQ1 within-author matched pairs (Wilcoxon signed-rank)', '']
    for design in ('author_median', 'time_matched'):
        for x, y in PAIRS:
            pairs = author_pairs(c, x, y, design)
            block = []
            for m in metrics:
                xv, yv = [], []
                for _, xs, ys in pairs:
                    a = [feats[p][m] for p in xs if m in feats[p]]
                    b = [feats[p][m] for p in ys if m in feats[p]]
                    if a and b:
                        xv.append(statistics.median(a))
                        yv.append(statistics.median(b))
                diffs = [a - b for a, b in zip(xv, yv)]
                W, p, r, n_nonzero = wilcoxon_signed_rank(diffs)
                block.append(dict(design=design, comparison=f'{x}_vs_{y}',
                                  metric=m, n_authors=len(diffs), n_nonzero=n_nonzero,
                                  median_x=statistics.median(xv), median_y=statistics.median(yv),
                                  median_diff=statistics.median(diffs), W=W, p=p, r=r))
            for row, adj in zip(block, benjamini_hochberg([b['p'] for b in block])):
                row['p_bh'] = adj
            rows += block

            gap = ''
            if design == 'time_matched':
                gap_days = [abs((c.rfc[xs[0]]['created'] - c.rfc[ys[0]]['created']).days)
                            for _, xs, ys in pairs]
                gap = f' | median gap between paired proposals: {statistics.median(gap_days):.0f} days'
            out.append(f'=== {design} | {x} vs {y} | authors = {len(pairs)}{gap} ===')
            out.append(f"{'metric':24s} {'Mdn ' + x[:3]:>10s} {'Mdn ' + y[:4]:>10s} "
                       f"{'Mdn diff':>10s} {'W':>9s} {'p':>8s} {'r':>7s}  sig  BH")
            for b in block:
                out.append(f"{b['metric']:24s} {b['median_x']:10.4f} {b['median_y']:10.4f} "
                           f"{b['median_diff']:+10.4f} {b['W']:9.1f} {b['p']:8.4f} {b['r']:+7.3f}  "
                           f"{star(b['p']):3s}  {star(b['p_bh'])}")
            out.append('')

    out.append('r is the matched-pairs rank-biserial correlation (positive: higher for the first '
               'outcome). sig uses raw p; BH is adjusted within each block of metrics.')
    write_csv(os.path.join(out_dir, 'rq1_matched_pairs.csv'), rows)
    with open(os.path.join(out_dir, 'rq1_matched_pairs_summary.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
