"""Paper tables I-VII from the results files.

Every value is read from the CSV files the analysis scripts write and rounded
half up, as in the paper (Python's round() rounds half to even). p-values are
the Benjamini-Hochberg-adjusted ones computed by the analysis scripts.
  Tables I-IV  rq1/rq1_profile.csv         one engagement metric by outcome
  Table V      rq1/rq1_matched_pairs.csv   within-author pairs, accepted vs rejected
  Table VI     rq1/rq1_profile.csv         GoEmotions signal screening
  Table VII    rq2/rq2_windows.csv         proposals still undecided at day 30 and day 60

Usage:  python3 analysis/make_tables.py <results_dir>
Writes: paper_tables.txt in <results_dir>
"""
import csv
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import EMOTIONS, OUTCOMES, PAIR_NAMES

PAIR_LABELS = ('Acc vs Rej', 'Acc vs Post', 'Rej vs Post')
# (table, title, column, decimals for the mean, decimals for the median; None = as is)
ENGAGEMENT = (('I', 'Unique Commenters', 'unique_commenters', 1, None),
              ('II', 'Influence Score', 'influence_score', 1, None),
              ('III', 'Echo Score', 'echo_score', 3, 3),
              ('IV', 'First Commenter Degree', 'first_commenter_degree', 1, None))
MATCHED = (('Unique Commenters', 'unique_commenters', 1), ('Influence Score', 'influence_score', 1),
           ('Echo Score', 'echo_score', 3), ('First Comm. Degree', 'first_commenter_degree', 1),
           ('GTT', 'tone', 3))
WINDOW = (('Comments', 'n_comments', None), ('Unique Commenters', 'unique_commenters', None),
          ('Influence Score', 'influence_score', None), ('Echo Score', 'echo_score', 3),
          ('GTT', 'tone', 3), ('Excitement', 'excitement', 5), ('Confusion', 'confusion', 5))


def rnd(value, places, sign=False):
    """Round half up to `places` decimals; places=None prints the value as is (11, 27.5)."""
    d = Decimal(value)
    d = d.normalize() if places is None else d.quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)
    return f'{d:+f}' if sign else f'{d:f}'


def pval(value):
    return '<0.001' if float(value) < .001 else rnd(value, 3)


def text(title, header, rows):
    """Fixed-width table; a row given as a string is a section heading."""
    cells = [header] + [r for r in rows if not isinstance(r, str)]
    widths = [max(len(r[i]) for r in cells) for i in range(len(header))]
    line = lambda r: '  '.join(c.ljust(w) if i == 0 else c.rjust(w)
                               for i, (c, w) in enumerate(zip(r, widths)))
    return '\n'.join([title, line(header)] + [r if isinstance(r, str) else line(r) for r in rows])


def engagement_table(row, mean_places, median_places):
    """Tables I-IV: mean and median per outcome next to the three pairwise tests."""
    return [[o.capitalize(), rnd(row[f'mean_{o}'], mean_places), rnd(row[f'median_{o}'], median_places),
             label, pval(row[f'p_bh_{pair}']), rnd(row[f'r_{pair}'], 3)]
            for o, pair, label in zip(OUTCOMES, PAIR_NAMES, PAIR_LABELS)]


def matched_table(matched):
    """Table V: within-author Wilcoxon signed-rank tests, accepted vs rejected."""
    rows = []
    for design, heading in (('author_median', 'Author medians:'),
                            ('time_matched', 'Time-matched pairs (median gap: rq1_matched_pairs_summary.txt):')):
        rows.append(heading)
        for label, metric, places in MATCHED:
            r = matched[design, 'accepted_vs_rejected', metric]
            rows.append([label, rnd(r['median_x'], places), rnd(r['median_y'], places),
                         rnd(r['median_diff'], places, sign=True), rnd(r['W'], 1),
                         pval(r['p_bh']), rnd(r['r'], 3, sign=True)])
    return rows


def emotion_table(profile):
    """Table VI: acceptance signals (BH p < .05 and r >= .10 for Acc vs Rej), then
    governance debt signals (the same for Acc vs Post), each ranked by that r."""
    def tier(pair, exclude=()):
        hits = [e for e in EMOTIONS if e not in exclude and float(profile[e][f'p_bh_{pair}']) < .05
                and float(profile[e][f'r_{pair}']) >= .10]
        hits.sort(key=lambda e: -float(profile[e][f'r_{pair}']))
        top = max(float(profile[e][f'p_bh_{pair}']) for e in hits)
        bound = next(b for b in ('0.001', '0.01', '0.05') if top < float(b))
        return hits, bound

    acceptance, acc_bound = tier('acc_vs_rej')
    debt, debt_bound = tier('acc_vs_post', exclude=acceptance)
    none = [e for e in EMOTIONS if all(float(profile[e][f'p_bh_{p}']) >= .05 for p in PAIR_NAMES)]
    other = [e for e in EMOTIONS if e not in acceptance + debt + none]

    rows = []
    for heading, emotions, pair in ((f'Acceptance signals - ranked by Acc vs Rej r (all p<{acc_bound}):',
                                     acceptance, 'acc_vs_rej'),
                                    (f'Governance debt signals - ranked by Acc vs Post r (all p<{debt_bound}):',
                                     debt, 'acc_vs_post')):
        rows.append(heading)
        rows += [[e] + [rnd(Decimal(profile[e][f'median_{o}']) * 100, 3) for o in OUTCOMES]
                 + [rnd(profile[e][f'r_{pair}'], 3)] for e in emotions]
    rows += ['Not significant in any comparison:', '  ' + ', '.join(none),
             'Significant in some comparison, but in neither tier (not listed in the paper):',
             '  ' + ', '.join(other)]
    return rows


def window_table(windows):
    """Table VII: proposals still undecided when the window ends; * marks BH p < .05
    within the window (bold in the paper)."""
    rows = []
    for days in ('30', '60'):
        n = windows[days, 'n_comments']
        rows.append(f"Day {days} (n = {n['n_accepted']}/{n['n_rejected']}/{n['n_postponed']}):")
        for label, feature, places in WINDOW:
            r = windows[days, feature]
            rows.append([label] + [rnd(r[f'median_{o}'], places) for o in OUTCOMES]
                        + [rnd(r[f'r_{p}'], 3) + ('*' if float(r[f'p_bh_{p}']) < .05 else ' ')
                           for p in PAIR_NAMES])
    return rows


def read(results, *path):
    return list(csv.DictReader(open(os.path.join(results, *path))))


def main(results):
    profile = {r['feature']: r for r in read(results, 'rq1', 'rq1_profile.csv')}
    matched = {(r['design'], r['comparison'], r['metric']): r
               for r in read(results, 'rq1', 'rq1_matched_pairs.csv')}
    windows = {(r['window_days'], r['feature']): r
               for r in read(results, 'rq2', 'rq2_windows.csv') if r['population'] == 'AT_RISK'}

    blocks = [text(f'TABLE {number}. {name} across outcome groups (Mann-Whitney U, BH-adjusted p)',
                   ['Group', 'Mean', 'Median', 'Comparison', 'p-value', 'r'],
                   engagement_table(profile[column], mean_places, median_places))
              for number, name, column, mean_places, median_places in ENGAGEMENT]
    n_authors = matched['author_median', 'accepted_vs_rejected', 'tone']['n_authors']
    blocks.append(text(f'TABLE V. Within-author matched pairs (Wilcoxon signed-rank, n = {n_authors} '
                       f'authors, BH-adjusted p)',
                       ['Metric', 'Acc', 'Rej', 'Diff', 'W', 'p-value', 'r'], matched_table(matched)))
    blocks.append(text('TABLE VI. GoEmotions signal screening (medians of per-proposal mean '
                       'probabilities, in percent; BH-adjusted p)',
                       ['Emotion', 'Acc Mdn', 'Rej Mdn', 'Post Mdn', 'r'], emotion_table(profile)))
    blocks.append(text('TABLE VII. Proposals still undecided at day 30 and day 60 '
                       '(* = BH-adjusted p < 0.05 within each window; bold in the paper)',
                       ['Metric', 'Acc Mdn', 'Rej Mdn', 'Pst Mdn', 'A/R r', 'A/P r', 'R/P r'],
                       window_table(windows)))

    out = '\n\n'.join(blocks) + '\n'
    with open(os.path.join(results, 'paper_tables.txt'), 'w') as fh:
        fh.write(out)
    print(out)


if __name__ == '__main__':
    main(sys.argv[1])
