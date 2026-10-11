"""RQ2: is the postponement signature visible early in deliberation?

Windows: all comments posted in the first 30 and 60 days after the PR was opened.
  ALL       every RFC
  AT_RISK   only RFCs still undecided when the window ends (early-warning test)
Same cohort: RFCs still undecided at day 60, measured at day 30 and at day 60,
to separate change over time from change in which RFCs remain open.

Usage:  python3 analysis/rq2_early_window.py <data_dir> <out_dir>
Writes:
  rq2_windows.csv       window x population x feature, descriptives and tests
  rq2_same_cohort.csv   the same-cohort comparison
  rq2_rfc_features.csv  one row per RFC still undecided at each window (for figures)
  rq2_summary.txt       readable summary
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import Corpus, OUTCOMES, compare, group_features, text_table, write_csv

WINDOWS = (30, 60)
ENGAGEMENT = ['n_comments', 'unique_commenters', 'influence_score', 'echo_score']
AFFECT = ['confusion', 'excitement']


def block(c, prs, days):
    names = ENGAGEMENT + c.tone_features + AFFECT
    feats = {p: c.features(p, days=days) for p in prs}
    return compare(group_features(feats, c.outcome, names), names)


def counts(c, prs):
    return '/'.join(str(sum(1 for p in prs if c.outcome(p) == o)) for o in OUTCOMES)


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(data_dir)
    prs = sorted(c.rfc, key=int)
    out = ['RQ2 early window | groups shown as accepted/rejected/postponed', '']

    window_rows = []
    for days in WINDOWS:
        for population in ('ALL', 'AT_RISK'):
            members = [p for p in prs if population == 'ALL' or c.decision_days(p) > days]
            rows = block(c, members, days)
            window_rows += [dict(window_days=days, population=population, **r) for r in rows]
            out += [f'=== first {days} days | {population} | n = {counts(c, members)} ===',
                    text_table(rows), '']
    write_csv(os.path.join(out_dir, 'rq2_windows.csv'), window_rows)

    per_rfc = []
    for days in WINDOWS:
        for p in prs:
            if c.decision_days(p) > days:
                per_rfc.append(dict(window_days=days, pr_number=p, outcome=c.outcome(p),
                                    **c.features(p, days=days)))
    write_csv(os.path.join(out_dir, 'rq2_rfc_features.csv'), per_rfc)

    cohort = [p for p in prs if c.decision_days(p) > 60]
    cohort_rows = []
    out.append(f'##### SAME COHORT: RFCs still undecided at day 60 | n = {counts(c, cohort)} #####')
    for days in WINDOWS:
        rows = block(c, cohort, days)
        cohort_rows += [dict(measured_at_day=days, **r) for r in rows]
        out += [f'--- measured at day {days} ---', text_table(rows), '']
    write_csv(os.path.join(out_dir, 'rq2_same_cohort.csv'), cohort_rows)

    out.append('Stars use raw p; BH-adjusted p-values (per block) are in the CSV files.')
    with open(os.path.join(out_dir, 'rq2_summary.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
