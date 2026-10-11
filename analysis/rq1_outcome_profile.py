"""RQ1: how engagement and affect differ across accepted, rejected, and postponed RFCs.

All measures use the whole discussion of each proposal.

Usage:  python3 analysis/rq1_outcome_profile.py <data_dir> <out_dir>
Writes:
  rq1_review_duration.csv        Review Duration (days from PR opening to decision), by outcome
  rq1_profile.csv                engagement, tone, 28 emotions
  rq1_rfc_features.csv           one row per RFC (for figures)
  rq1_summary.txt                readable summary
"""
import collections
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corpus import (Corpus, EMOTIONS, OUTCOMES, PAIR_NAMES, compare, group_features,
                    text_table, write_csv)

ENGAGEMENT = ['n_comments', 'unique_commenters', 'influence_score', 'echo_score',
              'first_commenter_degree']


def core_concentration(c):
    """Share of participants and participations held by the k most connected people overall."""
    degree = collections.Counter(p for pr in c.rfc for p in {x[2] for x in c.by_pr.get(pr, [])})
    top = [p for p, _ in degree.most_common(c.core_size)]
    return (f'Top-{c.core_size} participants by overall degree: {len(top) / len(degree):.2%} of '
            f'{len(degree)} participants, {sum(degree[p] for p in top) / sum(degree.values()):.1%} '
            f'of participations (person-proposal pairs)')


def main(data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    c = Corpus(data_dir)
    features = ENGAGEMENT + c.tone_features + list(EMOTIONS)
    prs = sorted(c.rfc, key=int)
    counts = {o: sum(1 for p in prs if c.outcome(p) == o) for o in OUTCOMES}
    out = [f'RQ1 outcome profile | n accepted/rejected/postponed = '
           f"{counts['accepted']}/{counts['rejected']}/{counts['postponed']}",
           f'excluded: {dict(c.excluded)}',
           f'Echo Score core size k (median Unique Commenters): {c.core_size}',
           core_concentration(c), '']

    # Review Duration
    days = group_features({p: {'review_duration_days': c.decision_days(p)} for p in prs},
                          c.outcome, ['review_duration_days'])
    rows = compare(days, ['review_duration_days'])
    for o in OUTCOMES:
        q1, _, q3 = statistics.quantiles(days[o]['review_duration_days'], n=4, method='inclusive')
        rows[0][f'q1_{o}'], rows[0][f'q3_{o}'], rows[0][f'iqr_{o}'] = q1, q3, q3 - q1
    write_csv(os.path.join(out_dir, 'rq1_review_duration.csv'), rows)
    out += ['Review Duration (days):',
            '  ' + '  '.join(f"{o}: median {rows[0][f'median_{o}']:.1f}, mean {rows[0][f'mean_{o}']:.1f}, "
                             f"IQR {rows[0][f'iqr_{o}']:.1f}" for o in OUTCOMES), '']

    feats = {p: c.features(p) for p in prs}
    per_rfc = [dict(pr_number=p, outcome=c.outcome(p),
                    review_duration_days=c.decision_days(p), **feats[p]) for p in prs]
    write_csv(os.path.join(out_dir, 'rq1_rfc_features.csv'), per_rfc)
    profile = compare(group_features(feats, c.outcome, features), features)
    write_csv(os.path.join(out_dir, 'rq1_profile.csv'), profile)

    out += ['=== Engagement metrics and tone ===',
            text_table([r for r in profile if r['feature'] in ENGAGEMENT + c.tone_features]), '',
            '=== Emotions with any BH-adjusted p < .05 ===',
            text_table([r for r in profile if r['feature'] in EMOTIONS
                        and min(r[f'p_bh_{p}'] for p in PAIR_NAMES) < .05]), '',
            'Stars use raw p; BH-adjusted p-values are in the CSV files.']

    with open(os.path.join(out_dir, 'rq1_summary.txt'), 'w') as fh:
        fh.write('\n'.join(out) + '\n')
    print('\n'.join(out))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
