"""Shared corpus loading, measures, and statistics for the RQ1 and RQ2 scripts.

Outcome labels: merged -> accepted; closed with the `postponed` label ->
postponed; other closed -> rejected. Excluded: PRs labelled not-rfc or
april-1st, the three disposition-mismatch PRs (3035, 3469, 3505), withdrawn
PRs, and accepted RFCs with no PR creation or merge date. PR 2071 carries two
accepted RFC numbers and is counted once.

All measures use the whole discussion of a proposal (every human comment on
its PR); RQ2 restricts this to the first N days after submission.

Prospective degree: distinct RFCs a person had commented on, by comment
timestamp, before the focal RFC was opened. The governance core for the Echo
Score is the k highest-degree participants at that moment, with k set to the
median Unique Commenters count over all proposals.
"""
import collections
import csv
import heapq
import math
import os
import statistics
from datetime import datetime

csv.field_size_limit(10**9)

BOTS = {'rfcbot', 'rust-highfive', 'bors', 'bors-servo', 'rust-rfcbot',
        'rustbot', 'renovate[bot]'}
AMBIGUOUS = {'3035', '3469', '3505'}
OUTCOMES = ('accepted', 'rejected', 'postponed')
PAIRS = (('accepted', 'rejected'), ('accepted', 'postponed'), ('rejected', 'postponed'))
PAIR_NAMES = ('acc_vs_rej', 'acc_vs_post', 'rej_vs_post')
EMOTIONS = ('admiration amusement anger annoyance approval caring confusion '
            'curiosity desire disappointment disapproval disgust embarrassment '
            'excitement fear gratitude grief joy love nervousness optimism pride '
            'realization relief remorse sadness surprise neutral').split()


def _ts(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')


def _data(root, name):
    """Resolve an input CSV in either a flat folder or the original RQ1/RQ2 layout."""
    for sub in ('', 'RQ1', 'RQ2'):
        path = os.path.join(root, sub, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{name} not found under {root}')


class Corpus:
    def __init__(self, root):
        self.excluded = collections.Counter()
        self.rfc = {}  # pr -> dict(outcome, created, decided, author)

        acc_by_rfcnum = {r['rfc_number']: r for r in
                         csv.DictReader(open(_data(root, 'rfc_accepted.csv')))}
        self.excluded['accepted without PR dates'] = sum(
            1 for a in acc_by_rfcnum.values() if not (a['pr_created_at'] and a['pr_merged_at']))

        for r in csv.DictReader(open(_data(root, 'rfc_rejected.csv'))):
            if r['outcome'] not in ('rejected', 'postponed'):
                self.excluded[r['outcome']] += 1
                continue
            labels = r['pr_labels'] or ''
            if 'not-rfc' in labels or 'april-1st' in labels:
                self.excluded['not-rfc / april-1st'] += 1
                continue
            if r['pr_number'] in AMBIGUOUS:
                self.excluded['disposition mismatch'] += 1
                continue
            self.rfc[r['pr_number']] = dict(outcome=r['outcome'], created=_ts(r['pr_created_at']),
                                            decided=_ts(r['pr_closed_at']),
                                            author=r['pr_author_login'])

        comments = []  # (time, pr, author, comment_id, body)
        seen_ids = set()  # PR 2071 (two RFC numbers) appears twice in the comment file
        for r in csv.DictReader(open(_data(root, 'rfc_comments_full.csv'))):
            if r['comment_id'] in seen_ids:
                continue
            seen_ids.add(r['comment_id'])
            pr = r['pr_number']
            if r['outcome'] == 'accepted' and pr not in self.rfc:
                a = acc_by_rfcnum.get(r['rfc_number'])
                if a and a['pr_created_at'] and a['pr_merged_at']:
                    self.rfc[pr] = dict(outcome='accepted', created=_ts(a['pr_created_at']),
                                        decided=_ts(a['pr_merged_at']),
                                        author=a['pr_author_login'])
            comments.append((_ts(r['comment_created_at']), pr, r['comment_author'],
                             r['comment_id'], r['comment_body'] or ''))
        comments.sort()

        self.tone = {r['comment_id']: float(r['rob_score'])
                     for r in csv.DictReader(open(_data(root, 'rfc_roberta_scores.csv')))}
        self.emo = {r['comment_id']: tuple(float(r['emo_' + e]) for e in EMOTIONS)
                    for r in csv.DictReader(open(_data(root, 'rfc_goemotions_scores.csv')))}
        # Optional robustness measure: tone from a three-class model with a neutral class.
        try:
            path3 = _data(root, 'rfc_sentiment_3class_scores.csv')
            self.tone3 = {r['comment_id']: float(r['rob_score']) for r in csv.DictReader(open(path3))}
        except FileNotFoundError:
            self.tone3 = {}
        self.tone_features = ['tone'] + (['tone_3class'] if self.tone3 else [])

        human = [c for c in comments if c[2] not in BOTS]
        self.by_pr = collections.defaultdict(list)
        for c in human:
            if c[1] in self.rfc:
                self.by_pr[c[1]].append(c)

        # Echo Score core size: median Unique Commenters over all proposals
        self.core_size = int(round(statistics.median(
            len({c[2] for c in self.by_pr.get(pr, [])}) for pr in self.rfc)))

        seen = collections.defaultdict(set)
        self.deg_at, self.core_at = {}, {}
        i = 0
        for pr in sorted(self.rfc, key=lambda p: self.rfc[p]['created']):
            t0 = self.rfc[pr]['created']
            while i < len(human) and human[i][0] < t0:
                seen[human[i][2]].add(human[i][1])
                i += 1
            self.deg_at[pr] = {p: len(s) for p, s in seen.items()}
            self.core_at[pr] = {p for p, _ in heapq.nlargest(
                self.core_size, self.deg_at[pr].items(), key=lambda kv: kv[1])}

    def outcome(self, pr):
        return self.rfc[pr]['outcome']

    def decision_days(self, pr):
        r = self.rfc[pr]
        return (r['decided'] - r['created']).total_seconds() / 86400

    def features(self, pr, days=None):
        """Engagement and affect for one RFC: over its whole discussion, or
        over the comments posted in the first `days` days after submission."""
        window = self.by_pr.get(pr, [])
        if days is not None:
            stop = self.rfc[pr]['created'].timestamp() + days * 86400
            window = [c for c in window if c[0].timestamp() < stop]
        people = {c[2] for c in window}
        deg = self.deg_at[pr]
        f = dict(n_comments=len(window), unique_commenters=len(people),
                 influence_score=sum(deg.get(p, 0) for p in people),
                 echo_score=(len(people & self.core_at[pr]) / len(people)) if people else 0.0)
        if window:
            f['first_commenter_degree'] = deg.get(window[0][2], 0)
        tones = [self.tone[c[3]] for c in window if c[3] in self.tone]
        if tones:
            f['tone'] = sum(tones) / len(tones)
        tones3 = [self.tone3[c[3]] for c in window if c[3] in self.tone3]
        if tones3:
            f['tone_3class'] = sum(tones3) / len(tones3)
        emos = [self.emo[c[3]] for c in window if c[3] in self.emo]
        if emos:
            for k, e in enumerate(EMOTIONS):
                f[e] = sum(x[k] for x in emos) / len(emos)
        return f


# ---- statistics -------------------------------------------------------------

def mann_whitney(a, b):
    """Two-sided Mann-Whitney U, normal approximation with tie correction.

    Returns (rank-biserial r, p).
    """
    values = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    n, i, ties, rank_a = len(values), 0, 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1][0] == values[i][0]:
            j += 1
        avg, t = (i + j) / 2 + 1, j - i + 1
        ties += t ** 3 - t
        rank_a += avg * sum(1 for k in range(i, j + 1) if values[k][1] == 0)
        i = j + 1
    n1, n2 = len(a), len(b)
    u = rank_a - n1 * (n1 + 1) / 2
    sd = math.sqrt(n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1))))
    z = (u - n1 * n2 / 2) / sd if sd else 0.0
    return abs(2 * u / (n1 * n2) - 1), math.erfc(abs(z) / math.sqrt(2))


def wilcoxon_signed_rank(diffs):
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are dropped; normal approximation with tie correction
    (matches scipy.stats.wilcoxon defaults for n > 50). Returns
    (W = min(W+, W-), p, matched-pairs rank-biserial r, n non-zero pairs).
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return float('nan'), float('nan'), float('nan'), 0
    values = sorted((abs(x), x > 0) for x in d)
    i, ties, w_plus = 0, 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and values[j + 1][0] == values[i][0]:
            j += 1
        avg, t = (i + j) / 2 + 1, j - i + 1
        ties += t ** 3 - t
        w_plus += avg * sum(1 for k in range(i, j + 1) if values[k][1])
        i = j + 1
    total = n * (n + 1) / 2
    w_minus = total - w_plus
    sd = math.sqrt((n * (n + 1) * (2 * n + 1) - ties / 2) / 24)
    z = (w_plus - total / 2) / sd if sd else 0.0
    return min(w_plus, w_minus), math.erfc(abs(z) / math.sqrt(2)), (w_plus - w_minus) / total, n


def benjamini_hochberg(pvalues):
    m = len(pvalues)
    order = sorted(range(m), key=lambda k: pvalues[k])
    adjusted, running = [0.0] * m, 1.0
    for rank in range(m, 0, -1):
        k = order[rank - 1]
        running = min(running, pvalues[k] * m / rank)
        adjusted[k] = running
    return adjusted


def median(x):
    return statistics.median(x) if x else float('nan')


def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'


def compare(groups, feature_names):
    """Rows of descriptives + pairwise tests; BH-adjusted across the whole table."""
    rows = []
    for name in feature_names:
        g = {o: groups[o][name] for o in OUTCOMES}
        row = dict(feature=name)
        for o in OUTCOMES:
            row[f'n_{o}'] = len(g[o])
            row[f'median_{o}'] = median(g[o])
            row[f'mean_{o}'] = statistics.fmean(g[o]) if g[o] else float('nan')
        for (x, y), pair in zip(PAIRS, PAIR_NAMES):
            row[f'r_{pair}'], row[f'p_{pair}'] = mann_whitney(g[x], g[y])
        rows.append(row)
    flat = [row[f'p_{pair}'] for row in rows for pair in PAIR_NAMES]
    adjusted = iter(benjamini_hochberg(flat))
    for row in rows:
        for pair in PAIR_NAMES:
            row[f'p_bh_{pair}'] = next(adjusted)
    return rows


def group_features(feature_by_pr, outcome_of, feature_names):
    groups = {o: {name: [] for name in feature_names} for o in OUTCOMES}
    for pr, f in feature_by_pr.items():
        for name in feature_names:
            if name in f:
                groups[outcome_of(pr)][name].append(f[name])
    return groups


def fmt(v):
    if isinstance(v, float):
        return '' if math.isnan(v) else f'{v:.6g}'
    return v


def write_csv(path, rows, extra=None):
    rows = [dict(extra or {}, **row) for row in rows]
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})


def text_table(rows):
    """Compact fixed-width table: medians and r with raw-p stars per pair."""
    lines = [f"{'feature':24s} {'Mdn A':>9s} {'Mdn R':>9s} {'Mdn P':>9s}   "
             f"{'A-R':14s} {'A-P':14s} {'R-P':14s}"]
    for row in rows:
        cells = '  '.join(f"{row[f'r_{p}']:.3f} {star(row[f'p_{p}']):3s} " for p in PAIR_NAMES)
        lines.append(f"{row['feature']:24s} {row['median_accepted']:9.4f} "
                     f"{row['median_rejected']:9.4f} {row['median_postponed']:9.4f}   {cells}")
    return '\n'.join(lines)
