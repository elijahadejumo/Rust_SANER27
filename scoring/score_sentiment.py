"""Score every comment's sentiment with a Hugging Face sequence classifier.

Text handling follows the original scoring: empty comments are scored as the
string "neutral", and comments are truncated to 512 tokens. Only rfcbot posts
are skipped, as in data/rfc_roberta_scores.csv; the analysis scripts remove
the remaining bots.

Output columns: comment metadata, one rob_<label> probability per model class,
rob_label (argmax), and rob_score = P(positive) - P(negative), in [-1, 1].

Usage:
  python score_sentiment.py --model siebert/sentiment-roberta-large-english \
      --comments ../data/rfc_comments_full.csv --out ../data/rfc_roberta_scores.csv
  # check a random sample against an existing scores file:
  python score_sentiment.py --model ... --comments ... --out sample.csv \
      --sample 300 --sample-from ../data/rfc_roberta_scores.csv

Requires: torch, transformers.
"""
import argparse
import csv
import random
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

csv.field_size_limit(10**9)
META = ['pr_number', 'rfc_number', 'outcome', 'comment_id', 'comment_author',
        'comment_order', 'comment_created_at']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--comments', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--sample', type=int, default=0, help='score a random sample of N comments')
    ap.add_argument('--sample-from', help='draw the sample from comment_ids in this CSV')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.comments)) if r['comment_author'] != 'rfcbot']
    if args.sample:
        pool = rows
        if args.sample_from:
            ids = {r['comment_id'] for r in csv.DictReader(open(args.sample_from))}
            pool = [r for r in rows if r['comment_id'] in ids]
        rows = random.Random(args.seed).sample(pool, args.sample)

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    lower = [l.lower() for l in labels]
    neg, pos = lower.index('negative'), lower.index('positive')
    print(f'{args.model} on {device}: labels {labels}, {len(rows):,} comments', file=sys.stderr)

    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(META + [f'rob_{l}' for l in lower] + ['rob_label', 'rob_score'])
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            texts = [r['comment_body'] if (r['comment_body'] or '').strip() else 'neutral' for r in batch]
            enc = tokenizer(texts, return_tensors='pt', truncation=True, max_length=512,
                            padding=True).to(device)
            with torch.no_grad():
                probs = F.softmax(model(**enc).logits, dim=-1).cpu().tolist()
            for r, p in zip(batch, probs):
                best = max(range(len(p)), key=p.__getitem__)
                w.writerow([r[k] for k in META] + [round(x, 4) for x in p]
                           + [labels[best], round(p[pos] - p[neg], 4)])
            if start // args.batch_size % 50 == 0:
                print(f'  {start + len(batch):,}/{len(rows):,}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
