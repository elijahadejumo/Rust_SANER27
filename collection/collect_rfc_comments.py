#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_rfc_comments.py
────────────────────────────────────────────────────────────────────────────────
Collects the full text of every comment on every RFC pull request.
This is the data collection step for RQ2 (affect and emotion analysis).

OUTPUT
  rfc_comments_full.csv     one row per comment
  rfc_comments_checkpoint.json  resumable progress

SCHEMA (one row per comment)
  pr_number           GitHub PR number
  rfc_number          RFC number (same as PR for accepted)
  outcome             accepted / rejected / postponed / withdrawn
  comment_id          GitHub comment ID (unique)
  comment_author      GitHub login of commenter
  comment_order       position in thread (1 = first comment)
  comment_created_at  ISO timestamp
  comment_body        full text of comment

USAGE
  export GITHUB_TOKEN=ghp_your_token_here
  python collect_rfc_comments.py \\
      --accepted rfc_features.csv \\
      --rejected rfc_rejected.csv \\
      --output   rfc_comments_full.csv

  Resume interrupted run (checkpoint auto-detected):
  python collect_rfc_comments.py \\
      --accepted rfc_features.csv \\
      --rejected rfc_rejected.csv \\
      --output   rfc_comments_full.csv

ESTIMATED TIME
  ~59,741 total comments across 1,543 RFCs
  ~2,140 API calls (most RFCs fit on one page of 100)
  At 5,000 req/hr authenticated = ~26 minutes

REQUIREMENTS
  Python 3.10+  pip install requests
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("ERROR: pip install requests")


# ── unbuffered output ─────────────────────────────────────────────────────────
def log(msg: str = "", end: str = "\n") -> None:
    print(msg, end=end, flush=True)


# ── constants ─────────────────────────────────────────────────────────────────
BASE_URL    = "https://api.github.com"
REPO        = "rust-lang/rfcs"
BACKOFF_MAX = 120
PER_PAGE    = 100

# Bots — we still collect their comments for completeness
# (RQ2 will filter them out during analysis)
# rfcbot comments contain governance signals (FCP started etc)
# so we keep them and let the analysis decide what to do


# ── progress ──────────────────────────────────────────────────────────────────
def progress_bar(current: int, total: int, width: int = 25) -> str:
    filled = int(width * current / total) if total else 0
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / total if total else 0
    return f"[{bar}] {current}/{total} ({pct:.0f}%)"


def eta_str(elapsed: float, done: int, left: int) -> str:
    if done == 0:
        return "ETA: --:--"
    rate = done / elapsed
    m, s = divmod(int(left / rate), 60)
    h, m = divmod(m, 60)
    return f"ETA: {h}h {m:02d}m" if h else f"ETA: {m:02d}m {s:02d}s"


# ── GitHub session ────────────────────────────────────────────────────────────
def make_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization":        f"Bearer {token}",
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent":           "rfc-comment-collector/1.0",
    })
    return s


def api_get(session: requests.Session,
            url: str,
            params: dict | None = None,
            delay: float = 0.5) -> list | dict | None:
    backoff = delay
    for attempt in range(7):
        time.sleep(delay if attempt == 0 else backoff)
        try:
            r = session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            log(f"\n    ⚠  Network error: {e} — retry {backoff:.0f}s")
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        remaining = int(r.headers.get("X-RateLimit-Remaining", 999))
        reset_ts  = int(r.headers.get("X-RateLimit-Reset", 0))
        if remaining < 20:
            wait = max(reset_ts - int(time.time()), 0) + 5
            log(f"\n    ⚠  Rate limit low ({remaining}) — sleeping {wait}s")
            time.sleep(wait)

        if r.status_code == 200:
            return r.json()
        if r.status_code in (403, 429):
            retry_after = int(r.headers.get("Retry-After", backoff * 2))
            log(f"\n    ⚠  Rate limited — sleeping {retry_after}s")
            time.sleep(retry_after)
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 410:
            # Resource gone (PR deleted)
            return None
        log(f"\n    ✗  HTTP {r.status_code}: {url}")
        return None

    log(f"\n    ✗  FAILED after retries: {url}")
    return None


def get_all_comments(session: requests.Session,
                     pr_number: int,
                     delay: float) -> list[dict]:
    """
    Fetch ALL comments for a PR, handling pagination.
    Returns list of raw GitHub comment objects.
    """
    all_comments = []
    page = 1

    while True:
        url  = f"{BASE_URL}/repos/{REPO}/issues/{pr_number}/comments"
        data = api_get(session, url,
                       params={"per_page": PER_PAGE, "page": page},
                       delay=delay)

        if not data:
            break

        all_comments.extend(data)

        # If we got fewer than PER_PAGE, we are on the last page
        if len(data) < PER_PAGE:
            break

        page += 1

    return all_comments


# ── checkpoint ────────────────────────────────────────────────────────────────
def load_checkpoint(path: str) -> set:
    """Returns set of pr_numbers already collected."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return set(data.get("completed_prs", []))
    return set()


def save_checkpoint(path: str, completed: set) -> None:
    with open(path, "w") as f:
        json.dump({"completed_prs": list(completed)}, f)


# ── build RFC list ────────────────────────────────────────────────────────────
def build_rfc_list(accepted_path: str,
                   rejected_path: str) -> list[dict]:
    """
    Returns a list of dicts:
      pr_number, rfc_number, outcome, pr_author_login
    for all 1,543 RFCs.
    """
    import csv as csv_mod

    rfcs = []

    # Accepted
    with open(accepted_path, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            rfcs.append({
                "pr_number":      int(row["rfc_number"]),
                "rfc_number":     int(row["rfc_number"]),
                "outcome":        "accepted",
                "pr_author_login": row.get("pr_author_login", ""),
            })

    # Rejected / postponed / withdrawn
    with open(rejected_path, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            rfcs.append({
                "pr_number":      int(row["pr_number"]),
                "rfc_number":     int(row.get("rfc_number",
                                              row["pr_number"])),
                "outcome":        row.get("outcome", "rejected"),
                "pr_author_login": row.get("pr_author_login", ""),
            })

    log(f"  Total RFCs to collect comments for: {len(rfcs):,}")
    return rfcs


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect full comment text for all RFC PRs."
    )
    parser.add_argument("--accepted",   required=True,
                        help="Path to rfc_features.csv")
    parser.add_argument("--rejected",   required=True,
                        help="Path to rfc_rejected.csv")
    parser.add_argument("--output",     default="rfc_comments_full.csv")
    parser.add_argument("--checkpoint",
                        default="rfc_comments_checkpoint.json")
    parser.add_argument("--delay",      type=float, default=0.5,
                        help="Seconds between API calls (default 0.5)")
    args = parser.parse_args()

    # ── token ─────────────────────────────────────────────────────────────────
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        sys.exit(
            "\nERROR: GITHUB_TOKEN not set.\n"
            "  export GITHUB_TOKEN=ghp_your_token_here\n"
        )

    session = make_session(token)

    log("━" * 60)
    log("RFC Comment Text Collector  v1.0")
    log("For RQ2 — Affect and Emotion Analysis")
    log("━" * 60)
    log("Verifying token ...", end=" ")
    me = api_get(session, f"{BASE_URL}/user", delay=0)
    if me is None:
        sys.exit("Authentication failed.")
    log(f"OK — {me.get('login')}")

    rate = api_get(session, f"{BASE_URL}/rate_limit", delay=0)
    if rate:
        core    = rate["resources"]["core"]
        reset_t = datetime.fromtimestamp(
            core["reset"]).strftime("%H:%M:%S")
        log(f"Rate limit: {core['remaining']:,}/{core['limit']:,} "
            f"(resets {reset_t})")
    log()

    # ── Build RFC list ─────────────────────────────────────────────────────────
    log("Building RFC list ...")
    rfcs = build_rfc_list(args.accepted, args.rejected)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    completed = load_checkpoint(args.checkpoint)
    if completed:
        log(f"Checkpoint: {len(completed)} RFCs already collected")
    remaining = [r for r in rfcs
                 if r["pr_number"] not in completed]
    log(f"RFCs remaining: {len(remaining):,}")

    # ── Open output file (append if resuming) ──────────────────────────────────
    file_exists = os.path.exists(args.output) and len(completed) > 0
    mode        = "a" if file_exists else "w"

    fieldnames = [
        "pr_number", "rfc_number", "outcome",
        "comment_id", "comment_author",
        "comment_order", "comment_created_at",
        "comment_body",
    ]

    out_f  = open(args.output, mode, newline="",
                  encoding="utf-8")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames,
                            extrasaction="ignore")
    if mode == "w":
        writer.writeheader()

    # ── Collection loop ────────────────────────────────────────────────────────
    log("━" * 60)
    log("Collecting comments ...")
    log("━" * 60)

    total      = len(remaining)
    start_time = time.time()
    done       = 0
    total_rows = 0

    for idx, rfc in enumerate(remaining, 1):
        pr_n    = rfc["pr_number"]
        outcome = rfc["outcome"]

        elapsed = time.time() - start_time
        bar     = progress_bar(idx, total)
        eta     = eta_str(elapsed, done, total - done)

        log(f"\r  {bar}  PR {pr_n:>5}  {eta}  "
            f"{outcome:<10} fetching...     ", end="")

        comments = get_all_comments(session, pr_n, args.delay)

        for order, comment in enumerate(comments, 1):
            body = comment.get("body") or ""
            # Clean up: remove excessive whitespace
            body = " ".join(body.split())

            writer.writerow({
                "pr_number":          pr_n,
                "rfc_number":         rfc["rfc_number"],
                "outcome":            outcome,
                "comment_id":         comment.get("id", ""),
                "comment_author":     (comment.get("user") or
                                       {}).get("login", ""),
                "comment_order":      order,
                "comment_created_at": comment.get("created_at", ""),
                "comment_body":       body,
            })
            total_rows += 1

        completed.add(pr_n)
        done += 1

        # Flush and checkpoint every 50 RFCs
        if done % 50 == 0:
            out_f.flush()
            save_checkpoint(args.checkpoint, completed)

        elapsed = time.time() - start_time
        bar     = progress_bar(idx, total)
        eta     = eta_str(elapsed, done, total - done)
        log(f"\r  {bar}  PR {pr_n:>5}  {eta}  "
            f"{outcome:<10}  "
            f"{len(comments):>4} comments  "
            f"total={total_rows:>7}     ")

    out_f.flush()
    out_f.close()
    save_checkpoint(args.checkpoint, completed)

    elapsed = time.time() - start_time
    m, s    = divmod(int(elapsed), 60)

    log()
    log("━" * 60)
    log("DONE")
    log("━" * 60)
    log(f"  Output file  : {args.output}")
    log(f"  Total rows   : {total_rows:,}  (one per comment)")
    log(f"  RFCs covered : {len(completed):,}")
    log(f"  Total time   : {m}m {s:02d}s")
    log(f"  Checkpoint   : {args.checkpoint}  (safe to delete)")
    log()
    log("  Next step: run VADER on rfc_comments_full.csv")
    log("  python rq2_vader.py --input rfc_comments_full.csv")
    log("━" * 60)


if __name__ == "__main__":
    main()
