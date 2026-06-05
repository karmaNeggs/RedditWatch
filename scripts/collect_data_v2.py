#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Enhanced Data Collection
Outputs two CSVs per run:
  data/v2/posts_YYYY-MM.csv       — one row per post
  data/v2/commenters_YYYY-MM.csv  — one row per (post, commenter)

Usage:
  python3 scripts/collect_data_v2.py                  # current month
  python3 scripts/collect_data_v2.py --month 2026-01  # historical backfill
"""

import argparse
import calendar
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reddit_auth import get_json as _get_json, USING_OAUTH

DATA_DIR       = ROOT / 'data' / 'v2'
SUBREDDITS_FILE = ROOT / 'subreddits.txt'
WORKERS        = 8 if USING_OAUTH else 3
EXCLUDED       = {'[deleted]', '[removed]', 'AutoModerator', 'None', 'nan'}


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--month', default=None,
                   help='Month to collect in YYYY-MM format. Omit for current month.')
    return p.parse_args()


def month_bounds(month_str):
    """Return (start_epoch, end_epoch, label) for a 'YYYY-MM' string."""
    year, month = map(int, month_str.split('-'))
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    last  = calendar.monthrange(year, month)[1]
    end   = int(datetime(year, month, last, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return start, end


# ── Collection helpers ────────────────────────────────────────────────────────

def fetch_posts(subreddit, month=None, limit=30):
    """
    Fetch top posts for a subreddit.
    - No month: current-month top listing (same as v1).
    - With month: search API with epoch time bounds (backfill).
    Returns list of raw post dicts (no author data yet).
    """
    collected_utc = int(time.time())

    if month:
        start, end = month_bounds(month)
        url = (f"https://www.reddit.com/r/{subreddit}/search.json"
               f"?q=&sort=top&t=all&restrict_sr=1"
               f"&after={start}&before={end}&limit={limit}")
    else:
        url = f"https://www.reddit.com/r/{subreddit}/top.json?t=month&limit={limit}"

    data = _get_json(url)
    if not data or 'data' not in data:
        return []

    posts = []
    for item in data['data']['children']:
        p = item['data']
        author = p.get('author', '[deleted]')
        if author in EXCLUDED:
            continue
        posts.append({
            'subreddit':    subreddit,
            'post_id':      p['id'],
            'collected_utc': collected_utc,
            'created_utc':  int(p['created_utc']),
            'title':        p['title'][:100],
            'score':        p['score'],
            'upvote_ratio': p['upvote_ratio'],
            'num_comments': p['num_comments'],
            'total_awards': p.get('total_awards_received', 0),
            'author':       author,
        })
    return posts


def fetch_comments(subreddit, post_id, n_top=10, n_first=5):
    """
    Fetch up to 30 comments for a post (sorted by top, depth ≤ 2).
    Returns a list of comment rows, each flagged as in_top10 and/or in_first5.

    in_top10  — among the N highest-scored comments in the batch
    in_first5 — among the N earliest depth-0 (direct) replies
    Both flags can be True for the same comment.
    """
    url = (f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
           f"?limit=30&sort=top&depth=2")
    data = _get_json(url)

    if not data or not isinstance(data, list) or len(data) < 2:
        return []

    raw = []
    for child in data[1]['data']['children']:
        if child.get('kind') != 't1':
            continue
        c      = child['data']
        author = c.get('author', '[deleted]')
        if author in EXCLUDED:
            continue
        ts = c.get('created_utc')
        raw.append({
            'comment_id':          c['id'],
            'author':              author,
            'is_submitter':        bool(c.get('is_submitter', False)),
            'comment_score':       c.get('score', 0),
            'comment_created_utc': int(ts) if ts else None,
            'comment_depth':       c.get('depth', 0),
        })

    if not raw:
        return []

    # Top N by score
    by_score  = sorted(raw, key=lambda x: -x['comment_score'])
    top_rank  = {c['comment_id']: i + 1 for i, c in enumerate(by_score[:n_top])}

    # First N depth-0 replies by timestamp
    direct    = [c for c in raw if c['comment_depth'] == 0 and c['comment_created_utc']]
    by_time   = sorted(direct, key=lambda x: x['comment_created_utc'])
    first_rank = {c['comment_id']: i + 1 for i, c in enumerate(by_time[:n_first])}

    included = set(top_rank) | set(first_rank)
    result = []
    for c in raw:
        if c['comment_id'] not in included:
            continue
        result.append({
            **c,
            'in_top10':    c['comment_id'] in top_rank,
            'top10_rank':  top_rank.get(c['comment_id']),
            'in_first5':   c['comment_id'] in first_rank,
            'first5_rank': first_rank.get(c['comment_id']),
        })
    return result


def fetch_user(username):
    """
    Fetch account profile for a single username.
    Returns dict with account_age_days, link_karma, comment_karma, verified_email.
    Returns None if the account is deleted / suspended / unavailable.
    """
    if not username or username in EXCLUDED:
        return None
    data = _get_json(f"https://www.reddit.com/user/{username}/about.json")
    if not data or 'data' not in data:
        return None
    d = data['data']
    created = d.get('created_utc')
    if not created:
        return None
    return {
        'account_age_days':   (time.time() - created) / 86400,
        'link_karma':         int(d.get('link_karma',    0)),
        'comment_karma':      int(d.get('comment_karma', 0)),
        'verified_email':     d.get('has_verified_email'),   # True/False/None
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    month = args.month or datetime.now().strftime('%Y-%m')

    print("\n" + "=" * 70)
    print("REDDIT BOT ANALYSIS V2 — DATA COLLECTION")
    print(f"Month     : {month}")
    print(f"Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    subreddits = []
    if SUBREDDITS_FILE.exists():
        subreddits = [
            l.strip() for l in SUBREDDITS_FILE.read_text().splitlines()
            if l.strip() and not l.startswith('#')
        ]
    if not subreddits:
        print("No subreddits found. Check subreddits.txt.")
        return
    print(f"Subreddits: {len(subreddits)}\n")

    # ── Phase 1: posts + comment metadata ────────────────────────────────────
    raw_posts    = []   # list of dicts  → posts CSV
    raw_comments = []   # list of dicts  → commenters CSV

    for sub in subreddits:
        print(f"  r/{sub}")
        posts = fetch_posts(sub, month=args.month)
        if not posts:
            print(f"    no posts returned")
            continue
        print(f"    {len(posts)} posts", end='', flush=True)

        for post in posts:
            comments = fetch_comments(sub, post['post_id'])
            for c in comments:
                raw_comments.append({'subreddit': sub, 'post_id': post['post_id'], **c})
            time.sleep(random.uniform(0.8, 1.3))

        raw_posts.extend(posts)
        print(f"  → {len(raw_comments)} comment rows so far")
        time.sleep(random.uniform(1.5, 2.5))

    print(f"\nPosts collected  : {len(raw_posts)}")
    print(f"Comment rows     : {len(raw_comments)}")

    # ── Phase 2: user profiles — deduplicated across posts + commenters ───────
    poster_names    = {p['author'] for p in raw_posts    if p['author'] not in EXCLUDED}
    commenter_names = {c['author'] for c in raw_comments if c['author'] not in EXCLUDED}
    all_names       = list(poster_names | commenter_names)

    print(f"\nUnique accounts  : {len(all_names)}"
          f"  ({len(poster_names)} posters + {len(commenter_names)} commenters, deduped)")

    user_cache = {}
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(fetch_user, name): name for name in all_names}
        for f in as_completed(futures):
            name, profile = futures[f], f.result()
            if profile:
                user_cache[name] = profile
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(all_names)} profiles fetched …")

    print(f"  Done: {len(user_cache)}/{len(all_names)} profiles retrieved")

    # ── Phase 3: enrich rows with author data ─────────────────────────────────
    def _author_cols(name):
        u = user_cache.get(name) or {}
        return {
            'author_account_age_days': round(u['account_age_days'], 1) if u else None,
            'author_link_karma':       u.get('link_karma'),
            'author_comment_karma':    u.get('comment_karma'),
            'author_verified_email':   u.get('verified_email'),
        }

    # Post rows
    POST_COLS = [
        'subreddit', 'post_id', 'collection_month', 'collected_utc', 'created_utc',
        'title', 'score', 'upvote_ratio', 'num_comments', 'total_awards',
        'author', 'author_account_age_days', 'author_link_karma',
        'author_comment_karma', 'author_verified_email',
    ]
    posts_rows = []
    for p in raw_posts:
        row = {**p, 'collection_month': month, **_author_cols(p['author'])}
        posts_rows.append({col: row.get(col) for col in POST_COLS})

    # Commenter rows
    COMM_COLS = [
        'subreddit', 'post_id', 'collection_month',
        'comment_id', 'author', 'in_top10', 'top10_rank', 'in_first5', 'first5_rank',
        'is_submitter', 'comment_score', 'comment_created_utc', 'comment_depth',
        'author_account_age_days', 'author_link_karma',
        'author_comment_karma', 'author_verified_email',
    ]
    comm_rows = []
    for c in raw_comments:
        row = {**c, 'collection_month': month, **_author_cols(c['author'])}
        comm_rows.append({col: row.get(col) for col in COMM_COLS})

    # ── Phase 4: write CSVs ───────────────────────────────────────────────────
    posts_path = DATA_DIR / f'posts_{month}.csv'
    comms_path = DATA_DIR / f'commenters_{month}.csv'

    pd.DataFrame(posts_rows).to_csv(posts_path, index=False)
    pd.DataFrame(comm_rows).to_csv(comms_path,  index=False)

    # Always keep a 'latest' copy for the analysis pipeline
    pd.DataFrame(posts_rows).to_csv(DATA_DIR / 'posts_latest.csv',      index=False)
    pd.DataFrame(comm_rows).to_csv( DATA_DIR / 'commenters_latest.csv', index=False)

    meta = {
        'version':         2,
        'month':           month,
        'timestamp':       datetime.now().isoformat(),
        'posts':           len(posts_rows),
        'comment_rows':    len(comm_rows),
        'unique_accounts': len(user_cache),
        'posts_file':      str(posts_path),
        'commenters_file': str(comms_path),
    }
    with open(DATA_DIR / f'metadata_{month}.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote : {posts_path.name}  ({len(posts_rows)} rows, {len(POST_COLS)} cols)")
    print(f"Wrote : {comms_path.name}  ({len(comm_rows)} rows, {len(COMM_COLS)} cols)")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
