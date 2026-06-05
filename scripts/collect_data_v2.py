#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Enhanced Data Collection
Outputs two CSVs per run:
  data/v2/posts_YYYY-MM.csv       — one row per post
  data/v2/commenters_YYYY-MM.csv  — one row per (post, commenter)

Checkpoint file (data/v2/checkpoint_YYYY-MM.json) survives interruptions.
Restart the same command to resume from the last completed subreddit.

Usage:
  python3 scripts/collect_data_v2.py                        # current month
  python3 scripts/collect_data_v2.py --month 2026-01        # backfill
  python3 scripts/collect_data_v2.py --subs india indiaspeaks  # test / subset
"""

import argparse
import calendar
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reddit_auth import get_json as _get_json, USING_OAUTH, print_auth_status

DATA_DIR        = ROOT / 'data' / 'v2'
LOGS_DIR        = ROOT / 'logs'
SUBREDDITS_FILE = ROOT / 'subreddits.txt'
WORKERS         = 8 if USING_OAUTH else 3
EXCLUDED        = {'[deleted]', '[removed]', 'AutoModerator', 'None', 'nan'}

POST_COLS = [
    'subreddit', 'post_id', 'collection_month', 'collected_utc', 'created_utc',
    'title', 'score', 'upvote_ratio', 'num_comments', 'total_awards',
    'author', 'author_account_age_days', 'author_link_karma',
    'author_comment_karma', 'author_verified_email',
]
COMM_COLS = [
    'subreddit', 'post_id', 'collection_month',
    'comment_id', 'author', 'in_top10', 'top10_rank', 'in_first5', 'first5_rank',
    'is_submitter', 'comment_score', 'comment_created_utc', 'comment_depth',
    'author_account_age_days', 'author_link_karma',
    'author_comment_karma', 'author_verified_email',
]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(month: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f'collect_v2_{month}.log'
    logger = logging.getLogger('collect_v2')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s  %(message)s', datefmt='%H:%M:%S')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Reddit Bot Analysis V2 — Data Collection')
    p.add_argument('--month', default=None,
                   help='Month to collect (YYYY-MM). Defaults to current month.')
    p.add_argument('--subs', nargs='+', default=None,
                   help='Override subreddits.txt with specific subs (useful for testing).')
    return p.parse_args()


def month_bounds(month_str: str):
    year, month = map(int, month_str.split('-'))
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    last  = calendar.monthrange(year, month)[1]
    end   = int(datetime(year, month, last, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return start, end


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _cp_path(month: str) -> Path:
    return DATA_DIR / f'checkpoint_{month}.json'


def load_checkpoint(month: str) -> dict:
    cp = _cp_path(month)
    if cp.exists():
        with open(cp) as f:
            data = json.load(f)
        return data
    return {'completed': [], 'posts': [], 'comments': [], 'users': {}}


def save_checkpoint(month: str, state: dict):
    with open(_cp_path(month), 'w') as f:
        json.dump({**state, 'saved_at': datetime.now().isoformat()}, f)


def clear_checkpoint(month: str):
    cp = _cp_path(month)
    if cp.exists():
        cp.unlink()


# ── Reddit fetch helpers ──────────────────────────────────────────────────────

def fetch_posts(subreddit: str, month=None, limit: int = 30) -> list:
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
        p      = item['data']
        author = p.get('author', '[deleted]')
        if author in EXCLUDED:
            continue
        posts.append({
            'subreddit':     subreddit,
            'post_id':       p['id'],
            'collected_utc': collected_utc,
            'created_utc':   int(p['created_utc']),
            'title':         p['title'][:100],
            'score':         p['score'],
            'upvote_ratio':  p['upvote_ratio'],
            'num_comments':  p['num_comments'],
            'total_awards':  p.get('total_awards_received', 0),
            'author':        author,
        })
    return posts


def fetch_comments(subreddit: str, post_id: str, n_top: int = 10, n_first: int = 5) -> list:
    """
    Fetch up to 30 comments sorted by top (depth ≤ 2).
    Tag each as in_top10 (highest-scored) and/or in_first5 (earliest direct reply).
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

    by_score   = sorted(raw, key=lambda x: -x['comment_score'])
    top_rank   = {c['comment_id']: i + 1 for i, c in enumerate(by_score[:n_top])}

    direct     = [c for c in raw if c['comment_depth'] == 0 and c['comment_created_utc']]
    by_time    = sorted(direct, key=lambda x: x['comment_created_utc'])
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


def fetch_user(username: str) -> dict | None:
    if not username or username in EXCLUDED:
        return None
    data = _get_json(f"https://www.reddit.com/user/{username}/about.json")
    if not data or 'data' not in data:
        return None
    d       = data['data']
    created = d.get('created_utc')
    if not created:
        return None
    return {
        'account_age_days': (time.time() - created) / 86400,
        'link_karma':       int(d.get('link_karma',    0)),
        'comment_karma':    int(d.get('comment_karma', 0)),
        'verified_email':   d.get('has_verified_email'),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    month = args.month or datetime.now().strftime('%Y-%m')
    log   = setup_logger(month)

    log.info('=' * 60)
    log.info(f'Reddit Bot Analysis V2 — collection  month={month}')
    print_auth_status()
    log.info('=' * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Subreddit list
    if args.subs:
        subreddits = args.subs
        log.info(f'Subreddits (override): {subreddits}')
    elif SUBREDDITS_FILE.exists():
        subreddits = [
            l.strip() for l in SUBREDDITS_FILE.read_text().splitlines()
            if l.strip() and not l.startswith('#')
        ]
        log.info(f'Subreddits (file): {len(subreddits)}')
    else:
        log.error('No subreddits found.')
        return

    # ── Resume from checkpoint if one exists ─────────────────────────────────
    state = load_checkpoint(month)
    if state['completed']:
        log.info(f'Resuming: {len(state["completed"])} subs already done '
                 f'({", ".join(state["completed"])})')

    raw_posts    = state['posts']
    raw_comments = state['comments']
    user_cache   = state['users']    # persisted across restarts
    completed    = set(state['completed'])

    # ── Phase 1: collect posts + comments ─────────────────────────────────────
    for sub in subreddits:
        if sub in completed:
            log.info(f'  r/{sub:<25} skip (checkpoint)')
            continue

        log.info(f'  r/{sub}')
        posts = fetch_posts(sub, month=args.month)
        if not posts:
            log.info(f'    no posts — skipping')
            completed.add(sub)
            save_checkpoint(month, {'completed': list(completed),
                                    'posts': raw_posts,
                                    'comments': raw_comments,
                                    'users': user_cache})
            continue

        n_comments = 0
        for post in posts:
            comments = fetch_comments(sub, post['post_id'])
            for c in comments:
                raw_comments.append({'subreddit': sub, 'post_id': post['post_id'], **c})
            n_comments += len(comments)
            time.sleep(random.uniform(0.8, 1.3))

        raw_posts.extend(posts)
        completed.add(sub)
        log.info(f'    {len(posts)} posts  {n_comments} comment-rows')

        # Checkpoint after every subreddit — safe to interrupt here
        save_checkpoint(month, {'completed': list(completed),
                                'posts': raw_posts,
                                'comments': raw_comments,
                                'users': user_cache})

        time.sleep(random.uniform(1.5, 2.5))

    log.info(f'Phase 1 done — {len(raw_posts)} posts  {len(raw_comments)} comment-rows')

    # ── Phase 2: user profiles (deduplicated; cache survives restarts) ────────
    poster_names    = {p['author']  for p in raw_posts    if p['author']    not in EXCLUDED}
    commenter_names = {c['author']  for c in raw_comments if c['author']    not in EXCLUDED}
    need_fetch      = [a for a in poster_names | commenter_names
                       if a not in EXCLUDED and a not in user_cache]

    log.info(f'Accounts: {len(poster_names)} posters + {len(commenter_names)} commenters '
             f'= {len(poster_names | commenter_names)} unique  '
             f'({len(need_fetch)} to fetch, {len(user_cache)} cached)')

    done = 0
    def _lookup(name):
        return name, fetch_user(name)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_lookup, a): a for a in need_fetch}
        for f in as_completed(futures):
            name, profile = f.result()   # _lookup returns (name, profile)
            if profile:
                user_cache[name] = profile
            done += 1
            if done % 100 == 0:
                log.info(f'  profiles {done}/{len(need_fetch)} …')
                # Persist cache so a crash here doesn't lose progress
                save_checkpoint(month, {'completed': list(completed),
                                        'posts': raw_posts,
                                        'comments': raw_comments,
                                        'users': user_cache})

    log.info(f'Profiles fetched: {len(user_cache)} total in cache')

    # ── Phase 3: enrich + write CSVs ─────────────────────────────────────────
    def _author_cols(name):
        u = user_cache.get(name) or {}
        return {
            'author_account_age_days': round(u['account_age_days'], 1) if u else None,
            'author_link_karma':       u.get('link_karma'),
            'author_comment_karma':    u.get('comment_karma'),
            'author_verified_email':   u.get('verified_email'),
        }

    posts_rows = []
    for p in raw_posts:
        row = {**p, 'collection_month': month, **_author_cols(p['author'])}
        posts_rows.append({col: row.get(col) for col in POST_COLS})

    comm_rows = []
    for c in raw_comments:
        row = {**c, 'collection_month': month, **_author_cols(c['author'])}
        comm_rows.append({col: row.get(col) for col in COMM_COLS})

    posts_path = DATA_DIR / f'posts_{month}.csv'
    comms_path = DATA_DIR / f'commenters_{month}.csv'

    pd.DataFrame(posts_rows).to_csv(posts_path, index=False)
    pd.DataFrame(comm_rows).to_csv(comms_path,  index=False)
    pd.DataFrame(posts_rows).to_csv(DATA_DIR / 'posts_latest.csv',      index=False)
    pd.DataFrame(comm_rows).to_csv( DATA_DIR / 'commenters_latest.csv', index=False)

    meta = {
        'version': 2, 'month': month,
        'timestamp': datetime.now().isoformat(),
        'posts': len(posts_rows), 'comment_rows': len(comm_rows),
        'unique_accounts': len(user_cache),
        'posts_file': str(posts_path), 'commenters_file': str(comms_path),
    }
    with open(DATA_DIR / f'metadata_{month}.json', 'w') as f:
        json.dump(meta, f, indent=2)

    clear_checkpoint(month)

    log.info(f'posts    → {posts_path.name}  ({len(posts_rows)} rows)')
    log.info(f'comments → {comms_path.name}  ({len(comm_rows)} rows)')
    log.info('Checkpoint cleared — collection complete.')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
