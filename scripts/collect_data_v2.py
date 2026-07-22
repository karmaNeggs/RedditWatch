#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Enhanced Data Collection

Two modes:
  --year           Paginate top.json?t=year (10 pages × 100 = up to 1000 posts per sub),
                   split by created_utc month, cap POSTS_CAP_PER_MONTH posts per (sub, month).
                   Comments collected only for top COMMENT_SAMPLE posts per (sub, month).
                   Writes one posts_YYYY-MM.csv + commenters_YYYY-MM.csv per calendar month.

  --month YYYY-MM  Single calendar month (default: previous calendar month).
                   Uses t=month (≤35 days ago) or t=year (≤380 days) filtered to bounds.
                   Same cap + comment-sample logic as --year.

Checkpoint file (data/v2/checkpoint_<key>.json) survives interruptions.
Re-run the same command to resume.

Usage:
  python3 scripts/collect_data_v2.py --year                      # full year backfill
  python3 scripts/collect_data_v2.py                             # previous calendar month
  python3 scripts/collect_data_v2.py --month 2026-05             # specific month
  python3 scripts/collect_data_v2.py --year --subs india ipl     # test subset
"""

import argparse
import calendar
import json
import logging
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
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

POSTS_CAP_PER_MONTH    = 40   # max posts kept per (sub, month) for analysis
COMMENT_SAMPLE         = 10   # top-N posts per (sub, month) that get deep comment fetch
THIN_MONTH_WARN        = 15   # warn if a (sub, month) has fewer than this many posts

POST_COLS = [
    'subreddit', 'post_id', 'collection_month', 'collected_utc', 'created_utc',
    'title', 'score', 'upvote_ratio', 'num_comments', 'total_awards',
    'author', 'author_account_age_days', 'author_link_karma',
    'author_comment_karma', 'author_verified_email',
    'is_top10_for_month',
]
COMM_COLS = [
    'subreddit', 'post_id', 'collection_month',
    'comment_id', 'author', 'in_top10', 'top10_rank', 'in_first5', 'first5_rank',
    'is_submitter', 'comment_score', 'comment_created_utc', 'comment_depth',
    'author_account_age_days', 'author_link_karma',
    'author_comment_karma', 'author_verified_email',
]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(key: str) -> logging.Logger:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f'collect_v2_{key}.log'
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
    grp = p.add_mutually_exclusive_group()
    grp.add_argument('--year',  action='store_true',
                     help='Collect rolling last year (1000 posts/sub, split by month).')
    grp.add_argument('--month', default=None,
                     help='Collect one calendar month (YYYY-MM). Default: previous month.')
    p.add_argument('--subs', nargs='+', default=None,
                   help='Override subreddits.txt (e.g. for testing).')
    return p.parse_args()


def month_bounds(month_str: str):
    year, month = map(int, month_str.split('-'))
    start = int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())
    last  = calendar.monthrange(year, month)[1]
    end   = int(datetime(year, month, last, 23, 59, 59, tzinfo=timezone.utc).timestamp())
    return start, end


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _cp_path(key: str) -> Path:
    return DATA_DIR / f'checkpoint_{key}.json'


def load_checkpoint(key: str) -> dict:
    cp = _cp_path(key)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {'completed': [], 'posts': [], 'comments': [], 'users': {}}


def save_checkpoint(key: str, state: dict):
    with open(_cp_path(key), 'w') as f:
        json.dump({**state, 'saved_at': datetime.now().isoformat()}, f)


def clear_checkpoint(key: str):
    cp = _cp_path(key)
    if cp.exists():
        cp.unlink()


# ── Reddit fetch helpers ──────────────────────────────────────────────────────

def _parse_post(item: dict, subreddit: str, collected_utc: int) -> dict | None:
    p      = item['data']
    author = p.get('author', '[deleted]')
    if author in EXCLUDED:
        return None
    return {
        'subreddit':     subreddit,
        'post_id':       p['id'],
        '_fullname':     f"t3_{p['id']}",
        'collected_utc': collected_utc,
        'created_utc':   int(p['created_utc']),
        'title':         p['title'][:100],
        'score':         p['score'],
        'upvote_ratio':  p['upvote_ratio'],
        'num_comments':  p['num_comments'],
        'total_awards':  p.get('total_awards_received', 0),
        'author':        author,
    }


def fetch_posts_year(subreddit: str) -> list:
    """
    Paginate top.json?t=year, up to 10 pages × 100 = 1000 posts.
    Returns all posts sorted by score descending (Reddit's natural order).
    """
    collected_utc = int(time.time())
    all_posts     = []
    after         = None

    for page in range(10):
        url = f"https://www.reddit.com/r/{subreddit}/top.json?t=year&limit=100"
        if after:
            url += f"&after={after}&count={page * 100}"

        data = _get_json(url)
        if not data or 'data' not in data:
            break

        children = data['data']['children']
        if not children:
            break

        for item in children:
            post = _parse_post(item, subreddit, collected_utc)
            if post:
                all_posts.append(post)

        after = children[-1]['data'].get('name') or f"t3_{children[-1]['data']['id']}"

        if len(children) < 100:
            break  # reached end of listing

        time.sleep(random.uniform(0.5, 0.9))

    return all_posts


def fetch_posts_month(subreddit: str, month: str) -> list:
    """
    Fetch top posts for a single calendar month with strict UTC bounds.
    Recent months (≤35 days) use t=month directly. Older months paginate
    t=year (like year mode) — a single 100-post page of the year's top only
    surfaces a handful of posts from any one month, which is what produced
    thin months in earlier backfills.
    """
    collected_utc = int(time.time())
    start, end    = month_bounds(month)
    days_ago      = (collected_utc - start) / 86400

    if days_ago <= 35:
        data = _get_json(f"https://www.reddit.com/r/{subreddit}/top.json?t=month&limit=60")
        if not data or 'data' not in data:
            return []
        raw = []
        for item in data['data']['children']:
            post = _parse_post(item, subreddit, collected_utc)
            if post:
                raw.append(post)
    else:
        raw = fetch_posts_year(subreddit)

    posts = [p for p in raw if start <= p['created_utc'] <= end]
    posts.sort(key=lambda x: -x['score'])
    return posts[:POSTS_CAP_PER_MONTH]


def cap_and_mark(posts: list) -> list:
    """
    Sort by score, cap at POSTS_CAP_PER_MONTH, mark top COMMENT_SAMPLE as is_top10_for_month.
    """
    posts = sorted(posts, key=lambda x: -x['score'])[:POSTS_CAP_PER_MONTH]
    for i, p in enumerate(posts):
        p['is_top10_for_month'] = (i < COMMENT_SAMPLE)
    return posts


def fetch_comments(subreddit: str, post_id: str, n_top: int = 10, n_first: int = 5) -> list:
    """
    Fetch up to 30 comments (sort=top, depth≤2).
    Tag top n_top by score (in_top10) and first n_first direct replies by timestamp (in_first5).
    """
    url  = (f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
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


# ── Shared phase 1 logic ──────────────────────────────────────────────────────

def collect_sub_year(sub: str, log) -> tuple[list, list]:
    """Fetch posts + comments for a subreddit in year mode. Returns (posts, comments)."""
    raw = fetch_posts_year(sub)
    if not raw:
        log.info(f'    r/{sub}: no posts returned')
        return [], []

    # Group by month, cap + mark
    by_month = defaultdict(list)
    for post in raw:
        m = datetime.utcfromtimestamp(post['created_utc']).strftime('%Y-%m')
        by_month[m].append(post)

    all_posts = []
    for m, month_posts in sorted(by_month.items()):
        month_posts = cap_and_mark(month_posts)
        by_month[m] = month_posts
        all_posts.extend(month_posts)

    thin = [m for m, mp in by_month.items() if len(mp) < THIN_MONTH_WARN]
    month_summary = {m: len(mp) for m, mp in sorted(by_month.items())}
    log.info(f'    r/{sub}: {len(raw)} fetched → {len(all_posts)} kept across '
             f'{len(by_month)} months  {month_summary}')
    if thin:
        log.info(f'    ⚠  thin months (<{THIN_MONTH_WARN} posts): {", ".join(thin)}')

    # Comments only for top-10-per-month posts
    comment_posts = [p for p in all_posts if p['is_top10_for_month']]
    all_comments  = []
    for post in comment_posts:
        comments = fetch_comments(sub, post['post_id'])
        for c in comments:
            all_comments.append({'subreddit': sub, 'post_id': post['post_id'], **c})
        time.sleep(random.uniform(0.8, 1.2))

    log.info(f'    r/{sub}: {len(comment_posts)} comment-sampled posts → '
             f'{len(all_comments)} comment rows')
    return all_posts, all_comments


def collect_sub_month(sub: str, month: str, log) -> tuple[list, list]:
    """Fetch posts + comments for a subreddit in single-month mode."""
    posts = fetch_posts_month(sub, month)
    if not posts:
        return [], []

    posts = cap_and_mark(posts)

    comments      = []
    comment_posts = [p for p in posts if p['is_top10_for_month']]
    for post in comment_posts:
        cs = fetch_comments(sub, post['post_id'])
        for c in cs:
            comments.append({'subreddit': sub, 'post_id': post['post_id'], **c})
        time.sleep(random.uniform(0.8, 1.3))

    log.info(f'    r/{sub}: {len(posts)} posts  '
             f'{len(comment_posts)} comment-sampled  {len(comments)} comment rows')
    return posts, comments


# ── Profile fetch phase ───────────────────────────────────────────────────────

def fetch_profiles(raw_posts, raw_comments, user_cache, cp_key, state, log):
    poster_names    = {p['author'] for p in raw_posts    if p['author'] not in EXCLUDED}
    commenter_names = {c['author'] for c in raw_comments if c['author'] not in EXCLUDED}
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
            name, profile = f.result()
            if profile:
                user_cache[name] = profile
            done += 1
            if done % 100 == 0:
                log.info(f'  profiles {done}/{len(need_fetch)} …')
                save_checkpoint(cp_key, state)

    log.info(f'Profiles fetched: {len(user_cache)} total in cache')


# ── Write output CSVs ─────────────────────────────────────────────────────────

def write_csvs(raw_posts, raw_comments, user_cache, log):
    """
    Enrich with profile data, split by collection_month, write per-month CSVs.
    Returns list of months written.
    """
    def _author_cols(name):
        u = user_cache.get(name) or {}
        return {
            'author_account_age_days': round(u['account_age_days'], 1) if u else None,
            'author_link_karma':       u.get('link_karma'),
            'author_comment_karma':    u.get('comment_karma'),
            'author_verified_email':   u.get('verified_email'),
        }

    # Derive collection_month from created_utc
    posts_by_month    = defaultdict(list)
    comments_by_month = defaultdict(list)

    for p in raw_posts:
        m = datetime.utcfromtimestamp(p['created_utc']).strftime('%Y-%m')
        row = {**p, 'collection_month': m, **_author_cols(p['author'])}
        posts_by_month[m].append({col: row.get(col) for col in POST_COLS})

    for c in raw_comments:
        # Find the parent post's month
        post_id = c['post_id']
        parent  = next((p for p in raw_posts if p['post_id'] == post_id), None)
        m = datetime.utcfromtimestamp(parent['created_utc']).strftime('%Y-%m') if parent else 'unknown'
        row = {**c, 'collection_month': m, **_author_cols(c['author'])}
        comments_by_month[m].append({col: row.get(col) for col in COMM_COLS})

    months_written = []
    for m in sorted(posts_by_month):
        p_rows = posts_by_month[m]
        c_rows = comments_by_month.get(m, [])

        posts_path = DATA_DIR / f'posts_{m}.csv'
        comms_path = DATA_DIR / f'commenters_{m}.csv'
        pd.DataFrame(p_rows).to_csv(posts_path, index=False)
        pd.DataFrame(c_rows).to_csv(comms_path,  index=False)
        log.info(f'  {m}: {len(p_rows)} posts ({sum(1 for r in p_rows if r["is_top10_for_month"])} comment-sampled)  '
                 f'{len(c_rows)} comment rows  → {posts_path.name}')
        months_written.append(m)

    # Also write latest symlinks
    if months_written:
        latest = months_written[-1]
        pd.DataFrame(posts_by_month[latest]).to_csv(   DATA_DIR / 'posts_latest.csv',      index=False)
        pd.DataFrame(comments_by_month.get(latest, [])).to_csv(DATA_DIR / 'commenters_latest.csv', index=False)

    return months_written


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    year_mode = args.year
    if year_mode:
        cp_key = 'year'
        label  = 'YEAR (rolling last 365 days)'
    else:
        if args.month:
            month = args.month
        else:
            now   = datetime.now()
            prev  = now.replace(day=1) - timedelta(days=1)
            month = prev.strftime('%Y-%m')
        cp_key = month
        label  = f'month={month}'

    log = setup_logger(cp_key)
    log.info('=' * 60)
    log.info(f'Reddit Bot Analysis V2 — collection  {label}')
    print_auth_status()
    log.info(f'Posts cap: {POSTS_CAP_PER_MONTH}/month  Comment sample: {COMMENT_SAMPLE}/month')
    log.info('=' * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

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

    # ── Resume from checkpoint ────────────────────────────────────────────────
    state      = load_checkpoint(cp_key)
    raw_posts  = state['posts']
    raw_comms  = state['comments']
    user_cache = state['users']
    completed  = set(state['completed'])

    if completed:
        log.info(f'Resuming: {len(completed)} subs done ({", ".join(sorted(completed))})')

    # ── Phase 1: posts + comments ─────────────────────────────────────────────
    for sub in subreddits:
        if sub in completed:
            log.info(f'  r/{sub:<25} skip (checkpoint)')
            continue

        if year_mode:
            posts, comments = collect_sub_year(sub, log)
        else:
            posts, comments = collect_sub_month(sub, month, log)

        if not posts:
            log.info(f'  r/{sub}: no posts — skipping')

        raw_posts.extend(posts)
        raw_comms.extend(comments)
        completed.add(sub)

        save_checkpoint(cp_key, {
            'completed': list(completed),
            'posts':     raw_posts,
            'comments':  raw_comms,
            'users':     user_cache,
        })
        time.sleep(random.uniform(1.5, 2.5))

    log.info(f'Phase 1 done — {len(raw_posts)} posts  {len(raw_comms)} comment rows')

    # ── Phase 2: user profiles ────────────────────────────────────────────────
    state_ref = {
        'completed': list(completed),
        'posts':     raw_posts,
        'comments':  raw_comms,
        'users':     user_cache,
    }
    fetch_profiles(raw_posts, raw_comms, user_cache, cp_key, state_ref, log)

    # ── Phase 3: write CSVs ───────────────────────────────────────────────────
    months_written = write_csvs(raw_posts, raw_comms, user_cache, log)

    meta = {
        'version':        2,
        'mode':           'year' if year_mode else 'month',
        'months_written': months_written,
        'timestamp':      datetime.now().isoformat(),
        'total_posts':    len(raw_posts),
        'total_comments': len(raw_comms),
        'unique_accounts': len(user_cache),
        'posts_cap_per_month':   POSTS_CAP_PER_MONTH,
        'comment_sample':        COMMENT_SAMPLE,
    }
    meta_path = DATA_DIR / (f'metadata_year.json' if year_mode else f'metadata_{month}.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    clear_checkpoint(cp_key)
    log.info(f'Months written: {months_written}')
    log.info('Checkpoint cleared — collection complete.')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
