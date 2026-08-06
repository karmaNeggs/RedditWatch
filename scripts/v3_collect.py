#!/usr/bin/env python3
"""V3 full collection: 45 subs x 24 months (2024-08 -> 2026-07).

Top-100-by-score posts + counter-sample (num_comments-matched, up to 20)
per sub-month; full comment-thread scan per post (compute-then-discard);
first-10 and top-10-by-score commenters extracted and stored per post,
including body text (needed for Stage 4 pair-level stylometry/template
features -- see V3_PLAN.md Sec 4.4 B3/B4).

Checkpointed and resumable: one (posts, commenters) zstd-NDJSON file pair
per (sub, month) cell, written atomically (tmp + rename). A cell whose
final files already exist is skipped, so a killed/restarted run only
redoes in-flight cells. Run with --workers N; per V3_PLAN.md Sec 2, start
conservative and confirm sustained throughput before raising N (restarting
with a higher --workers is cheap -- completed cells are skipped instantly).
"""
import argparse
import csv
import json
import os
import random
import sys
import threading
import time
import datetime as dt
import urllib.request
import urllib.parse
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import zstandard as zstd

B = 'https://arctic-shift.photon-reddit.com/api'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, 'data', 'v3', 'raw')
DEFAULT_CSV = os.path.join(ROOT, 'subreddits_v3.csv')

random.seed(42)

BOT_SELF_DECLARE = ['i am a bot', 'beep boop', 'performed automatically',
                     'this action was performed automatically']


# ---------- HTTP ----------

def get(url, stats=None, retries=5):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'v3-collect/1.0'})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode())
            if stats:
                stats.add_request()
            return data
        except urllib.error.HTTPError as e:
            if stats:
                stats.add_request(err=True)
            if e.code == 404:
                return None
            time.sleep(2 * (a + 1))
        except Exception:
            if stats:
                stats.add_request(err=True)
            time.sleep(2 * (a + 1))
    return None


def month_bounds(y, m):
    a = int(dt.datetime(y, m, 1).timestamp())
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    b = int(dt.datetime(ny, nm, 1).timestamp())
    return a, b


def month_range(start_ym, end_ym):
    """Inclusive [start_ym, end_ym], each 'YYYY-MM'."""
    sy, sm = map(int, start_ym.split('-'))
    ey, em = map(int, end_ym.split('-'))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


# ---------- fetch ----------

def fetch_month_posts(sub, after, before, stats):
    out = {}
    cur = after
    pages = 0
    while True:
        d = get(f'{B}/posts/search?subreddit={urllib.parse.quote(sub)}&after={cur}'
                 f'&before={before}&limit=auto&sort=asc', stats)
        pages += 1
        if not d:
            break
        items = d.get('data', [])
        if not items:
            break
        for it in items:
            out[it['id']] = it
        last = max(i['created_utc'] for i in items)
        if last >= before:
            break
        newcur = last * 1000 - 1
        if newcur == cur:
            break
        cur = newcur
        if pages > 3000:
            break
    return list(out.values())


def fetch_all_comments(post_id, link_t3, stats):
    out = {}
    cur = None
    while True:
        after_part = f'&after={cur}' if cur is not None else ''
        d = get(f'{B}/comments/search?link_id={link_t3}{after_part}&limit=auto&sort=asc', stats)
        if not d:
            break
        items = d.get('data', [])
        if not items:
            break
        for it in items:
            out[it['id']] = it
        last = max(i['created_utc'] for i in items)
        if cur == last * 1000 - 1:
            break
        cur = last * 1000 - 1
        if len(out) > 20000:
            break
    return list(out.values())


# ---------- derived metrics ----------

def gini(values):
    v = sorted(x for x in values if x is not None)
    n = len(v)
    if n < 2 or sum(v) == 0:
        return None
    cum = 0
    s = sum(v)
    for i, x in enumerate(v, 1):
        cum += i * x
    return (2 * cum) / (n * s) - (n + 1) / n


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def matched_random_sample(top_posts, rest_posts, k=20):
    if not rest_posts or k <= 0:
        return []
    top_nc = sorted((p.get('num_comments') or 0) for p in top_posts)
    if k == 1 or len(top_nc) <= 1:
        targets = [top_nc[len(top_nc) // 2] if top_nc else 0] * k
    else:
        targets = [top_nc[int(q * (len(top_nc) - 1))] for q in
                   [i / (k - 1) for i in range(k)]]
    pool = list(rest_posts)
    picked = []
    for t in targets:
        if not pool:
            break
        best = min(pool, key=lambda p: abs((p.get('num_comments') or 0) - t))
        picked.append(best)
        pool.remove(best)
    return picked


def derive_post_metrics(post):
    S = post.get('score') or 0
    r = post.get('upvote_ratio')
    m = {}
    if r and r > 0.5 and S > 0:
        denom = 2 * r - 1
        m['implied_votes'] = S / denom
        m['implied_downs'] = S * (1 - r) / denom
        m['implied_ups'] = S * r / denom
    m['contested_share'] = (1 - r) if r is not None else None
    return m


def is_bot_comment(c):
    author = (c.get('author') or '')
    if author == 'AutoModerator' or author.endswith('Bot') or author.endswith('bot'):
        return True
    body = (c.get('body') or '').lower()
    return any(p in body for p in BOT_SELF_DECLARE)


def compute_depths(comments):
    """depth via parent_id chain-walk (comment `depth` isn't returned by
    search -- V3_PLAN.md Sec 3). Memoized; unresolved chains (parent not in
    the fetched set, e.g. a removed ancestor) fall back to depth 1."""
    by_id = {c['id']: c for c in comments if c.get('id')}
    depth_cache = {}

    def depth_of(cid, guard=0):
        if cid in depth_cache:
            return depth_cache[cid]
        if guard > 200:
            return 1
        c = by_id.get(cid)
        if c is None:
            return 1
        pid = c.get('parent_id') or ''
        if pid.startswith('t3_'):
            d = 0
        else:
            parent_cid = pid[3:] if pid.startswith('t1_') else pid
            if parent_cid not in by_id:
                d = 1
            else:
                d = 1 + depth_of(parent_cid, guard + 1)
        depth_cache[cid] = d
        return d

    return {cid: depth_of(cid) for cid in by_id}


def reply_reciprocity(comments):
    """Share of directed comment->comment reply edges (child author -> parent
    author) whose reverse edge also exists in this thread. P10 in
    V3_PLAN.md Sec 4.2."""
    by_id = {c['id']: c for c in comments if c.get('id')}
    edges = set()
    for c in comments:
        pid = c.get('parent_id') or ''
        if not pid.startswith('t1_'):
            continue
        parent = by_id.get(pid[3:])
        if not parent:
            continue
        u = c.get('author')
        v = parent.get('author')
        if not u or not v or u == v or u in ('[deleted]', 'AutoModerator') or v in ('[deleted]', 'AutoModerator'):
            continue
        edges.add((u, v))
    if not edges:
        return None
    reciprocal = sum(1 for (u, v) in edges if (v, u) in edges)
    return reciprocal / len(edges)


def comment_derived_fields_and_rows(comments, sub, month_label, pid, role):
    """Everything derivable from a post's full comment list: the post-row
    comment-derived fields, plus the first10/top10 commenter rows. Shared by
    process_post (initial collection) and the truncation-repair script so a
    repaired post gets identical treatment to a freshly-collected one."""
    n_obs = len(comments)
    unique_commenters = set(c.get('author') for c in comments
                             if c.get('author') not in (None, '[deleted]', 'AutoModerator'))
    scores = sorted(c.get('score') for c in comments if c.get('score') is not None)
    top_level = [c for c in comments if (c.get('parent_id') or '').startswith('t3_')]
    removed = [c for c in comments if (c.get('_meta') or {}).get('removal_type') or c.get('removed_by_category')]
    tomb = [c for c in comments if c.get('author') == '[deleted]' and (c.get('body') or '').strip() == '[removed]']
    bot_c = [c for c in comments if is_bot_comment(c)]
    submitter_replies = [c for c in comments if c.get('is_submitter')]

    depths = compute_depths(comments)
    depth_vals = list(depths.values())

    comments_sorted_time = sorted(comments, key=lambda c: c.get('created_utc') or 0)
    first10 = []
    seen_auth = set()
    for c in comments_sorted_time:
        a = c.get('author')
        if a in (None, '[deleted]', 'AutoModerator') or a in seen_auth:
            continue
        seen_auth.add(a)
        first10.append(c)
        if len(first10) >= 10:
            break
    gaps = []
    if len(first10) >= 2:
        ts = sorted(c['created_utc'] for c in first10)
        gaps = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]

    comments_sorted_score = sorted([c for c in comments if c.get('score') is not None],
                                    key=lambda c: -c['score'])
    top10 = []
    seen_auth2 = set()
    for c in comments_sorted_score:
        a = c.get('author')
        if a in (None, '[deleted]', 'AutoModerator') or a in seen_auth2:
            continue
        seen_auth2.add(a)
        top10.append(c)
        if len(top10) >= 10:
            break

    fields = {
        'n_comments_observed': n_obs,
        'n_unique_commenters': len(unique_commenters),
        'comment_score_p50': percentile(scores, 0.50),
        'comment_score_p90': percentile(scores, 0.90),
        'comment_score_max': (scores[-1] if scores else None),
        'comment_score_gini': gini(scores),
        'max_depth': (max(depth_vals) if depth_vals else None),
        'mean_depth': (sum(depth_vals) / len(depth_vals) if depth_vals else None),
        'pct_toplevel': (len(top_level) / n_obs if n_obs else None),
        'reply_reciprocity': reply_reciprocity(comments),
        'removed_comment_rate': (len(removed) / n_obs if n_obs else None),
        'tombstone_rate': (len(tomb) / n_obs if n_obs else None),
        'bot_comment_rate': (len(bot_c) / n_obs if n_obs else None),
        'submitter_reply_rate': (len(submitter_replies) / n_obs if n_obs else None),
        'first10_arrival_gaps': gaps,
        'first10_authors': [c.get('author') for c in first10],
        'top10_authors': [c.get('author') for c in top10],
    }

    commenter_rows = []
    for tag, group in [('first10', first10), ('top10', top10)]:
        for c in group:
            cid = c.get('id')
            commenter_rows.append({
                'sub': sub, 'month': month_label, 'post_id': pid, 'role': role,
                'commenter_tag': tag, 'author': c.get('author'),
                'author_fullname': c.get('author_fullname'),
                'comment_id': cid, 'created_utc': c.get('created_utc'),
                'score': c.get('score'), 'controversiality': c.get('controversiality'),
                'is_submitter': c.get('is_submitter'),
                'parent_is_post': (c.get('parent_id') or '').startswith('t3_'),
                'depth': depths.get(cid),
                'body': c.get('body'),
                'body_len': len(c.get('body') or ''),
                'distinguished': c.get('distinguished'),
                'meta_removal_type': (c.get('_meta') or {}).get('removal_type'),
                'meta_was_deleted_later': (c.get('_meta') or {}).get('was_deleted_later'),
            })

    return fields, commenter_rows


def process_post(post, sub, month_label, role, tier, stats):
    pid = post['id']
    link_t3 = f"t3_{pid}"
    comments = fetch_all_comments(pid, link_t3, stats)

    post_row = {
        'sub': sub, 'month': month_label, 'role': role, 'incentive_tier': tier,
        'post_id': pid, 'author': post.get('author'),
        'author_fullname': post.get('author_fullname'),
        'created_utc': post.get('created_utc'),
        'title_len': len(post.get('title') or ''),
        'selftext_len': len(post.get('selftext') or ''),
        'score': post.get('score'), 'upvote_ratio': post.get('upvote_ratio'),
        'num_comments_reported': post.get('num_comments'),
        'num_crossposts': post.get('num_crossposts'),
        'is_self': post.get('is_self'), 'domain': post.get('domain'),
        'link_flair_text': post.get('link_flair_text'),
        'over_18': post.get('over_18'),
        'removed_by_category': post.get('removed_by_category'),
        'meta_removal_type': (post.get('_meta') or {}).get('removal_type'),
        'meta_was_deleted_later': (post.get('_meta') or {}).get('was_deleted_later'),
        'meta_is_edited': (post.get('_meta') or {}).get('is_edited'),
        'subreddit_subscribers': post.get('subreddit_subscribers'),
    }
    post_row.update(derive_post_metrics(post))

    fields, commenter_rows = comment_derived_fields_and_rows(comments, sub, month_label, pid, role)
    post_row.update(fields)

    return post_row, commenter_rows


# ---------- stats / progress ----------

class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.posts_done = 0
        self.cells_done = 0
        self.start = time.time()

    def add_request(self, err=False):
        with self.lock:
            self.requests += 1
            if err:
                self.errors += 1

    def add_post(self):
        with self.lock:
            self.posts_done += 1

    def add_cell(self):
        with self.lock:
            self.cells_done += 1

    def line(self, total_cells):
        with self.lock:
            elapsed = time.time() - self.start
            rps = self.requests / elapsed if elapsed else 0
            err_rate = self.errors / self.requests if self.requests else 0
            pps = self.posts_done / elapsed if elapsed else 0
            return (f'[{elapsed/60:6.1f}m] cells {self.cells_done}/{total_cells}  '
                     f'posts {self.posts_done}  req {self.requests} ({rps:.2f}/s, '
                     f'err {err_rate:.1%})  {pps:.2f} posts/s')


# ---------- I/O ----------

def post_path(sub, month_label, out_dir):
    return os.path.join(out_dir, 'posts', f'{sub}__{month_label}.ndjson.zst')


def commenter_path(sub, month_label, out_dir):
    return os.path.join(out_dir, 'commenters', f'{sub}__{month_label}.ndjson.zst')


def write_ndjson_zst(path, rows, level=12):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    cctx = zstd.ZstdCompressor(level=level)
    with open(tmp, 'wb') as f:
        with cctx.stream_writer(f) as zf:
            for r in rows:
                zf.write((json.dumps(r, ensure_ascii=False) + '\n').encode('utf-8'))
    os.replace(tmp, path)


def cell_done(sub, month_label, out_dir):
    return os.path.exists(post_path(sub, month_label, out_dir)) and \
        os.path.exists(commenter_path(sub, month_label, out_dir))


# ---------- cell processing ----------

def process_cell(sub, tier, year, month, out_dir, workers, stats):
    month_label = f'{year}-{month:02d}'
    if cell_done(sub, month_label, out_dir):
        return 'skip'

    after, before = month_bounds(year, month)
    t0 = time.time()
    posts = fetch_month_posts(sub, after, before, stats)

    if not posts:
        write_ndjson_zst(post_path(sub, month_label, out_dir), [])
        write_ndjson_zst(commenter_path(sub, month_label, out_dir), [])
        stats.add_cell()
        return 'empty'

    posts.sort(key=lambda p: -(p.get('score') or 0))
    top = posts[:100]
    rest = posts[100:]
    counter = matched_random_sample(top, rest, k=min(20, len(rest)))
    to_process = [(p, 'top') for p in top] + [(p, 'counter') for p in counter]

    all_posts_out = []
    all_commenters_out = []
    out_lock = threading.Lock()

    def worker(item):
        p, role = item
        return process_post(p, sub, month_label, role, tier, stats)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, item) for item in to_process]
        for fut in as_completed(futs):
            pr, cr = fut.result()
            with out_lock:
                all_posts_out.append(pr)
                all_commenters_out.extend(cr)
            stats.add_post()

    write_ndjson_zst(post_path(sub, month_label, out_dir), all_posts_out)
    write_ndjson_zst(commenter_path(sub, month_label, out_dir), all_commenters_out)
    stats.add_cell()
    print(f'  [{sub}/{month_label}] {len(to_process)} posts, {len(all_commenters_out)} '
          f'commenter rows, {time.time()-t0:.0f}s', flush=True)
    return 'done'


def load_subs(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return [(r['sub_name'].replace('r/', ''), r['incentive_tier']) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4,
                     help='concurrent comment-thread fetches per cell (default 4; '
                          'confirm sustained throughput before raising -- see V3_PLAN.md Sec 2)')
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--subs-csv', default=DEFAULT_CSV)
    ap.add_argument('--start', default='2024-08')
    ap.add_argument('--end', default='2026-07')
    ap.add_argument('--only-sub', default=None, help='restrict to one sub (name w/o r/), for validation runs')
    ap.add_argument('--only-months', default=None, help='comma-separated YYYY-MM list, for validation runs')
    ap.add_argument('--progress-every', type=int, default=1, help='print a stats line every N cells')
    args = ap.parse_args()

    subs = load_subs(args.subs_csv)
    if args.only_sub:
        subs = [(s, t) for (s, t) in subs if s == args.only_sub]
        if not subs:
            print(f'sub {args.only_sub!r} not found in {args.subs_csv}', file=sys.stderr)
            sys.exit(1)

    if args.only_months:
        months = [tuple(map(int, ym.split('-'))) for ym in args.only_months.split(',')]
    else:
        months = month_range(args.start, args.end)

    total_cells = len(subs) * len(months)
    print(f'{len(subs)} subs x {len(months)} months = {total_cells} cells -> {args.out_dir}', flush=True)

    stats = Stats()
    n = 0
    for sub, tier in subs:
        for (y, m) in months:
            result = process_cell(sub, tier, y, m, args.out_dir, args.workers, stats)
            n += 1
            if result != 'skip' and n % args.progress_every == 0:
                print(stats.line(total_cells), flush=True)

    print('\n' + stats.line(total_cells))
    print('DONE.')


if __name__ == '__main__':
    main()
