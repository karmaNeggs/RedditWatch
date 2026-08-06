#!/usr/bin/env python3
"""V3 pilot collection: 3 subs (one per incentive tier) x 3 months.
Top-100-by-score posts + 20 num_comments-matched random counter-sample per
sub-month; full comment-thread scan per post (compute-then-discard); first-10
and top-10-by-score commenters extracted and stored per post."""
import json, time, math, random, sys
import datetime as dt
import urllib.request, urllib.parse, urllib.error

B = 'https://arctic-shift.photon-reddit.com/api'
SP = '/private/tmp/claude-501/-Users-anupamvashist-Documents-Project-writeups-Analysis-Report--reddit-bot-analysis/e3d807eb-e38c-4468-a2f9-f863e737a2ad/scratchpad/'
OUT = SP + 'pilot/'
import os
os.makedirs(OUT, exist_ok=True)

random.seed(42)

SUBS = [
    ('IndiaSpeaks', 'high'),
    ('IndianStockMarket', 'medium'),
    ('ISRO', 'low'),
]
MONTHS = [(2026, 5), (2026, 6), (2026, 7)]

def get(url, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'v3-pilot/1.0'})
            with urllib.request.urlopen(req, timeout=90) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (a + 1))
        except Exception:
            time.sleep(2 * (a + 1))
    return None

def month_bounds(y, m):
    a = int(dt.datetime(y, m, 1).timestamp())
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    b = int(dt.datetime(ny, nm, 1).timestamp())
    return a, b

def fetch_month_posts(sub, after, before):
    out = {}
    cur = after
    pages = 0
    while True:
        d = get(f'{B}/posts/search?subreddit={urllib.parse.quote(sub)}&after={cur}'
                 f'&before={before}&limit=auto&sort=asc')
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
        if newcur == cur:  # no progress -- boundary post keeps re-returning, stop
            break
        cur = newcur
        if pages % 20 == 0:
            print(f'    ...page {pages}, {len(out)} posts so far', flush=True)
        if pages > 2000:  # sanity cap
            break
    return list(out.values())

def fetch_all_comments(post_id, link_t3):
    """Paginate the full comment set for a post via comments/search."""
    out = {}
    cur = None  # omit `after` on first page -- after=0 is a 400 Bad Request
    while True:
        after_part = f'&after={cur}' if cur is not None else ''
        d = get(f'{B}/comments/search?link_id={link_t3}{after_part}&limit=auto&sort=asc')
        if not d:
            break
        items = d.get('data', [])
        if not items:
            break
        for it in items:
            out[it['id']] = it
        last = max(i['created_utc'] for i in items)
        if cur == last * 1000 - 1:  # no progress guard
            break
        cur = last * 1000 - 1
        if len(out) > 20000:
            break
    return list(out.values())

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

def matched_random_sample(top_posts, rest_posts, k=20):
    """Pick k posts from rest_posts whose num_comments distribution matches
    top_posts' distribution as closely as possible (nearest-neighbour on
    num_comments, sampled without replacement)."""
    if not rest_posts:
        return []
    top_nc = sorted((p.get('num_comments') or 0) for p in top_posts)
    targets = [top_nc[int(q * (len(top_nc) - 1))] for q in
               [i / (k - 1) for i in range(k)]] if len(top_nc) > 1 else [top_nc[0]] * k
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

BOT_SELF_DECLARE = ['i am a bot', 'beep boop', 'performed automatically', 'this action was performed automatically']

def is_bot_comment(c):
    author = (c.get('author') or '')
    if author == 'AutoModerator' or author.endswith('Bot') or author.endswith('bot'):
        return True
    body = (c.get('body') or '').lower()
    return any(p in body for p in BOT_SELF_DECLARE)

def process_post(post, sub, month_label, role):
    pid = post['id']
    link_t3 = f"t3_{pid}"
    comments = fetch_all_comments(pid, link_t3)

    post_row = {
        'sub': sub, 'month': month_label, 'role': role,  # role: 'top' or 'counter'
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
        'removed_by_category': post.get('removed_by_category'),
        'meta_removal_type': (post.get('_meta') or {}).get('removal_type'),
        'subreddit_subscribers': post.get('subreddit_subscribers'),
    }
    post_row.update(derive_post_metrics(post))

    n_obs = len(comments)
    unique_commenters = set(c.get('author') for c in comments if c.get('author') not in (None, '[deleted]', 'AutoModerator'))
    scores = [c.get('score') for c in comments if c.get('score') is not None]
    top_level = [c for c in comments if (c.get('parent_id') or '').startswith('t3_')]
    removed = [c for c in comments if (c.get('_meta') or {}).get('removal_type') or c.get('removed_by_category')]
    tomb = [c for c in comments if c.get('author') == '[deleted]' and (c.get('body') or '').strip() == '[removed]']
    bot_c = [c for c in comments if is_bot_comment(c)]
    submitter_replies = [c for c in comments if c.get('is_submitter')]

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
        gaps = [ts[i+1] - ts[i] for i in range(len(ts) - 1)]

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

    post_row.update({
        'n_comments_observed': n_obs,
        'n_unique_commenters': len(unique_commenters),
        'comment_score_p50': (sorted(scores)[len(scores)//2] if scores else None),
        'comment_score_max': (max(scores) if scores else None),
        'comment_score_gini': gini(scores),
        'pct_toplevel': (len(top_level) / n_obs if n_obs else None),
        'removed_comment_rate': (len(removed) / n_obs if n_obs else None),
        'tombstone_rate': (len(tomb) / n_obs if n_obs else None),
        'bot_comment_rate': (len(bot_c) / n_obs if n_obs else None),
        'submitter_reply_rate': (len(submitter_replies) / n_obs if n_obs else None),
        'first10_arrival_gaps': gaps,
        'first10_authors': [c.get('author') for c in first10],
        'top10_authors': [c.get('author') for c in top10],
    })

    commenter_rows = []
    for tag, group in [('first10', first10), ('top10', top10)]:
        for c in group:
            commenter_rows.append({
                'sub': sub, 'month': month_label, 'post_id': pid, 'role': role,
                'commenter_tag': tag, 'author': c.get('author'),
                'author_fullname': c.get('author_fullname'),
                'comment_id': c.get('id'), 'created_utc': c.get('created_utc'),
                'score': c.get('score'), 'controversiality': c.get('controversiality'),
                'is_submitter': c.get('is_submitter'),
                'parent_is_post': (c.get('parent_id') or '').startswith('t3_'),
                'body_len': len(c.get('body') or ''),
                'distinguished': c.get('distinguished'),
            })

    return post_row, commenter_rows

def process_cell(sub, tier, year, month):
    month_label = f'{year}-{month:02d}'
    after, before = month_bounds(year, month)
    t0 = time.time()
    print(f'  [{sub}/{month_label}] fetching month posts...', flush=True)
    posts = fetch_month_posts(sub, after, before)
    print(f'  [{sub}/{month_label}] {len(posts)} posts fetched in {time.time()-t0:.0f}s', flush=True)
    if not posts:
        return [], []

    posts.sort(key=lambda p: -(p.get('score') or 0))
    top = posts[:100]
    rest = posts[100:]
    counter = matched_random_sample(top, rest, k=min(20, len(rest)))

    all_rows, all_commenters = [], []
    total = len(top) + len(counter)
    for i, p in enumerate(top, 1):
        pr, cr = process_post(p, sub, month_label, 'top')
        pr['incentive_tier'] = tier
        all_rows.append(pr); all_commenters.extend(cr)
        if i % 25 == 0:
            print(f'    top {i}/{len(top)}  ({time.time()-t0:.0f}s elapsed)', flush=True)
    for i, p in enumerate(counter, 1):
        pr, cr = process_post(p, sub, month_label, 'counter')
        pr['incentive_tier'] = tier
        all_rows.append(pr); all_commenters.extend(cr)
    print(f'  [{sub}/{month_label}] DONE {total} posts, {len(all_commenters)} commenter rows, '
          f'{time.time()-t0:.0f}s total', flush=True)
    return all_rows, all_commenters

def main():
    all_posts, all_commenters = [], []
    for sub, tier in SUBS:
        for (y, m) in MONTHS:
            pr, cr = process_cell(sub, tier, y, m)
            all_posts.extend(pr)
            all_commenters.extend(cr)
            # checkpoint after every cell
            with open(OUT + 'posts.json', 'w') as f:
                json.dump(all_posts, f)
            with open(OUT + 'commenters.json', 'w') as f:
                json.dump(all_commenters, f)
    print(f'\nTOTAL: {len(all_posts)} post rows, {len(all_commenters)} commenter rows', flush=True)

if __name__ == '__main__':
    main()
