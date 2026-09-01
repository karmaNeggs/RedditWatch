#!/usr/bin/env python3
"""V3 tracker feasibility: which of the 45 subs have real, current data for
the last 12 months, assuming an ongoing monthly collector? Per-month
posts/comments, with explicit recency and continuity checks -- not just
totals, since a dead sub and a healthy sub can have the same 24-month sum."""
import csv, json, time, datetime as dt
import urllib.request, urllib.parse, urllib.error

B = 'https://arctic-shift.photon-reddit.com/api'
# Output dir. Was a hardcoded absolute path into a long-dead session scratchpad, which made this
# script unrunnable -- fixed 2026-09-01 to write beside the repo's other outputs.
import os
SP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'v3') + os.sep

def get(url, retries=3):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'v3-research/1.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (a + 1))
        except Exception:
            time.sleep(2 * (a + 1))
    return None

def series(sub, kind, after, before):
    d = get(f'{B}/time_series?key=r/{urllib.parse.quote(sub)}/{kind}/count&precision=month'
             f'&after={after}&before={before}')
    if not d:
        return []
    data = d.get('data', d) if isinstance(d, dict) else d
    return data if isinstance(data, list) else []

def count_via_search(sub, endpoint, after, before, max_pages=60):
    """Fallback for subs whose time_series index is empty (e.g. unitedstatesofindia).
    Paginate the raw search endpoint and count rows in-window. Hard page cap
    so a mis-triggered fallback (e.g. a transient time_series timeout) can't
    turn into an unbounded pagination of a high-volume sub."""
    n = 0
    cur = after
    pages = 0
    while pages < max_pages:
        d = get(f'{B}/{endpoint}?subreddit={urllib.parse.quote(sub)}&after={cur}'
                 f'&before={before}&limit=auto&sort=asc')
        pages += 1
        if not d:
            break
        items = d.get('data', [])
        if not items:
            break
        n += len(items)
        last = max(i['created_utc'] for i in items)
        if last >= before:
            break
        cur = last * 1000 - 1
    return n, pages >= max_pages  # (count, was_capped)

with open('/Users/anupamvashist/Documents/Project writeups/Analysis Report- reddit-bot-analysis/subreddits_v3.csv') as f:
    rows = list(csv.DictReader(f))
subs = [r['sub_name'].replace('r/', '') for r in rows]

today = dt.date.today()
# most recently COMPLETE month (current month is partial, exclude it)
last_complete = (dt.date(today.year, today.month, 1) - dt.timedelta(days=1)).replace(day=1)
end_month = dt.date(last_complete.year, last_complete.month, 1)
# 12 complete months ending at last_complete
start_month = dt.date(end_month.year, end_month.month, 1)
for _ in range(11):
    start_month = (start_month - dt.timedelta(days=1)).replace(day=1)
after_ts = int(dt.datetime(start_month.year, start_month.month, 1).timestamp())
before_ts = int(dt.datetime(end_month.year, end_month.month, 1).timestamp()) + 31*86400  # include last month fully
before_ts = int(dt.datetime(
    end_month.year + (1 if end_month.month == 12 else 0),
    1 if end_month.month == 12 else end_month.month + 1, 1).timestamp())

print(f'Tracker window: 12 complete months, {start_month} -> {end_month} (inclusive)\n', flush=True)

months = []
m = start_month
while m <= end_month:
    months.append(m)
    m = (dt.date(m.year + (1 if m.month == 12 else 0), 1 if m.month == 12 else m.month + 1, 1))

results = {}
for i, sub in enumerate(subs, 1):
    pts = series(sub, 'posts', after_ts, before_ts)
    cts = series(sub, 'comments', after_ts, before_ts)

    by_month_p = {dt.datetime.utcfromtimestamp(p['date']).strftime('%Y-%m'): p.get('value', 0) for p in pts}
    by_month_c = {dt.datetime.utcfromtimestamp(c['date']).strftime('%Y-%m'): c.get('value', 0) for c in cts}

    used_fallback = False
    capped = False
    if not pts and not cts:
        # time_series index gap -- fall back to direct search + count (last month only, to keep it cheap)
        used_fallback = True
        last_m_start = int(dt.datetime(end_month.year, end_month.month, 1).timestamp())
        last_m_end = before_ts
        p_last, p_capped = count_via_search(sub, 'posts/search', last_m_start, last_m_end)
        c_last, c_capped = count_via_search(sub, 'comments/search', last_m_start, last_m_end)
        capped = p_capped or c_capped
        by_month_p = {end_month.strftime('%Y-%m'): p_last}
        by_month_c = {end_month.strftime('%Y-%m'): c_last}

    month_labels = [mo.strftime('%Y-%m') for mo in months]
    p_series = [by_month_p.get(ml, 0) for ml in month_labels]
    c_series = [by_month_c.get(ml, 0) for ml in month_labels]

    n_months_with_posts = sum(1 for v in p_series if v and v > 0)
    latest_posts = p_series[-1] if p_series else 0
    latest_comments = c_series[-1] if c_series else 0
    prev6_avg_posts = (sum(p_series[-7:-1]) / 6) if len(p_series) >= 7 else (sum(p_series[:-1]) / max(1, len(p_series) - 1))
    prev6_avg_comments = (sum(c_series[-7:-1]) / 6) if len(c_series) >= 7 else (sum(c_series[:-1]) / max(1, len(c_series) - 1))

    still_collecting = latest_posts > 0 and (prev6_avg_posts == 0 or latest_posts >= 0.1 * prev6_avg_posts)
    comment_cliff = prev6_avg_comments > 0 and latest_comments < 0.1 * prev6_avg_comments

    results[sub] = {
        'months_with_data': n_months_with_posts,
        'coverage_12mo': round(n_months_with_posts / 12, 2),
        'latest_month': month_labels[-1] if month_labels else None,
        'latest_posts': latest_posts,
        'latest_comments': latest_comments,
        'prev6mo_avg_posts': round(prev6_avg_posts, 1),
        'prev6mo_avg_comments': round(prev6_avg_comments, 1),
        'still_collecting': still_collecting,
        'comment_cliff_flag': comment_cliff,
        'used_search_fallback': used_fallback,
        'monthly_posts': dict(zip(month_labels, p_series)),
        'monthly_comments': dict(zip(month_labels, c_series)),
    }

    flag = ''
    if not still_collecting:
        flag = ' <-- STALE/NOT COLLECTING'
    elif comment_cliff:
        flag = ' <-- comment cliff'
    if used_fallback:
        flag += f' [fallback: search-count, last month only{", CAPPED" if capped else ""}]'

    print(f'[{i:2d}/{len(subs)}] r/{sub:<24} coverage={n_months_with_posts:>2}/12  '
          f'latest({month_labels[-1] if month_labels else "?"})  '
          f'posts={latest_posts:>6}  cmts={latest_comments:>7}  '
          f'prev6avg_p={prev6_avg_posts:>7.0f}  prev6avg_c={prev6_avg_comments:>8.0f}{flag}',
          flush=True)
    time.sleep(0.25)

with open(SP + 'tracker_freshness.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '=' * 78)
stale = [s for s, r in results.items() if not r['still_collecting']]
cliff = [s for s, r in results.items() if r['comment_cliff_flag'] and r['still_collecting']]
partial = [s for s, r in results.items() if r['coverage_12mo'] < 1.0]
print(f'Fully current (posting through {end_month.strftime("%Y-%m")}): {len(subs) - len(stale)}/{len(subs)}')
if stale:
    print(f'STALE / not collecting: {stale}')
if cliff:
    print(f'Comment-cliff (posts fine, comments collapsed): {cliff}')
if partial:
    print(f'Partial 12mo coverage (<12 months with data): {partial}')
print(f'\nSaved -> {SP}tracker_freshness.json')
