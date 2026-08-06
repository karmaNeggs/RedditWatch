#!/usr/bin/env python3
"""Small go/no-go check: does the plan survive? Sample ~150 accounts/sub from
the pilot pool, pull last-50-comments timing footprint (capped, fields-limited
-- NOT full history), compute real behavioral features (interval entropy,
burstiness, circadian dead hours, subreddit entropy), and test for genuine
multimodal structure -- the thing engagement-count features failed to show
cleanly in the first pilot pass."""
import json, math, time, random
import urllib.request, urllib.parse, urllib.error
import numpy as np
import pandas as pd

B = 'https://arctic-shift.photon-reddit.com/api'
SP = '/private/tmp/claude-501/-Users-anupamvashist-Documents-Project-writeups-Analysis-Report--reddit-bot-analysis/e3d807eb-e38c-4468-a2f9-f863e737a2ad/scratchpad/'
OUT = SP + 'pilot/'

random.seed(42)

def get(url, retries=3):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'v3-pilot/1.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (a + 1))
        except Exception:
            time.sleep(1.5 * (a + 1))
    return None

def fetch_recent(author, cap=50):
    d = get(f'{B}/comments/search?author={urllib.parse.quote(author)}&limit={cap}'
             f'&sort=desc&fields=author,created_utc,subreddit')
    if not d:
        return []
    return d.get('data', [])

def shannon_entropy(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return None
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def compute_features(author, comments):
    ts = sorted(c['created_utc'] for c in comments)
    n = len(ts)
    row = {'author': author, 'n_comments_fetched': n}
    if n < 3:
        return row  # not enough for interval-based features

    gaps = np.diff(ts)
    row['active_span_days'] = round((ts[-1] - ts[0]) / 86400, 2)

    # interval entropy -- log-binned gap histogram
    log_gaps = np.log10(np.clip(gaps, 1, None))
    bins = np.linspace(0, 6, 13)  # 1s .. ~11.5 days, 12 bins
    hist, _ = np.histogram(log_gaps, bins=bins)
    row['interval_entropy'] = shannon_entropy(hist)

    # burstiness, plain Goh-Barabasi (length-bias caveat noted in report)
    mu, sigma = gaps.mean(), gaps.std()
    row['burstiness'] = round((sigma - mu) / (sigma + mu), 4) if (sigma + mu) > 0 else None
    row['n_gaps'] = len(gaps)

    # circadian dead hours -- 24 UTC bins
    hours = [time.gmtime(t).tm_hour for t in ts]
    hour_counts = np.bincount(hours, minlength=24)
    row['circadian_dead_hours'] = int((hour_counts == 0).sum())
    row['circadian_entropy'] = shannon_entropy(hour_counts)

    # subreddit entropy / diversity
    subs = [c.get('subreddit') for c in comments if c.get('subreddit')]
    sub_counts = pd.Series(subs).value_counts().values if subs else []
    row['n_unique_subs'] = len(set(subs))
    row['subreddit_entropy'] = shannon_entropy(sub_counts) if len(sub_counts) else None

    return row

def main():
    acc = pd.read_csv(OUT + 'pilot_accounts.csv')
    samples = []
    for sub, g in acc.groupby('sub'):
        n = min(150, len(g))
        s = g.sample(n=n, random_state=42)
        samples.append(s)
    sample = pd.concat(samples).reset_index(drop=True)
    print(f'Sampled {len(sample)} accounts: {sample.groupby("sub").size().to_dict()}\n', flush=True)

    rows = []
    t0 = time.time()
    for i, r in sample.iterrows():
        comments = fetch_recent(r['author'], cap=50)
        feat = compute_features(r['author'], comments)
        feat['sub'] = r['sub']
        feat['incentive_tier'] = r['incentive_tier']
        feat['pilot_n_appearances'] = r['n_appearances']
        rows.append(feat)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(sample)}  ({time.time()-t0:.0f}s elapsed)', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT + 'behavioral_check.csv', index=False)
    print(f'\nDone: {len(df)} accounts, {time.time()-t0:.0f}s total', flush=True)
    print(f'Saved -> {OUT}behavioral_check.csv')
    print(f'\ncoverage: n>=3 comments (usable for interval features): {(df.n_comments_fetched>=3).sum()}/{len(df)}')

if __name__ == '__main__':
    main()
