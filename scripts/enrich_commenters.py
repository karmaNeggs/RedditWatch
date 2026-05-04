#!/usr/bin/env python3
"""
One-time enrichment: add commenter karma/age data to an existing CSV.

Reads data/reddit_data_latest.csv (which has post_id + subreddit),
fetches top 5 commenters per post, looks up their account stats, and
writes three new columns back to the CSV:
  commenter_avg_kpd       — avg karma/day across top commenters
  commenter_suspicious    — count flagged (>200 kpd OR new+>50 kpd)
  commenters_checked      — number of commenters successfully fetched

Then re-runs analyze_data.py and generate_site.py automatically.
Usage: python3 scripts/enrich_commenters.py
"""

import requests
import pandas as pd
import time
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
]

def headers():
    return {'User-Agent': random.choice(USER_AGENTS)}

def get_json(url, retries=8, wait=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers(), timeout=20)
            if r.status_code == 429:
                sleep = wait * (1.5 ** attempt) * random.uniform(0.8, 1.2)
                print(f"    Rate limited, waiting {sleep:.1f}s…")
                time.sleep(sleep)
                continue
            if r.status_code in (404, 403):
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                time.sleep(wait * random.uniform(0.8, 1.2))
    return None

def fetch_top_commenters(subreddit, post_id, limit=5):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json?limit={limit}&sort=top&depth=1"
    data = get_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return []
    names = []
    for child in data[1]['data']['children']:
        author = child['data'].get('author', '[deleted]')
        if author and author not in ('[deleted]', 'AutoModerator'):
            names.append(author)
        if len(names) >= limit:
            break
    return names

def fetch_user(username):
    data = get_json(f"https://www.reddit.com/user/{username}/about.json")
    if not data or 'data' not in data:
        return None
    d = data['data']
    created = d.get('created_utc')
    if not created:
        return None
    age_days = (datetime.now().timestamp() - created) / 86400
    karma = d.get('link_karma', 0) + d.get('comment_karma', 0)
    return {'age_days': age_days, 'karma_per_day': karma / max(age_days, 1)}


def main():
    csv_path = DATA_DIR / 'reddit_data_latest.csv'
    if not csv_path.exists():
        print("No reddit_data_latest.csv found. Run collect_data.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    if 'commenters_checked' in df.columns:
        already = df['commenters_checked'].notna().sum()
        print(f"CSV already has commenter columns ({already} rows filled).")
        ans = input("Re-enrich anyway? [y/N] ").strip().lower()
        if ans != 'y':
            sys.exit(0)

    print(f"\nLoaded {len(df)} posts across {df['subreddit'].nunique()} subreddits.")
    print("Step 1: Fetching top 5 commenters per post…\n")

    post_commenters = {}
    total = len(df)
    for i, row in df.iterrows():
        post_id  = row['post_id']
        subreddit = row['subreddit']
        if i % 30 == 0:
            print(f"  Posts: {i}/{total}")
        names = fetch_top_commenters(subreddit, post_id)
        post_commenters[post_id] = names
        time.sleep(random.uniform(0.8, 1.5))

    all_commenters = list({n for names in post_commenters.values() for n in names})
    print(f"\nStep 2: Looking up {len(all_commenters)} unique commenter accounts…\n")

    user_cache = {}
    for idx, name in enumerate(all_commenters):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(all_commenters)}")
        data = fetch_user(name)
        if data:
            user_cache[name] = data
        time.sleep(random.uniform(1.0, 1.8))

    print(f"  Fetched: {len(user_cache)}/{len(all_commenters)}")

    print("\nStep 3: Computing per-post commenter stats…")
    avg_kpds, suspicions, checked = [], [], []
    for _, row in df.iterrows():
        names = post_commenters.get(row['post_id'], [])
        accounts = [user_cache[n] for n in names if n in user_cache]
        if not accounts:
            avg_kpds.append(None)
            suspicions.append(None)
            checked.append(0)
            continue
        kpds = [a['karma_per_day'] for a in accounts]
        ages = [a['age_days'] for a in accounts]
        susp = sum(1 for k, ag in zip(kpds, ages)
                   if k > 200 or (ag < 90 and k > 50))
        avg_kpds.append(sum(kpds) / len(kpds))
        suspicions.append(susp)
        checked.append(len(accounts))

    df['commenter_avg_kpd']    = avg_kpds
    df['commenter_suspicious'] = suspicions
    df['commenters_checked']   = checked

    df.to_csv(csv_path, index=False)
    # Also save timestamped copy
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    df.to_csv(DATA_DIR / f'reddit_data_{ts}.csv', index=False)
    print(f"Saved enriched CSV → {csv_path.name}")

    print("\nStep 4: Re-running analysis and site generation…")
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'analyze_data.py')], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts' / 'generate_site.py')], check=True)
    print("\nDone. Commit and push when ready:\n  git add -A && git commit -m 'Enrich May 2026 with commenter data' && git push")


if __name__ == '__main__':
    main()
