#!/usr/bin/env python3
"""
Fetch data for additional subreddits and append to existing dataset.
Useful when adding new subreddits mid-month without re-scraping all.
Usage: python3 scripts/append_subreddits.py Sub1 Sub2 Sub3
"""

import sys
import pandas as pd
import time
import random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from collect_data import fetch_posts, fetch_user_data, DATA_DIR

def main():
    new_subs = sys.argv[1:]
    if not new_subs:
        print("Usage: python3 scripts/append_subreddits.py Sub1 Sub2 Sub3")
        sys.exit(1)

    print(f"\nAppending data for: {', '.join('r/'+s for s in new_subs)}")

    latest_csv = DATA_DIR / 'reddit_data_latest.csv'
    if latest_csv.exists():
        existing = pd.read_csv(latest_csv)
        already = set(existing['subreddit'].unique())
        print(f"Existing data has: {', '.join('r/'+s for s in sorted(already))}")
    else:
        existing = pd.DataFrame()
        already = set()

    to_fetch = [s for s in new_subs if s not in already]
    skipped  = [s for s in new_subs if s in already]
    if skipped:
        print(f"Already in dataset, skipping: {', '.join('r/'+s for s in skipped)}")
    if not to_fetch:
        print("Nothing new to fetch.")
        return

    all_posts = []
    for sub in to_fetch:
        posts = fetch_posts(sub, limit=30)
        all_posts.extend(posts)
        time.sleep(random.uniform(2, 4))

    if not all_posts:
        print("No posts collected.")
        return

    df_new = pd.DataFrame(all_posts)
    unique_authors = [a for a in df_new['author'].unique()
                      if a not in ('[deleted]', 'AutoModerator')]
    print(f"\nFetching user data for {len(unique_authors)} authors...")

    all_users = {}
    for idx, author in enumerate(unique_authors):
        if idx > 0 and idx % 20 == 0:
            print(f"  Progress: {idx}/{len(unique_authors)}")
        u = fetch_user_data(author)
        if u:
            all_users[author] = u
        time.sleep(random.uniform(1, 2))

    df_new['total_karma']      = df_new['author'].map(lambda x: all_users.get(x, {}).get('total_karma'))
    df_new['account_age_days'] = df_new['author'].map(lambda x: all_users.get(x, {}).get('account_age_days'))
    df_new['karma_per_day']    = df_new['author'].map(lambda x: all_users.get(x, {}).get('karma_per_day'))

    combined = pd.concat([existing, df_new], ignore_index=True)
    combined.to_csv(latest_csv, index=False)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined.to_csv(DATA_DIR / f'reddit_data_{timestamp}.csv', index=False)

    print(f"\nDone. Dataset now has {len(combined)} rows across "
          f"{combined['subreddit'].nunique()} subreddits.")

if __name__ == '__main__':
    main()
