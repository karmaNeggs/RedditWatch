#!/usr/bin/env python3
"""
Reddit Bot Analysis - Data Collection Script
Fetches top 30 posts from the last 30 days for each subreddit in subreddits.txt.
Usage: python3 scripts/collect_data.py
"""

import requests
import pandas as pd
import time
import random
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUBREDDITS_FILE = ROOT / 'subreddits.txt'
DATA_DIR = ROOT / 'data'

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
]


def load_subreddits():
    if not SUBREDDITS_FILE.exists():
        print(f"Warning: {SUBREDDITS_FILE} not found, using defaults")
        return ['india', 'unitedstatesofindia', 'indiaspeaks', 'teenindia', 'indiasocial']
    subs = [line.strip() for line in SUBREDDITS_FILE.read_text().splitlines() if line.strip() and not line.startswith('#')]
    print(f"Loaded {len(subs)} subreddits from {SUBREDDITS_FILE.name}")
    return subs


def get_headers():
    return {'User-Agent': random.choice(USER_AGENTS)}


def fetch_with_retry(url, max_retries=12, initial_wait=6):
    wait_time = initial_wait
    for attempt in range(max_retries):
        try:
            print(f"    Attempt {attempt + 1}/{max_retries}...", end=' ', flush=True)
            response = requests.get(url, headers=get_headers(), timeout=30)

            if response.status_code == 429:
                jitter = random.uniform(0.8, 1.5)
                actual_wait = wait_time * jitter
                print(f"Rate limited. Waiting {actual_wait:.1f}s...")
                time.sleep(actual_wait)
                wait_time = min(wait_time * 2, 120)
                continue

            response.raise_for_status()
            print("OK")
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                jitter = random.uniform(0.8, 1.5)
                time.sleep(wait_time * jitter)
                wait_time = min(wait_time * 2, 120)

    print("Failed after all retries")
    return None


def fetch_posts(subreddit, limit=30):
    """Fetch top posts from last 30 days."""
    print(f"\n  Fetching r/{subreddit}...")
    # t=month = top posts from the last ~30 days (rolling window)
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t=month"
    data = fetch_with_retry(url)

    if not data or 'data' not in data:
        print(f"    Failed to get posts for r/{subreddit}")
        return []

    posts = []
    for post in data['data']['children']:
        p = post['data']
        posts.append({
            'subreddit': subreddit,
            'post_id': p['id'],
            'title': p['title'][:80],
            'score': p['score'],
            'upvote_ratio': p['upvote_ratio'],
            'num_comments': p['num_comments'],
            'created_utc': p['created_utc'],
            'author': p.get('author', '[deleted]'),
        })

    print(f"    Got {len(posts)} posts")
    return posts


def fetch_user_data(username, max_retries=8):
    if not username or username in ('[deleted]', 'AutoModerator'):
        return None

    url = f"https://www.reddit.com/user/{username}/about.json"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=get_headers(), timeout=20)

            if response.status_code == 429:
                time.sleep(random.uniform(4, 8))
                continue
            if response.status_code in (404, 403):
                return None

            response.raise_for_status()
            user_data = response.json()

            if 'data' in user_data:
                d = user_data['data']
                created_utc = d.get('created_utc')
                total_karma = d.get('link_karma', 0) + d.get('comment_karma', 0)

                if created_utc:
                    account_age_days = (datetime.now().timestamp() - created_utc) / 86400
                    karma_per_day = total_karma / max(account_age_days, 1)
                    return {
                        'username': username,
                        'account_age_days': account_age_days,
                        'total_karma': total_karma,
                        'karma_per_day': karma_per_day,
                    }
            return None

        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(random.uniform(2, 5))

    return None


def main():
    print("\n" + "=" * 70)
    print("REDDIT BOT ANALYSIS — DATA COLLECTION")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    DATA_DIR.mkdir(exist_ok=True)
    subreddits = load_subreddits()

    all_posts = []
    for subreddit in subreddits:
        posts = fetch_posts(subreddit, limit=30)
        all_posts.extend(posts)
        time.sleep(random.uniform(2, 4))

    if not all_posts:
        print("No posts collected. Exiting.")
        return None

    df = pd.DataFrame(all_posts)
    print(f"\nTotal posts collected: {len(df)}")

    unique_authors = [a for a in df['author'].unique() if a not in ('[deleted]', 'AutoModerator')]
    print(f"Fetching user data for {len(unique_authors)} unique authors...")

    all_users = {}
    for idx, author in enumerate(unique_authors):
        if idx > 0 and idx % 20 == 0:
            print(f"  Progress: {idx}/{len(unique_authors)}")

        user_data = fetch_user_data(author)
        if user_data:
            all_users[author] = user_data
        time.sleep(random.uniform(1, 2))

    print(f"  User data fetched: {len(all_users)}/{len(unique_authors)}")

    df['total_karma'] = df['author'].map(lambda x: all_users.get(x, {}).get('total_karma'))
    df['account_age_days'] = df['author'].map(lambda x: all_users.get(x, {}).get('account_age_days'))
    df['karma_per_day'] = df['author'].map(lambda x: all_users.get(x, {}).get('karma_per_day'))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = DATA_DIR / f'reddit_data_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    df.to_csv(DATA_DIR / 'reddit_data_latest.csv', index=False)

    metadata = {
        'timestamp': datetime.now().isoformat(),
        'collection_period': 'last_30_days',
        'posts_per_subreddit': 30,
        'total_posts': len(df),
        'total_users_fetched': len(all_users),
        'subreddits': subreddits,
        'file': str(output_file),
    }
    with open(DATA_DIR / 'metadata_latest.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nData saved: {output_file}")
    print(f"Rows: {len(df)}  |  Columns: {', '.join(df.columns)}")
    print("=" * 70 + "\n")
    return str(output_file)


if __name__ == '__main__':
    main()
