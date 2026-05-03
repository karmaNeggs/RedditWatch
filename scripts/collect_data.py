#!/usr/bin/env python3
"""
Reddit Bot Analysis - Data Collection Script
Run manually to collect fresh data from all 5 subreddits
Usage: python3 collect_data.py
"""

import requests
import pandas as pd
import time
import random
import json
from datetime import datetime
import sys

# Multiple User-Agent strings to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'RedditBotAnalyzer/1.0 by /u/AnalysisBot',
]

SUBREDDITS = ['india', 'unitedstatesofindia', 'indiaspeaks', 'teenindia', 'indiasocial']

def get_headers():
    """Return headers with rotating User-Agent"""
    return {
        'User-Agent': random.choice(USER_AGENTS)
    }

def fetch_with_retry(url, max_retries=15, initial_wait=5):
    """Fetch URL with exponential backoff and jitter"""
    wait_time = initial_wait
    
    for attempt in range(max_retries):
        try:
            headers = get_headers()
            print(f"    Attempt {attempt + 1}/{max_retries}...", end=' ', flush=True)
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 429:
                jitter = random.uniform(0.5, 1.5)
                actual_wait = wait_time * jitter
                print(f"Rate limited. Waiting {actual_wait:.1f}s...")
                time.sleep(actual_wait)
                wait_time = min(wait_time * 2, 120)
                continue
            
            response.raise_for_status()
            print("✓")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            if attempt < max_retries - 1:
                jitter = random.uniform(0.5, 1.5)
                actual_wait = wait_time * jitter
                print(f"    Retrying in {actual_wait:.1f}s...")
                time.sleep(actual_wait)
                wait_time = min(wait_time * 2, 120)
    
    print("Failed after all retries")
    return None

def fetch_posts(subreddit, limit=50):
    """Fetch posts from subreddit with retry logic"""
    print(f"\n  📥 Fetching posts from r/{subreddit}...")
    posts_data = []
    
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit={limit}&t=year"
    data = fetch_with_retry(url, max_retries=15, initial_wait=8)
    
    if not data or 'data' not in data or 'children' not in data['data']:
        print(f"    ✗ Failed to get data")
        return posts_data
    
    for post in data['data']['children']:
        p = post['data']
        posts_data.append({
            'subreddit': subreddit,
            'post_id': p['id'],
            'title': p['title'][:50],
            'score': p['score'],
            'upvote_ratio': p['upvote_ratio'],
            'num_comments': p['num_comments'],
            'created_utc': p['created_utc'],
            'author': p.get('author', '[deleted]'),
        })
    
    print(f"    ✓ Fetched {len(posts_data)} posts")
    return posts_data

def fetch_user_data(username, max_retries=10):
    """Fetch user data with retry logic"""
    if username == '[deleted]' or not username:
        return None
    
    url = f"https://www.reddit.com/user/{username}/about.json"
    
    for attempt in range(max_retries):
        try:
            headers = get_headers()
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 429:
                jitter = random.uniform(0.5, 1.5)
                time.sleep(5 * jitter)
                continue
            
            if response.status_code == 404 or response.status_code == 403:
                return None
            
            response.raise_for_status()
            user_data = response.json()
            
            if 'data' in user_data:
                data = user_data['data']
                created_utc = data.get('created_utc')
                total_karma = data.get('link_karma', 0) + data.get('comment_karma', 0)
                
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
                jitter = random.uniform(0.5, 1.5)
                time.sleep(3 * jitter)
    
    return None

def main():
    print("\n" + "="*80)
    print("REDDIT BOT ANALYSIS - DATA COLLECTION")
    print("Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    all_posts = []
    all_users = {}
    
    # Collect posts from all subreddits
    for subreddit in SUBREDDITS:
        posts = fetch_posts(subreddit, limit=50)
        all_posts.extend(posts)
        time.sleep(random.uniform(2, 4))
    
    # Create DataFrame
    df_posts = pd.DataFrame(all_posts)
    print(f"\n✓ Total posts collected: {len(df_posts)}")
    
    # Fetch user data for unique authors
    print(f"\n📊 Fetching user data for {df_posts['author'].nunique()} unique authors...")
    unique_authors = df_posts['author'].unique()
    users_fetched = 0
    users_failed = 0
    
    for idx, author in enumerate(unique_authors):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(unique_authors)}")
        
        user_data = fetch_user_data(author)
        
        if user_data:
            all_users[author] = user_data
            users_fetched += 1
        else:
            users_failed += 1
        
        time.sleep(random.uniform(1, 2))
    
    print(f"\n✅ Successfully fetched data for {users_fetched} users")
    print(f"❌ Failed to fetch data for {users_failed} users")
    
    # Merge user data with posts
    df_posts['total_karma'] = df_posts['author'].map(lambda x: all_users.get(x, {}).get('total_karma'))
    df_posts['account_age_days'] = df_posts['author'].map(lambda x: all_users.get(x, {}).get('account_age_days'))
    df_posts['karma_per_day'] = df_posts['author'].map(lambda x: all_users.get(x, {}).get('karma_per_day'))
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/reddit_data_{timestamp}.csv'
    df_posts.to_csv(output_file, index=False)
    
    # Also save as latest
    df_posts.to_csv('data/reddit_data_latest.csv', index=False)
    
    print(f"\n✅ Data saved to: {output_file}")
    print(f"   Total rows: {len(df_posts)}")
    print(f"   Columns: {', '.join(df_posts.columns)}")
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_posts': len(df_posts),
        'total_users': len(all_users),
        'subreddits': SUBREDDITS,
        'file': output_file
    }
    
    with open('data/metadata_latest.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80 + "\n")
    return output_file

if __name__ == '__main__':
    main()
