#!/usr/bin/env python3
"""
Reddit Bot Analysis - Scoring Engine
Run after collect_data.py. Writes timestamped JSON to output/.
Usage: python3 scripts/analyze_data.py
"""

import pandas as pd
import numpy as np
import json
import os
import glob
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'


def load_latest_data():
    latest = DATA_DIR / 'reddit_data_latest.csv'
    if latest.exists():
        df = pd.read_csv(latest)
    else:
        files = sorted(glob.glob(str(DATA_DIR / 'reddit_data_*.csv')))
        if not files:
            raise FileNotFoundError("No data files found. Run collect_data.py first.")
        df = pd.read_csv(max(files, key=os.path.getctime))

    subs_file = ROOT / 'subreddits.txt'
    if subs_file.exists():
        allowed = {s.strip() for s in subs_file.read_text().splitlines() if s.strip()}
        df = df[df['subreddit'].isin(allowed)]
    return df


# ── Component 1: User account patterns (35%) ─────────────────────────────────
# Covers both post authors and top 5 commenters per post (when available).

def _suspicious_score(kpd_series, age_series):
    """Given karma_per_day and account_age_days series, return 0-100 score."""
    df = pd.DataFrame({'kpd': kpd_series, 'age': age_series}).dropna()
    if df.empty:
        return 0.0, 0
    suspicious_mask = (df['kpd'] > 200) | ((df['age'] < 90) & (df['kpd'] > 50))
    very_suspicious  = (df['kpd'] > 1000)
    n = len(df)
    susp_pct  = suspicious_mask.sum() / n * 100
    vsusp_pct = very_suspicious.sum()  / n * 100
    return float(min(max(0, susp_pct * 1.2 + vsusp_pct * 0.28), 100)), n

def analyze_users(df):
    has_commenters = 'commenter_avg_kpd' in df.columns
    result = {}
    for sub in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == sub]

        # Poster score
        poster_score, n_posters = _suspicious_score(
            sub_df['karma_per_day'], sub_df.get('account_age_days', pd.Series(dtype=float))
        )

        # Commenter score (only if new-style CSV has those columns)
        commenter_score = 0.0
        n_commenters = 0
        if has_commenters:
            c_df = sub_df[['commenter_avg_kpd', 'commenters_checked']].dropna()
            if not c_df.empty:
                total_checked = c_df['commenters_checked'].sum()
                # Approximate: treat avg_kpd per post as a single account proxy
                suspicious_posts = (c_df['commenter_avg_kpd'] > 200).sum()
                susp_pct = suspicious_posts / len(c_df) * 100
                commenter_score = float(min(max(0, susp_pct * 1.2), 100))
                n_commenters = int(total_checked)

        # Weight: 60% poster, 40% commenter (falls back to poster-only if no data)
        if n_commenters > 0:
            score = poster_score * 0.6 + commenter_score * 0.4
        else:
            score = poster_score

        poster_df = sub_df.dropna(subset=['karma_per_day'])
        result[sub] = {
            'users_analyzed': n_posters + n_commenters,
            'posters_analyzed': n_posters,
            'commenters_analyzed': n_commenters,
            'avg_karma_per_day': float(poster_df['karma_per_day'].mean()) if not poster_df.empty else 0,
            'poster_score': float(poster_score),
            'commenter_score': float(commenter_score),
            'user_score': float(score),
        }
    return result


# ── Component 2: Engagement patterns (30%) ───────────────────────────────────

def analyze_engagement(df):
    result = {}
    for sub in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == sub]

        avg_score = float(sub_df['score'].mean())
        avg_comments = float(sub_df['num_comments'].mean())
        avg_upvote_ratio = float(sub_df['upvote_ratio'].mean())
        ucr = avg_score / max(avg_comments, 1)

        # High UCR = bots upvote without commenting → up to 65 pts
        # UCR > 30 is very suspicious
        ucr_component = min((ucr / 30) * 65, 65)

        # High upvote consensus = coordinated voting
        # Maps 85%–100% ratio → 0–35 pts; never goes negative
        upvote_component = max(0, min((avg_upvote_ratio - 0.85) / 0.15 * 35, 35))

        engagement_score = min(max(0, ucr_component + upvote_component), 100)

        result[sub] = {
            'posts_analyzed': len(sub_df),
            'avg_score': avg_score,
            'median_score': float(sub_df['score'].median()),
            'avg_comments': avg_comments,
            'median_comments': float(sub_df['num_comments'].median()),
            'avg_upvote_ratio': avg_upvote_ratio,
            'ucr': float(ucr),
            'engagement_score': float(engagement_score),
        }
    return result


# ── Component 3: Temporal patterns (20%) ─────────────────────────────────────

def analyze_temporal(df):
    df = df.copy()
    df['hour'] = pd.to_datetime(df['created_utc'], unit='s').dt.hour
    result = {}

    for sub in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == sub]
        hour_counts = sub_df['hour'].value_counts().sort_index()

        peak_hour = int(hour_counts.idxmax())
        top3 = hour_counts.nlargest(3).sum()
        concentration = float((top3 / len(sub_df)) * 100)

        probs = hour_counts / len(sub_df)
        entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
        entropy_max = np.log2(24)  # 4.585 — perfectly uniform across 24h

        # High concentration in top 3 hours → up to 60 pts
        concentration_component = min(concentration * 1.0, 60)
        # Low entropy = predictable = automated → up to 40 pts
        entropy_component = max(0, (1.0 - entropy / entropy_max) * 40)
        temporal_score = min(max(0, concentration_component + entropy_component), 100)

        result[sub] = {
            'total_posts': len(sub_df),
            'peak_hour_utc': peak_hour,
            'peak_hour_count': int(hour_counts.max()),
            'top_3_hours_concentration': concentration,
            'entropy': entropy,
            'hours_with_posts': len(hour_counts),
            'temporal_score': float(temporal_score),
        }
    return result


# ── Component 4: Score distribution anomalies (15%) ──────────────────────────

def analyze_distribution(df):
    result = {}
    for sub in df['subreddit'].unique():
        sub_df = df[df['subreddit'] == sub]

        score_mean = sub_df['score'].mean()
        score_std = sub_df['score'].std()
        score_cv = score_std / score_mean if score_mean > 0 else 0

        comments_mean = sub_df['num_comments'].mean()
        comments_cv = sub_df['num_comments'].std() / comments_mean if comments_mean > 0 else 0

        # Low CV = suspiciously uniform votes (bots give similar counts to all posts)
        # Organic communities: CV typically 1.5–3.0; bots: CV < 0.5–0.8
        # Threshold 0.8: below it is suspicious, maps linearly to 0–70 pts
        score_uniformity = max(0, min((0.8 - score_cv) / 0.8 * 70, 70))
        comment_uniformity = max(0, min((0.8 - comments_cv) / 0.8 * 30, 30))
        distribution_score = min(max(0, score_uniformity + comment_uniformity), 100)

        result[sub] = {
            'score_cv': float(score_cv),
            'comments_cv': float(comments_cv),
            'score_skew': float(sub_df['score'].skew()),
            'comments_skew': float(sub_df['num_comments'].skew()),
            'distribution_score': float(distribution_score),
        }
    return result


# ── Unified scoring ───────────────────────────────────────────────────────────

def calculate_scores(user_a, engagement_a, temporal_a, distribution_a):
    scores = {}
    for sub in user_a:
        if sub not in engagement_a or sub not in temporal_a or sub not in distribution_a:
            continue

        u = user_a[sub]['user_score']
        e = engagement_a[sub]['engagement_score']
        t = temporal_a[sub]['temporal_score']
        d = distribution_a[sub]['distribution_score']

        final = (u * 0.35) + (e * 0.30) + (t * 0.20) + (d * 0.15)
        scores[sub] = {
            'final_score': float(final),
            'user_score': float(u),
            'engagement_score': float(e),
            'temporal_score': float(t),
            'distribution_score': float(d),
        }
    return scores


def severity(score):
    if score >= 70: return 'CRITICAL'
    if score >= 50: return 'HIGH'
    if score >= 30: return 'MODERATE'
    return 'LOW'


def main():
    print("\n" + "=" * 70)
    print("REDDIT BOT ANALYSIS — SCORING ENGINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    OUTPUT_DIR.mkdir(exist_ok=True)

    print("\nLoading data...")
    df = load_latest_data()
    print(f"  {len(df)} posts across {df['subreddit'].nunique()} subreddits")

    print("\nRunning analyses...")
    user_a = analyze_users(df)
    print("  User account patterns done")
    engagement_a = analyze_engagement(df)
    print("  Engagement patterns done")
    temporal_a = analyze_temporal(df)
    print("  Temporal patterns done")
    distribution_a = analyze_distribution(df)
    print("  Distribution analysis done")

    scores = calculate_scores(user_a, engagement_a, temporal_a, distribution_a)

    print("\n" + "=" * 70)
    print("BOT ACTIVITY RANKINGS")
    print("=" * 70)
    for rank, (sub, sc) in enumerate(
        sorted(scores.items(), key=lambda x: x[1]['final_score'], reverse=True), 1
    ):
        sev = severity(sc['final_score'])
        print(f"\n#{rank} r/{sub:<25} Score: {sc['final_score']:5.1f}/100  [{sev}]")
        print(f"     User:{sc['user_score']:5.1f}  Engagement:{sc['engagement_score']:5.1f}  "
              f"Temporal:{sc['temporal_score']:5.1f}  Distribution:{sc['distribution_score']:5.1f}")

    output = {
        'analysis_date': datetime.now().isoformat(),
        'user_analysis': user_a,
        'post_analysis': engagement_a,
        'temporal_analysis': temporal_a,
        'distribution_analysis': distribution_a,
        'unified_scores': scores,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f'analysis_{timestamp}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    with open(OUTPUT_DIR / 'analysis_latest.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved: {out_path}")
    print("=" * 70 + "\n")
    return str(out_path)


if __name__ == '__main__':
    main()
