#!/usr/bin/env python3
"""
Reddit Bot Analysis - Comprehensive Analysis Script
Run after data collection to perform multi-faceted bot detection
Usage: python3 analyze_data.py
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob
import os

def load_latest_data():
    """Load the latest collected data"""
    if os.path.exists('data/reddit_data_latest.csv'):
        return pd.read_csv('data/reddit_data_latest.csv')
    else:
        # Find the most recent file
        files = glob.glob('data/reddit_data_*.csv')
        if not files:
            raise FileNotFoundError("No data files found. Run collect_data.py first.")
        latest_file = max(files, key=os.path.getctime)
        return pd.read_csv(latest_file)

def analyze_user_level(df):
    """Analyze user-level bot indicators"""
    user_analysis = {}
    
    for sub in df['subreddit'].unique():
        sub_data = df[df['subreddit'] == sub].dropna(subset=['total_karma'])
        
        if len(sub_data) == 0:
            continue
        
        suspicious_accounts = len(sub_data[sub_data['karma_per_day'] > 100])
        very_suspicious = len(sub_data[sub_data['karma_per_day'] > 500])
        
        user_analysis[sub] = {
            'users_analyzed': len(sub_data),
            'avg_account_age_days': sub_data['account_age_days'].mean(),
            'avg_total_karma': sub_data['total_karma'].mean(),
            'avg_karma_per_day': sub_data['karma_per_day'].mean(),
            'median_karma_per_day': sub_data['karma_per_day'].median(),
            'suspicious_accounts_count': suspicious_accounts,
            'suspicious_accounts_pct': (suspicious_accounts / len(sub_data)) * 100,
            'very_suspicious_count': very_suspicious,
            'karma_per_day_std': sub_data['karma_per_day'].std(),
        }
    
    return user_analysis

def analyze_engagement(df):
    """Analyze post-level engagement patterns"""
    post_analysis = {}
    
    for sub in df['subreddit'].unique():
        sub_posts = df[df['subreddit'] == sub]
        
        avg_score = sub_posts['score'].mean()
        avg_comments = sub_posts['num_comments'].mean()
        avg_upvote_ratio = sub_posts['upvote_ratio'].mean()
        
        ucr = avg_score / max(avg_comments, 1)
        
        post_analysis[sub] = {
            'posts_analyzed': len(sub_posts),
            'avg_score': avg_score,
            'median_score': sub_posts['score'].median(),
            'avg_comments': avg_comments,
            'median_comments': sub_posts['num_comments'].median(),
            'avg_upvote_ratio': avg_upvote_ratio,
            'median_upvote_ratio': sub_posts['upvote_ratio'].median(),
            'ucr': ucr,
            'score_std': sub_posts['score'].std(),
            'comments_std': sub_posts['num_comments'].std(),
        }
    
    return post_analysis

def analyze_temporal(df):
    """Analyze temporal posting patterns"""
    df['created_datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['hour_of_day'] = df['created_datetime'].dt.hour
    
    temporal_analysis = {}
    
    for sub in df['subreddit'].unique():
        sub_posts = df[df['subreddit'] == sub]
        
        hour_dist = sub_posts['hour_of_day'].value_counts().sort_index()
        
        peak_hour = hour_dist.idxmax()
        peak_count = hour_dist.max()
        
        top_3_hours = hour_dist.nlargest(3).sum()
        concentration = (top_3_hours / len(sub_posts)) * 100
        
        hour_probs = hour_dist / len(sub_posts)
        entropy = -np.sum(hour_probs * np.log2(hour_probs + 1e-10))
        
        temporal_analysis[sub] = {
            'total_posts': len(sub_posts),
            'peak_hour_utc': int(peak_hour),
            'peak_hour_count': int(peak_count),
            'top_3_hours_concentration': concentration,
            'entropy': entropy,
            'hours_with_posts': len(hour_dist),
        }
    
    return temporal_analysis

def analyze_distribution(df):
    """Analyze statistical distribution patterns"""
    distribution_analysis = {}
    
    for sub in df['subreddit'].unique():
        sub_posts = df[df['subreddit'] == sub]
        
        score_cv = sub_posts['score'].std() / sub_posts['score'].mean()
        comments_cv = sub_posts['num_comments'].std() / sub_posts['num_comments'].mean()
        
        score_skew = sub_posts['score'].skew()
        comments_skew = sub_posts['num_comments'].skew()
        
        q1_score = sub_posts['score'].quantile(0.25)
        q3_score = sub_posts['score'].quantile(0.75)
        iqr_score = q3_score - q1_score
        outliers_score = len(sub_posts[(sub_posts['score'] > q3_score + 1.5*iqr_score)])
        
        distribution_analysis[sub] = {
            'score_cv': score_cv,
            'comments_cv': comments_cv,
            'score_skew': score_skew,
            'comments_skew': comments_skew,
            'score_outliers': outliers_score,
            'outlier_pct': (outliers_score / len(sub_posts)) * 100,
        }
    
    return distribution_analysis

def calculate_unified_scores(user_analysis, post_analysis, temporal_analysis, distribution_analysis):
    """Calculate unified bot activity scores"""
    unified_scores = {}
    
    for sub in user_analysis.keys():
        if sub not in post_analysis or sub not in temporal_analysis:
            continue
        
        # Component 1: User-level (35%)
        user_score = min(
            (user_analysis[sub]['suspicious_accounts_pct'] * 0.6) +
            (min(user_analysis[sub]['avg_karma_per_day'] / 10, 100) * 0.4),
            100
        )
        
        # Component 2: Engagement (30%)
        ucr = post_analysis[sub]['ucr']
        upvote_ratio = post_analysis[sub]['avg_upvote_ratio']
        engagement_score = min(
            ((ucr / 30) * 50) +
            ((upvote_ratio - 0.90) * 500 * 0.5),
            100
        )
        
        # Component 3: Temporal (20%)
        temporal_score = min(
            (temporal_analysis[sub]['top_3_hours_concentration'] * 2) +
            ((3 - temporal_analysis[sub]['entropy']) * 10),
            100
        )
        
        # Component 4: Distribution (15%)
        distribution_score = min(
            (distribution_analysis[sub]['score_cv'] * 20) +
            (distribution_analysis[sub]['outlier_pct'] * 2),
            100
        )
        
        # Final score
        final_score = (
            (user_score * 0.35) +
            (engagement_score * 0.30) +
            (temporal_score * 0.20) +
            (distribution_score * 0.15)
        )
        
        unified_scores[sub] = {
            'final_score': final_score,
            'user_score': user_score,
            'engagement_score': engagement_score,
            'temporal_score': temporal_score,
            'distribution_score': distribution_score,
        }
    
    return unified_scores

def main():
    print("\n" + "="*80)
    print("REDDIT BOT ANALYSIS - COMPREHENSIVE ANALYSIS")
    print("Timestamp: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    # Load data
    print("\n📂 Loading data...")
    df = load_latest_data()
    print(f"✓ Loaded {len(df)} posts")
    
    # Run analyses
    print("\n🔍 Running analyses...")
    user_analysis = analyze_user_level(df)
    print("  ✓ User-level analysis complete")
    
    post_analysis = analyze_engagement(df)
    print("  ✓ Engagement analysis complete")
    
    temporal_analysis = analyze_temporal(df)
    print("  ✓ Temporal analysis complete")
    
    distribution_analysis = analyze_distribution(df)
    print("  ✓ Distribution analysis complete")
    
    # Calculate unified scores
    print("\n📊 Calculating unified scores...")
    unified_scores = calculate_unified_scores(
        user_analysis, post_analysis, temporal_analysis, distribution_analysis
    )
    
    # Sort by final score
    sorted_scores = sorted(unified_scores.items(), key=lambda x: x[1]['final_score'], reverse=True)
    
    print("\n" + "="*80)
    print("FINAL BOT ACTIVITY RANKINGS")
    print("="*80)
    
    for rank, (sub, scores) in enumerate(sorted_scores, 1):
        severity = 'CRITICAL' if scores['final_score'] > 70 else 'HIGH' if scores['final_score'] > 55 else 'MODERATE' if scores['final_score'] > 40 else 'LOW'
        print(f"\n#{rank} {sub:25} | Score: {scores['final_score']:6.1f}/100 | {severity}")
        print(f"     User: {scores['user_score']:6.1f} | Engagement: {scores['engagement_score']:6.1f} | Temporal: {scores['temporal_score']:6.1f} | Distribution: {scores['distribution_score']:6.1f}")
    
    # Save comprehensive results
    comprehensive_data = {
        'analysis_date': datetime.now().isoformat(),
        'user_analysis': user_analysis,
        'post_analysis': post_analysis,
        'temporal_analysis': temporal_analysis,
        'distribution_analysis': distribution_analysis,
        'unified_scores': unified_scores,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'output/analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    # Also save as latest
    with open('output/analysis_latest.json', 'w') as f:
        json.dump(comprehensive_data, f, indent=2, default=str)
    
    print(f"\n✅ Analysis saved to: {output_file}")
    print("="*80 + "\n")
    
    return output_file

if __name__ == '__main__':
    main()
