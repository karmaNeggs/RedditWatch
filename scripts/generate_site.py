#!/usr/bin/env python3
"""
Generate static site data from analysis outputs.
Reads all output/analysis_*.json files and writes docs/data/ files
consumed by the GitHub Pages website.
Usage: python3 scripts/generate_site.py
"""

import json
import glob
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'output'
DOCS_DATA_DIR = ROOT / 'docs' / 'data'


def get_severity(score):
    if score >= 70: return 'critical'
    if score >= 40: return 'high'
    if score >= 20: return 'moderate'
    return 'low'


def load_analysis_files():
    files = sorted(glob.glob(str(OUTPUT_DIR / 'analysis_*.json')))
    results = []
    for f in files:
        try:
            with open(f) as fh:
                raw = json.load(fh)
            dt = datetime.fromisoformat(raw['analysis_date'])
            month = dt.strftime('%Y-%m')
            results.append((month, dt, raw))
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")
    return results


def build_month_doc(month, dt, raw):
    scores     = raw.get('unified_scores', {})
    user_a     = raw.get('user_analysis', {})
    post_a     = raw.get('post_analysis', {})
    temporal_a = raw.get('temporal_analysis', {})
    dist_a     = raw.get('distribution_analysis', {})
    network_a  = raw.get('network_analysis', {})
    regular_a  = raw.get('post_regularity', {})
    eng_str_a  = raw.get('engagement_structure', {})
    astro_a    = raw.get('astroturf_density', {})

    subreddits = {}
    for sub, sc in scores.items():
        u  = user_a.get(sub, {})
        p  = post_a.get(sub, {})
        t  = temporal_a.get(sub, {})
        d  = dist_a.get(sub, {})
        n  = network_a.get(sub, {})
        r  = regular_a.get(sub, {})
        es = eng_str_a.get(sub, {})
        at = astro_a.get(sub, {})

        subreddits[sub] = {
            'final_score':        round(sc.get('final_score', 0), 1),
            'user_score':         round(sc.get('user_score', 0), 1),
            'engagement_score':   round(sc.get('engagement_score', 0), 1),
            'temporal_score':     round(sc.get('temporal_score', 0), 1),
            'distribution_score': round(sc.get('distribution_score', 0), 1),
            'details': {
                'posts_analyzed':       p.get('posts_analyzed', 0),
                'users_analyzed':       u.get('users_analyzed', 0),
                'avg_account_age_days': round(u.get('avg_account_age_days', 0)),
                'avg_karma_per_day':    round(u.get('avg_karma_per_day', 0), 1),
                'suspicious_pct':       round(u.get('suspicious_accounts_pct', 0), 1),
                'avg_upvote_ratio':     round(p.get('avg_upvote_ratio', 0), 3),
                'avg_score':            round(p.get('avg_score', 0)),
                'avg_comments':         round(p.get('avg_comments', 0), 1),
                'ucr':                  round(p.get('ucr', 0), 1),
                'peak_hour_utc':        t.get('peak_hour_utc', 0),
                'top_3_concentration':  round(t.get('top_3_hours_concentration', 0), 1),
                'entropy':              round(t.get('entropy', 0), 2),
                'score_cv':             round(d.get('score_cv', 0), 3),
                'comments_cv':          round(d.get('comments_cv', 0), 3),
                # new signals
                'cross_sub_author_pct':    n.get('cross_sub_author_pct'),
                'multi_sub_author_pct':    n.get('multi_sub_author_pct'),
                'interval_cv':             r.get('interval_cv'),
                'mean_interval_hours':     r.get('mean_interval_hours'),
                'score_comment_corr':      es.get('score_comment_corr'),
                'upvote_ratio_std':        es.get('upvote_ratio_std'),
                'fully_coordinated_pct':   at.get('fully_coordinated_pct'),
                'poster_only_susp_pct':    at.get('poster_only_susp_pct'),
                'commenter_only_susp_pct': at.get('commenter_only_susp_pct'),
            },
        }

    net_summary = network_a.get('_network', {})
    return {
        'month':         month,
        'analysis_date': dt.isoformat(),
        'network': {
            'total_unique_authors':     net_summary.get('total_unique_authors', 0),
            'cross_sub_authors_2plus':  net_summary.get('cross_sub_2plus', 0),
            'cross_sub_authors_3plus':  net_summary.get('cross_sub_3plus', 0),
            'top_cross_sub_authors':    net_summary.get('top_cross_sub_authors', {}),
        },
        'subreddits': subreddits,
    }


def main():
    print("\n" + "=" * 70)
    print("GENERATE SITE DATA")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    entries = load_analysis_files()
    if not entries:
        print("No analysis files found in output/. Run analyze_data.py first.")
        return

    # Keep only the latest run per calendar month
    by_month = {}
    for month, dt, raw in entries:
        if month not in by_month or dt > by_month[month][0]:
            by_month[month] = (dt, raw)

    history_months = []
    for month in sorted(by_month):
        dt, raw = by_month[month]
        doc = build_month_doc(month, dt, raw)

        out_path = DOCS_DATA_DIR / f'{month}.json'
        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"  Wrote {out_path.name}")

        all_scores = [d['final_score'] for d in doc['subreddits'].values()]
        history_months.append({
            'month': month,
            'analysis_date': dt.isoformat(),
            'data_file': f'{month}.json',
            'aggregate': {
                'avg_score': round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
                'pct_moderate_plus': round(sum(1 for s in all_scores if s >= 20) / len(all_scores) * 100) if all_scores else 0,
                'pct_high_plus': round(sum(1 for s in all_scores if s >= 40) / len(all_scores) * 100) if all_scores else 0,
                'sub_count': len(all_scores),
                'max_score': round(max(all_scores), 1) if all_scores else 0,
                'min_score': round(min(all_scores), 1) if all_scores else 0,
            },
            'summary': {
                sub: {
                    'final_score': d['final_score'],
                    'severity': get_severity(d['final_score']),
                }
                for sub, d in doc['subreddits'].items()
            },
        })

    history = {'months': history_months}
    hist_path = DOCS_DATA_DIR / 'history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Wrote {hist_path.name}")

    print(f"\nDone. {len(by_month)} month(s) written to docs/data/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
