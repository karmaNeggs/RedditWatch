#!/usr/bin/env python3
"""
Generate static site data from analysis outputs.
V1: reads output/analysis_*.json        → docs/data/
V2: reads output/v2/analysis_*.json     → docs/data_v2/

Usage:
  python3 scripts/generate_site.py           # V1
  python3 scripts/generate_site.py --v2      # V2
"""

import argparse
import json
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v2', action='store_true', help='Generate V2 site data')
    p.add_argument('--month', default=None, help='Limit to specific YYYY-MM month')
    p.add_argument('--min-sub-coverage', type=float, default=0.6,
                   help='Warn (not block) if a month covers fewer than this fraction of tracked subreddits')
    return p.parse_args()


def expected_sub_count() -> int:
    subs_file = ROOT / 'subreddits.txt'
    if not subs_file.exists():
        return 0
    return len([l for l in subs_file.read_text().splitlines()
                if l.strip() and not l.startswith('#')])


def validate_month(month, doc, min_coverage, expected_subs):
    """
    Collection/scoring bugs upstream (thin scrape, a broken analysis run) should be
    loud at publish time, not silently baked into the public dashboard. Non-blocking —
    warnings are surfaced in the JSON itself and printed, but the month still publishes,
    since a genuinely thin in-progress current month is a legitimate case, not a bug.
    """
    warnings = []
    subs = doc.get('subreddits', {})

    if expected_subs and len(subs) < expected_subs * min_coverage:
        warnings.append(f'only {len(subs)}/{expected_subs} tracked subreddits present '
                         f'(<{min_coverage*100:.0f}% coverage) — check for a partial/failed collection run')

    bad_scores = [sub for sub, d in subs.items()
                  if d.get('final_score') is None or not isinstance(d.get('final_score'), (int, float))]
    if bad_scores:
        warnings.append(f'{len(bad_scores)} subreddit(s) missing a valid final_score: {bad_scores[:5]}')

    if warnings:
        print(f"  ⚠ VALIDATION WARNINGS for {month}:")
        for w in warnings:
            print(f"      - {w}")
    return warnings


def _dirs(v2: bool):
    if v2:
        return ROOT / 'output' / 'v2', ROOT / 'docs' / 'data_v2'
    return ROOT / 'output', ROOT / 'docs' / 'data'


OUTPUT_DIR    = ROOT / 'output'        # default; overridden in main()
DOCS_DATA_DIR = ROOT / 'docs' / 'data' # default; overridden in main()


def get_severity(score):
    if score >= 70: return 'critical'
    if score >= 40: return 'high'
    if score >= 20: return 'moderate'
    return 'low'


def load_analysis_files(output_dir: Path, month_filter=None):
    files = sorted(glob.glob(str(output_dir / 'analysis_*.json')))
    results = []
    for f in files:
        if 'latest' in os.path.basename(f):
            continue
        try:
            with open(f) as fh:
                raw = json.load(fh)
            # V2 files store month explicitly; V1 derives from analysis_date
            if 'month' in raw:
                month = raw['month']
                dt = datetime.fromisoformat(raw['analysis_date'])
            else:
                dt = datetime.fromisoformat(raw['analysis_date'])
                month = dt.strftime('%Y-%m')
            if month_filter and month != month_filter:
                continue
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


def build_month_doc_v2(month, dt, raw):
    """Build month doc from a V2 analysis JSON (5-component schema)."""
    scores   = raw.get('unified_scores', {})
    acct_a   = raw.get('account_analysis', {})
    ring_a   = raw.get('ring_analysis', {})
    eng_a    = raw.get('engagement_analysis', {})
    temp_a   = raw.get('temporal_analysis', {})
    dist_a   = raw.get('distribution_analysis', {})

    subreddits = {}
    for sub, sc in scores.items():
        ac = acct_a.get(sub, {})
        ri = ring_a.get(sub, {})
        en = eng_a.get(sub, {})
        te = temp_a.get(sub, {})
        di = dist_a.get(sub, {})

        subreddits[sub] = {
            'final_score':        round(sc.get('final_score', 0), 1),
            'account_score':      round(sc.get('account_score', 0), 1),
            'ring_score':         round(sc.get('ring_score', 0), 1),
            'engagement_score':   round(sc.get('engagement_score', 0), 1),
            'temporal_score':     round(sc.get('temporal_score', 0), 1),
            'distribution_score': round(sc.get('distribution_score', 0), 1),
            'details': {
                'new_poster_pct':      ac.get('new_poster_pct'),
                'new_commenter_pct':   ac.get('new_commenter_pct'),
                'unverified_comm_pct': ac.get('unverified_comm_pct'),
                'burst_score':         ri.get('burst_score'),
                'fast_ttfc_pct':       ri.get('fast_ttfc_pct'),
                'avg_ttfc_minutes':    ri.get('avg_ttfc_minutes'),
                'recurrence_rate':     ri.get('recurrence_rate'),
                'self_amp_rate':       ri.get('self_amp_rate'),
                'overlap_rate':        ri.get('overlap_rate'),
                'score_comment_corr':  en.get('score_comment_corr'),
                'upvote_ratio_std':    en.get('upvote_ratio_std'),
                'ucr':                 en.get('ucr'),
                'avg_awards':          en.get('avg_awards'),
                'interval_cv':         te.get('interval_cv'),
                'top3_concentration':  te.get('top3_concentration'),
                'entropy':             te.get('entropy'),
                'peak_hour_utc':       te.get('peak_hour_utc'),
                'score_cv':            di.get('score_cv'),
                'avg_comment_depth':   di.get('avg_comment_depth'),
            },
        }

    return {
        'version':       2,
        'month':         month,
        'analysis_date': dt.isoformat(),
        'weights':       raw.get('weights', {}),
        'subreddits':    subreddits,
    }


def main():
    args = parse_args()
    v2   = args.v2

    output_dir, docs_dir = _dirs(v2)

    print("\n" + "=" * 70)
    print(f"GENERATE SITE DATA  (V{'2' if v2 else '1'})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    docs_dir.mkdir(parents=True, exist_ok=True)

    entries = load_analysis_files(output_dir, month_filter=args.month)
    if not entries:
        print(f"No analysis files found in {output_dir}. Run analyze_data{'_v2' if v2 else ''}.py first.")
        return

    # Keep only the latest run per calendar month
    by_month = {}
    for month, dt, raw in entries:
        if month not in by_month or dt > by_month[month][0]:
            by_month[month] = (dt, raw)

    expected_subs = expected_sub_count()
    history_months = []
    for month in sorted(by_month):
        dt, raw = by_month[month]
        doc = build_month_doc_v2(month, dt, raw) if v2 else build_month_doc(month, dt, raw)

        warnings = validate_month(month, doc, args.min_sub_coverage, expected_subs)
        if warnings:
            doc['validation_warnings'] = warnings

        out_path = docs_dir / f'{month}.json'
        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"  Wrote {out_path.name}")

        all_scores = [d['final_score'] for d in doc['subreddits'].values()]
        history_months.append({
            'month':         month,
            'analysis_date': dt.isoformat(),
            'data_file':     f'{month}.json',
            'aggregate': {
                'avg_score':        round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
                'pct_moderate_plus': round(sum(1 for s in all_scores if s >= 20) / len(all_scores) * 100) if all_scores else 0,
                'pct_high_plus':     round(sum(1 for s in all_scores if s >= 40) / len(all_scores) * 100) if all_scores else 0,
                'sub_count':    len(all_scores),
                'max_score':    round(max(all_scores), 1) if all_scores else 0,
                'min_score':    round(min(all_scores), 1) if all_scores else 0,
            },
            'summary': {
                sub: {'final_score': d['final_score'], 'severity': get_severity(d['final_score'])}
                for sub, d in doc['subreddits'].items()
            },
        })

    history   = {'version': 2 if v2 else 1, 'months': history_months}
    hist_path = docs_dir / 'history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Wrote {hist_path.name}")

    if v2:
        findings_src = ROOT / 'reports' / 'findings.json'
        if findings_src.exists():
            shutil.copy(findings_src, docs_dir / 'findings.json')
            print(f"  Copied findings.json")

    print(f"\nDone. {len(by_month)} month(s) written to {docs_dir.relative_to(ROOT)}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
