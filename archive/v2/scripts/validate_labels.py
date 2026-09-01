#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Weak-label validation

Every score in this pipeline (component weights, anomaly detection) has been
calibrated with zero labeled bot/human ground truth. This script builds one
weak, noisy but real label — current account status — and checks whether it
actually correlates with what the pipeline already flags as suspicious.

Logic: accounts genuinely run for spam/manipulation get suspended or banned
over time at a higher rate than ordinary accounts. If IsolationForest's
top-anomaly accounts (output/v2/anomaly_accounts.json) are now gone/suspended
at a meaningfully higher rate than a random baseline sample, that's real
(if noisy) evidence the score means something. If the rates are indistinguishable,
that's evidence the score isn't tracking anything reddit itself considers bad.

Read-only Reddit API calls only (about.json per account) — same call the
collector already makes at scale, just re-checking status on a small sample.

Usage:
  python3 scripts/validate_labels.py --n-top 100 --n-baseline 100
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reddit_auth import get_json, print_auth_status
from anomaly_detection import load_corpus, build_account_features, score_anomalies

OUTPUT_DIR = ROOT / 'output' / 'v2'


def check_account_gone(username: str) -> bool:
    """True if the account is suspended, deleted, or otherwise unreachable now."""
    data = get_json(f"https://www.reddit.com/user/{username}/about.json")
    if not data:
        return True
    d = data.get('data', {})
    if d.get('is_suspended'):
        return True
    return False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n-top', type=int, default=100, help='Sample size from top-anomaly accounts')
    p.add_argument('--n-baseline', type=int, default=100, help='Sample size from random baseline')
    p.add_argument('--contamination', type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    print('\n' + '=' * 70)
    print('WEAK-LABEL VALIDATION — current account status vs. anomaly score')
    print('=' * 70)
    print_auth_status()

    print('\nLoading corpus + scoring accounts…')
    posts, comms = load_corpus()
    accounts = build_account_features(posts, comms)
    scored   = score_anomalies(accounts, args.contamination)

    n = len(scored)
    top_cut = scored['anomaly_score'].quantile(0.90)
    bottom_cut = scored['anomaly_score'].quantile(0.50)

    top_pool      = scored[scored['anomaly_score'] >= top_cut]['account'].tolist()
    baseline_pool = scored[scored['anomaly_score'] <= bottom_cut]['account'].tolist()

    random.seed(42)
    top_sample      = random.sample(top_pool, min(args.n_top, len(top_pool)))
    baseline_sample = random.sample(baseline_pool, min(args.n_baseline, len(baseline_pool)))

    print(f'  Population: {n:,} accounts')
    print(f'  Top-anomaly sample (>=p90, score>={top_cut:.1f}): {len(top_sample)} accounts')
    print(f'  Baseline sample (<=p50, score<={bottom_cut:.1f}): {len(baseline_sample)} accounts')

    def _check_group(names, label):
        gone = 0
        t0 = time.time()
        for i, name in enumerate(names, 1):
            if check_account_gone(name):
                gone += 1
            if i % 25 == 0:
                print(f'    {label}: {i}/{len(names)}  ({time.time()-t0:.0f}s elapsed)')
        return gone

    print('\nChecking top-anomaly group…')
    top_gone = _check_group(top_sample, 'top-anomaly')
    print('Checking baseline group…')
    baseline_gone = _check_group(baseline_sample, 'baseline')

    top_rate      = top_gone / len(top_sample) if top_sample else 0.0
    baseline_rate = baseline_gone / len(baseline_sample) if baseline_sample else 0.0
    ratio = (top_rate / baseline_rate) if baseline_rate > 0 else float('inf') if top_rate > 0 else 1.0

    result = {
        'generated': datetime.now().isoformat(),
        'method': 'current suspended/deleted status as a weak, noisy proxy for "bad account" — '
                  'NOT a validated bot label; ordinary account deletion/inactivity is also captured',
        'top_anomaly_group': {'n': len(top_sample), 'n_gone': top_gone, 'gone_rate': round(top_rate, 3)},
        'baseline_group':    {'n': len(baseline_sample), 'n_gone': baseline_gone, 'gone_rate': round(baseline_rate, 3)},
        'rate_ratio': round(ratio, 2) if ratio != float('inf') else None,
        'interpretation': (
            'ratio > ~1.5 suggests the anomaly score tracks something real (higher-flagged '
            'accounts vanish more); ratio ~1 means the score is not distinguishing meaningfully '
            'from ordinary account churn on this sample size'
        ),
    }

    print('\n' + '-' * 70)
    print(f"  Top-anomaly group:  {top_gone}/{len(top_sample)} gone  ({top_rate*100:.1f}%)")
    print(f"  Baseline group:     {baseline_gone}/{len(baseline_sample)} gone  ({baseline_rate*100:.1f}%)")
    print(f"  Rate ratio:         {result['rate_ratio']}")
    print('-' * 70)

    out_path = OUTPUT_DIR / 'label_validation.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
