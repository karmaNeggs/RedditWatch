#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Predictive backtest (does an early score forecast
later attrition, not just describe accounts already circling the drain?)

Every validation so far in this pipeline (validate_labels.py, scale_weak_labels.py,
the old referee_weights.py) is a single cross-sectional snapshot: score computed
from the *same* activity window as the outcome it's compared against. That can't
distinguish "this score predicts future bad behavior" from "this score just
redescribes accounts that are already in decline right now" — both would show
the same correlation in a snapshot test.

This script instead compares two non-overlapping time windows against one
outcome measured now:
  - EARLY window: each subreddit's pct_high_risk averaged over its first ~4
    collected months (2025-06 through 2025-09)
  - LATE window: each subreddit's pct_high_risk averaged over its most recent
    ~4 collected months (2026-04 through 2026-07)
  - OUTCOME: current subreddit gone_rate (subreddit_gone_rates.json, from
    scale_weak_labels.py — run that with a large --n first)

If EARLY correlates with the OUTCOME about as well as LATE does, that's real
evidence of forward-looking predictive content (a reading from ~10-13 months
before the gone-check still tracked who'd have elevated attrition). If EARLY is
much weaker than LATE, the score is more of a concurrent descriptor than a
forecast — still useful, just a narrower claim.

Caveat (stated plainly, not hidden): this is a subreddit-level, window-level
check, not an account-level point-in-time forecast — a fully rigorous version
would re-featurize each account using only data available as of month T, which
this pipeline doesn't currently support (account features are aggregated across
the full corpus regardless of when you "ask"). n=25 subreddits is small; treat
directionally, not as a precise number.

Usage:
  python3 scripts/backtest_predictive.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from anomaly_detection import rollup_to_subreddit, month_relative_high_risk

DATA_DIR   = ROOT / 'data' / 'v2'
OUTPUT_DIR = ROOT / 'output' / 'v2'
FINDINGS   = ROOT / 'reports' / 'findings.json'

EARLY_MONTHS = ['2025-06', '2025-07', '2025-08', '2025-09']
LATE_MONTHS  = ['2026-04', '2026-05', '2026-06', '2026-07']


def window_pct_high_risk(months: list, risk_scores: pd.DataFrame) -> dict:
    """Per-subreddit pct_high_risk averaged across the given months. "High
    risk" is month-relative (top decile of THAT month's active accounts,
    matching production — see anomaly_detection.month_relative_high_risk),
    not a fixed global threshold, and top-sample only (matching production —
    analyze_data_v2.py's analyze_account_risk)."""
    per_sub_month = []
    for month in months:
        pf, cf = DATA_DIR / f'posts_{month}.csv', DATA_DIR / f'commenters_{month}.csv'
        if not pf.exists() or not cf.exists():
            print(f'  WARNING: missing data for {month}, skipping')
            continue
        posts, comms = pd.read_csv(pf), pd.read_csv(cf)
        if 'sample_type' in posts.columns:
            posts = posts[posts['sample_type'] == 'top']
        if posts.empty:
            continue
        risk_month, _ = month_relative_high_risk(posts, comms, risk_scores)
        rollup = rollup_to_subreddit(posts, comms, risk_month, score_col='high_risk')
        rollup['month'] = month
        per_sub_month.append(rollup)
    if not per_sub_month:
        return {}
    allr = pd.concat(per_sub_month, ignore_index=True)
    avg = allr.groupby('subreddit')['avg_score'].mean() * 100
    return avg.round(1).to_dict()


def main():
    print('\n' + '=' * 70)
    print('PREDICTIVE BACKTEST — early window vs. late window vs. current gone-rate')
    print('=' * 70)

    risk_path = OUTPUT_DIR / 'account_risk_scores.csv'
    gone_path = OUTPUT_DIR / 'subreddit_gone_rates.json'
    if not risk_path.exists():
        raise FileNotFoundError(f'{risk_path} not found — run score_accounts.py first')
    if not gone_path.exists():
        raise FileNotFoundError(f'{gone_path} not found — run scale_weak_labels.py first')

    risk = pd.read_csv(risk_path)[['account', 'risk_score']]
    with open(gone_path) as f:
        gone_data = json.load(f)
    gone_rates = {r['subreddit']: r['gone_rate'] for r in gone_data['subreddit_gone_rates']}
    thin = set(gone_data.get('thin_subs_low_confidence', []))
    print(f"\nOutcome: gone_rate from {gone_path.name} "
          f"(n_sampled={gone_data.get('n_sampled')}, {len(thin)} thin subs excluded)")

    print(f'\nEarly window: {EARLY_MONTHS}')
    early = window_pct_high_risk(EARLY_MONTHS, risk)
    print(f'Late window:  {LATE_MONTHS}')
    late = window_pct_high_risk(LATE_MONTHS, risk)

    subs = sorted(set(early) & set(late) & set(gone_rates) - thin)
    print(f'\nComparable subs: {len(subs)}')

    def _corr(predictor):
        x = [predictor[s] for s in subs]
        y = [gone_rates[s] for s in subs]
        r, p = spearmanr(x, y)
        return round(float(r), 3), round(float(p), 3)

    early_r, early_p = _corr(early)
    late_r, late_p   = _corr(late)

    print('\n' + '-' * 70)
    print(f'  EARLY window (2025-06..09) pct_high_risk vs. current gone_rate:  '
          f'r={early_r:+.3f}  p={early_p:.3f}')
    print(f'  LATE window  (2026-04..07) pct_high_risk vs. current gone_rate:  '
          f'r={late_r:+.3f}  p={late_p:.3f}')
    print('-' * 70)

    if early_r >= late_r * 0.7:
        verdict = ('EARLY tracks the outcome about as well as LATE — real forward-looking '
                   'signal: a reading from ~10-13 months before the gone-check still predicted '
                   'who would show elevated attrition, not just concurrent decline.')
    elif early_r > 0.15:
        verdict = ('EARLY is weaker than LATE but still positive — some forward-looking signal, '
                   'but the score is more informative closer to the outcome (partly concurrent-'
                   'descriptor, not purely predictive).')
    else:
        verdict = ('EARLY shows little/no relationship while LATE does — the score mostly '
                   'describes accounts already in decline at measurement time, not a real '
                   'forecast. Treat final_score as a concurrent health indicator, not a predictor.')
    print(f'\nVerdict: {verdict}')

    result = {
        'generated': datetime.now().isoformat(),
        'method': 'Spearman correlation of subreddit pct_high_risk (averaged over a 4-month '
                  'window) against current gone_rate, compared for an early window '
                  '(2025-06..09) vs. a late window (2026-04..07) — tests whether an early '
                  'reading predicts later attrition or the score only tracks concurrent decline.',
        'early_window': EARLY_MONTHS, 'late_window': LATE_MONTHS,
        'n_subs': len(subs),
        'early_window_spearman_r': early_r, 'early_window_p_value': early_p,
        'late_window_spearman_r':  late_r,  'late_window_p_value':  late_p,
        'verdict': verdict,
        'caveat': 'subreddit-level, window-level check, not an account-level point-in-time '
                  'forecast — see module docstring. n=25 subs is small; directional evidence, '
                  'not a precise estimate.',
    }
    out_path = OUTPUT_DIR / 'backtest_predictive.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')

    if FINDINGS.exists():
        with open(FINDINGS) as f:
            findings = json.load(f)
        findings['predictive_backtest'] = result
        with open(FINDINGS, 'w') as f:
            json.dump(findings, f, indent=2, default=str)
        print(f'Wrote predictive_backtest to {FINDINGS}')

    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
