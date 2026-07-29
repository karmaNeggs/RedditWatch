#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Scale up the weak-label check + roll up to subreddit level

train_weak_label_classifier.py (n=500) showed a real but small-sample signal.
This scales the same unbiased random check to a larger n and, new here, rolls
the per-account gone/not-gone label up to a per-subreddit gone-rate (weighted
by how much of each account's activity happened in that sub — same approach
as anomaly_detection.per_subreddit_rollup). That per-subreddit rate is what
referee_weights.py uses as ground truth to decide between the CV-calibrated,
PCA, and 6-component (network-included) weight candidates in findings.json —
whichever produces a final_score that correlates best with actual observed
account attrition per subreddit wins.

Usage:
  python3 scripts/scale_weak_labels.py --n 2000
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reddit_auth import print_auth_status
from anomaly_detection import load_corpus, build_account_features, FEATURE_COLS
from validate_labels import check_account_gone

OUTPUT_DIR = ROOT / 'output' / 'v2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--seed', type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    print('\n' + '=' * 70)
    print(f'SCALED WEAK-LABEL CHECK (n={args.n}) + SUBREDDIT ROLLUP')
    print('=' * 70)
    print_auth_status()

    print('\nLoading corpus + building account features…')
    posts, comms = load_corpus()
    accounts = build_account_features(posts, comms)
    print(f'  {len(accounts):,} unique accounts')

    random.seed(args.seed)
    sample_idx = random.sample(range(len(accounts)), min(args.n, len(accounts)))
    sample = accounts.iloc[sample_idx].reset_index(drop=True)
    print(f'  Random sample (unbiased): {len(sample)} accounts')

    print('\nChecking current account status (this is the long part)…')
    t0 = time.time()
    gone = []
    for i, name in enumerate(sample['account'], 1):
        gone.append(check_account_gone(name))
        if i % 100 == 0:
            n_gone = sum(gone)
            elapsed = time.time() - t0
            eta_min = (elapsed / i) * (len(sample) - i) / 60
            print(f'    {i}/{len(sample)}  ({elapsed:.0f}s elapsed, {n_gone} gone so far, '
                  f'{n_gone/i*100:.1f}%, ETA {eta_min:.0f} min)')
    sample['gone'] = gone

    n_gone = sample['gone'].sum()
    print(f'\n  Total: {n_gone}/{len(sample)} gone ({n_gone/len(sample)*100:.1f}%)')

    # ── logistic regression (larger n = more stable than the n=500 run) ────────
    X = sample[FEATURE_COLS].fillna(0)
    y = sample['gone'].astype(int)
    Xs = StandardScaler().fit_transform(X)
    model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
    model.fit(Xs, y)
    cv_scores = []
    try:
        cv_scores = cross_val_score(model, Xs, y, cv=5, scoring='roc_auc').tolist()
    except Exception as e:
        print(f'  Cross-val AUC skipped: {e}')
    coefs = dict(sorted(zip(FEATURE_COLS, model.coef_[0].tolist()), key=lambda x: -abs(x[1])))

    print('\n  Logistic regression coefficients (standardized, +ve = predicts "gone"):')
    for feat, c in coefs.items():
        print(f'    {feat:<20} {c:+.3f}')
    if cv_scores:
        print(f'\n  5-fold CV ROC-AUC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})')

    classifier_result = {
        'generated': datetime.now().isoformat(),
        'method': 'LogisticRegression(class_weight=balanced) on standardized features, '
                  'weak label = current suspended/deleted status',
        'n_sampled': int(len(sample)), 'n_gone': int(n_gone),
        'gone_rate': round(float(n_gone / len(sample)), 3),
        'features': FEATURE_COLS,
        'coefficients': {k: round(float(v), 3) for k, v in coefs.items()},
        'cv_roc_auc_mean': round(float(np.mean(cv_scores)), 3) if cv_scores else None,
        'cv_roc_auc_std':  round(float(np.std(cv_scores)), 3) if cv_scores else None,
        'caveat': 'weak/noisy label — ordinary account deletion/inactivity is also captured, '
                  'not just enforcement action',
    }
    with open(OUTPUT_DIR / 'weak_label_classifier.json', 'w') as f:
        json.dump(classifier_result, f, indent=2, default=str)

    # ── roll up to subreddit level (activity-weighted, like anomaly_detection) ─
    print('\nRolling up gone-rate to subreddit level…')
    p_act = posts.rename(columns={'author': 'account'})[['account', 'subreddit']]
    c_act = comms.rename(columns={'author': 'account'})[['account', 'subreddit']]
    activity = pd.concat([p_act, c_act], ignore_index=True)

    merged = activity.merge(sample[['account', 'gone']], on='account', how='inner')
    rollup = merged.groupby('subreddit').agg(
        gone_rate=('gone', 'mean'),
        n_activity_rows=('account', 'count'),
        n_unique_accounts=('account', 'nunique'),
    ).reset_index()
    rollup['gone_rate'] = (rollup['gone_rate'] * 100).round(1)
    rollup = rollup.sort_values('gone_rate', ascending=False)

    print('\n  Subreddit gone-rates (ground truth for referee_weights.py):')
    for _, row in rollup.iterrows():
        print(f"    r/{row['subreddit']:<25} gone_rate={row['gone_rate']:5.1f}%  "
              f"n_activity={int(row['n_activity_rows']):4d}  n_accounts={int(row['n_unique_accounts']):4d}")

    thin_subs = rollup[rollup['n_unique_accounts'] < 5]['subreddit'].tolist()

    rollup_result = {
        'generated': datetime.now().isoformat(),
        'n_sampled': int(len(sample)),
        'seed': args.seed,
        'method': 'activity-weighted rollup: each checked account contributes to every subreddit '
                  'it posted/commented in, weighted by row count',
        'subreddit_gone_rates': rollup.to_dict(orient='records'),
        'thin_subs_low_confidence': thin_subs,
        'caveat': 'subs with <5 unique sampled accounts have unreliable gone-rate estimates — '
                  'treat with caution in the referee comparison',
    }
    with open(OUTPUT_DIR / 'subreddit_gone_rates.json', 'w') as f:
        json.dump(rollup_result, f, indent=2, default=str)

    print(f"\nSaved: {OUTPUT_DIR / 'weak_label_classifier.json'}")
    print(f"Saved: {OUTPUT_DIR / 'subreddit_gone_rates.json'}")
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
