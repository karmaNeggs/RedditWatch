#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Supervised fit on weak labels

validate_labels.py showed top-anomaly accounts are suspended/deleted at ~2.5x
the baseline rate on a small sample (n=60/group) — real but noisy. This script
scales that up with a random (unbiased, not anomaly-stratified) sample across
the whole account population, then fits a plain logistic regression: current
gone-status ~ account features. Deliberately not a neural net — a few hundred
labeled examples with a handful of features is exactly the regime where a
regularized linear model is appropriate and a net would just overfit (see the
abhaybd/Reddit-Bot-Detector project this project's README search turned up:
620 examples, small net, authors themselves flag likely overfitting).

Caveat unchanged from validate_labels.py: "gone" is a weak, noisy proxy for
"bad account" (ordinary deletion/inactivity is also captured). This is about
checking which *features* the weak label actually moves with — direction and
significance — not producing a final, trustworthy bot-probability score.

Usage:
  python3 scripts/train_weak_label_classifier.py --n 500
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from reddit_auth import print_auth_status
from anomaly_detection import load_corpus, build_account_features, FEATURE_COLS
from validate_labels import check_account_gone

OUTPUT_DIR = ROOT / 'output' / 'v2'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=500, help='Random sample size from the full account population')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print('\n' + '=' * 70)
    print('SUPERVISED FIT ON WEAK LABELS (logistic regression)')
    print('=' * 70)
    print_auth_status()

    print('\nLoading corpus + building account features…')
    posts, comms = load_corpus()
    accounts = build_account_features(posts, comms)
    print(f'  {len(accounts):,} unique accounts')

    random.seed(args.seed)
    sample_idx = random.sample(range(len(accounts)), min(args.n, len(accounts)))
    sample = accounts.iloc[sample_idx].reset_index(drop=True)
    print(f'  Random sample (unbiased, not anomaly-stratified): {len(sample)} accounts')

    print('\nChecking current account status…')
    t0 = time.time()
    gone = []
    for i, name in enumerate(sample['account'], 1):
        gone.append(check_account_gone(name))
        if i % 50 == 0:
            n_gone = sum(gone)
            print(f'    {i}/{len(sample)}  ({time.time()-t0:.0f}s elapsed, {n_gone} gone so far, {n_gone/i*100:.1f}%)')
    sample['gone'] = gone

    n_gone = sample['gone'].sum()
    print(f'\n  Total: {n_gone}/{len(sample)} gone ({n_gone/len(sample)*100:.1f}%)')

    if n_gone < 8:
        print('  Too few positive labels for a stable fit — reporting rate only, skipping regression.')
        result = {
            'generated': datetime.now().isoformat(),
            'n_sampled': int(len(sample)), 'n_gone': int(n_gone),
            'gone_rate': round(float(n_gone / len(sample)), 3),
            'note': 'insufficient positive labels for logistic regression at this sample size',
        }
    else:
        X = sample[FEATURE_COLS].fillna(0)
        y = sample['gone'].astype(int)
        Xs = StandardScaler().fit_transform(X)

        model = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
        model.fit(Xs, y)

        cv_scores = []
        try:
            cv_scores = cross_val_score(model, Xs, y, cv=min(5, n_gone), scoring='roc_auc').tolist()
        except Exception as e:
            print(f'  Cross-val AUC skipped: {e}')

        coefs = dict(zip(FEATURE_COLS, model.coef_[0].tolist()))
        coefs_sorted = dict(sorted(coefs.items(), key=lambda x: -abs(x[1])))

        print('\n  Logistic regression coefficients (standardized features, +ve = predicts "gone"):')
        for feat, c in coefs_sorted.items():
            print(f'    {feat:<20} {c:+.3f}')
        if cv_scores:
            print(f'\n  5-fold CV ROC-AUC: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})  n_folds={len(cv_scores)}')

        result = {
            'generated': datetime.now().isoformat(),
            'method': 'LogisticRegression(class_weight=balanced) on standardized features, '
                      'weak label = current suspended/deleted status',
            'n_sampled': int(len(sample)), 'n_gone': int(n_gone),
            'gone_rate': round(float(n_gone / len(sample)), 3),
            'features': FEATURE_COLS,
            'coefficients': {k: round(float(v), 3) for k, v in coefs_sorted.items()},
            'cv_roc_auc_mean': round(float(np.mean(cv_scores)), 3) if cv_scores else None,
            'cv_roc_auc_std':  round(float(np.std(cv_scores)), 3) if cv_scores else None,
            'cv_folds': len(cv_scores),
            'caveat': 'weak/noisy label (ordinary account deletion is also captured, not just '
                      'enforcement action); coefficients show direction/significance of each '
                      'feature, not a validated bot-probability score',
        }

    out_path = OUTPUT_DIR / 'weak_label_classifier.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
