#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Scale up the weak-label check + roll up to subreddit level

train_weak_label_classifier.py (n=500) showed a real but small-sample signal.
This scales the same unbiased random check to a larger n and rolls the per-
account gone/not-gone label up to a per-subreddit gone-rate (weighted by how
much of each account's activity happened in that sub — same approach as
anomaly_detection.per_subreddit_rollup).

score_accounts.py builds the production final_score from this script's fitted
coefficients. Also persists the raw per-account labeled sample — account,
features, gone — to weak_label_sample.csv, plus the model intercept and
StandardScaler mean_/scale_ (previously computed and thrown away every run).
Two consumers need this: backtest_predictive.py (Phase 2's temporal check)
needs the actual labeled accounts, not just the aggregate rate; and a future
refit can build a properly calibrated predict_proba instead of
score_accounts.py's current intercept-free linear-ranking approximation.

Checkpointed/resumable — same pattern as collect_data_v2.py. At large --n
(each account check is one live, rate-limited API call, ~1.1s/call under
OAuth) a single run can take hours, longer than any one process should be
expected to stay alive uninterrupted. Each invocation checks at most
--batch-size *new* accounts, saves progress to
output/v2/weak_label_checkpoint_n<N>_seed<S>.json, and exits. Re-run the exact
same command to continue; once the full sample is checked, that run fits the
model, writes the real outputs, and clears the checkpoint.

Usage:
  python3 scripts/scale_weak_labels.py --n 8000               # first call starts, subsequent calls resume
  python3 scripts/scale_weak_labels.py --n 8000 --batch-size 400
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
    p.add_argument('--n', type=int, default=8000)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=400,
                    help='Max new accounts to check this invocation (~400 ≈ 7-8 min at 54 req/min, '
                         'safely under a single-call time limit). Re-run to continue.')
    p.add_argument('--refit-only', action='store_true',
                    help='Skip live gone-checks entirely — reuse the gone labels already saved in '
                         'weak_label_sample.csv, rebuild features fresh from the corpus (picks up '
                         'any FEATURE_COLS changes since that sample was checked), and refit. For '
                         'iterating on feature engineering without re-spending ~2.5h of API calls.')
    return p.parse_args()


def _cp_path(n: int, seed: int) -> Path:
    return OUTPUT_DIR / f'weak_label_checkpoint_n{n}_seed{seed}.json'


def load_checkpoint(n: int, seed: int) -> dict:
    cp = _cp_path(n, seed)
    if cp.exists():
        with open(cp) as f:
            return json.load(f)
    return {'sample_accounts': None, 'results': {}}


def save_checkpoint(n: int, seed: int, state: dict):
    with open(_cp_path(n, seed), 'w') as f:
        json.dump({**state, 'saved_at': datetime.now().isoformat()}, f)


def clear_checkpoint(n: int, seed: int):
    cp = _cp_path(n, seed)
    if cp.exists():
        cp.unlink()


def finalize(sample: pd.DataFrame, posts: pd.DataFrame, comms: pd.DataFrame):
    """Fit the model + write all three output files. Called once the full
    sample has been checked."""
    n_gone = sample['gone'].sum()
    print(f'\n  Total: {n_gone}/{len(sample)} gone ({n_gone/len(sample)*100:.1f}%)')

    X = sample[FEATURE_COLS].fillna(0)
    y = sample['gone'].astype(int)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
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
    print(f'  intercept: {model.intercept_[0]:+.3f}')
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
        'intercept': round(float(model.intercept_[0]), 3),
        'scaler_mean':  {c: round(float(m), 4) for c, m in zip(FEATURE_COLS, scaler.mean_)},
        'scaler_scale': {c: round(float(s), 4) for c, s in zip(FEATURE_COLS, scaler.scale_)},
        'cv_roc_auc_mean': round(float(np.mean(cv_scores)), 3) if cv_scores else None,
        'cv_roc_auc_std':  round(float(np.std(cv_scores)), 3) if cv_scores else None,
        'caveat': 'weak/noisy label — ordinary account deletion/inactivity is also captured, '
                  'not just enforcement action',
    }
    with open(OUTPUT_DIR / 'weak_label_classifier.json', 'w') as f:
        json.dump(classifier_result, f, indent=2, default=str)

    sample_out = sample[['account', 'gone'] + FEATURE_COLS].copy()
    sample_out.to_csv(OUTPUT_DIR / 'weak_label_sample.csv', index=False)
    print(f"\nSaved raw labeled sample: {OUTPUT_DIR / 'weak_label_sample.csv'} ({len(sample_out)} rows)")

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

    print('\n  Subreddit gone-rates (ground truth for score_accounts.py):')
    for _, row in rollup.iterrows():
        print(f"    r/{row['subreddit']:<25} gone_rate={row['gone_rate']:5.1f}%  "
              f"n_activity={int(row['n_activity_rows']):4d}  n_accounts={int(row['n_unique_accounts']):4d}")

    thin_subs = rollup[rollup['n_unique_accounts'] < 5]['subreddit'].tolist()

    rollup_result = {
        'generated': datetime.now().isoformat(),
        'n_sampled': int(len(sample)),
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

    if args.refit_only:
        sample_path = OUTPUT_DIR / 'weak_label_sample.csv'
        if not sample_path.exists():
            raise FileNotFoundError(f'{sample_path} not found — need a completed full run first '
                                     '(--refit-only reuses its gone labels, not a substitute for '
                                     'the initial live check).')
        prior = pd.read_csv(sample_path)[['account', 'gone']]
        print(f'\n--refit-only: reusing {len(prior)} gone labels from {sample_path.name}, '
              f'rebuilding features fresh (picks up any FEATURE_COLS changes)…')
        sample = accounts.merge(prior, on='account', how='inner')
        missing = len(prior) - len(sample)
        if missing:
            print(f'  ({missing} previously-labeled accounts no longer in the corpus — skipped)')
        finalize(sample, posts, comms)
        print('=' * 70 + '\n')
        return

    state = load_checkpoint(args.n, args.seed)
    if state['sample_accounts'] is None:
        random.seed(args.seed)
        sample_idx = random.sample(range(len(accounts)), min(args.n, len(accounts)))
        state['sample_accounts'] = accounts.iloc[sample_idx]['account'].tolist()
        state['results'] = {}
        print(f"  New sample (unbiased, seed={args.seed}): {len(state['sample_accounts'])} accounts")
    else:
        print(f"  Resuming checkpoint: {len(state['results'])}/{len(state['sample_accounts'])} "
              f"already checked")

    remaining = [a for a in state['sample_accounts'] if a not in state['results']]
    if remaining:
        batch = remaining[:args.batch_size]
        print(f'\nChecking {len(batch)} accounts this run '
              f'({len(remaining)} remaining before this batch)…')
        t0 = time.time()
        for i, name in enumerate(batch, 1):
            state['results'][name] = check_account_gone(name)
            if i % 50 == 0:
                elapsed = time.time() - t0
                n_gone = sum(state['results'].values())
                print(f'    {i}/{len(batch)} this batch  ({elapsed:.0f}s elapsed, '
                      f'{n_gone} gone so far overall)')
                save_checkpoint(args.n, args.seed, state)
        save_checkpoint(args.n, args.seed, state)
        remaining = [a for a in state['sample_accounts'] if a not in state['results']]

    total_checked = len(state['results'])
    print(f"\nProgress: {total_checked}/{len(state['sample_accounts'])} accounts checked")

    if remaining:
        eta_min = len(remaining) * 1.15 / 60 * (args.batch_size / min(args.batch_size, len(remaining) + args.batch_size))
        print(f"  {len(remaining)} accounts left — re-run the same command to continue "
              f"(~{len(remaining)/args.batch_size:.1f} more batches).")
        print('=' * 70 + '\n')
        return

    # Full sample checked — finalize.
    sample = accounts[accounts['account'].isin(state['sample_accounts'])].copy()
    sample['gone'] = sample['account'].map(state['results'])
    finalize(sample, posts, comms)
    clear_checkpoint(args.n, args.seed)
    print('\nCheckpoint cleared — weak-label collection complete.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
