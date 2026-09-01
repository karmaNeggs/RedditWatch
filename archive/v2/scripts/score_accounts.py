#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Production account-risk model

scale_weak_labels.py already fit a LogisticRegression against the one real
label this project has (current suspended/gone status) and found it
meaningfully non-trivial (see output/v2/weak_label_classifier.json for the
current AUC). That model was previously a side-diagnostic — analyze_data_v2.py
instead summed six hand-weighted heuristic components that were never
individually checked against this label. This script makes the validated
model the actual production scorer.

We don't re-fit here (that needs a fresh batch of live gone-checks — see
scale_weak_labels.py / Phase 2's larger re-run). Instead we reuse the already-
fit, already-validated coefficients and apply them to every account in the
corpus:

  1. Load the full account population (anomaly_detection.build_account_features).
  2. Standardize its features with a StandardScaler fit on the *population*
     (thousands of accounts), not the original ~2,000-account labeled sample —
     the original sample's scaler object was never persisted. Since the
     sample was drawn unbiased/random from this same population, population
     stats should closely approximate the original sample stats. This is a
     documented approximation, not an exact replay of the original fit —
     Phase 2's larger refit will persist scaler + intercept directly and
     remove the need for this substitution.
  3. Score each account as a linear combination of (coefficient × standardized
     feature) — no intercept is applied (none was persisted), so this is a
     *relative risk ranking*, not a calibrated probability. Min-max rescaled
     to 0-100, same convention as anomaly_detection.py's anomaly_score.
  4. Roll up to (subreddit, month) as **% of that period's activity coming
     from accounts in the population's top risk decile** — not the mean
     risk_score. Tested both on the current corpus: averaging a continuous
     score across ~100+ accounts per sub-month regresses almost entirely to
     the population mean (std 2.4 across subs, effectively no discrimination —
     expected from the CLT given how weak the per-account signal already is).
     The %-in-top-decile composition metric keeps ~3.5x more between-subreddit
     variance (std 8.5) and is a more legible claim anyway ("18% of active
     accounts here fall in our elevated-risk tier" vs. an opaque blended
     average). This is the same pattern anomaly_detection.py already uses
     (pct_flagged alongside avg_anomaly_score) — we're just making it the
     primary metric instead of a side stat.
  5. Derive percentile-based severity bands from the actual observed
     pct_high_risk distribution across every month already collected —
     replacing the old fixed 20/40/70 thresholds, which were carried over
     from the heuristic-sum score and don't mean anything on this new scale.

Output:
  output/v2/account_risk_scores.csv   — one row per account: risk_score, high_risk flag, raw features
  reports/findings.json['account_model']     — model provenance + caveats
  reports/findings.json['severity_bands']    — percentile cutoffs from historical rollups

Usage:
  python3 scripts/score_accounts.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
from anomaly_detection import (load_corpus, build_account_features, FEATURE_COLS,
                                rollup_to_subreddit, month_relative_high_risk)

DATA_DIR    = ROOT / 'data' / 'v2'
OUTPUT_DIR  = ROOT / 'output' / 'v2'
FINDINGS    = ROOT / 'reports' / 'findings.json'
CLASSIFIER  = OUTPUT_DIR / 'weak_label_classifier.json'


def load_validated_coefficients() -> dict:
    if not CLASSIFIER.exists():
        raise FileNotFoundError(
            f'{CLASSIFIER} not found — run scale_weak_labels.py first to fit and '
            'validate the logistic regression this script builds on.'
        )
    with open(CLASSIFIER) as f:
        d = json.load(f)
    if 'coefficients' not in d:
        raise ValueError(f'{CLASSIFIER} has no fitted coefficients (too few positive labels?) — '
                          're-run scale_weak_labels.py with a larger --n.')
    return d


TOP_RISK_PERCENTILE = 90  # accounts at/above this MONTH-relative percentile count as "high risk"


def score_population(accounts: pd.DataFrame, coefficients: dict) -> pd.DataFrame:
    """risk_score only — a fixed lifetime value per account. NOT thresholded
    here: "high risk" is now computed month-relatively (see
    anomaly_detection.month_relative_high_risk), at analysis time, not baked
    into this population-wide file. A global threshold here would silently
    track population composition drift rather than actual relative risk —
    see month_relative_high_risk's docstring for why this changed."""
    X = accounts[FEATURE_COLS].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    coef_vec = np.array([coefficients[c] for c in FEATURE_COLS])
    raw = Xs @ coef_vec

    accounts = accounts.copy()
    accounts['risk_raw'] = raw
    lo, hi = raw.min(), raw.max()
    accounts['risk_score'] = ((raw - lo) / (hi - lo) * 100).round(1) if hi > lo else 0.0
    return accounts


def historical_rollup_distribution(risk_df: pd.DataFrame) -> list:
    """Recompute the (subreddit, month) pct_high_risk rollup for every month
    already collected, to derive severity bands from the actual observed
    distribution — using the SAME month-relative threshold production
    scoring uses (analyze_data_v2.py's analyze_account_risk), so bands are
    calibrated against exactly what gets published, not a different
    methodology.

    Filters to sample_type == 'top' where that column exists (Phase 3 onward)
    — production final_score is top-sample only. Months collected before
    Phase 3 have no sample_type column at all — every row in them already
    *is* top-sample, nothing to filter."""
    post_files = sorted(DATA_DIR.glob('posts_20*.csv'))
    all_scores = []
    for pf in post_files:
        month = pf.stem.replace('posts_', '')
        cf = DATA_DIR / f'commenters_{month}.csv'
        if not cf.exists():
            continue
        posts = pd.read_csv(pf)
        comms = pd.read_csv(cf)
        if 'sample_type' in posts.columns:
            posts = posts[posts['sample_type'] == 'top']
        if posts.empty:
            continue
        risk_month, _ = month_relative_high_risk(posts, comms, risk_df, TOP_RISK_PERCENTILE)
        rollup = rollup_to_subreddit(posts, comms, risk_month, score_col='high_risk')
        all_scores.extend((rollup['avg_score'] * 100).tolist())
    return all_scores


def main():
    print('\n' + '=' * 70)
    print('PRODUCTION ACCOUNT-RISK MODEL')
    print('=' * 70)

    classifier = load_validated_coefficients()
    coefficients = classifier['coefficients']
    print(f"\nUsing coefficients from {CLASSIFIER.name} "
          f"(fit on n={classifier.get('n_sampled')}, "
          f"cv_roc_auc={classifier.get('cv_roc_auc_mean')}, "
          f"gone_rate={classifier.get('gone_rate')})")

    print('\nLoading full corpus + building account features…')
    posts, comms = load_corpus()
    accounts = build_account_features(posts, comms)
    print(f'  {len(accounts):,} unique accounts')

    print(f'\nScoring population (population-standardized, no intercept — relative risk ranking)…')
    scored = score_population(accounts, coefficients)
    print(f"  risk_score range: {scored['risk_score'].min():.1f} – {scored['risk_score'].max():.1f}  "
          f"median: {scored['risk_score'].median():.1f}")
    print(f'  "high risk" (top {100-TOP_RISK_PERCENTILE}%) is now computed month-relatively at '
          f'analysis time, not baked into this file — see month_relative_high_risk.')

    out_csv = OUTPUT_DIR / 'account_risk_scores.csv'
    cols = ['account', 'risk_score'] + FEATURE_COLS
    scored[cols].to_csv(out_csv, index=False)
    print(f'\nSaved: {out_csv}')

    print('\nRecomputing historical (subreddit, month) pct_high_risk rollup for severity bands '
          '(month-relative threshold, matching production)…')
    dist = historical_rollup_distribution(scored[['account', 'risk_score']])
    if len(dist) >= 20:
        p50, p80, p95 = (float(np.percentile(dist, q)) for q in (50, 80, 95))
        print(f'  n={len(dist)} sub-month rollups  p50={p50:.1f}  p80={p80:.1f}  p95={p95:.1f}')
    else:
        p50, p80, p95 = 20.0, 40.0, 70.0
        print(f'  Only {len(dist)} sub-month rollups — too few to derive percentiles, '
              f'falling back to legacy fixed bands (20/40/70)')

    findings = {}
    if FINDINGS.exists():
        with open(FINDINGS) as f:
            findings = json.load(f)

    findings['account_model'] = {
        'generated': datetime.now().isoformat(),
        'method': 'Linear combination of LogisticRegression coefficients (from '
                  'weak_label_classifier.json) applied to population-standardized account '
                  'features. No intercept (none persisted from the original fit) — this is a '
                  'relative risk ranking (min-max rescaled 0-100), not a calibrated probability.',
        'coefficients': coefficients,
        'coefficient_source': str(CLASSIFIER.relative_to(ROOT)),
        'source_n_sampled': classifier.get('n_sampled'),
        'source_gone_rate': classifier.get('gone_rate'),
        'source_cv_roc_auc_mean': classifier.get('cv_roc_auc_mean'),
        'source_cv_roc_auc_std': classifier.get('cv_roc_auc_std'),
        'n_accounts_scored': int(len(scored)),
        'top_risk_percentile': TOP_RISK_PERCENTILE,
        'final_score_definition': '% of a (subreddit, month)\'s posting/commenting activity that '
                  'comes from accounts in that MONTH\'s own top-decile risk_score tier (not a '
                  'fixed global population threshold — see month_relative_high_risk). Originally '
                  'used a fixed global threshold; changed after finding it produced a near-'
                  'monotonic 13-month score climb (4.6→23.0 avg) driven almost entirely by the '
                  'active population trending younger over time, not by increasing relative risk '
                  '— confirmed by the fact that ALL 25 subreddits rose in lockstep regardless of '
                  'topic, and that median poster account age genuinely fell (1180→683 days) over '
                  'the same window. Month-relative thresholding removes that drift (flat ~10 avg '
                  'across all 14 months) while barely moving predictive validity (backtest '
                  'early/late r: 0.668/0.794 → 0.659/0.759).',
        'caveat': 'AUC ~0.6 means this ranks risk better than chance but is far from a '
                  'confident classifier — treat final_score as a weak-to-moderate signal, '
                  'not a verdict. It answers "risky relative to this month\'s contemporaries", '
                  'not "risky in absolute/historical terms" — see final_score_definition.',
    }
    findings['severity_bands'] = {
        'generated': datetime.now().isoformat(),
        'method': 'percentiles of the (subreddit, month) pct_high_risk rollup (% of activity from '
                  'top-risk-decile accounts) across every month already collected',
        'n_sub_months': len(dist),
        'moderate': round(p50, 1),
        'high':     round(p80, 1),
        'critical': round(p95, 1),
    }

    with open(FINDINGS, 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    print(f'\nWrote account_model + severity_bands to {FINDINGS}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
