#!/usr/bin/env python3
"""V3 feature sanitisation for Stage 3 (V3_PLAN.md Sec 5), building a
model-ready view on top of account_features in data/v3/analysis/v3.duckdb.
Does NOT build the Stage 3 model itself -- this only prepares its input.

Four things, each because its absence produces a confident wrong answer
(Sec 5's framing):

1. **Zero-inflation hurdle indicators (Sec 5.2).** For every feature where
   a single value carries >=25% of the mass (mostly 0), add an explicit
   `{feature}_nonzero` boolean alongside the raw magnitude, rather than
   relying on a transform to paper over the spike. Tree splits don't need
   this to find the same boundary (Sec 5.2: "skew transforms are pointless
   for XGBoost, splits depend only on ordering") -- it's for
   interpretability (SHAP can separate "did this ever happen" from "how
   much, given it happened") and for the linear/composite reporting side
   Sec 5.3 keeps around.
2. **Volume normalisation (Sec 5.4).** n_high_tier/n_medium_tier/n_low_tier
   were RAW COUNTS -- a 50-comment account and a 2-comment account with the
   same n_high_tier=2 look identical on that column despite completely
   different behaviour (2/50 vs 2/2). Replaced with shares of
   n_comments_sample (the natural per-account denominator here), with
   log1p(n_comments_sample) kept alongside as the explicit volume
   covariate per Sec 5.4's prescription.
3. **VIF pruning WITHIN evidence families only (Sec 5.3).** Families below
   mirror Sec 4.3's own grouping (provenance / footprint / reception /
   username morphology / timing / removal-quality / engagement-pattern).
   Iteratively drops the highest-VIF member of a family while VIF>10,
   never comparing across families (a timing feature correlating with a
   reception feature is a finding, not redundancy to prune).
4. **No hand-built composites as model input (Sec 5.3).** The bot-marker
   percentile family (removal_rate_pctl, deleted_later_rate_pctl,
   thin_history_score, reception_spread_pctl, botmarker_composite,
   n_markers_available) is EXCLUDED here -- three of those four _pctl
   columns are pure monotonic transforms of a raw column already in the
   model set (removal_rate, deleted_later_rate, reception_spread;
   thin_history_score is -n_comments_sample percentile-ranked), so they're
   redundant for a tree model, not just philosophically composite. The
   composite itself and karma_extremeness / karma_per_post_extremeness
   (which have no raw non-percentile equivalent persisted) are kept
   available as reporting-only columns in the view, flagged
   MODEL_READY=False, not fed to Stage 3 as inputs.

Also drops n_distinct_threads: verified byte-for-byte identical to
n_threads_active across all 347,886 rows (exact duplicate column, not
just correlated) -- confirmed 2026-08-06 before writing this script."""
import os
import time

import duckdb
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

POINT_MASS_THRESHOLD = 0.25
VIF_CEILING = 10.0

# raw model-input candidates by evidence family (Sec 5.3) -- excludes
# identifiers/timestamps, booleans, tier-2-only timing features handled
# separately, and the bot-marker percentile family (see docstring point 4)
FAMILIES = {
    'footprint': ['n_comments_sample', 'n_posts_sample', 'n_subs_active',
                  'n_threads_active', 'high_tier_share', 'medium_tier_share',
                  'low_tier_share', 'n_low_tier_subs', 'subreddit_entropy'],
    'reception': ['mean_comment_score', 'median_comment_score', 'score_stddev',
                  'controversiality_rate', 'is_submitter_rate', 'mean_depth',
                  'mean_body_len', 'mean_post_score', 'best_sub_mean_score',
                  'worst_sub_mean_score', 'reception_spread'],
    'provenance_age': ['account_ordinal', 'observed_span_days',
                        'comments_per_day_observed', 'sample_score_per_day_observed',
                        'days_since_first_seen', 'comments_per_day_since_first_seen',
                        'posts_per_day_since_first_seen', 'karma_per_day_since_first_seen'],
    'username_morphology': ['username_char_entropy', 'username_digit_suffix_len'],
    'timing_tier2': ['interval_entropy', 'burstiness_kimjo', 'interval_quantization_rate'],
    'removal_quality': ['removal_rate', 'deleted_later_rate'],
    'engagement_pattern': ['repeat_engagement_rate', 'own_post_reply_rate',
                            'n_threads_with_repeat', 'n_own_posts_with_comments',
                            'n_subs_rejected_but_returned'],
    'bot_marker_extremeness': ['karma_extremeness', 'karma_per_post_extremeness'],
}

# booleans pass through unchanged, not VIF-pruned (not continuous)
PASSTHROUGH_BOOL = ['ever_automation_seed', 'hobby_absence', 'username_is_default_pattern',
                     'has_timing_features', 'shows_silo_mismatch_pattern']

# reporting-only, never fed to Stage 3 (Sec 5.3 point 4)
REPORTING_ONLY = ['removal_rate_pctl', 'deleted_later_rate_pctl', 'thin_history_score',
                  'reception_spread_pctl', 'botmarker_composite', 'n_markers_available']

DROPPED_DUPLICATE = ['n_distinct_threads']  # identical to n_threads_active, verified


def add_tier_shares(df):
    denom = df['n_comments_sample'].replace(0, np.nan)
    df['high_tier_share'] = df['n_high_tier'] / denom
    df['medium_tier_share'] = df['n_medium_tier'] / denom
    df['low_tier_share'] = df['n_low_tier'] / denom
    return df


def hurdle_indicators(df, candidate_cols):
    added = []
    for col in candidate_cols:
        s = df[col]
        if s.isna().all():
            continue
        vc = s.value_counts(dropna=True)
        if len(vc) == 0:
            continue
        top_val, top_count = vc.index[0], vc.iloc[0]
        share = top_count / s.notna().sum()
        if share >= POINT_MASS_THRESHOLD:
            ind_name = f'{col}_nonzero' if top_val == 0 else f'{col}_ne_{top_val}'
            df[ind_name] = (s != top_val) & s.notna()
            added.append((col, float(top_val), float(share), ind_name))
    return df, added


def vif_prune_family(df, cols, ceiling=VIF_CEILING):
    cols = [c for c in cols if c in df.columns]
    sub = df[cols].astype('float64').copy()
    for c in cols:
        sub[c] = sub[c].fillna(sub[c].median())
    # drop zero-variance columns up front -- VIF is undefined for them
    keep = [c for c in cols if sub[c].std() > 1e-12]
    dropped_zero_var = [c for c in cols if c not in keep]
    removed = []
    while len(keep) > 1:
        X = sub[keep].values.astype(float)
        vifs = []
        for i in range(X.shape[1]):
            try:
                v = variance_inflation_factor(X, i)
            except Exception:
                v = float('inf')
            vifs.append(v)
        worst_i = int(np.argmax(vifs))
        if vifs[worst_i] <= ceiling or not np.isfinite(vifs[worst_i]):
            if not np.isfinite(vifs[worst_i]):
                removed.append((keep[worst_i], vifs[worst_i]))
                keep.pop(worst_i)
                continue
            break
        removed.append((keep[worst_i], vifs[worst_i]))
        keep.pop(worst_i)
    return keep, removed, dropped_zero_var


def main():
    t0 = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute('SELECT * FROM account_features').fetchdf()
    con.close()
    n0 = len(df)
    print(f'Loaded account_features: {n0} rows, {df.shape[1]} cols ({time.time()-t0:.1f}s)\n')

    df = add_tier_shares(df)

    all_family_cols = sorted({c for cols in FAMILIES.values() for c in cols})
    df, hurdles = hurdle_indicators(df, [c for c in all_family_cols if c in df.columns])
    print('=== ZERO-INFLATION HURDLE INDICATORS ADDED ===')
    for col, top_val, share, ind_name in hurdles:
        print(f'  {col:<30} point mass {100*share:.0f}% @ {top_val:g}  ->  {ind_name}')
    print()

    print('=== VIF PRUNING, WITHIN FAMILY ONLY (ceiling=%.0f) ===\n' % VIF_CEILING)
    model_cols = []
    for family, cols in FAMILIES.items():
        keep, removed, dropped_zero_var = vif_prune_family(df, cols)
        print(f'[{family}] candidates={len(cols)} kept={len(keep)}')
        if dropped_zero_var:
            print(f'  dropped (zero-variance): {dropped_zero_var}')
        for name, vif in removed:
            vs = 'inf' if not np.isfinite(vif) else f'{vif:.1f}'
            print(f'  dropped (VIF={vs}): {name}')
        print(f'  kept: {keep}\n')
        model_cols.extend(keep)

    hurdle_cols = [h[3] for h in hurdles]
    model_cols = model_cols + hurdle_cols + PASSTHROUGH_BOOL
    model_cols = [c for c in model_cols if c in df.columns]

    print('=== FINAL MODEL-READY FEATURE MANIFEST ===')
    print(f'{len(model_cols)} columns for Stage 3 (author + label columns added separately at fit time):\n')
    for c in sorted(model_cols):
        print(f'  {c}')

    print(f'\nExcluded as reporting-only (Sec 5.3, not fed to model): {REPORTING_ONLY}')
    print(f'Excluded as verified duplicate: {DROPPED_DUPLICATE}')

    keep_cols = ['author'] + model_cols + REPORTING_ONLY
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols]

    con = duckdb.connect(DB_PATH)  # brief read-write, close immediately after
    try:
        con.register('sanitised_df', out)
        con.execute('CREATE OR REPLACE TABLE account_features_model AS SELECT * FROM sanitised_df')
    finally:
        con.close()
    print(f'\nWrote account_features_model ({len(out)} rows, {out.shape[1]} cols) to {DB_PATH}')
    print(f'DONE in {time.time()-t0:.1f}s.')


if __name__ == '__main__':
    main()
