#!/usr/bin/env python3
"""V3: cross-sample boundary discovery + conjunctive (AND) bot-flag rule.

User-directed alternative to Stage 3's supervised, removal-label-based
channels. Rationale (user's own words): Reddit's moderation actions don't
even agree with each other (V2: admin-removal vs suspension near-disjoint;
Stage 3: self_deletion vs comment_removed_ambiguous overlap 68.6%), so
treating any of them as ground truth for "bot behavior" is shaky. Instead:
find behavior metrics that show a REPLICATING bimodal split across
independent samples (not just once, on the whole corpus, the way Stage 1
did), then build a rule where an account is flagged ONLY if it lands on
the "bot side" of every one of those metrics simultaneously -- a
conjunctive filter, not a compensatory score or a predicted probability.
Removal/deletion/edit-derived features are excluded from the candidate
pool entirely (not just checked for leakage) -- they ARE the external
labels being avoided. Removal_rate is checked against the FINAL rule only
as an external plausibility read, strictly after the rule is fixed, per
explicit instruction not to let it feed back into which metrics or
thresholds make the rule.

Reuses the corrected Stage-1 bimodality methodology unchanged (iterative
point-mass strip, signed-log1p for signed heavy-tailed features, real
KDE-valley check) -- see v3_stage1_univariate.py's docstring for why the
naive version (single-mode strip only, no valley check) is not trustworthy.
"""
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from diptest import diptest
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


RNG_SEED = 42
SCREEN_N = 20_000
MULTI_MASS_THRESHOLD = 0.01
MAX_MASSES_TO_STRIP = 8
MAX_TOTAL_STRIP_SHARE = 0.5
DEGENERATE_SEPARATION_CEILING = 20.0
REAL_SEPARATION_FLOOR = 0.5
KDE_VALLEY_CEILING = 0.9

# -------------------------------------------------------------------------
# Removal/deletion/edit-derived columns: excluded from the candidate pool
# entirely (Step 2). pc_removed_comment_rate/pc_tombstone_rate/
# pc_bot_comment_rate are kept -- they describe the THREAD's context, not
# this account's own moderation record.
# -------------------------------------------------------------------------
LABEL_DERIVED_EXCLUDE = {
    'removal_rate', 'deleted_later_rate', 'removal_rate_nonzero', 'deleted_later_rate_nonzero',
    'removal_rate_pctl', 'deleted_later_rate_pctl', 'thin_history_score',
    'reception_spread_pctl', 'botmarker_composite', 'n_markers_available',
    'post_edit_rate', 'post_edit_rate_nonzero',
}

# candidate continuous/count features: (col, transform, direction_hint)
# direction_hint: 'high' = higher value is the a-priori suspicious direction
# (per this session's own established findings), 'low' = lower value is
# suspicious, None = no prior, use minority-component default.
CANDIDATES = [
    ('n_comments_sample', 'log1p', None),
    ('n_posts_sample', 'log1p', None),
    ('high_tier_share', 'none', None),
    ('medium_tier_share', 'none', None),
    ('low_tier_share', 'none', None),
    ('n_low_tier_subs', 'log1p', None),
    ('subreddit_entropy', 'none', None),
    ('score_stddev', 'log1p', None),
    ('controversiality_rate', 'none', 'high'),
    ('is_submitter_rate', 'none', None),
    ('mean_depth', 'none', None),
    ('mean_body_len', 'log1p', None),
    ('mean_post_score', 'log1p', None),
    ('worst_sub_mean_score', 'signed_log1p', None),
    ('reception_spread', 'log1p', 'high'),
    ('account_ordinal', 'log1p', 'high'),   # newer accounts: 13.3% vs 3.5% removal, session finding
    ('observed_span_days', 'log1p', None),
    ('comments_per_day_observed', 'log1p', 'high'),
    ('sample_score_per_day_observed', 'signed_log1p', None),
    ('days_since_first_seen', 'none', None),
    ('comments_per_day_since_first_seen', 'log1p', 'high'),
    ('posts_per_day_since_first_seen', 'log1p', 'high'),
    ('karma_per_day_since_first_seen', 'signed_log1p', None),
    ('username_char_entropy', 'none', 'low'),          # low entropy = more auto-generated-looking
    ('username_digit_suffix_len', 'none', 'high'),      # long digit suffix = default-pattern-like
    ('interval_entropy', 'none', 'low'),                # low = more mechanical/regular timing
    ('burstiness_kimjo', 'none', 'high'),               # high = more bursty, less organic
    ('interval_quantization_rate', 'none', 'high'),     # high = cron-signature
    ('repeat_engagement_rate', 'none', None),
    ('own_post_reply_rate', 'none', 'low'),             # post-and-leave hypothesis
    ('n_threads_with_repeat', 'log1p', None),
    ('n_own_posts_with_comments', 'log1p', None),
    ('n_subs_rejected_but_returned', 'log1p', 'high'),  # silo-mismatch pattern
    ('karma_extremeness', 'log1p', 'high'),
    ('karma_per_post_extremeness', 'log1p', 'high'),
    ('n_distinct_posts_ctx', 'log1p', None),
    ('pc_contested_share', 'none', None),
    ('pc_comment_score_gini', 'none', None),
    ('pc_reply_reciprocity', 'none', None),
    ('pc_removed_comment_rate', 'none', 'high'),
    ('pc_removed_comment_rate_max', 'none', 'high'),
    ('pc_tombstone_rate', 'none', 'high'),
    ('pc_tombstone_rate_max', 'none', 'high'),
    ('pc_bot_comment_rate', 'none', 'high'),
    ('pc_bot_comment_rate_max', 'none', 'high'),
    ('pc_submitter_reply_rate', 'none', None),
    ('pc_upvote_ratio', 'none', None),
    ('pc_pct_toplevel', 'none', None),
    ('pc_mean_depth', 'none', None),
    ('pc_num_crossposts', 'log1p', None),
    ('pc_log_subscribers', 'none', None),
    ('pc_n_unique_commenters', 'log1p', None),
    ('pc_n_comments_observed', 'log1p', None),
    ('pc_is_self_rate', 'none', None),
    ('pc_over18_rate', 'none', None),
    ('own_repeat_rate', 'none', 'high'),
    ('url_rate', 'none', 'high'),
    ('outsider_influx_share', 'none', None),
    ('title_body_ratio', 'log1p', None),
    ('score_per_word', 'signed_log1p', None),
    ('sub_month_spike_share', 'none', 'low'),  # Stage 2: spike-exposed accounts measured LOWER risk
    ('coappear_degree', 'log1p', None),
    ('coappear_hhi', 'none', 'high'),          # concentrated on same partners = more suspicious
    ('domain_hhi', 'none', 'high'),
]


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def strip_point_masses(raw, threshold=MULTI_MASS_THRESHOLD, max_strip=MAX_MASSES_TO_STRIP,
                        max_total_share=MAX_TOTAL_STRIP_SHARE):
    n = len(raw)
    vals, counts = np.unique(raw, return_counts=True)
    order = np.argsort(-counts)
    stripped = []
    running_share = 0.0
    for idx in order:
        share = counts[idx] / n
        if share < threshold or len(stripped) >= max_strip or running_share + share > max_total_share:
            break
        stripped.append((float(vals[idx]), share))
        running_share += share
    if not stripped:
        return raw, stripped
    stripped_vals = np.array([v for v, _ in stripped])
    remainder = raw[~np.isin(raw, stripped_vals)]
    return remainder, stripped


def kde_valley_ratio(x, means):
    lo, hi = float(min(means)), float(max(means))
    if hi - lo < 1e-9:
        return None
    try:
        kde = gaussian_kde(x)
    except Exception:
        return None
    grid = np.linspace(lo, hi, 200)
    density = kde(grid)
    d_lo, d_hi = kde(np.array([lo]))[0], kde(np.array([hi]))[0]
    peak_ref = min(d_lo, d_hi)
    if peak_ref < 1e-12:
        return None
    return float(density.min() / peak_ref)


def fit_gmm_bic(x):
    x = x.reshape(-1, 1)
    best = None
    for k in (1, 2, 3):
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=3).fit(x)
        bic = gmm.bic(x)
        if best is None or bic < best[0]:
            best = (bic, k, gmm)
    _, k, gmm = best
    return k, gmm


def separation_score(gmm, n_total):
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    valid = [i for i in range(len(weights)) if weights[i] * n_total >= 30]
    if len(valid) < 2:
        return None, None, None
    order = sorted(valid, key=lambda i: -weights[i])[:2]
    i, j = order
    pooled = np.sqrt((stds[i] ** 2 + stds[j] ** 2) / 2)
    if pooled < 1e-9:
        return None, None, None
    sep = abs(means[i] - means[j]) / pooled
    return sep, i, j


def screen_one(raw, transform, rng, orig_min=None):
    """Run the corrected Stage-1 bimodality test on one array. Returns a
    dict with verdict + the boundary (midpoint between component means,
    in ORIGINAL units) if REAL CANDIDATE, else None boundary."""
    n = len(raw)
    if n < 200:
        return {'verdict': 'skip: n<200', 'boundary_orig': None}

    remainder, stripped = strip_point_masses(raw)
    if len(remainder) < 200:
        return {'verdict': 'skip: remainder<200 after strip', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped)}

    x_orig = remainder
    if transform == 'log1p':
        shift = -x_orig.min() if x_orig.min() < 0 else 0.0
        x = np.log1p(x_orig + shift)
    elif transform == 'signed_log1p':
        x = signed_log1p(x_orig)
        shift = 0.0
    else:
        x = x_orig
        shift = 0.0

    if len(x) <= SCREEN_N:
        x_screen, x_screen_orig = x, x_orig
    else:
        idx = rng.choice(len(x), SCREEN_N, replace=False)
        x_screen, x_screen_orig = x[idx], x_orig[idx]

    dip_stat, dip_p = diptest(x_screen)
    k, gmm = fit_gmm_bic(x_screen)
    if k == 1:
        return {'verdict': 'unimodal', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k}

    sep, i, j = separation_score(gmm, len(x_screen))
    if sep is None:
        return {'verdict': 'k>1 but extra component near-empty', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k}
    if sep > DEGENERATE_SEPARATION_CEILING:
        return {'verdict': f'DEGENERATE (sep={sep:.0f})', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k, 'sep': sep}
    if dip_p > 0.05:
        return {'verdict': 'dip not significant', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k, 'sep': sep}
    if sep < REAL_SEPARATION_FLOOR:
        return {'verdict': 'weak separation', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k, 'sep': sep}

    means = gmm.means_.flatten()
    two_means_transformed = sorted([means[i], means[j]])
    valley = kde_valley_ratio(x_screen, two_means_transformed)
    if valley is None or valley > KDE_VALLEY_CEILING:
        return {'verdict': f'no KDE valley (ratio={valley})', 'boundary_orig': None,
                'stripped_share': sum(s for _, s in stripped), 'dip_p': dip_p, 'bic_k': k, 'sep': sep,
                'valley': valley}

    # boundary in transformed space = midpoint between the two component means
    boundary_t = (two_means_transformed[0] + two_means_transformed[1]) / 2.0
    # map back to original units for reporting/thresholding
    if transform == 'log1p':
        boundary_orig = np.expm1(boundary_t) - shift
    elif transform == 'signed_log1p':
        boundary_orig = np.sign(boundary_t) * (np.expm1(abs(boundary_t)))
    else:
        boundary_orig = boundary_t

    minority_weight = float(min(gmm.weights_[i], gmm.weights_[j]))
    minority_is_high = (gmm.weights_[i] < gmm.weights_[j] and means[i] > means[j]) or \
                        (gmm.weights_[j] < gmm.weights_[i] and means[j] > means[i])

    return {
        'verdict': 'REAL CANDIDATE', 'boundary_orig': float(boundary_orig),
        'boundary_t': float(boundary_t), 'dip_p': dip_p, 'bic_k': k, 'sep': sep, 'valley': valley,
        'stripped_share': sum(s for _, s in stripped), 'minority_weight': minority_weight,
        'minority_is_high_side': bool(minority_is_high), 'transform': transform,
    }


def load_data(con):
    log('loading + joining all feature tables on author')
    df = con.execute('SELECT * FROM account_features_model').fetchdf()
    for tbl in ['account_post_context', 'account_tier1_repeat_url', 'account_tier1_post_context',
                'account_tier2_regime', 'account_tier2_coappear', 'account_tier3_domain']:
        t = con.execute(f'SELECT * FROM {tbl}').fetchdf()
        df = df.merge(t, on='author', how='left')
    log(f'joined shape: {df.shape}')
    return df


def stratified_3way_split(df, rng_seed=RNG_SEED):
    """Stratify by (primary tier bucket, account_ordinal decile) so the
    three parts are balanced on tier + rough temporal cohort, not just
    randomly assigned."""
    d = df.copy()
    tier_bucket = np.select(
        [d['high_tier_share'] >= 0.5, d['medium_tier_share'] >= 0.5],
        ['high', 'medium'], default='low_or_mixed')
    ord_valid = d['account_ordinal'].dropna()
    try:
        ordinal_decile = pd.qcut(d['account_ordinal'], 10, labels=False, duplicates='drop')
    except Exception:
        ordinal_decile = pd.Series(0, index=d.index)
    strata = pd.Series(tier_bucket, index=d.index).astype(str) + '_' + ordinal_decile.astype(str)

    rng = np.random.RandomState(rng_seed)
    part_assignment = pd.Series(index=d.index, dtype=object)
    for s, idx in d.groupby(strata).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        n1 = n // 3
        n2 = n // 3
        part_assignment.loc[idx[:n1]] = 'part1'
        part_assignment.loc[idx[n1:n1 + n2]] = 'part2'
        part_assignment.loc[idx[n1 + n2:]] = 'part3'
    d['part'] = part_assignment
    return d


def check_split_balance(d):
    log('=== split composition check ===')
    for part in ['part1', 'part2', 'part3']:
        sub = d[d['part'] == part]
        log(f'{part}: n={len(sub)}  high_tier_share mean={sub["high_tier_share"].mean():.4f}  '
            f'medium_tier_share mean={sub["medium_tier_share"].mean():.4f}  '
            f'account_ordinal mean={sub["account_ordinal"].mean():.0f} median={sub["account_ordinal"].median():.0f}  '
            f'days_since_first_seen mean={sub["days_since_first_seen"].mean():.1f}')


def run_replication_screen(d):
    log('=== Step 3: per-metric replication screen on Part1 vs Part2 ===')
    results = []
    rng1 = np.random.RandomState(RNG_SEED)
    rng2 = np.random.RandomState(RNG_SEED + 1)
    p1 = d[d['part'] == 'part1']
    p2 = d[d['part'] == 'part2']

    for col, transform, direction_hint in CANDIDATES:
        if col not in d.columns:
            log(f'  {col:<32} MISSING FROM JOIN -- skip')
            continue
        raw1 = p1[col].dropna().to_numpy(dtype=float)
        raw2 = p2[col].dropna().to_numpy(dtype=float)
        if len(raw1) < 200 or len(raw2) < 200:
            log(f'  {col:<32} too few non-null rows (n1={len(raw1)}, n2={len(raw2)}) -- skip')
            continue
        r1 = screen_one(raw1, transform, rng1)
        r2 = screen_one(raw2, transform, rng2)
        standout = (r1['verdict'] == 'REAL CANDIDATE' and r2['verdict'] == 'REAL CANDIDATE')
        row = {
            'feature': col, 'transform': transform, 'direction_hint': direction_hint,
            'verdict1': r1['verdict'], 'verdict2': r2['verdict'],
            'boundary1': r1.get('boundary_orig'), 'boundary2': r2.get('boundary_orig'),
            'minority_high1': r1.get('minority_is_high_side'), 'minority_high2': r2.get('minority_is_high_side'),
            'sep1': r1.get('sep'), 'sep2': r2.get('sep'),
            'standout': standout,
        }
        results.append(row)
        flag = '*** STANDOUT ***' if standout else ''
        b1 = f'{row["boundary1"]:.4g}' if row['boundary1'] is not None else '--'
        b2 = f'{row["boundary2"]:.4g}' if row['boundary2'] is not None else '--'
        log(f'  {col:<32} P1={r1["verdict"]:<28} P2={r2["verdict"]:<28} bnd1={b1:>10} bnd2={b2:>10} {flag}')

    return pd.DataFrame(results)


def month_tier_robustness(d, standout_rows):
    log('=== month/tier robustness check on standout metrics ===')
    d = d.copy()
    d['tier_bucket'] = np.select(
        [d['high_tier_share'] >= 0.5, d['medium_tier_share'] >= 0.5],
        ['high', 'medium'], default='low_or_mixed')
    p12 = d[d['part'].isin(['part1', 'part2'])].dropna(subset=['account_ordinal']).copy()
    ord_med = p12['account_ordinal'].median()
    p12['era'] = np.where(p12['account_ordinal'].to_numpy() <= ord_med, 'older_half', 'newer_half')

    robustness = {}
    for _, row in standout_rows.iterrows():
        col = row['feature']
        transform = row['transform']
        sub_results = {}
        for tier_val in ['high', 'medium', 'low_or_mixed']:
            raw = p12.loc[p12['tier_bucket'] == tier_val, col].dropna().to_numpy(dtype=float)
            if len(raw) < 500:
                sub_results[f'tier={tier_val}'] = 'n<500'
                continue
            r = screen_one(raw, transform, np.random.RandomState(0))
            sub_results[f'tier={tier_val}'] = r['verdict']
        for era_val in ['older_half', 'newer_half']:
            raw = p12.loc[p12['era'] == era_val, col].dropna().to_numpy(dtype=float)
            if len(raw) < 500:
                sub_results[f'era={era_val}'] = 'n<500'
                continue
            r = screen_one(raw, transform, np.random.RandomState(0))
            sub_results[f'era={era_val}'] = r['verdict']
        n_real = sum(1 for v in sub_results.values() if v == 'REAL CANDIDATE')
        robustness[col] = {'slices': sub_results, 'n_real_of_5': n_real}
        log(f'  {col:<32} {sub_results}  ({n_real}/5 slices REAL)')
    return robustness


def build_and_rule(d, standout_rows):
    log('=== Step 4: incremental AND-rule ===')
    p1 = d[d['part'] == 'part1']
    p2 = d[d['part'] == 'part2']

    # order standout metrics by mean of sep1/sep2 (cleanest replication first)
    standout_rows = standout_rows.copy()
    standout_rows['avg_sep'] = (standout_rows['sep1'].fillna(0) + standout_rows['sep2'].fillna(0)) / 2
    standout_rows = standout_rows.sort_values('avg_sep', ascending=False).reset_index(drop=True)

    conds1 = pd.Series(True, index=p1.index)
    conds2 = pd.Series(True, index=p2.index)
    curve = []
    for _, row in standout_rows.iterrows():
        col = row['feature']
        boundary1 = row['boundary1']
        # direction: prefer explicit hint, else use minority-side-is-bot-side default
        hint = row['direction_hint']
        if hint == 'high':
            is_bot1 = p1[col] >= boundary1
            is_bot2 = p2[col] >= row['boundary2']
        elif hint == 'low':
            is_bot1 = p1[col] <= boundary1
            is_bot2 = p2[col] <= row['boundary2']
        else:
            # default: minority component's side is the "bot side"
            minority_high1 = row['minority_high1']
            is_bot1 = (p1[col] >= boundary1) if minority_high1 else (p1[col] <= boundary1)
            minority_high2 = row['minority_high2']
            is_bot2 = (p2[col] >= row['boundary2']) if minority_high2 else (p2[col] <= row['boundary2'])
        is_bot1 = is_bot1.fillna(False)
        is_bot2 = is_bot2.fillna(False)
        conds1 = conds1 & is_bot1
        conds2 = conds2 & is_bot2
        n1 = int(conds1.sum())
        n2 = int(conds2.sum())
        curve.append({'added_feature': col, 'direction': hint or ('minority-default'),
                      'n_flagged_part1': n1, 'n_flagged_part2': n2,
                      'pct_part1': 100 * n1 / len(p1), 'pct_part2': 100 * n2 / len(p2)})
        log(f'  +{col:<32} dir={hint or "minority":<10} flagged: P1={n1:>7} ({100*n1/len(p1):.3f}%)  '
            f'P2={n2:>7} ({100*n2/len(p2):.3f}%)')

    return pd.DataFrame(curve), standout_rows, conds1, conds2


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = load_data(con)
    d = stratified_3way_split(df)
    check_split_balance(d)

    standout_df = run_replication_screen(d)
    standouts = standout_df[standout_df['standout']]
    log(f'\n{len(standouts)}/{len(standout_df)} candidate metrics replicated (REAL CANDIDATE in both Part1 and Part2)')
    log('Standout metrics: ' + ', '.join(standouts['feature'].tolist()))

    robustness = month_tier_robustness(d, standouts) if len(standouts) else {}

    curve_df, standouts_ordered, conds1, conds2 = build_and_rule(d, standouts) if len(standouts) else (None, None, None, None)

    out = {
        'split_balance': {p: {
            'n': int((d['part'] == p).sum()),
            'high_tier_share_mean': float(d.loc[d['part'] == p, 'high_tier_share'].mean()),
            'medium_tier_share_mean': float(d.loc[d['part'] == p, 'medium_tier_share'].mean()),
            'account_ordinal_mean': float(d.loc[d['part'] == p, 'account_ordinal'].mean()),
            'days_since_first_seen_mean': float(d.loc[d['part'] == p, 'days_since_first_seen'].mean()),
        } for p in ['part1', 'part2', 'part3']},
        'n_candidates_tested': int(len(standout_df)),
        'n_standout': int(len(standouts)),
        'standout_features': standouts[['feature', 'transform', 'direction_hint', 'boundary1', 'boundary2',
                                         'sep1', 'sep2', 'minority_high1', 'minority_high2']].to_dict('records'),
        'robustness': robustness,
        'and_rule_curve': curve_df.to_dict('records') if curve_df is not None else [],
    }

    import json
    out_path = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'boundary_discovery_data.json')

    # Step 5: validate EVERY prefix cutoff (top-k conditions, k=1..N) on Part3,
    # not just the trivial full-AND -- the curve above showed the rule collapses
    # to 0 well before all N conditions are required, so the informative cutoffs
    # are the ones just before collapse, not the endpoint.
    if standouts_ordered is not None and len(standouts_ordered):
        log('\n=== Step 5: validating each prefix cutoff (thresholds fixed from Part1) on held-out Part3 ===')
        p3 = d[d['part'] == 'part3']
        p1 = d[d['part'] == 'part1']
        cond3 = pd.Series(True, index=p3.index)
        cond1_running = pd.Series(True, index=p1.index)
        prefix_results = []
        for k, (_, row) in enumerate(standouts_ordered.iterrows(), start=1):
            col = row['feature']
            hint = row['direction_hint']
            if hint == 'high':
                is_bot3 = p3[col] >= row['boundary1']
                is_bot1 = p1[col] >= row['boundary1']
            elif hint == 'low':
                is_bot3 = p3[col] <= row['boundary1']
                is_bot1 = p1[col] <= row['boundary1']
            else:
                is_bot3 = (p3[col] >= row['boundary1']) if row['minority_high1'] else (p3[col] <= row['boundary1'])
                is_bot1 = (p1[col] >= row['boundary1']) if row['minority_high1'] else (p1[col] <= row['boundary1'])
            cond3 = cond3 & is_bot3.fillna(False)
            cond1_running = cond1_running & is_bot1.fillna(False)
            n3, n1 = int(cond3.sum()), int(cond1_running.sum())
            prefix_results.append({
                'k': k, 'added_feature': col,
                'n_flagged_part1_using_part1_thresholds': n1,
                'n_flagged_part3_using_part1_thresholds': n3,
                'pct_part3': 100 * n3 / len(p3),
            })
            log(f'  k={k:<2} +{col:<28} Part1(own thresh)={n1:>7}  Part3(holdout, Part1 thresh)={n3:>7} ({100*n3/len(p3):.4f}%)')
        out['part3_prefix_validation'] = prefix_results

        # pick the last non-collapsed prefix (largest k with n3 > 0) as "the rule"
        non_zero = [r for r in prefix_results if r['n_flagged_part3_using_part1_thresholds'] > 0]
        chosen = non_zero[-1] if non_zero else prefix_results[0]
        chosen_k = chosen['k']
        log(f'\nChosen cutoff for plausibility check: k={chosen_k} (last non-collapsed prefix on Part3)')

        cond1_chosen = pd.Series(True, index=p1.index)
        p2 = d[d['part'] == 'part2']
        cond2_chosen = pd.Series(True, index=p2.index)
        cond3_chosen = pd.Series(True, index=p3.index)
        for _, row in standouts_ordered.iloc[:chosen_k].iterrows():
            col = row['feature']
            hint = row['direction_hint']
            for part_df, cond_name in [(p1, 'cond1_chosen'), (p2, 'cond2_chosen'), (p3, 'cond3_chosen')]:
                pass
            if hint == 'high':
                b1 = row['boundary1']
                is_bot1 = p1[col] >= b1
                is_bot2 = p2[col] >= b1
                is_bot3 = p3[col] >= b1
            elif hint == 'low':
                b1 = row['boundary1']
                is_bot1 = p1[col] <= b1
                is_bot2 = p2[col] <= b1
                is_bot3 = p3[col] <= b1
            else:
                b1 = row['boundary1']
                is_bot1 = (p1[col] >= b1) if row['minority_high1'] else (p1[col] <= b1)
                is_bot2 = (p2[col] >= b1) if row['minority_high1'] else (p2[col] <= b1)
                is_bot3 = (p3[col] >= b1) if row['minority_high1'] else (p3[col] <= b1)
            cond1_chosen &= is_bot1.fillna(False)
            cond2_chosen &= is_bot2.fillna(False)
            cond3_chosen &= is_bot3.fillna(False)

        # Step 6: plausibility check against removal_rate, STRICTLY AFTER finalization
        log('\n=== Step 6: external plausibility check (removal_rate) -- NOT used for tuning ===')
        removal_all = con.execute('SELECT author, removal_rate, deleted_later_rate FROM account_features_model').fetchdf()
        d2 = d.merge(removal_all, on='author', how='left', suffixes=('', '_ext'))
        flagged_authors = set(p1.loc[cond1_chosen, 'author']) | set(p2.loc[cond2_chosen, 'author']) | \
                           set(p3.loc[cond3_chosen, 'author'])
        d2['flagged'] = d2['author'].isin(flagged_authors)
        removal_flagged = d2.loc[d2['flagged'], 'removal_rate'].mean()
        removal_pop = d2['removal_rate'].mean()
        deleted_flagged = d2.loc[d2['flagged'], 'deleted_later_rate'].mean()
        deleted_pop = d2['deleted_later_rate'].mean()
        log(f'  chosen rule: top-{chosen_k} conditions: {list(standouts_ordered["feature"].iloc[:chosen_k])}')
        log(f'  n_flagged (all 3 parts, each using thresholds fixed from Part1): {d2["flagged"].sum()}  '
            f'(part1={int(cond1_chosen.sum())}, part2={int(cond2_chosen.sum())}, part3={int(cond3_chosen.sum())})')
        log(f'  removal_rate: flagged={removal_flagged:.4f}  population={removal_pop:.4f}  ratio={removal_flagged/removal_pop if removal_pop else float("nan"):.2f}x')
        log(f'  deleted_later_rate: flagged={deleted_flagged:.4f}  population={deleted_pop:.4f}  ratio={deleted_flagged/deleted_pop if deleted_pop else float("nan"):.2f}x')
        out['plausibility_check'] = {
            'chosen_k': chosen_k, 'chosen_features': list(standouts_ordered['feature'].iloc[:chosen_k]),
            'n_flagged_total': int(d2['flagged'].sum()),
            'n_flagged_part1': int(cond1_chosen.sum()), 'n_flagged_part2': int(cond2_chosen.sum()),
            'n_flagged_part3': int(cond3_chosen.sum()),
            'removal_rate_flagged': float(removal_flagged), 'removal_rate_population': float(removal_pop),
            'deleted_later_rate_flagged': float(deleted_flagged), 'deleted_later_rate_population': float(deleted_pop),
        }

    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
