#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Account-level Anomaly Detection

Complement to analyze_data_v2.py: the production scorer only ever aggregates
to (subreddit, month) — this script works at the account level instead, using
the same already-scraped corpus, so specific suspicious accounts surface
rather than only "this subreddit feels bot-y".

Unsupervised (IsolationForest) — same caveat as the rest of the pipeline: no
labeled bot/human ground truth exists, so "anomalous" means "far from the
account population in feature space", not "confirmed bot". Useful as (a) a
second, independent read on which subreddits look worst, to sanity-check the
subreddit-level score, and (b) a starting point for a labeled sample.

Usage:
  python3 scripts/anomaly_detection.py
  python3 scripts/anomaly_detection.py --contamination 0.03
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / 'data' / 'v2'
OUTPUT_DIR = ROOT / 'output' / 'v2'

OLD_ACCOUNT_DAYS = 180  # threshold for the old-account activity interaction tested below

FEATURE_COLS = [
    'kpd_log', 'link_ratio_log', 'account_age_days', 'unverified',
    'n_posts', 'n_comments', 'n_subs',
    'pct_first5', 'pct_top10',
    'comments_per_day',
]
# Input-sanitization pass (see project notes): a distributional check found
# kpd and link_ratio severely right-skewed (skew 27 and 41 — max value
# 1,000-1,500x the median), so a handful of outliers dominated their
# standardized coefficients under plain StandardScaler. log1p-transformed
# here (kpd_log, link_ratio_log) — this alone let both contribute real
# weight (+0.03 -> +0.18 and +0.08 respectively) instead of being squashed.
# A VIF check found n_months severely collinear (VIF=15.4, over the "strong
# multicollinearity" line of 10) with n_subs/n_comments — three overlapping
# "activity breadth" measures competing for the same variance, which likely
# explains n_months' persistently counter-intuitive negative sign across
# every earlier test. Dropped. msgs_per_day (n_posts+n_comments combined)
# replaced with comments_per_day alone — comments dominate accounts' activity
# in this corpus (n_posts is 75% zero, since being a top-40 POST author is
# far rarer than commenting), and SHAP importance showed msgs_per_day's
# coefficient overstated its real per-observation contribution (rank #3 by
# |coefficient| vs. #7 by mean|SHAP|) relative to comments_per_day alone.
#
# Net effect: best CV-AUC of any variant tested (0.6632 vs. 0.6582 before).
# Backtest correlation is a genuine trade, not a clean win (early_r
# 0.695->0.660, late_r 0.734->0.774, both still highly significant) — adopted
# anyway on independent statistical-hygiene grounds (real skew, real VIF>10
# collinearity, real SHAP/coefficient divergence all fixed), not because the
# noisy n=25 backtest moved in one clean direction. First deviation this
# project has made from requiring an unambiguous backtest win before shipping
# a feature change — a deliberate, discussed exception, not a relaxed bar
# going forward.
#
# avg_comment_depth dropped earlier: tested and confirmed genuinely inert
# (coefficient was already 0.000) — removing it produced IDENTICAL AUC and
# backtest correlations to three decimal places. unverified was tested
# alongside it (mixed, tiny effect in both directions, within CV noise) and
# kept — no demonstrated improvement from dropping it. old_x_msgs_per_day
# (age>=180d AND top-quartile activity-for-old-accounts interaction) was
# tested and dropped: a real-looking bivariate ~2x gone-rate lift didn't
# survive the full multivariate fit (near-zero net contribution, AUC flat-to-
# worse) — classic confounding with the other already-included activity/
# tenure features. Batch-mate-count and cross-sub-timing (network-style
# candidates) showed the same pattern: real bivariate signal, redundant once
# fit alongside existing features. pct_controversial (Reddit's own
# controversiality flag, backfilled corpus-wide) had the largest stable
# coefficient of any tested candidate (~-1.0) and passed full-population-
# coverage testing, but never moved the backtest number at all — flagged as
# a real account-level signal our current n=25 validation instrument likely
# lacks the resolution to confirm at the aggregate level, not shipped.


def load_corpus():
    post_files = sorted(DATA_DIR.glob('posts_20*.csv'))
    comm_files = sorted(DATA_DIR.glob('commenters_20*.csv'))
    if not post_files:
        raise FileNotFoundError(f'No data in {DATA_DIR}. Run collect_data_v2.py --year first.')
    posts = pd.concat([pd.read_csv(f) for f in post_files], ignore_index=True)
    comms = pd.concat([pd.read_csv(f) for f in comm_files], ignore_index=True)
    return posts, comms


def build_account_features(posts: pd.DataFrame, comms: pd.DataFrame) -> pd.DataFrame:
    """One row per unique account, aggregated across every post/comment it made."""
    p = posts.rename(columns={'author': 'account'}).copy()
    c = comms.rename(columns={'author': 'account'}).copy()

    p_agg = p.groupby('account').agg(
        account_age_days=('author_account_age_days', 'first'),
        link_karma=('author_link_karma', 'first'),
        comment_karma=('author_comment_karma', 'first'),
        verified=('author_verified_email', 'first'),
        n_posts=('post_id', 'count'),
        subs_posted=('subreddit', lambda s: set(s)),
        months_posted=('collection_month', lambda s: set(s)),
    )

    c_agg = c.groupby('account').agg(
        c_age=('author_account_age_days', 'first'),
        c_link_karma=('author_link_karma', 'first'),
        c_comment_karma=('author_comment_karma', 'first'),
        c_verified=('author_verified_email', 'first'),
        n_comments=('comment_id', 'count'),
        subs_commented=('subreddit', lambda s: set(s)),
        months_commented=('collection_month', lambda s: set(s)),
        pct_first5=('in_first5', 'mean'),
        pct_top10=('in_top10', 'mean'),
        avg_comment_depth=('comment_depth', 'mean'),
    )

    df = p_agg.join(c_agg, how='outer')

    df['account_age_days'] = df['account_age_days'].fillna(df['c_age'])
    df['link_karma']       = df['link_karma'].fillna(df['c_link_karma'])
    df['comment_karma']    = df['comment_karma'].fillna(df['c_comment_karma'])
    df['verified']         = df['verified'].fillna(df['c_verified'])
    df['n_posts']          = df['n_posts'].fillna(0)
    df['n_comments']       = df['n_comments'].fillna(0)
    df['pct_first5']       = df['pct_first5'].fillna(0)
    df['pct_top10']        = df['pct_top10'].fillna(0)
    df['avg_comment_depth'] = df['avg_comment_depth'].fillna(0)

    def _union(a, b):
        a = a if isinstance(a, set) else set()
        b = b if isinstance(b, set) else set()
        return a | b

    df['subs_posted']    = df['subs_posted'].where(df['subs_posted'].notna(), pd.Series([set()] * len(df), index=df.index))
    df['subs_commented'] = df['subs_commented'].where(df['subs_commented'].notna(), pd.Series([set()] * len(df), index=df.index))
    df['all_subs']       = [ _union(a, b) for a, b in zip(df['subs_posted'], df['subs_commented']) ]
    df['months_posted']    = df['months_posted'].where(df['months_posted'].notna(), pd.Series([set()] * len(df), index=df.index))
    df['months_commented'] = df['months_commented'].where(df['months_commented'].notna(), pd.Series([set()] * len(df), index=df.index))
    df['all_months']       = [ _union(a, b) for a, b in zip(df['months_posted'], df['months_commented']) ]

    df['n_subs']   = df['all_subs'].apply(len)
    df['n_months'] = df['all_months'].apply(len)

    df['kpd']        = (df['link_karma'].fillna(0) + df['comment_karma'].fillna(0)) / df['account_age_days'].clip(lower=1)
    df['link_ratio'] = df['link_karma'].fillna(0) / (df['comment_karma'].fillna(0).clip(lower=0) + 1)
    df['unverified'] = (df['verified'] == False).astype(int)  # noqa: E712 — True/False/NaN, not truthy check

    # kpd/link_ratio are severely right-skewed (see FEATURE_COLS comment) —
    # log1p-transformed versions are what actually go into the model.
    # comments_per_day: n_posts/n_comments are only what showed up in *our*
    # top-40 sample, not the account's true Reddit-wide activity — a narrower
    # proxy, not a real site-wide rate. Comments only, not posts+comments
    # (msgs_per_day) — see FEATURE_COLS comment for why.
    df['kpd_log']         = np.log1p(df['kpd'].clip(lower=0))
    df['link_ratio_log']  = np.log1p(df['link_ratio'].clip(lower=0))
    df['comments_per_day'] = df['n_comments'] / df['account_age_days'].clip(lower=1)
    df['msgs_per_day']     = (df['n_posts'] + df['n_comments']) / df['account_age_days'].clip(lower=1)  # kept for diagnostics/reference, not in FEATURE_COLS

    df = df.dropna(subset=['account_age_days'])
    df.index.name = 'account'
    return df.reset_index()


def score_anomalies(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    X = df[FEATURE_COLS].fillna(0).values
    Xs = StandardScaler().fit_transform(X)

    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=42, n_jobs=-1)
    iso.fit(Xs)
    # decision_function: higher = more normal. Flip + rescale to 0-100, higher = more anomalous.
    raw = -iso.decision_function(Xs)
    df = df.copy()
    df['anomaly_raw']  = raw
    lo, hi = raw.min(), raw.max()
    df['anomaly_score'] = ((raw - lo) / (hi - lo) * 100).round(1) if hi > lo else 0.0
    df['flagged']      = iso.predict(Xs) == -1
    return df


def per_subreddit_rollup(df_scored: pd.DataFrame, posts: pd.DataFrame, comms: pd.DataFrame) -> pd.DataFrame:
    """
    Roll account anomaly scores up to subreddit level by weighting each account's
    score by how much of its activity happened in that sub, so a prolific
    cross-sub account doesn't fully count against every sub it touched once.
    """
    p = posts.rename(columns={'author': 'account'})[['account', 'subreddit']]
    c = comms.rename(columns={'author': 'account'})[['account', 'subreddit']]
    activity = pd.concat([p, c], ignore_index=True)

    merged = activity.merge(df_scored[['account', 'anomaly_score', 'flagged']], on='account', how='inner')
    rollup = merged.groupby('subreddit').agg(
        avg_anomaly_score=('anomaly_score', 'mean'),
        pct_flagged=('flagged', 'mean'),
        n_activity_rows=('account', 'count'),
    ).reset_index()
    rollup['pct_flagged'] = (rollup['pct_flagged'] * 100).round(1)
    rollup['avg_anomaly_score'] = rollup['avg_anomaly_score'].round(1)
    return rollup.sort_values('avg_anomaly_score', ascending=False)


def rollup_to_subreddit(posts: pd.DataFrame, comms: pd.DataFrame, account_scores: pd.DataFrame,
                          score_col: str = 'score') -> pd.DataFrame:
    """
    Generic version of per_subreddit_rollup's activity-weighting for any
    per-account score column (not just IsolationForest's anomaly_score) —
    used by score_accounts.py to roll the production risk score up to
    (subreddit, month) using that period's own posts/comms activity rows.
    """
    p = posts.rename(columns={'author': 'account'})[['account', 'subreddit']]
    c = comms.rename(columns={'author': 'account'})[['account', 'subreddit']]
    activity = pd.concat([p, c], ignore_index=True)

    merged = activity.merge(account_scores[['account', score_col]], on='account', how='inner')
    rollup = merged.groupby('subreddit').agg(
        avg_score=(score_col, 'mean'),
        n_activity_rows=('account', 'count'),
    ).reset_index()
    rollup['avg_score'] = rollup['avg_score'].round(2)
    return rollup


def month_active_accounts(posts_top: pd.DataFrame, comms: pd.DataFrame) -> pd.Series:
    """Accounts active this month: top-sample posters + top-10 commenters."""
    posters    = posts_top['author']
    commenters = comms[comms['in_top10']]['author']
    return pd.concat([posters, commenters], ignore_index=True)


def month_relative_high_risk(posts_top: pd.DataFrame, comms: pd.DataFrame,
                               risk_score_df: pd.DataFrame, percentile: int = 90) -> tuple[pd.DataFrame, float]:
    """
    "High risk" relative to THIS MONTH's active accounts, not a fixed global
    population threshold. risk_score itself is a fixed lifetime value per
    account (age/kpd/etc. captured once, at first-seen — see
    build_account_features), so a *global* top-decile threshold quietly
    tracks the composition of who's active each month rather than any
    account's risk changing: if the whole active population trends younger
    over time, a fixed global cutoff flags a growing share of every single
    month even with zero change in relative behavior. Recomputing the top
    decile fresh each month, among that month's own active accounts, removes
    that drift and answers "is this subreddit's activity disproportionately
    risky compared to its contemporaries right now" instead of "...compared
    to the full multi-year history." Tested (see project notes): removes an
    almost-perfectly monotonic 13-month score climb (4.6→23.0 avg) down to a
    flat ~10 baseline, while barely moving predictive validity (early/late
    backtest correlation 0.668/0.794 → 0.659/0.759).

    Falls back to the population-wide percentile if a month has too few
    active accounts (<10) for a stable month-specific cutoff.
    """
    active = month_active_accounts(posts_top, comms).unique()
    active_scores = risk_score_df[risk_score_df['account'].isin(active)]
    if len(active_scores) < 10:
        threshold = float(risk_score_df['risk_score'].quantile(percentile / 100))
    else:
        threshold = float(active_scores['risk_score'].quantile(percentile / 100))
    out = risk_score_df.copy()
    out['high_risk'] = (out['risk_score'] >= threshold).astype(int)
    return out, threshold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--contamination', type=float, default=0.05,
                    help='Expected fraction of anomalous accounts (default 0.05)')
    p.add_argument('--top-n', type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    print('\n' + '=' * 70)
    print('ACCOUNT-LEVEL ANOMALY DETECTION (IsolationForest)')
    print('=' * 70)

    print('\nLoading corpus…')
    posts, comms = load_corpus()
    print(f'  {len(posts):,} posts  {len(comms):,} comment rows')

    print('Building account-level features…')
    accounts = build_account_features(posts, comms)
    print(f'  {len(accounts):,} unique accounts')

    print(f'Scoring anomalies (contamination={args.contamination})…')
    scored = score_anomalies(accounts, args.contamination)

    print('Rolling up to subreddit level…')
    rollup = per_subreddit_rollup(scored, posts, comms)

    print('\nTop 15 subreddits by avg account anomaly score (independent of final_score):')
    for _, row in rollup.head(15).iterrows():
        print(f"  r/{row['subreddit']:<25} avg_anomaly={row['avg_anomaly_score']:5.1f}  "
              f"flagged={row['pct_flagged']:5.1f}%  n={int(row['n_activity_rows'])}")

    top_accounts = scored.sort_values('anomaly_score', ascending=False).head(args.top_n)
    out = {
        'generated': datetime.now().isoformat(),
        'method': 'IsolationForest, unsupervised, no ground-truth labels — anomalous means '
                  'far from the account population in feature space, not confirmed-bot',
        'contamination': args.contamination,
        'n_accounts': int(len(accounts)),
        'features_used': FEATURE_COLS,
        'subreddit_rollup': rollup.to_dict(orient='records'),
        'top_anomalous_accounts': [
            {
                'account': r['account'], 'anomaly_score': r['anomaly_score'],
                'kpd': round(float(r['kpd']), 1), 'account_age_days': round(float(r['account_age_days']), 1),
                'n_subs': int(r['n_subs']), 'n_posts': int(r['n_posts']), 'n_comments': int(r['n_comments']),
            }
            for _, r in top_accounts.iterrows()
        ],
    }
    out_path = OUTPUT_DIR / 'anomaly_accounts.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
