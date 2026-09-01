"""
RedditWatch 1.0 — monthly refresh for the bot/spam detection compass.

Re-run this monthly (after the corpus has been extended with a new month of
posts/comments) to:
  1. Retrain the final model (15 features, tuned XGBoost, clubbed banned+deleted
     target, account_ordinal excluded — see docs/v3-research/charts/model_analysis.html
     for why) on the current hand-verified label set (output/v3/ground_truth_labels.csv).
  2. Score the full account population.
  3. Recompute the subreddit-month bot/spam prevalence table (top-30-posts-by-karma
     methodology) across every month in the corpus.
  4. Regenerate docs/bot-spam-compass.html with the refreshed data (via v3_stage9).

No manual steps beyond running this file. Takes well under a minute.
"""
import argparse
import duckdb, pandas as pd, numpy as np, json
from pathlib import Path
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / 'data/v3/analysis/v3.duckdb'
OUT = ROOT / 'output/v3'

# account_features columns that are exact duplicates of another column (the "_1" variants) --
# found and confirmed via corr()==1.000000 during model finalization; drop them, don't split
# importance between identical copies.
DUP_COLS = ['removal_rate_pctl_1', 'deleted_later_rate_pctl_1', 'thin_history_score_1',
            'karma_extremeness_1', 'karma_per_post_extremeness_1', 'reception_spread_pctl_1',
            'botmarker_composite_1', 'n_markers_available_1']
# near-duplicate columns (rho > 0.85 with another kept column, confirmed via the final
# correlation sweep) -- different names/definitions but carry essentially the same
# information for modeling purposes; kept the higher-importance member of each cluster:
#   n_own_posts_with_comments kept over: n_posts_sample, posts_per_day_since_first_seen
#   n_subs_rejected_but_returned kept over: shows_silo_mismatch_pattern
#   n_comments_sample kept over: n_gaps, n_threads_active, n_distinct_threads
#   removal_rate_pctl kept over: removal_rate, deleted_later_rate
#   comments_per_day_observed kept over: comments_per_day_since_first_seen (see note below)
#   sample_score_per_day_observed kept over: karma_per_day_since_first_seen (see note below)
# BUG FOUND AND FIXED 2026-08-22: comments_per_day_since_first_seen / posts_per_day_since_first_seen /
# karma_per_day_since_first_seen were computed in v3_account_features.py using epoch(now()) --
# wall-clock "today" -- as their time denominator instead of the account's actual last_seen_utc.
# Every refresh, "now" advances for every account, so these rates mechanically shrank release-over-
# release regardless of real behavior -- a corpus-aging artifact that was silently deflating the
# whole 24-month subreddit-prevalence trend toward "everything looks safer lately." Fixed to use
# last_seen_utc (matching the already-correct comments_per_day_observed / sample_score_per_day_observed
# pattern) -- which makes them EXACT duplicates of those two already-kept columns, so they're now
# pruned here instead, same as any other near-duplicate.
NEAR_DUP_COLS = ['n_posts_sample', 'posts_per_day_since_first_seen', 'shows_silo_mismatch_pattern',
                 'n_gaps', 'n_threads_active', 'n_distinct_threads', 'removal_rate',
                 'deleted_later_rate', 'contribs_per_month',
                 'comments_per_day_since_first_seen', 'karma_per_day_since_first_seen']
# Feature-count reduction, two passes:
# 49->18 via backward elimination (drop the single lowest-importance feature, refit, repeat).
# AUC held or improved down to ~17-20 features, confirming most of the 49 were redundant. One
# manual swap: removal_rate_pctl kept over deleted_later_rate_pctl (rho=0.886, removal_rate_pctl
# has more solo signal: bivariate AUC 0.625 vs 0.590).
# 18->15 from fixing a wall-clock time-denominator bug (days_since_first_seen /
# comments_per_day_since_first_seen / karma_per_day_since_first_seen used epoch(now()) instead of
# last_seen_utc -- see v3_account_features.py); the fix made all 3 exact duplicates of
# already-kept columns.
# 15->10 by user request ("limit to ~10-12 params"): a second backward-elimination pass, same
# method, confirmed a real but modest AUC cost (0.789 -> 0.778, consistent across n=10/11/12, not
# noise) with clean correlation (max rho=0.73, well under the 0.85 threshold) -- accepted as the
# right tradeoff for a simpler model. Below is every account_features / custom column NOT in the
# final 10.
LOW_IMPORTANCE_COLS = ['n_comments_sample', 'mean_post_score', 'n_subs_active', 'mean_comment_score',
                       'median_comment_score', 'controversiality_rate', 'is_submitter_rate',
                       'ever_automation_seed', 'n_low_tier', 'n_low_tier_subs', 'hobby_absence',
                       'subreddit_entropy', 'username_is_default_pattern', 'username_char_entropy',
                       'username_digit_suffix_len', 'interval_entropy', 'interval_quantization_rate',
                       'has_timing_features', 'repeat_engagement_rate', 'n_threads_with_repeat',
                       'best_sub_mean_score', 'reception_spread', 'deleted_later_rate_pctl',
                       'karma_extremeness', 'karma_per_post_extremeness', 'reception_spread_pctl',
                       'botmarker_composite', 'n_markers_available', 'churn_ratio',
                       'net_karma_spread_by_sub', 'median_contrib_score',
                       'score_stddev', 'mean_depth', 'burstiness_kimjo', 'own_post_reply_rate',
                       'n_subs']
# collection-snapshot timing artifacts (mechanically shift once an account is removed --
# symptom of removal, not a behavioral precursor) PLUS account_ordinal, excluded on purpose:
# it only bought +0.002 AUC once tuned/deduplicated, and manually-verified samples
# showed it made the model's tiers barely separate high from low risk -- reads as "just
# flags new accounts," not worth the interpretability cost. days_since_first_seen joins this
# list post wall-clock-bug-fix (see NEAR_DUP_COLS note) -- it's now mathematically identical
# to observed_span_days, so the same leakage reasoning applies.
EXCLUDE_COLS = ['last_seen_utc', 'first_seen_utc', 'observed_span_days', 'account_ordinal',
                'days_since_first_seen']
# columns where NaN means "has no posts" (semantically 0, not the population median)
POST_ZERO_COLS = ['n_own_posts_with_comments']

# random_state pinned 2026-09-01: subsample/colsample make the fit stochastic, so two retrains
# on identical data landed on different trees -- one more source of silent restatement.
BEST_CFG = {'max_depth': 5, 'n_estimators': 150, 'learning_rate': 0.1, 'min_child_weight': 1,
            'subsample': 0.9, 'colsample_bytree': 0.7, 'reg_lambda': 1, 'random_state': 42}
ACTIVITY_FLOOR = 10   # min total (comments+posts) contributions to be scored at all
HIGH_RISK_THRESHOLD = 0.7
# Sampling depth for the subreddit-prevalence metric. These are the v1.0 published values and are
# NOT changed in v1.1 -- widening them is a live proposal, deliberately deferred.
#
# The case for widening, measured 2026-09-01 and held for a separate decision: the collector
# already stores 100 top-by-score posts per sub-month and up to 10 top + 10 first commenters per
# post, so the pipeline reads a fraction of what is on disk. Raising these to 100 / 30 costs no
# new collection and would take median unique posters 25 -> 73 (p10 16 -> 42) and commenters
# 128 -> 966. But it changes what the metric MEANS -- influence over a sub's top ~100 posts rather
# than its top 30 -- and measured prevalence falls ~1.5pp as a result (12.72% -> 11.26%), because
# the elite-30 slice runs hotter: repeat karma-farmers chase exactly the top spot (V3_PLAN.md
# 2026-08-22). Spearman between the two definitions is 0.867. That is a new metric, not a refresh,
# so it belongs in its own SERIES_VERSION with both series published -- not folded into v1.1.
TOP_POSTS_PER_CELL = 30
COMMENTERS_PER_POST = 5
# Minimum SCORED accounts before a (sub, month, role) cell gets a published prevalence figure.
# Matches v3_stage7_monthly_score.py's MIN_ACCOUNTS_PER_SUB_MONTH -- that floor existed there but
# was never applied here, the path that actually feeds the dashboard. Found 2026-09-01: 19 of
# 1,120 published subreddit-months sat below it and 3 below n=3. IndiaTrending 2026-03 published
# a figure derived from ONE scored account (22 influencers, 4.5% coverage), so it read 100% in one
# vintage and 0% in the next -- a coin flip rendered as a severity band. Cells below the floor are
# now suppressed to NaN (stage9 drops NaN rows) rather than published as if they were measurements.
MIN_SCORED_PER_CELL = 5

# Frozen-vintage scoring, 2026-09-01. Measured against the 2026-08-22 refresh: re-running this
# script restated 94.5% of the 1,076 published subreddit-months (mean |delta| 1.74pp, max 18.18pp)
# and flipped 24.6% of published severity labels. 24.4% of those flips traced to account scores
# moving, only 3.4% to severity bands -- so the scoring function is what had to be pinned:
#   1. train_final_model() ran every refresh. Now only under --retrain, which bumps MODEL_VERSION.
#   2. Imputation used the population median at score time but the labeled-set median at train
#      time -- a moving reference AND a train/serve skew. Medians are now fit once, persisted,
#      and reused verbatim when scoring.
#   3. random_state (above).
# Still unfixed and still moves history: account_features aggregates each account's whole 24-month
# history, so extending the corpus changes features -- and scores -- for months already published.
# Fix is point-in-time features; see V3_PLAN.md.
MODEL_VERSION = '1.1.0'
MODEL_PATH = OUT / 'final_bot_model.json'
MODEL_META_PATH = OUT / 'final_bot_model_meta.json'


def load_features(con):
    af = con.execute('SELECT * FROM account_features').fetchdf()
    af = af.drop(columns=[c for c in EXCLUDE_COLS + DUP_COLS + NEAR_DUP_COLS + LOW_IMPORTANCE_COLS if c in af.columns])

    contribs = con.execute('''
        WITH per_contrib AS (
            SELECT author FROM commenters_dedup
            UNION ALL
            SELECT author FROM posts
        )
        SELECT author, count(*) AS total_contribs FROM per_contrib GROUP BY author
    ''').fetchdf()

    af = af.merge(contribs, on='author', how='left')

    for c in POST_ZERO_COLS:
        if c in af.columns:
            af[c] = af[c].fillna(0)
    return af


def get_feature_cols(af):
    return [c for c in af.columns if c not in ('author', 'total_contribs')
            and pd.api.types.is_numeric_dtype(af[c])]


def train_final_model(af, feature_cols):
    allt = pd.read_csv(OUT / 'ground_truth_labels.csv')
    allt['flag'] = allt['flag'].astype(str).str.strip().str.lower()
    allt = allt[allt['flag'].isin(['ok', 'banned', 'deleted', 'mod'])].drop_duplicates('author')
    allt['y'] = allt['flag'].isin(['banned', 'deleted']).astype(int)

    train_df = allt[['author', 'y']].merge(af, on='author', how='left')
    X_train = train_df[feature_cols].astype('float64')
    # These medians ARE the model's definition of "missing" -- persist them and reuse at score
    # time. Recomputing them per-population was the train/serve skew described at MODEL_VERSION.
    impute_medians = X_train.median()
    X_train = X_train.fillna(impute_medians)
    y_train = train_df['y'].values

    model = xgb.XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                               eval_metric='logloss', **BEST_CFG)
    model.fit(X_train, y_train)
    model.save_model(str(MODEL_PATH))
    with open(OUT / 'final_bot_model_features.json', 'w') as f:
        json.dump(feature_cols, f)
    meta = {
        'model_version': MODEL_VERSION,
        'n_train': int(len(train_df)),
        'n_positive': int(y_train.sum()),
        'feature_cols': feature_cols,
        'impute_medians': {k: (None if pd.isna(v) else float(v))
                           for k, v in impute_medians.items()},
        'best_cfg': BEST_CFG,
        'activity_floor': ACTIVITY_FLOOR,
        'high_risk_threshold': HIGH_RISK_THRESHOLD,
    }
    with open(MODEL_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Trained model_version={MODEL_VERSION} on n={len(train_df)} labeled accounts '
          f'({y_train.sum()} banned/deleted), {len(feature_cols)} features (RedditWatch 1.0: '
          f'no account_ordinal, deduplicated, backward-eliminated to the minimal set, tuned).')
    return model, meta


def load_pinned_model():
    """Load the frozen model + the feature list and impute medians it was fit with.

    Refuses rather than falling back to a retrain: silently retraining is exactly the behavior
    that restated published history, so a missing artifact must be an explicit --retrain."""
    if not (MODEL_PATH.exists() and MODEL_META_PATH.exists()):
        raise SystemExit(
            f'No pinned model at {MODEL_PATH.name} / {MODEL_META_PATH.name}.\n'
            'Run once with --retrain to fit and pin one. Refusing to retrain implicitly: that '
            'restates every published month (see MODEL_VERSION note in this file).')
    meta = json.loads(MODEL_META_PATH.read_text())
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"Loaded pinned model_version={meta['model_version']} "
          f"({len(meta['feature_cols'])} features, trained on n={meta['n_train']}).")
    return model, meta


def score_population(af, model, meta):
    feature_cols = meta['feature_cols']
    missing = [c for c in feature_cols if c not in af.columns]
    if missing:
        raise SystemExit(
            f'account_features is missing {len(missing)} column(s) the pinned model needs: '
            f'{missing}. The corpus rebuild changed the feature schema -- rerun with --retrain '
            'and bump MODEL_VERSION rather than scoring against a mismatched feature set.')
    pop = af[af['total_contribs'] >= ACTIVITY_FLOOR].copy()
    X = pop[feature_cols].astype('float64')
    medians = pd.Series({k: (np.nan if v is None else v)
                         for k, v in meta['impute_medians'].items()})
    X = X.fillna(medians)
    pop['bot_score'] = model.predict_proba(X)[:, 1]
    pop[['author', 'bot_score']].to_parquet(OUT / 'final_bot_scores.parquet', index=False)
    print(f'Scored {len(pop)} accounts (>= {ACTIVITY_FLOOR} contributions) '
          f"under model_version={meta['model_version']}.")
    return pop[['author', 'bot_score']]


def build_subreddit_prevalence(con, scores):
    max_month = con.execute("SELECT max(month) FROM posts").fetchone()[0]
    # Two correctness fixes, 2026-09-01, both affecting which posts enter the influencer set:
    #
    # 1. `AND role = 'top'`. The collector stores 100 top-by-score posts per sub-month PLUS a
    #    separately-tagged ~20-post counter-sample drawn from BELOW the top 100 (a deliberate
    #    control group, never intended as "top content"). The published query had no role filter,
    #    so counter-sample posts could enter the top-30 set.
    # 2. `, post_id` as a tiebreaker. row_number() over an unstable sort is non-deterministic:
    #    re-running the identical query on the identical database returned between 4 and 22
    #    counter-sample posts inside the top-30 across five consecutive executions, because 99 of
    #    1,120 sub-months (8.8%) have a score tie exactly at the 30/31 cutoff. Published v1.0
    #    numbers were therefore not reproducible even from unchanged data. post_id is stable and
    #    arbitrary, which is what a tiebreaker should be.
    top_posts = con.execute(f'''
        SELECT sub, month, post_id, author AS poster,
               row_number() OVER (PARTITION BY sub, month ORDER BY score DESC, post_id) AS rn
        FROM posts WHERE month <= '{max_month}' AND role = 'top'
        QUALIFY rn <= {TOP_POSTS_PER_CELL}
    ''').fetchdf()
    con.register('top_post_ids', top_posts[['post_id']].drop_duplicates())

    top_commenters = con.execute(f'''
        SELECT post_id, author FROM (
            SELECT c.post_id, c.author, c.score,
                   row_number() OVER (PARTITION BY c.post_id ORDER BY c.score DESC, c.author) AS rn
            FROM commenters_dedup c JOIN top_post_ids t USING(post_id)
        ) WHERE rn <= {COMMENTERS_PER_POST}
    ''').fetchdf()
    latest_commenters = con.execute(f'''
        SELECT post_id, author FROM (
            SELECT c.post_id, c.author, c.created_utc,
                   row_number() OVER (PARTITION BY c.post_id ORDER BY c.created_utc DESC, c.author) AS rn
            FROM commenters_dedup c JOIN top_post_ids t USING(post_id)
        ) WHERE rn <= {COMMENTERS_PER_POST}
    ''').fetchdf()

    posters = top_posts[['sub', 'month', 'post_id', 'poster']].rename(columns={'poster': 'author'})
    posters['role'] = 'poster'
    tc = top_commenters.merge(top_posts[['sub', 'month', 'post_id']], on='post_id')
    lc = latest_commenters.merge(top_posts[['sub', 'month', 'post_id']], on='post_id')
    commenters = pd.concat([tc[['sub', 'month', 'author']], lc[['sub', 'month', 'author']]], ignore_index=True)
    commenters = commenters.drop_duplicates(['sub', 'month', 'author'])
    commenters['role'] = 'commenter'

    # combined = union of both roles, deduped per (sub, month, author) -- an account posting
    # AND commenting in the same sub-month counts once, same definition used historically
    combined = pd.concat([posters[['sub', 'month', 'author']], commenters[['sub', 'month', 'author']]],
                          ignore_index=True).drop_duplicates(['sub', 'month', 'author'])
    combined['role'] = 'combined'

    all_roles = pd.concat([posters[['sub', 'month', 'author', 'role']],
                            commenters[['sub', 'month', 'author', 'role']],
                            combined[['sub', 'month', 'author', 'role']]], ignore_index=True)
    all_roles = all_roles.merge(scores, on='author', how='left')
    all_roles['scored'] = all_roles['bot_score'].notna()
    all_roles['high_risk'] = all_roles['scored'] & (all_roles['bot_score'] >= HIGH_RISK_THRESHOLD)

    def summarize(g):
        n_total = len(g)
        n_scored = g['scored'].sum()
        n_hr = g['high_risk'].sum()
        # Below the floor the ratio is noise, not a measurement -- report the counts (so the
        # suppression is visible and auditable) but withhold the rate.
        enough = n_scored >= MIN_SCORED_PER_CELL
        return pd.Series({
            'n': n_total, 'n_scored': n_scored,
            'coverage_pct': 100 * n_scored / n_total if n_total else np.nan,
            'pct_high_risk': 100 * n_hr / n_scored if enough else np.nan,
            'mean_bot_score': g.loc[g['scored'], 'bot_score'].mean() if enough else np.nan,
        })

    by_role = all_roles.groupby(['sub', 'month', 'role']).apply(summarize, include_groups=False).reset_index()
    n_suppressed = int((by_role['n_scored'] < MIN_SCORED_PER_CELL).sum())
    if n_suppressed:
        print(f'Suppressed {n_suppressed} (sub, month, role) cell(s) below the '
              f'MIN_SCORED_PER_CELL={MIN_SCORED_PER_CELL} floor.')
    wide = by_role.pivot(index=['sub', 'month'], columns='role',
                          values=['n', 'n_scored', 'coverage_pct', 'pct_high_risk', 'mean_bot_score'])
    wide.columns = [f'{role}_{metric}' for metric, role in wide.columns]
    monthly = wide.reset_index()
    # back-compat aliases: the combined role keeps the original (pre-split) column names, since
    # every earlier chart/script/whitepaper section refers to these by name
    monthly['n_influencers'] = monthly['combined_n']
    monthly['coverage_pct'] = monthly['combined_coverage_pct']
    monthly['pct_high_risk_of_scored'] = monthly['combined_pct_high_risk']
    monthly['mean_bot_score_scored'] = monthly['combined_mean_bot_score']
    # Comment self-deletion rate -- checked 2026-08-23 against 1,114 subreddit-months: the one
    # content-moderation signal (of four candidates) that's genuinely complementary to account
    # risk, not redundant with it. Spearman rho=0.32 vs combined_pct_high_risk -- moderate and
    # real, not near-1.0 (same signal) or near-0 (noise). Post removal rates went the *wrong*
    # direction to use naively (more active moderation reads as *lower* apparent risk, since
    # it's catching things before they become "influencers") and were dropped; "total posts"
    # was dropped too -- the posts table is capped at ~120/sub/month by collection design, so
    # it measures the cap, not real activity. Reported as its own column, not blended into the
    # risk score, same transparency-over-black-box reasoning as the poster/commenter split.
    comment_self_del = con.execute('''
        SELECT sub, month, count(*) AS n_comments_sampled,
               sum(CASE WHEN meta_removal_type = 'deleted' THEN 1 ELSE 0 END) AS n_self_deleted
        FROM commenters_dedup GROUP BY 1, 2
    ''').fetchdf()
    comment_self_del['comment_self_del_rate'] = (
        100 * comment_self_del['n_self_deleted'] / comment_self_del['n_comments_sampled'])
    monthly = monthly.merge(comment_self_del[['sub', 'month', 'comment_self_del_rate']],
                             on=['sub', 'month'], how='left')

    # Subreddit subscriber growth -- a second sub-level signal, checked 2026-08-23 the same way:
    # weaker than comment self-deletion (rho=0.18 vs combined risk, vs 0.32) but real and
    # consistent, not noise. Interesting shape: it shows up on the commenter side (rho=0.18) not
    # the poster side (rho=0.03) -- fast subscriber growth associates with more risky commenting
    # activity, not a different mix of who gets to the top of the leaderboard. Raw subscriber
    # COUNT barely matters (rho=0.09) -- it's the rate of change that carries signal, not size.
    subscribers = con.execute('''
        SELECT sub, month, median(subreddit_subscribers) AS subscribers
        FROM posts GROUP BY 1, 2
    ''').fetchdf().sort_values(['sub', 'month'])
    subscribers['subscriber_growth_pct'] = subscribers.groupby('sub')['subscribers'].pct_change() * 100
    monthly = monthly.merge(subscribers[['sub', 'month', 'subscribers', 'subscriber_growth_pct']],
                             on=['sub', 'month'], how='left')

    monthly.to_csv(OUT / 'subreddit_bot_prevalence_mom.csv', index=False)
    print(f'Subreddit-month prevalence (poster/commenter/combined split): {len(monthly)} rows through {max_month}.')
    return monthly


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--retrain', action='store_true',
                    help='Refit the model on the current label set and re-pin it. This restates '
                         'every published month, so bump MODEL_VERSION and treat the result as a '
                         'new vintage. Without this flag the pinned model is loaded and only '
                         'scoring runs -- the normal monthly path.')
    args = ap.parse_args()

    con = duckdb.connect(str(DB), read_only=True)
    af = load_features(con)
    if args.retrain:
        model, meta = train_final_model(af, get_feature_cols(af))
    else:
        model, meta = load_pinned_model()
    scores = score_population(af, model, meta)
    monthly = build_subreddit_prevalence(con, scores)

    import subprocess
    subprocess.run(['python3', str(ROOT / 'scripts/v3_stage9_generate_dashboard_data.py')], check=True)
    subprocess.run(['python3', str(ROOT / 'scripts/v3_build_whitepaper_html.py')], check=True)

    print('\nDone. Re-run this script monthly after the corpus is extended.')


if __name__ == '__main__':
    main()
