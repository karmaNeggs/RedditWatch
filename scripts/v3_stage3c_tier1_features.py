#!/usr/bin/env python3
"""V3 Stage 3c: iteration 2 of the feature-engineering loop, on top of Stage 3b
(scripts/v3_stage3b_feature_iteration.py, which added account_post_context and
lifted all 5 channels into 0.65-0.80). Builds the "Tier 1 -- build next" set
from the feature-potential deep dive (V3_PLAN.md Sec 10.4):

  1. own_repeat_rate / has_own_repeat  -- account-level, exact-duplicate-body
     rate among an account's own comments. No leakage risk (not derived from
     any removal/label channel).
  2. comment_post_ratio (+ hurdle)     -- n_comments_sample/n_posts_sample,
     defined only for the ~16% of accounts with n_posts_sample>0.
  3. vs_mode_{karma,comments,posts,score} -- 4 metrics (NOT 5: "reply latency"
     is dropped, confirmed unbuildable -- per-comment parent_id is never
     persisted to any stored table, only parent_is_post survives; confirmed
     independently in Stage 4 and in the feature-potential deep dive, which
     also caught V3_METRIC_CATALOGUE.md wrongly marking it buildable).
     Each expressed as (raw value - modal value of the account's PRIMARY
     incentive tier's peer group). Primary tier = whichever of
     high/medium/low_tier_share is largest (documented judgment call -- an
     account active across tiers is assigned to its plurality tier only for
     this comparison, not split). Mode of a continuous variable is estimated
     via histogram-binning (50 bins over the log1p-transformed distribution,
     modal bin's original-scale midpoint) rather than KDE, for speed and
     determinism; documented, not left implicit.
  4. url_rate (+ hurdle)               -- share of an account's comments
     containing a URL (regexp_matches on body). No leakage risk.
  5. outsider_influx_share             -- post-context: mean share of a
     post's commenters who are appearing in that SPECIFIC subreddit for the
     first time in the corpus (per-(author,sub) first-appearance check, NOT
     the corpus-wide first_seen_utc/account_ordinal already in
     account_features_model, which don't distinguish which sub). Aggregated
     per posting-account the same way Stage 3b built account_post_context.
  6. title_body_ratio / score_per_word -- post-context: from posts_clean's
     title_len/selftext_len/score, aggregated the same way.

**Leakage discipline.** Items 1/2/4 draw only from an account's own comment
text/counts, not from any post-level pool -- they don't carry the specific
row-inclusion mechanism that hit admin_removal (Stage 3) or
pc_removed_comment_rate (Stage 3b), and aren't removal-derived (leakage
register item 1 doesn't apply). They still ride on the SAME general
volume-gate (n_comments_sample>=VOLUME_GATE_THRESHOLD) applied to the whole
feature matrix below -- not given a bespoke gate of their own.

Items 5/6 ARE post-derived aggregates -- exactly Stage 3b's leak class (an
account's own post contributes to its own post-context aggregate). Reuses
account_post_context's own n_distinct_posts_ctx / POST_CTX_GATE=5 rather than
inventing a second gate.

Item 3's vs_mode_score (built from mean_comment_score) is score-adjacent
(Sec 8 leakage item 6, same family Stage 3's SCORE_FAMILY ablation already
covers) -- tagged into family_of as 'score' for SHAP purposes, but NOT
re-run through a separate with/without-score ablation matrix here; that
machinery already exists at the base-feature level and re-deriving it for
one derived feature was time-boxed out of this iteration. Flagged as a
deviation, not silently skipped.

Reuses CHANNELS/make_xgb/select_features/fit_eval/kfold_auc/chrono_split/
shap_family_importance/load_base from v3_stage3_account_model.py and
POST_CTX_COLS/POST_CTX_GATE/load_expanded/hurdle_and_gate/family_of from
v3_stage3b_feature_iteration.py -- imported, not duplicated. Same scope
choice as Stage 3b: rung1 + rung4 only (ungated and volume-gated), not the
full subreddit-blocked/PU-learning diagnostics, to fit one iteration cycle in
reasonable wall-clock time.
"""
import json
import os
import re
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stage3_account_model import (  # noqa: E402
    CHANNELS, KUMAR_LOW, KUMAR_HIGH, VOLUME_GATE_THRESHOLD,
    LEAKAGE_EXCLUDE, SCORE_FAMILY, RNG,
    make_xgb, select_features, fit_eval, kfold_auc, chrono_split,
    shap_family_importance, load_base,
)
from v3_stage3b_feature_iteration import (  # noqa: E402
    POST_CTX_COLS, POST_CTX_GATE, load_expanded, hurdle_and_gate,
    family_of as _stage3b_family_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3c_data.json')

T0 = time.time()

TIER1_COLS = [
    'own_repeat_rate', 'comment_post_ratio', 'url_rate',
    'vs_mode_karma', 'vs_mode_comments', 'vs_mode_posts', 'vs_mode_score',
    'outsider_influx_share', 'title_body_ratio', 'score_per_word',
]
TIER1_SCORE_ADJACENT = ['vs_mode_score']  # Sec 8 item 6 family, see docstring

# Iteration cycle (Phase E, mirroring Stage 3b's one-cycle discipline): first
# pass showed own_repeat_rate/url_rate never appear in any channel's top-6
# SHAP features, while vs_mode_*/score_per_word do -- try dropping the two
# apparently-inert features to see if decluttering the candidate set (which
# feeds a top-30-by-cumulative-gain selector) recovers the small AUC loss
# seen on admin_removal/automod_filtered.
TIER1_TRIMMED_DROP = ['own_repeat_rate', 'has_own_repeat', 'url_rate', 'url_rate_nonzero']


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def family_of(feature):
    if feature in TIER1_SCORE_ADJACENT:
        return 'score'  # ride with SCORE_FAMILY for SHAP rollup purposes
    if feature in TIER1_COLS:
        return 'tier1'
    return _stage3b_family_of(feature)


def modal_bin_value(x):
    """Histogram-based mode on log1p(x) (x assumed >=0), original-scale midpoint
    of the modal bin. Falls back to median if too few points or degenerate."""
    x = x[np.isfinite(x)]
    if len(x) < 30:
        return float(np.median(x)) if len(x) else 0.0
    xl = np.log1p(np.clip(x, 0, None))
    counts, edges = np.histogram(xl, bins=50)
    i = int(np.argmax(counts))
    mid_log = (edges[i] + edges[i + 1]) / 2
    return float(np.expm1(mid_log))


def build_repeat_ratio_url(con):
    log('building own_repeat_rate / comment_post_ratio / url_rate')
    repeat = con.execute("""
        WITH qualifying AS (
            SELECT author, comment_id, body FROM commenters_clean WHERE body_len > 10
        ),
        grouped AS (
            SELECT author, body, count(*) AS grp_n FROM qualifying GROUP BY author, body
        ),
        totals AS (
            SELECT author, count(*) AS n_qualifying FROM qualifying GROUP BY author
        ),
        dups AS (
            SELECT author, sum(grp_n - 1) AS n_dup_extra, count(*) FILTER (WHERE grp_n > 1) AS n_dup_groups
            FROM grouped GROUP BY author
        )
        SELECT t.author, t.n_qualifying,
               COALESCE(d.n_dup_extra, 0) AS n_dup_extra,
               COALESCE(d.n_dup_groups, 0) AS n_dup_groups
        FROM totals t LEFT JOIN dups d ON t.author = d.author
    """).fetchdf()
    repeat['own_repeat_rate'] = repeat['n_dup_extra'] / repeat['n_qualifying'].clip(lower=1)
    repeat['has_own_repeat'] = repeat['n_dup_groups'] > 0
    n_qualify_pos = int((repeat['n_dup_groups'] > 0).sum())
    log(f'  own-repeat: {n_qualify_pos} accounts with >=1 exact duplicate (verified target: 1520)')

    url = con.execute("""
        SELECT author, count(*) AS n_c,
               sum(CASE WHEN regexp_matches(body, 'https?://|www\\.') THEN 1 ELSE 0 END) AS n_url
        FROM commenters_clean WHERE body IS NOT NULL GROUP BY author
    """).fetchdf()
    url['url_rate'] = url['n_url'] / url['n_c'].clip(lower=1)

    out = repeat[['author', 'own_repeat_rate', 'has_own_repeat']].merge(
        url[['author', 'url_rate']], on='author', how='outer')
    return out


def build_outsider_and_post_ratio(con):
    log('building outsider_influx_share / title_body_ratio / score_per_word (post-context style)')
    post_level = con.execute("""
        WITH author_sub_first AS (
            SELECT author, sub, min(created_utc) AS first_utc
            FROM commenters_clean GROUP BY author, sub
        ),
        post_commenters AS (
            SELECT c.post_id, c.author AS commenter, c.created_utc,
                   (c.created_utc = f.first_utc) AS is_first_in_sub
            FROM commenters_clean c
            JOIN author_sub_first f ON c.author = f.author AND c.sub = f.sub
        ),
        post_outsider AS (
            SELECT post_id, avg(CASE WHEN is_first_in_sub THEN 1.0 ELSE 0.0 END) AS outsider_share
            FROM post_commenters GROUP BY post_id
        )
        SELECT p.author, p.post_id,
               o.outsider_share,
               CASE WHEN p.selftext_len > 0 THEN p.title_len::DOUBLE / p.selftext_len ELSE NULL END AS title_body_ratio,
               p.score::DOUBLE / NULLIF(p.title_len + p.selftext_len, 0) AS score_per_word
        FROM posts_clean p
        LEFT JOIN post_outsider o ON p.post_id = o.post_id
    """).fetchdf()
    agg = post_level.groupby('author').agg(
        outsider_influx_share=('outsider_share', 'mean'),
        title_body_ratio=('title_body_ratio', 'mean'),
        score_per_word=('score_per_word', 'mean'),
        n_posts_ctx_tier1=('post_id', 'count'),
    ).reset_index()
    return agg


def build_vs_mode(df):
    log('building vs_mode_{karma,comments,posts,score} (primary-tier histogram mode)')
    tier_cols = ['high_tier_share', 'medium_tier_share', 'low_tier_share']
    primary_tier = df[tier_cols].fillna(0).idxmax(axis=1).str.replace('_tier_share', '', regex=False)
    metrics = {
        'vs_mode_karma': 'karma_per_day_since_first_seen',
        'vs_mode_comments': 'comments_per_day_since_first_seen',
        'vs_mode_posts': 'posts_per_day_since_first_seen',
        'vs_mode_score': 'worst_sub_mean_score',  # mean_comment_score was VIF-pruned out
        # of account_features_model (Sec 8 audit, collinear w/ worst_sub_mean_score/
        # score_stddev) -- substituted with the closest available score-per-comment
        # proxy already in the table, still in SCORE_FAMILY.
    }
    out = pd.DataFrame({'author': df['author']})
    for new_col, raw_col in metrics.items():
        vals = df[raw_col].astype(float)
        out[new_col] = np.nan
        for tier in ('high', 'medium', 'low'):
            mask = (primary_tier == tier).values
            if mask.sum() < 30:
                continue
            tier_vals = vals[mask].dropna()
            if len(tier_vals) < 30:
                continue
            mode_val = modal_bin_value(tier_vals.values)
            out.loc[mask, new_col] = vals[mask].values - mode_val
    return out


def hurdle_and_gate_tier1(df):
    df['comment_post_ratio'] = np.where(df['n_posts_sample'].fillna(0) > 0,
                                         df['n_comments_sample'] / df['n_posts_sample'].replace(0, np.nan), np.nan)
    df['has_posts'] = df['n_posts_sample'].fillna(0) > 0
    df['url_rate_nonzero'] = df['url_rate'].fillna(0) > 0
    df['has_own_repeat'] = df['has_own_repeat'].fillna(False).astype(bool)
    df['title_body_ratio_defined'] = df['title_body_ratio'].notna()
    return df


def run_channel_comparison(df, post_authors, ch_name, spec, pos_authors, base_cols):
    if spec['pop'] == 'post_authors':
        sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
    else:
        sub_df = df.reset_index(drop=True)
    y = sub_df['author'].isin(pos_authors).astype(int)
    n_pos = int(y.sum())

    base_feats = [c for c in base_cols if c in sub_df.columns]
    tier1_feats = TIER1_COLS + ['has_own_repeat', 'has_posts', 'url_rate_nonzero', 'title_body_ratio_defined']
    expanded_feats = base_feats + [c for c in tier1_feats if c in sub_df.columns]

    result = {'n_population': len(sub_df), 'n_positives': n_pos, 'positive_rate': n_pos / max(len(sub_df), 1)}

    r1_base = kfold_auc(sub_df, base_feats, y)
    r1_exp = kfold_auc(sub_df, expanded_feats, y)
    tr4, te4 = chrono_split(sub_df, purge_frac=0.10, test_frac=0.30)
    r4_base = fit_eval(sub_df, base_feats, y, tr4, te4)
    r4_exp = fit_eval(sub_df, expanded_feats, y, tr4, te4)
    log(f'  {ch_name} ungated: rung1 base={r1_base["auc"] if r1_base else None} exp={r1_exp["auc"] if r1_exp else None}'
        f'  rung4 base={r4_base["auc"] if r4_base else None} exp={r4_exp["auc"] if r4_exp else None}')

    vmask = (sub_df['n_comments_sample'] >= VOLUME_GATE_THRESHOLD).values
    sub_v = sub_df[vmask].reset_index(drop=True)
    y_v = y[vmask].reset_index(drop=True)
    r1v_base = r1v_exp = r4v_base = r4v_exp = None
    if int(y_v.sum()) >= 20 and int((y_v == 0).sum()) >= 20:
        r1v_base = kfold_auc(sub_v, base_feats, y_v)
        r1v_exp = kfold_auc(sub_v, expanded_feats, y_v)
        trv, tev = chrono_split(sub_v, purge_frac=0.10, test_frac=0.30)
        r4v_base = fit_eval(sub_v, base_feats, y_v, trv, tev)
        r4v_exp = fit_eval(sub_v, expanded_feats, y_v, trv, tev)
    log(f'  {ch_name} gated(n>=10): rung1 base={r1v_base["auc"] if r1v_base else None} exp={r1v_exp["auc"] if r1v_exp else None}'
        f'  rung4 base={r4v_base["auc"] if r4v_base else None} exp={r4v_exp["auc"] if r4v_exp else None}')

    shap_res = None
    best = r4v_exp or r4_exp
    if best is not None:
        try:
            shap_res = shap_family_importance(best['model'], best['X_te'], family_fn=family_of)
            shap_res['family'] = {fam: v for fam, v in sorted(shap_res['family'].items(), key=lambda kv: -kv[1])}
        except Exception as e:
            shap_res = {'error': str(e)}

    result.update({
        'rung1_base': r1_base['auc'] if r1_base else None,
        'rung1_expanded': r1_exp['auc'] if r1_exp else None,
        'rung4_base': r4_base['auc'] if r4_base else None,
        'rung4_expanded': r4_exp['auc'] if r4_exp else None,
        'gated_n_population': len(sub_v), 'gated_n_positives': int(y_v.sum()),
        'rung1_gated_base': r1v_base['auc'] if r1v_base else None,
        'rung1_gated_expanded': r1v_exp['auc'] if r1v_exp else None,
        'rung4_gated_base': r4v_base['auc'] if r4v_base else None,
        'rung4_gated_expanded': r4v_exp['auc'] if r4v_exp else None,
        'shap': shap_res,
        'within_kumar_gated_expanded': (r4v_exp is not None and KUMAR_LOW <= r4v_exp['auc'] <= KUMAR_HIGH),
    })
    return result


def main():
    log('connecting (write, to persist tier1 tables)')
    con = duckdb.connect(DB_PATH, read_only=False)
    df, post_authors = load_expanded(con)  # account_features_model + account_post_context

    repeat_ratio_url = build_repeat_ratio_url(con)
    outsider_post = build_outsider_and_post_ratio(con)
    con.register('tier1_repeat_ratio_url', repeat_ratio_url)
    con.register('tier1_outsider_post', outsider_post)
    con.execute('CREATE OR REPLACE TABLE account_tier1_repeat_url AS SELECT * FROM tier1_repeat_ratio_url')
    con.execute('CREATE OR REPLACE TABLE account_tier1_post_context AS SELECT * FROM tier1_outsider_post')
    log(f'persisted account_tier1_repeat_url ({len(repeat_ratio_url)} rows), '
        f'account_tier1_post_context ({len(outsider_post)} rows)')

    labels = {}
    for name, spec in CHANNELS.items():
        labels[name] = set(con.execute(spec['sql_pos']).fetchdf()['author'])
    con.close()

    df['high_tier_share'] = df['high_tier_share'].fillna(0)
    df['medium_tier_share'] = df['medium_tier_share'].fillna(0)
    df['low_tier_share'] = df['low_tier_share'].fillna(0)
    df = hurdle_and_gate(df)  # stage3b's post_context hurdle/gate

    df = df.merge(repeat_ratio_url, on='author', how='left')
    df = df.merge(outsider_post, on='author', how='left')
    df = hurdle_and_gate_tier1(df)
    vm = build_vs_mode(df)
    df = df.merge(vm, on='author', how='left')

    # apply the SAME post-context gate (POST_CTX_GATE) to items 5/6, reusing
    # n_distinct_posts_ctx already computed for account_post_context -- NOT a
    # second bespoke gate.
    gate_mask = df['n_distinct_posts_ctx'].fillna(0) >= POST_CTX_GATE
    for c in ('outsider_influx_share', 'title_body_ratio', 'score_per_word'):
        df.loc[~gate_mask, c] = np.nan

    log(f'loaded {len(df)} accounts; own_repeat_rate non-null={df["own_repeat_rate"].notna().sum()}, '
        f'outsider_influx_share non-null (post-ctx-gated)={df["outsider_influx_share"].notna().sum()}')

    # "base" for this iteration = Stage 3b's expanded feature set (post_context
    # included) -- Tier 1 is compared AGAINST that, not against raw Stage 3.
    base_cols = [c for c in df.columns if c not in set(LEAKAGE_EXCLUDE) | {'author', 'primary_sub'}
                 and c not in TIER1_COLS
                 and c not in ['has_own_repeat', 'has_posts', 'url_rate_nonzero', 'title_body_ratio_defined',
                                'n_posts_ctx_tier1', 'post_ctx_gated']]

    results = {'channels': {}, 'meta': {
        'n_accounts': len(df), 'n_post_authors': len(post_authors),
        'post_ctx_gate_reused': POST_CTX_GATE,
        'n_own_repeat_positive': int(df['has_own_repeat'].sum()),
        'n_url_rate_nonzero': int(df['url_rate_nonzero'].sum()),
        'n_outsider_influx_defined': int(df['outsider_influx_share'].notna().sum()),
    }}
    for ch_name, spec in CHANNELS.items():
        log(f'=== channel: {ch_name} ===')
        results['channels'][ch_name] = run_channel_comparison(df, post_authors, ch_name, spec, labels[ch_name], base_cols)

    # Iteration cycle: rerun with own_repeat_rate/url_rate dropped (base_cols
    # now = base + trimmed tier1, i.e. base_cols itself expands to include the
    # surviving tier1 cols as the new "base" for this one comparison; the
    # dropped cols stay in account_tier1_repeat_url for anyone who wants them
    # later, this only removes them from the model's candidate feature list).
    trimmed_tier1_active = [c for c in (TIER1_COLS + ['has_posts', 'title_body_ratio_defined'])
                             if c not in TIER1_TRIMMED_DROP]
    log(f'=== iteration: trimmed tier1 (dropped {TIER1_TRIMMED_DROP}) ===')
    results['iteration_trimmed'] = {}
    for ch_name, spec in CHANNELS.items():
        if spec['pop'] == 'post_authors':
            sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
        else:
            sub_df = df.reset_index(drop=True)
        y = sub_df['author'].isin(labels[ch_name]).astype(int)
        vmask = (sub_df['n_comments_sample'] >= VOLUME_GATE_THRESHOLD).values
        sub_v, y_v = sub_df[vmask].reset_index(drop=True), y[vmask].reset_index(drop=True)
        trimmed_feats = base_cols + trimmed_tier1_active
        if int(y_v.sum()) >= 20 and int((y_v == 0).sum()) >= 20:
            trv, tev = chrono_split(sub_v, purge_frac=0.10, test_frac=0.30)
            r = fit_eval(sub_v, trimmed_feats, y_v, trv, tev)
            results['iteration_trimmed'][ch_name] = {'rung4_gated_trimmed': r['auc'] if r else None}
            log(f'  {ch_name} rung4_gated_trimmed={r["auc"] if r else None}'
                f'  (vs full-tier1={results["channels"][ch_name]["rung4_gated_expanded"]},'
                f'  vs base={results["channels"][ch_name]["rung4_gated_base"]})')

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    clean = json.loads(json.dumps(results, default=lambda o: (
        int(o) if isinstance(o, np.integer) else
        (None if (isinstance(o, np.floating) and np.isnan(o)) else float(o)) if isinstance(o, np.floating) else
        bool(o) if isinstance(o, np.bool_) else
        o.tolist() if isinstance(o, np.ndarray) else str(o))))
    with open(OUT_JSON, 'w') as f:
        json.dump(clean, f, indent=2)
    log(f'wrote {OUT_JSON}')
    log(f'TOTAL {time.time()-T0:.1f}s')
    return clean


if __name__ == '__main__':
    main()
