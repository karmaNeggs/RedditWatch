#!/usr/bin/env python3
"""V3 Stage 3e: iteration 4 of the feature-engineering loop, on top of
Stage 3d (scripts/v3_stage3d_tier2_features.py, a clean wash). Builds the
"Tier 3 -- interesting, low-confidence" set from the feature-potential deep
dive (V3_PLAN.md Sec 10.4). Two of the four proposed items were rejected
BEFORE building the full pipeline around them, based on a direct data check
-- that check is itself the useful output here, not a formality:

  1. domain_hhi -- BUILT (post-author population). Herfindahl over
     posts_clean.domain. Real caveat found on inspection: domain is
     overwhelmingly Reddit's own hosts (i.redd.it 52,184 / v.redd.it 25,506 /
     reddit.com 14,814) or `self.<subreddit>` for self-posts (self.* entries
     literally encode subreddit choice, not link diversity). So domain_hhi
     mostly measures "does this account stick to one subreddit + media type,"
     which overlaps with subreddit_entropy already in the model -- built and
     reported with this caveat attached, not silently presented as a novel
     signal. self-posts (is_self=True, 25,292/127,961) kept as their own
     domain value `self.<sub>` rather than excluded, so the HHI reflects the
     real self-vs-link-vs-media mix, not an arbitrary carve-out.

  2. within-thread activity Gini -- REJECTED before building the account
     rollup. Computed post-level Gini of comment-count-per-commenter (NOT
     score Gini, which already exists as posts_clean.comment_score_gini /
     account_post_context.pc_comment_score_gini) for the 120,882 posts with
     >=2 commenters. Correlation against contested_share (0.047) and
     n_unique_commenters (-0.050) is low -- NOT redundant by correlation.
     But the metric itself is close to degenerate: median 0.0, 75th
     percentile 0.058, max 0.17 (real Gini ranges 0-1) -- because at median
     14 commenters/thread and the bipartite first-10/top-10 sampling design,
     one account rarely comments twice in the same thread. This is
     substantively the same underlying event as the already-existing
     account-level `repeat_engagement_rate` feature ("does this account ever
     comment >1x in the same sampled thread"), just recomputed at the post
     level and diluted further by aggregation. Not built -- a near-constant
     feature adds selection-inside-fold cost (89th candidate column) for a
     signal already captured more directly elsewhere.

  3. flair_diversity -- REJECTED, no defensible encoding found. Checked the
     real distribution first, per the brief's instruction not to assume:
     1,406 distinct link_flair_text values on 120,532/127,961 posts (94.2%
     coverage, matches deep-dive's figure). NOT a small set of clean
     categories -- heavy per-subreddit string fragmentation of the same
     semantic idea ("Discussion" / "Discuss" / "Discussions" /
     "Discussion/Opinion" / "#Discussion \U0001F4AC" all separately present in the
     top 20 alone; same pattern for "Meme"/"MEME"). Raw-string Shannon
     entropy over this would mostly measure how many different subreddits'
     flair vocabularies an account has touched -- i.e. it would be a noisy
     re-derivation of subreddit_entropy/n_subs_active, both already in the
     model, not a new signal. Semantic clustering of 1,406 free-text labels
     into real categories is a much bigger lift than a "worth trying,
     low-confidence" tier item -- skipped, not forced.

  4. Two stated-mechanism interactions, exactly these two, per Sec 7's rule
     against blanket interaction sweeps (V2's old_x_msgs_per_day failed for
     lacking exactly this discipline):
       - karma_extremeness_x_reception_spread -- mechanism: does an account
         show BOTH a narrow engagement footprint AND polarized reception at
         once, a signature distinct from either alone.
       - bot_rate_x_coappear_degree -- mechanism: does an account with a
         high co-appearance degree ALSO concentrate that reach in
         already-bot-heavy threads (pc_bot_comment_rate), i.e. is its broad
         reach specifically into low-quality territory. No other interaction
         terms considered were added; one candidate (removal_rate x
         n_comments_sample) was discussed and rejected outright -- it would
         reintroduce a removal-derived feature that's hard-excluded from
         every channel's model input by Sec 8 item 1, an interaction does
         not launder that exclusion.

**Leakage discipline.** domain_hhi and the two interaction terms are content/
footprint features, not removal-derived on their face -- re-verified
directly (see leakage_recheck_correlations in the output), not assumed safe
by category, matching Stage 3d's post_edit_rate lesson: a feature that looks
population-thin or purely descriptive can still be a removal-derived leak in
disguise, so this stage checks correlation against removal_rate directly for
every new feature, not just the ones that look risky on their face.

**Candidate pool discipline, per explicit instruction: do NOT hand-pick from
prior runs' SHAP.** base_cols = Stage 3d's full expanded set (base + post
context + Tier 1 + Tier 2, including the two Tier-1 features that looked
SHAP-inert and the two Tier-2 features that looked SHAP-marginal) --
inside-fold selection does the winnowing.

Reuses CHANNELS/make_xgb/select_features/fit_eval/kfold_auc/chrono_split/
shap_family_importance/load_base/LEAKAGE_EXCLUDE/VOLUME_GATE_THRESHOLD/
KUMAR_LOW/KUMAR_HIGH from v3_stage3_account_model.py, POST_CTX_COLS/
POST_CTX_GATE/load_expanded/hurdle_and_gate from v3_stage3b_feature_iteration.py,
TIER1_COLS/build_repeat_ratio_url/build_outsider_and_post_ratio/build_vs_mode/
hurdle_and_gate_tier1 from v3_stage3c_tier1_features.py, TIER2_COLS/
build_sub_month_regime/build_coappearance_degree_concentration/
build_post_edit_rate/hurdle_tier2/TIER2_LEAKAGE_EXCLUDE from
v3_stage3d_tier2_features.py -- imported, not duplicated. Same scoped-down
ladder as Stage 3b/3c/3d: rung1 + gated rung4 only.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stage3_account_model import (  # noqa: E402
    CHANNELS, KUMAR_LOW, KUMAR_HIGH, VOLUME_GATE_THRESHOLD,
    LEAKAGE_EXCLUDE, RNG,
    make_xgb, select_features, fit_eval, kfold_auc, chrono_split,
    shap_family_importance,
)
from v3_stage3b_feature_iteration import (  # noqa: E402
    POST_CTX_COLS, POST_CTX_GATE, load_expanded, hurdle_and_gate,
)
from v3_stage3c_tier1_features import (  # noqa: E402
    TIER1_COLS, build_repeat_ratio_url, build_outsider_and_post_ratio,
    build_vs_mode,
)
from v3_stage3d_tier2_features import (  # noqa: E402
    TIER2_COLS, TIER2_LEAKAGE_EXCLUDE,
    build_sub_month_regime, build_coappearance_degree_concentration,
    build_post_edit_rate, hurdle_tier2,
    family_of as _stage3d_family_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3e_data.json')

T0 = time.time()

TIER3_COLS = ['domain_hhi', 'karma_extremeness_x_reception_spread', 'bot_rate_x_coappear_degree']
TIER1_FULL_ACTIVE = TIER1_COLS + ['has_own_repeat', 'has_posts', 'url_rate_nonzero', 'title_body_ratio_defined']
TIER2_FULL_ACTIVE = [c for c in TIER2_COLS + ['post_edit_rate_nonzero'] if c not in TIER2_LEAKAGE_EXCLUDE]


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def family_of(feature):
    if feature in TIER3_COLS:
        return 'tier3'
    return _stage3d_family_of(feature)


def build_domain_hhi(con):
    log('building domain_hhi (post-author accounts)')
    out = con.execute("""
        WITH counts AS (
            SELECT author, domain, count(*) AS n
            FROM posts_clean WHERE domain IS NOT NULL
            GROUP BY 1, 2
        ), tot AS (
            SELECT author, sum(n) AS total, sum(n*n) AS sumsq
            FROM counts GROUP BY author
        )
        SELECT author, sumsq::DOUBLE / (total * total) AS domain_hhi, total AS n_posts_with_domain
        FROM tot
    """).fetchdf()
    return out[['author', 'domain_hhi']]


def check_activity_gini_redundancy(con):
    """Rejected before building the account rollup -- see module docstring.
    Kept as a standalone diagnostic, not merged into the feature pipeline."""
    log('checking within-thread activity-Gini redundancy (diagnostic only, not a feature)')
    df = con.execute("""
        SELECT post_id, count(*) AS n_commenters, list(cast(n_c as double)) AS counts
        FROM (SELECT post_id, author, count(*) AS n_c FROM commenters_clean GROUP BY 1, 2) t
        GROUP BY post_id HAVING count(*) >= 2
    """).fetchdf()

    def gini(x):
        x = np.sort(np.array(x, dtype=float))
        n = len(x)
        cum = np.cumsum(x)
        if cum[-1] == 0:
            return 0.0
        return (n + 1 - 2 * (cum.sum() / cum[-1])) / n

    df['activity_gini'] = df['counts'].apply(gini)
    pc = con.execute('SELECT post_id, contested_share, n_unique_commenters FROM posts_clean').fetchdf()
    m = df.merge(pc, on='post_id')
    corr_contested = float(np.corrcoef(m['activity_gini'], m['contested_share'].fillna(0))[0, 1])
    corr_nuniq = float(np.corrcoef(m['activity_gini'], m['n_unique_commenters'].fillna(0))[0, 1])
    stats = {
        'n_posts': int(len(m)),
        'median': float(df['activity_gini'].median()),
        'p75': float(df['activity_gini'].quantile(0.75)),
        'max': float(df['activity_gini'].max()),
        'corr_vs_contested_share': corr_contested,
        'corr_vs_n_unique_commenters': corr_nuniq,
        'verdict': 'rejected: near-degenerate distribution (median 0), substantively redundant with existing repeat_engagement_rate; not correlation-redundant with contested_share/n_unique_commenters but not worth a 89th candidate column for this little variance',
    }
    log(f'  activity_gini median={stats["median"]:.4f} p75={stats["p75"]:.4f} '
        f'corr_contested={corr_contested:.3f} corr_nuniq={corr_nuniq:.3f} -- REJECTED')
    return stats


def check_flair_diversity_feasibility(con):
    """Rejected before building -- see module docstring."""
    log('checking flair_diversity encoding feasibility (diagnostic only, not a feature)')
    n_distinct = con.execute('SELECT count(DISTINCT link_flair_text) FROM posts_clean').fetchone()[0]
    coverage = con.execute('SELECT count(*), count(link_flair_text) FROM posts_clean').fetchone()
    top = con.execute("""SELECT link_flair_text, count(*) c FROM posts_clean
                          WHERE link_flair_text IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10""").fetchdf()
    stats = {
        'n_distinct_flairs': int(n_distinct),
        'coverage': f'{coverage[1]}/{coverage[0]}',
        'top10_sample': top['link_flair_text'].tolist(),
        'verdict': ('rejected: 1,406 distinct free-text values, heavy per-subreddit string '
                     'fragmentation of the same semantic categories (Discussion/Discuss/Discussions '
                     'all separately present in the top 10) -- no defensible entropy encoding without '
                     'semantic clustering, which is out of scope for this tier'),
    }
    log(f'  {n_distinct} distinct flair strings, top10={top["link_flair_text"].tolist()[:5]}... -- REJECTED')
    return stats


def run_channel_comparison(df, post_authors, ch_name, spec, pos_authors, base_cols):
    if spec['pop'] == 'post_authors':
        sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
    else:
        sub_df = df.reset_index(drop=True)
    y = sub_df['author'].isin(pos_authors).astype(int)
    n_pos = int(y.sum())

    base_feats = [c for c in base_cols if c in sub_df.columns]
    tier3_feats = [c for c in TIER3_COLS if c in sub_df.columns]
    expanded_feats = base_feats + tier3_feats

    result = {'n_population': len(sub_df), 'n_positives': n_pos, 'positive_rate': n_pos / max(len(sub_df), 1)}

    r1_base = kfold_auc(sub_df, base_feats, y)
    r1_exp = kfold_auc(sub_df, expanded_feats, y)
    log(f'  {ch_name} ungated rung1: base={r1_base["auc"] if r1_base else None} exp={r1_exp["auc"] if r1_exp else None}')

    vmask = (sub_df['n_comments_sample'] >= VOLUME_GATE_THRESHOLD).values
    sub_v = sub_df[vmask].reset_index(drop=True)
    y_v = y[vmask].reset_index(drop=True)
    r4v_base = r4v_exp = None
    if int(y_v.sum()) >= 20 and int((y_v == 0).sum()) >= 20:
        trv, tev = chrono_split(sub_v, purge_frac=0.10, test_frac=0.30)
        r4v_base = fit_eval(sub_v, base_feats, y_v, trv, tev)
        r4v_exp = fit_eval(sub_v, expanded_feats, y_v, trv, tev)
    log(f'  {ch_name} gated(n>=10) rung4: base={r4v_base["auc"] if r4v_base else None} exp={r4v_exp["auc"] if r4v_exp else None}')

    corr_note = {}
    for c in TIER3_COLS:
        if c in sub_df.columns:
            m = sub_df[c].notna()
            if m.sum() > 30:
                corr_note[f'{c}_vs_removal_rate'] = float(
                    np.corrcoef(sub_df.loc[m, c], sub_df.loc[m, 'removal_rate'])[0, 1])

    shap_res = None
    best = r4v_exp
    if best is not None:
        try:
            shap_res = shap_family_importance(best['model'], best['X_te'], family_fn=family_of)
            shap_res['family'] = {fam: v for fam, v in sorted(shap_res['family'].items(), key=lambda kv: -kv[1])}
        except Exception as e:
            shap_res = {'error': str(e)}

    result.update({
        'rung1_base': r1_base['auc'] if r1_base else None,
        'rung1_expanded': r1_exp['auc'] if r1_exp else None,
        'gated_n_population': len(sub_v), 'gated_n_positives': int(y_v.sum()),
        'rung4_gated_base': r4v_base['auc'] if r4v_base else None,
        'rung4_gated_expanded': r4v_exp['auc'] if r4v_exp else None,
        'leakage_recheck_correlations': corr_note,
        'shap': shap_res,
        'within_kumar_gated_expanded': (r4v_exp is not None and KUMAR_LOW <= r4v_exp['auc'] <= KUMAR_HIGH),
    })
    return result


def main():
    log('connecting (write, to persist tier3 table)')
    con = duckdb.connect(DB_PATH, read_only=False)
    df, post_authors = load_expanded(con)

    repeat_ratio_url = con.execute('SELECT * FROM account_tier1_repeat_url').fetchdf()
    outsider_post = con.execute('SELECT * FROM account_tier1_post_context').fetchdf()
    regime = con.execute('SELECT * FROM account_tier2_regime').fetchdf()
    coappear = con.execute('SELECT * FROM account_tier2_coappear').fetchdf()
    edit_rate = con.execute('SELECT * FROM account_tier2_edit').fetchdf()

    domain_hhi = build_domain_hhi(con)
    gini_check = check_activity_gini_redundancy(con)
    flair_check = check_flair_diversity_feasibility(con)

    con.register('t3_domain', domain_hhi)
    con.execute('CREATE OR REPLACE TABLE account_tier3_domain AS SELECT * FROM t3_domain')
    log(f'persisted account_tier3_domain ({len(domain_hhi)})')

    labels = {}
    for name, spec in CHANNELS.items():
        labels[name] = set(con.execute(spec['sql_pos']).fetchdf()['author'])
    con.close()

    df['high_tier_share'] = df['high_tier_share'].fillna(0)
    df['medium_tier_share'] = df['medium_tier_share'].fillna(0)
    df['low_tier_share'] = df['low_tier_share'].fillna(0)
    df = hurdle_and_gate(df)

    df = df.merge(repeat_ratio_url, on='author', how='left')
    df = df.merge(outsider_post, on='author', how='left')
    df = df.merge(regime, on='author', how='left')
    df = df.merge(coappear, on='author', how='left')
    df = df.merge(edit_rate, on='author', how='left')
    df = df.merge(domain_hhi, on='author', how='left')

    df['comment_post_ratio'] = np.where(df['n_posts_sample'].fillna(0) > 0,
                                         df['n_comments_sample'] / df['n_posts_sample'].replace(0, np.nan), np.nan)
    df['has_posts'] = df['n_posts_sample'].fillna(0) > 0
    df['url_rate_nonzero'] = df['url_rate'].fillna(0) > 0
    df['has_own_repeat'] = df['has_own_repeat'].fillna(False).astype(bool)
    df['title_body_ratio_defined'] = df['title_body_ratio'].notna()
    vm = build_vs_mode(df)
    df = df.merge(vm, on='author', how='left')
    df = hurdle_tier2(df)

    gate_mask = df['n_distinct_posts_ctx'].fillna(0) >= POST_CTX_GATE
    for c in ('outsider_influx_share', 'title_body_ratio', 'score_per_word'):
        df.loc[~gate_mask, c] = np.nan

    # the two stated-mechanism interactions
    df['karma_extremeness_x_reception_spread'] = df['karma_extremeness'] * df['reception_spread']
    df['bot_rate_x_coappear_degree'] = df['pc_bot_comment_rate'].fillna(0) * np.log1p(df['coappear_degree'].fillna(0))

    log(f'loaded {len(df)} accounts; domain_hhi non-null={df["domain_hhi"].notna().sum()}, '
        f'interactions built for all rows')

    # base_cols = Stage 3d's FULL expanded set (base through Tier2), per
    # explicit instruction not to hand-pick based on prior runs' SHAP.
    exclude_cols = (set(LEAKAGE_EXCLUDE) | {'author', 'primary_sub'} | set(TIER3_COLS)
                    | set(TIER2_LEAKAGE_EXCLUDE)  # post_edit_rate + _nonzero -- Sec 8 item 1 leak, see Stage 3d docstring
                    | {'n_posts_ctx_tier1', 'post_ctx_gated',
                       'n_regime_tagged', 'n_posts_for_edit'})
    base_cols = [c for c in df.columns if c not in exclude_cols]

    results = {'channels': {}, 'meta': {
        'n_accounts': len(df), 'n_post_authors': len(post_authors),
        'n_domain_hhi_defined': int(df['domain_hhi'].notna().sum()),
        'activity_gini_redundancy_check': gini_check,
        'flair_diversity_feasibility_check': flair_check,
        'interactions_built': ['karma_extremeness_x_reception_spread', 'bot_rate_x_coappear_degree'],
        'interaction_rejected': 'removal_rate x n_comments_sample -- would reintroduce a hard-excluded '
                                 'removal-derived feature via the back door; an interaction term does not '
                                 'launder Sec 8 item 1\'s exclusion',
    }}
    for ch_name, spec in CHANNELS.items():
        log(f'=== channel: {ch_name} ===')
        results['channels'][ch_name] = run_channel_comparison(df, post_authors, ch_name, spec, labels[ch_name], base_cols)

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
