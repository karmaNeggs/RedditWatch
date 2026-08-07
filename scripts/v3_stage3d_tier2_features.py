#!/usr/bin/env python3
"""V3 Stage 3d: iteration 3 of the feature-engineering loop, on top of
Stage 3c (scripts/v3_stage3c_tier1_features.py, which added Tier-1 features --
a wash, small mixed-direction effects, confirmed genuine not noise after a
2026-08-07 variance audit). Builds the "Tier 2 -- worth trying" set from the
feature-potential deep dive (V3_PLAN.md Sec 10.4):

  1. Sentiment/toxicity -- SKIPPED, not built. Spot-checked VADER (nltk)
     against 25 random real comments from this corpus before building
     anything on top of it, per the deep-dive's own caveat. Confirmed
     degradation: the corpus (Indian subreddits) is heavily Hinglish/
     code-mixed ("Let see ki kitne der me hatega!", "Bro kitane ka pada ??",
     "Floor and room number bhi bata de") -- VADER's English-only lexicon
     returns compound=0.00 on these, indistinguishable from genuine
     neutrality. Also scores 0.00 trivially on the many URL-only/image-link
     "comments" in this corpus. Roughly a third of the spot-check sample hit
     one of these two zero-inflation-via-blindness modes, not real
     neutrality. Shipping this feature would silently encode "is this
     comment in English with no URL" as if it were sentiment. Skipped per
     the deep-dive's own instruction to skip rather than ship something
     broken -- not attempted with a caveat, actually not built.
  2. sub_month_spike_share -- account-level. Operationalizes Stage 2's S15
     finding (only 7/45 subs show genuine GMM/BIC 2-component baseline/spike
     monthly structure; spike-exposed accounts measured LOWER risk, 5.3% vs
     6.9%) as an actual column instead of leaving it as prose. Re-derives the
     exact method from scripts/v3_stage2_bivariate.py's s15_regime_detection
     (log1p(monthly comment volume) per sub, GMM k=1 vs k=2 by BIC, spike
     component must be <45% weight to count as a genuine minority regime,
     not a symmetric split) rather than reinventing it. Feature = share of an
     account's own tagged threads (in one of the regime-defined subs) that
     fall in a spike month; NaN for accounts with zero activity in any of the
     7 regime-defined subs (most accounts -- this is a thin, sub-specific
     signal, not a population-wide one).
  3. coappear_degree / coappear_hhi -- account-level, NOT pair-level.
     Deliberately sidesteps Stage 4's label-construction problem (which made
     that stage inconclusive) by building pure description, no same-operator
     label needed. Reuses the SAME corrected distinct-(author,post_id)-
     presence method Stage 4's audit established (v3_stage4_pair_model.py's
     docstring: the naive self-join double-counts when an account has >1
     comment on the same post -- 43,141 such (author,post_id) pairs existed
     in the corpus). degree = count of distinct other accounts this account
     ever shares a thread with; hhi = Herfindahl concentration of this
     account's co-appearance events across those partners (close to 1 =
     mostly the same few partners; close to 1/degree = diffuse, a different
     set each time).
  4. post_edit_rate -- BUILT, then HARD-EXCLUDED from the model feature set
     after a leakage check the first run of this script did NOT do (it only
     flagged the population-thinness caveat, not this). Direct check on
     posts_clean: edit_rate by meta_removal_type = None 6.2%,
     automod_filtered 47.3%, moderator 9.4%, reddit(admin) 60.2% -- a 7.6x-10x
     jump for the exact categories this stage's labels are built from. This
     is Sec 8 leakage item 1 ("removal-derived features cannot be used
     against removal targets") applied to an edit flag instead of a removal
     flag -- almost certainly users editing a post to try to fix/appeal
     whatever triggered removal, not an independent behavioral signal.
     Caught because automod_filtered's first run jumped to gated rung4=0.880
     (above the 0.65-0.80 Kumar ceiling, same red flag pattern as
     admin_removal's original 0.896) with post_edit_rate as the single
     dominant SHAP feature (0.797, more than double the next feature) --
     investigated rather than reported as a clean win. Excluded via
     TIER2_LEAKAGE_EXCLUDE below; kept in account_tier2_edit for the record,
     just not fed to the model.

**Leakage discipline.** sub_month_spike_share and coappear_degree/hhi are
NOT removal-derived (Sec 8 item 1 doesn't apply) and NOT computed from a
post's own comment-quality pool the way Stage 3b's post_context family was
(an account's OWN co-appearance-partner count isn't "contaminated" by
whether that account's content got removed) -- re-verified directly in this
run's own numbers (see results), not just assumed safe by category.
post_edit_rate rides the existing population-thinness caveat already
documented for the post-author-only channels, nothing new.

**Candidate pool discipline, per explicit instruction: do NOT hand-pick which
Tier 1 features to keep based on Stage 3c's one-run SHAP results.** base_cols
here = Stage 3c's full expanded set (Stage 3's 62 + post_context + ALL 6
Tier 1 features, including own_repeat_rate/url_rate which looked SHAP-inert
in that one run) -- inside-fold selection does the winnowing, not a manual
pre-filter, per Sec 7/Ambroise & McLachlan.

Reuses CHANNELS/make_xgb/select_features/fit_eval/kfold_auc/chrono_split/
shap_family_importance/load_base from v3_stage3_account_model.py,
POST_CTX_COLS/POST_CTX_GATE/load_expanded/hurdle_and_gate from
v3_stage3b_feature_iteration.py, and TIER1_COLS/hurdle_and_gate_tier1/
build_vs_mode/build_repeat_ratio_url/build_outsider_and_post_ratio from
v3_stage3c_tier1_features.py -- imported, not duplicated. Same scope choice
as Stage 3b/3c: rung1 + gated rung4 only, to fit one iteration cycle in
reasonable wall-clock time.
"""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_stage3_account_model import (  # noqa: E402
    CHANNELS, KUMAR_LOW, KUMAR_HIGH, VOLUME_GATE_THRESHOLD,
    LEAKAGE_EXCLUDE, RNG,
    make_xgb, select_features, fit_eval, kfold_auc, chrono_split,
    shap_family_importance, load_base,
)
from v3_stage3b_feature_iteration import (  # noqa: E402
    POST_CTX_COLS, POST_CTX_GATE, load_expanded, hurdle_and_gate,
)
from v3_stage3c_tier1_features import (  # noqa: E402
    TIER1_COLS, build_repeat_ratio_url, build_outsider_and_post_ratio,
    build_vs_mode, hurdle_and_gate_tier1, family_of as _stage3c_family_of,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3d_data.json')

T0 = time.time()

TIER2_COLS = ['sub_month_spike_share', 'coappear_degree', 'coappear_hhi', 'post_edit_rate']
# post_edit_rate hard-excluded from the model feature set -- Sec 8 leakage
# item 1, see module docstring. Kept in TIER2_COLS (for the correlation
# recheck / reporting) but stripped out of tier2_feats below before fitting.
TIER2_LEAKAGE_EXCLUDE = ['post_edit_rate', 'post_edit_rate_nonzero']
TIER1_FULL_ACTIVE = TIER1_COLS + ['has_own_repeat', 'has_posts', 'url_rate_nonzero', 'title_body_ratio_defined']


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def family_of(feature):
    if feature in TIER2_COLS:
        return 'tier2'
    return _stage3c_family_of(feature)


def build_sub_month_regime(con):
    """Re-derives v3_stage2_bivariate.py's s15_regime_detection exactly
    (GMM k=1 vs k=2 by BIC on log1p monthly comment volume per sub, spike
    component must be <45% weight), then rolls up to an account-level
    spike-exposure share instead of Stage 2's pooled group comparison."""
    log('re-deriving S15 regime (GMM/BIC per sub, same method as Stage 2)')
    df = con.execute("""
        SELECT sub, month, sum(n_comments_observed) AS c
        FROM posts_clean WHERE role = 'top'
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchdf()
    regime_rows = []
    n_two_comp = 0
    for sub, g in df.groupby('sub'):
        g = g.sort_values('month')
        y = np.log1p(g['c'].values.astype(float)).reshape(-1, 1)
        if len(y) < 12:
            continue
        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=3).fit(y)
        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(y)
        two_component = gmm2.bic(y) < gmm1.bic(y)
        if two_component:
            labels = gmm2.predict(y)
            means = gmm2.means_.flatten()
            spike_comp = int(np.argmax(means))
            weights = gmm2.weights_
            if weights[spike_comp] >= 0.45:
                two_component = False
        if two_component:
            n_two_comp += 1
            regime = np.where(labels == spike_comp, 'spike', 'baseline')
        else:
            regime = np.array(['baseline'] * len(g))
        for m, r in zip(g['month'], regime):
            regime_rows.append({'sub': sub, 'month': m, 'regime': r})
    log(f'  {n_two_comp} subs with genuine 2-component regime (Stage 2 found 7/45)')

    regime_df = pd.DataFrame(regime_rows)
    con.execute('CREATE OR REPLACE TEMP TABLE _regime_t2 AS SELECT * FROM regime_df')
    account_metric = con.execute("""
        WITH tagged AS (
            SELECT c.author, r.regime
            FROM commenters_clean c
            JOIN posts_clean p ON p.post_id = c.post_id AND p.role = 'top'
            JOIN _regime_t2 r ON r.sub = p.sub AND r.month = p.month
        )
        SELECT author,
               sum(CASE WHEN regime = 'spike' THEN 1.0 ELSE 0.0 END) / count(*) AS sub_month_spike_share,
               count(*) AS n_regime_tagged
        FROM tagged GROUP BY author
    """).fetchdf()
    return account_metric[['author', 'sub_month_spike_share', 'n_regime_tagged']]


def build_coappearance_degree_concentration(con):
    """Account-level degree/concentration over the co-appearance graph, NOT
    pair-level -- no same-operator label needed, pure description. Reuses
    the corrected distinct-(author,post_id)-presence method Stage 4's audit
    established (naive self-join double-counts when an account has >1
    comment on the same post)."""
    log('building coappear_degree / coappear_hhi (account-level graph description)')
    con.execute('CREATE OR REPLACE TEMP TABLE presence_t2 AS SELECT DISTINCT author, post_id FROM commenters_clean')
    con.execute("""CREATE OR REPLACE TEMP TABLE pairs_t2 AS
        SELECT a.author AS a1, b.author AS a2, count(*) AS n_coappear
        FROM presence_t2 a JOIN presence_t2 b ON a.post_id = b.post_id AND a.author < b.author
        GROUP BY 1, 2""")
    n_pairs = con.execute('SELECT count(*) FROM pairs_t2').fetchone()[0]
    log(f'  {n_pairs:,} distinct co-appearing pairs (corrected dedup method)')
    out = con.execute("""
        WITH both_dir AS (
            SELECT a1 AS author, a2 AS partner, n_coappear FROM pairs_t2
            UNION ALL
            SELECT a2 AS author, a1 AS partner, n_coappear FROM pairs_t2
        ), agg AS (
            SELECT author, count(*) AS coappear_degree,
                   sum(n_coappear) AS total_events,
                   sum(n_coappear * n_coappear) AS sumsq
            FROM both_dir GROUP BY author
        )
        SELECT author, coappear_degree,
               sumsq::DOUBLE / (total_events * total_events) AS coappear_hhi
        FROM agg
    """).fetchdf()
    return out


def build_post_edit_rate(con):
    log('building post_edit_rate')
    out = con.execute("""
        SELECT author, avg(CASE WHEN meta_is_edited THEN 1.0 ELSE 0.0 END) AS post_edit_rate,
               count(*) AS n_posts_for_edit
        FROM posts_clean GROUP BY author
    """).fetchdf()
    return out[['author', 'post_edit_rate']]


def hurdle_tier2(df):
    df['post_edit_rate_nonzero'] = df['post_edit_rate'].fillna(0) > 0
    return df


def run_channel_comparison(df, post_authors, ch_name, spec, pos_authors, base_cols):
    if spec['pop'] == 'post_authors':
        sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
    else:
        sub_df = df.reset_index(drop=True)
    y = sub_df['author'].isin(pos_authors).astype(int)
    n_pos = int(y.sum())

    base_feats = [c for c in base_cols if c in sub_df.columns]
    tier2_feats = [c for c in TIER2_COLS + ['post_edit_rate_nonzero'] if c not in TIER2_LEAKAGE_EXCLUDE]
    expanded_feats = base_feats + [c for c in tier2_feats if c in sub_df.columns]

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

    # leakage re-verification, not assumed: correlate the two new non-post
    # non-removal features against removal_rate directly on this population
    corr_note = {}
    if 'sub_month_spike_share' in sub_df.columns:
        m = sub_df['sub_month_spike_share'].notna()
        if m.sum() > 30:
            corr_note['spike_share_vs_removal_rate'] = float(
                np.corrcoef(sub_df.loc[m, 'sub_month_spike_share'], sub_df.loc[m, 'removal_rate'])[0, 1])
    if 'coappear_degree' in sub_df.columns:
        corr_note['coappear_degree_vs_removal_rate'] = float(
            np.corrcoef(sub_df['coappear_degree'].fillna(0), sub_df['removal_rate'])[0, 1])

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
    log('connecting (write, to persist tier2 tables)')
    con = duckdb.connect(DB_PATH, read_only=False)
    df, post_authors = load_expanded(con)  # account_features_model + account_post_context

    # rebuild tier1 too (not persisted as a merge-ready table by Stage 3c
    # beyond its own two tier1 tables -- reuse those directly instead of
    # recomputing, they're already in the DB)
    repeat_ratio_url = con.execute('SELECT * FROM account_tier1_repeat_url').fetchdf()
    outsider_post = con.execute('SELECT * FROM account_tier1_post_context').fetchdf()

    regime = build_sub_month_regime(con)
    coappear = build_coappearance_degree_concentration(con)
    edit_rate = build_post_edit_rate(con)

    con.register('t2_regime', regime)
    con.register('t2_coappear', coappear)
    con.register('t2_edit', edit_rate)
    con.execute('CREATE OR REPLACE TABLE account_tier2_regime AS SELECT * FROM t2_regime')
    con.execute('CREATE OR REPLACE TABLE account_tier2_coappear AS SELECT * FROM t2_coappear')
    con.execute('CREATE OR REPLACE TABLE account_tier2_edit AS SELECT * FROM t2_edit')
    log(f'persisted account_tier2_regime ({len(regime)}), account_tier2_coappear ({len(coappear)}), '
        f'account_tier2_edit ({len(edit_rate)})')

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

    log(f'loaded {len(df)} accounts; sub_month_spike_share non-null={df["sub_month_spike_share"].notna().sum()}, '
        f'coappear_degree median={df["coappear_degree"].median()}, '
        f'post_edit_rate non-null={df["post_edit_rate"].notna().sum()}')

    # base_cols = Stage 3c's FULL expanded set (base-through-Tier1), per
    # explicit instruction not to hand-pick based on one run's SHAP.
    base_cols = [c for c in df.columns if c not in set(LEAKAGE_EXCLUDE) | {'author', 'primary_sub'}
                 and c not in TIER2_COLS
                 and c not in ['post_edit_rate_nonzero', 'n_posts_ctx_tier1', 'post_ctx_gated', 'n_regime_tagged',
                                'n_posts_for_edit']]

    results = {'channels': {}, 'meta': {
        'n_accounts': len(df), 'n_post_authors': len(post_authors),
        'n_regime_defined': int(df['sub_month_spike_share'].notna().sum()),
        'n_coappear_degree_gt0': int((df['coappear_degree'].fillna(0) > 0).sum()),
        'n_post_edit_defined': int(df['post_edit_rate'].notna().sum()),
        'sentiment_skipped': True,
        'sentiment_skip_reason': ('VADER spot-checked against 25 random corpus comments before building; '
                                   'corpus is heavily Hinglish/code-mixed and URL-heavy, VADER returns 0.00 '
                                   '(indistinguishable from neutral) on both failure modes -- skipped, not '
                                   'shipped with a silent caveat'),
        'post_edit_rate_excluded': True,
        'post_edit_rate_exclusion_reason': ('leakage register item 1: edit_rate by meta_removal_type = '
                                             'None 6.2%, automod_filtered 47.3%, moderator 9.4%, reddit 60.2% '
                                             '-- caught after automod_filtered gated rung4 jumped to 0.880 '
                                             '(above Kumar ceiling) with post_edit_rate as dominant SHAP feature'),
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
