#!/usr/bin/env python3
"""V3 Stage 3b: one feature-engineering iteration cycle on top of Stage 3
(scripts/v3_stage3_account_model.py). Adds a `post_context` feature family --
account-level aggregates of the posts each account commented on (posts_clean
has ~33 computed post-level features, e.g. contested_share, bot_comment_rate,
comment_score_gini, that were never joined up to account_features_model) --
then re-runs the existing 5-channel PU/XGBoost pipeline with the expanded
feature set and compares against the Stage 3 baseline.

**New leakage risk, same class as Stage 3's admin_removal bug, checked
explicitly, not assumed away:** several post_context features are themselves
computed FROM the pool of comments observed on a post (comment_score_gini,
bot_comment_rate, removed_comment_rate, tombstone_rate, reply_reciprocity,
pct_toplevel, mean_depth, n_unique_commenters, n_comments_observed) -- an
account's own comment on that post is one of the inputs. At median 14
comments/thread this is a much smaller share per post than Stage 3's
per-account leak (median 2 comments/account -> up to 100% self), but it's
non-zero and compounds across an account's posts. Treated with the same
volume-gating discipline Stage 3 adopted (not leave-one-out -- that was
tried in Stage 3 and produced a WORSE leak via XGBoost's NaN-as-label-proxy
handling): distinct-post count distribution here is median 1, mean 4.5, only
33,258/347,886 (9.6%) accounts have >=10 distinct posts. Chose
POST_CTX_GATE=5 (117,927 accounts, 34%) as a workable middle ground given
this stage's tighter population -- NOT validated by a sweep the way Stage
3's threshold=10 was (that took a dedicated audit fork); flagged as a
threshold chosen for tractability, not swept, and should be revisited if
this family shows up as a leading SHAP contributor.

Two other post_context features (title_len/selftext_len/domain/is_self/
over_18/subreddit_subscribers/num_crossposts/upvote_ratio -- properties of
the post submission itself, not aggregates over its comment pool) don't have
this specific leak mechanism (an account's own comment doesn't change a
post's title length or subscriber count) but are still included under the
same gate for simplicity and because "posts this account tends to show up
on" is itself a selection-effect signal worth the same population
restriction for comparability.

Reuses CHANNELS, make_xgb, select_features, fit_eval, kfold_auc,
chrono_split, shap_family_importance from v3_stage3_account_model.py rather
than re-implementing -- imported, not duplicated. Scoped down from the full
4-rung ladder to rung1 (random CV, reference) + rung4 (grouped+blocked+
purged, "the number that counts" per Sec 8) only, both ungated and
volume-gated (n_comments_sample>=10, Stage 3's own gate, applied on TOP of
the post_context-specific gate above where both apply) -- chosen to fit one
iteration cycle in reasonable wall-clock time; the full subreddit-blocked/
PU-learning diagnostics are not rerun here, they didn't change population
composition in a way expected to interact with this feature family.
"""
import json
import os
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
    shap_family_importance, family_of as _base_family_of, load_base,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3b_data.json')
OUT_HTML = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3b.html')

POST_CTX_GATE = 5  # n_distinct_posts_ctx >= this; see module docstring, not swept
T0 = time.time()

POST_CTX_COLS = [
    'pc_contested_share', 'pc_comment_score_gini', 'pc_reply_reciprocity',
    'pc_removed_comment_rate', 'pc_tombstone_rate', 'pc_bot_comment_rate',
    'pc_bot_comment_rate_max', 'pc_removed_comment_rate_max', 'pc_tombstone_rate_max',
    'pc_submitter_reply_rate', 'pc_upvote_ratio',
    'pc_pct_toplevel', 'pc_mean_depth', 'pc_num_crossposts',
    'pc_log_subscribers', 'pc_n_unique_commenters', 'pc_n_comments_observed',
    'pc_is_self_rate', 'pc_over18_rate',
]
# leakage register (V3_PLAN.md Sec 8) applied to the new family, same logic
# Stage 3 applied to removal_rate/score family: item 1 (removal-adjacent) and
# item 6 (score-adjacent) get a with/without ablation, not a silent choice.
POST_CTX_REMOVAL_ADJACENT = ['pc_removed_comment_rate', 'pc_tombstone_rate', 'pc_bot_comment_rate', 'pc_bot_comment_rate_max']
POST_CTX_SCORE_ADJACENT = ['pc_comment_score_gini', 'pc_upvote_ratio', 'pc_contested_share']
# _nonzero hurdle indicators newly derived FROM the post_context family --
# excluded from `base_feats` alongside POST_CTX_COLS itself. NOTE: this list
# must be enumerated explicitly, not via a blanket `col.endswith('_nonzero')`
# filter -- account_features_model already carries ~15 pre-existing
# `_nonzero`/`_ne_*` hurdle columns from Stage 3's original feature set
# (e.g. `high_tier_share_nonzero`, `mean_depth_nonzero`), and a blanket
# suffix filter silently strips those too, producing a `base_feats` that is
# NOT the same feature set Stage 3 originally reported numbers on. Found via
# a dedicated variance-audit 2026-08-07: this bug (plus 2 stray `pc_*_max`
# columns -- added to POST_CTX_COLS above -- that weren't in any exclusion
# list and leaked into "base" as if pre-existing) fully explains the
# previously-unexplained ~0.01-0.02 AUC drift between Stage 3's reported
# numbers and this script's own "base" recomputation of the same feature
# set. It was never sampling noise or model non-determinism -- both were
# checked directly (identical data + identical code reproduces bit-identical
# AUC across repeated runs, including fresh process invocations) and ruled
# out before this bug was found.
POST_CTX_NONZERO_COLS = [f'{c}_nonzero' for c in
    ['pc_removed_comment_rate', 'pc_tombstone_rate', 'pc_bot_comment_rate',
     'pc_submitter_reply_rate', 'pc_pct_toplevel', 'pc_num_crossposts']]


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def family_of(feature):
    if feature in POST_CTX_COLS:
        return 'post_context'
    return _base_family_of(feature)


def load_expanded(con):
    df, post_authors = load_base(con)
    pc = con.execute('SELECT * FROM account_post_context').fetchdf()
    df = df.merge(pc, on='author', how='left')
    return df, post_authors


def hurdle_and_gate(df):
    for c in ['pc_removed_comment_rate', 'pc_tombstone_rate', 'pc_bot_comment_rate',
              'pc_submitter_reply_rate', 'pc_pct_toplevel', 'pc_num_crossposts']:
        df[f'{c}_nonzero'] = (df[c].fillna(0) > 0)
    df['post_ctx_gated'] = (df['n_distinct_posts_ctx'].fillna(0) >= POST_CTX_GATE)
    return df


def run_channel_comparison(df, post_authors, ch_name, spec, pos_authors):
    if spec['pop'] == 'post_authors':
        sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
    else:
        sub_df = df.reset_index(drop=True)
    y = sub_df['author'].isin(pos_authors).astype(int)
    n_pos = int(y.sum())

    # base_feats = Stage 3's original feature set, unmodified -- only excludes
    # POST_CTX_COLS/POST_CTX_NONZERO_COLS (the new family) and the two
    # gating-only columns, NOT a blanket `_nonzero` suffix strip (see the
    # module-level comment above POST_CTX_NONZERO_COLS for why that was wrong).
    base_feats = [c for c in sub_df.columns if c not in set(LEAKAGE_EXCLUDE) | {'author', 'primary_sub'}
                  and c not in POST_CTX_COLS and c not in POST_CTX_NONZERO_COLS
                  and c != 'post_ctx_gated' and c != 'n_distinct_posts_ctx']
    expanded_feats = base_feats + POST_CTX_COLS + POST_CTX_NONZERO_COLS + ['n_distinct_posts_ctx']

    result = {'n_population': len(sub_df), 'n_positives': n_pos, 'positive_rate': n_pos / len(sub_df)}

    # ungated: rung1 + rung4, base vs expanded
    r1_base = kfold_auc(sub_df, base_feats, y)
    r1_exp = kfold_auc(sub_df, expanded_feats, y)
    tr4, te4 = chrono_split(sub_df, purge_frac=0.10, test_frac=0.30)
    r4_base = fit_eval(sub_df, base_feats, y, tr4, te4)
    r4_exp = fit_eval(sub_df, expanded_feats, y, tr4, te4)
    log(f'  {ch_name} ungated: rung1 base={r1_base["auc"] if r1_base else None:.3f} exp={r1_exp["auc"] if r1_exp else None:.3f}'
        f'  rung4 base={r4_base["auc"] if r4_base else None:.3f} exp={r4_exp["auc"] if r4_exp else None:.3f}')

    # volume-gated (Stage 3's own gate, n_comments_sample>=10) -- the number that counts
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
    log(f'  {ch_name} volume-gated(n>=10): rung1 base={r1v_base["auc"] if r1v_base else None} exp={r1v_exp["auc"] if r1v_exp else None}'
        f'  rung4 base={r4v_base["auc"] if r4v_base else None} exp={r4v_exp["auc"] if r4v_exp else None}')

    shap_res = None
    if r4v_exp is not None:
        try:
            shap_res = shap_family_importance(r4v_exp['model'], r4v_exp['X_te'], family_fn=family_of)
            shap_res['family'] = {fam: v for fam, v in
                                   sorted(shap_res['family'].items(), key=lambda kv: -kv[1])}
        except Exception as e:
            shap_res = {'error': str(e)}
    elif r4_exp is not None:
        try:
            shap_res = shap_family_importance(r4_exp['model'], r4_exp['X_te'], family_fn=family_of)
        except Exception as e:
            shap_res = {'error': str(e)}

    result.update({
        'rung1_base': r1_base['auc'] if r1_base else None,
        'rung1_expanded': r1_exp['auc'] if r1_exp else None,
        'rung4_base': r4_base['auc'] if r4_base else None,
        'rung4_expanded': r4_exp['auc'] if r4_exp else None,
        'n_features_base': len(base_feats), 'n_features_expanded': len(expanded_feats),
        'rung4_n_selected_base': r4_base['n_features_selected'] if r4_base else None,
        'rung4_n_selected_expanded': r4_exp['n_features_selected'] if r4_exp else None,
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
    log('connecting (write, to build account_post_context if missing)')
    con = duckdb.connect(DB_PATH, read_only=False)
    df, post_authors = load_expanded(con)
    labels = {}
    for name, spec in CHANNELS.items():
        labels[name] = set(con.execute(spec['sql_pos']).fetchdf()['author'])
    con.close()
    log(f'loaded {len(df)} accounts, {len(post_authors)} post authors, '
        f'{df["n_distinct_posts_ctx"].notna().sum()} have post_context')

    df['high_tier_share'] = df['high_tier_share'].fillna(0)
    df['medium_tier_share'] = df['medium_tier_share'].fillna(0)
    df['low_tier_share'] = df['low_tier_share'].fillna(0)
    df = hurdle_and_gate(df)

    results = {'channels': {}, 'meta': {'n_accounts': len(df), 'n_post_authors': len(post_authors),
                                         'post_ctx_gate': POST_CTX_GATE,
                                         'n_with_post_ctx_gate': int(df['post_ctx_gated'].sum())}}
    for ch_name, spec in CHANNELS.items():
        log(f'=== channel: {ch_name} ===')
        results['channels'][ch_name] = run_channel_comparison(df, post_authors, ch_name, spec, labels[ch_name])

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
