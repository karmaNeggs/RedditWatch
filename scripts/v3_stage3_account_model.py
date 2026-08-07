#!/usr/bin/env python3
"""V3 Stage 3 (V3_PLAN.md Sec 7, "Stage 3 -- account model"): XGBoost per
label channel, on account_features_model (data/v3/analysis/v3.duckdb).
Expect 0.65-0.80 (Kumar et al. 2017) and say so -- this stage exists to be
superseded by Stage 4 (pair model), not to be oversold as the deliverable.

**Channel set is a data-driven adaptation of Sec 6's literal 6-channel
table, not the literal 6 -- verified directly, don't re-derive:**
meta_removal_type has different granularity in the two source tables.
commenters: only {None, 'deleted', 'removed', 'removed by reddit'} -- comment
level "removed" does NOT distinguish automod from moderator. posts: full
granularity {None, 'automod_filtered', 'moderator', 'reddit', 'deleted',
'content_takedown', 'copyright_takedown', 'author'} but posts only cover the
~13% of accounts who authored a sampled post. meta_was_deleted_later is a
separate boolean signal from meta_removal_type=='deleted'. Five channels
built from what's actually resolvable:
  - admin_removal:      meta_removal_type in ('reddit','removed by reddit')  [full population]
  - self_deletion:      meta_removal_type=='deleted' OR meta_was_deleted_later [full population]
  - comment_removed_ambiguous: meta_removal_type=='removed' on commenters (automod/mod indistinguishable) [full population]
  - automod_filtered:   meta_removal_type=='automod_filtered' on posts        [post-author population only]
  - moderator_removed:  meta_removal_type=='moderator' on posts               [post-author population only]
Confirmed automation is a seed set, not a target (already excluded from
account_features_model's source population at Stage 0). Suspension is
skipped -- infeasible in bulk, already deprioritized alongside the on-hold
base36 calibration (see V3_PLAN.md Sec 10.4). All 5 channels checked to have
positive counts (3,204-47,730 for full-population channels; 4,395/4,970 out
of 45,861 post authors) well above V2's workable range (~28-334), so none
dropped.

**Leakage register (Sec 8), applied literally:**
1. Hard exclude removal_rate, deleted_later_rate, removal_rate_nonzero,
   deleted_later_rate_nonzero, and the 6 reporting-only columns
   (removal_rate_pctl, deleted_later_rate_pctl, thin_history_score,
   reception_spread_pctl, botmarker_composite, n_markers_available) from
   every channel's feature matrix -- every channel here is removal/deletion
   derived. NOTE: account_features_model's docstring claims
   karma_extremeness/karma_per_post_extremeness are "reporting-only,
   MODEL_READY=False", but the actual table has no MODEL_READY column and
   those two ARE present as regular (non-_pctl) columns, structurally
   identical to any other model-ready feature -- a discrepancy between
   scripts/v3_feature_sanitise.py's docstring and its own code, flagged here
   not fixed (out of this stage's scope). Treated as ordinary features,
   AND as part of the score-derived family below, since Sec 8 leakage item 6
   is about what they measure, not what a docstring calls them.
6. Score-derived family (mean_post_score, score_stddev, karma_extremeness,
   karma_per_post_extremeness, reception_spread, worst_sub_mean_score) run
   both included and excluded per channel, not silently one or the other.
7. Volume leakage: mandatory log1p(n_comments_sample)-only baseline below,
   plus a volume-decile-stratified permutation floor (not full-random --
   preserves whatever the real volume/label correlation is, isolating
   whether the model finds anything BEYOND volume).
5. **Revisited 2026-08-06 after admin_removal's 0.896 (well above the 0.65-0.80
   Kumar ceiling) triggered a dedicated leak audit -- this was NOT actually
   resolved by (1).** (1) only excludes the removal-RATE columns; it does
   nothing about the fact that every "behavioral" feature (footprint,
   reception, engagement, provenance-rate) is aggregated in
   v3_account_features.py over ALL of an account's sampled rows, INCLUDING
   the row(s) that define the label. For a thin-history account (median 2
   comments/account corpus-wide) this means the "behavioral" features are
   substantially the label-defining row's own metadata, not independent
   past behaviour.
   - **Confirmed via a volume-threshold sweep** (restrict population to
     n_comments_sample >= k, so the label row is at most 1/k of each
     account's aggregate): admin_removal rung4 AUC 0.896 (k=1) -> 0.800
     (k=2) -> 0.822 (k=5) -> **0.743 (k=10, lands inside 0.65-0.80)**.
     self_deletion 0.778->0.695 and comment_removed_ambiguous 0.805->0.641
     show the same monotonic decline (the latter crosses BELOW the Kumar
     floor once corrected -- a reversal worth flagging on its own).
     automod_filtered (0.653->0.725) and moderator_removed (0.637->0.687)
     do NOT decline with this gate -- expected, since their label is
     post-level and the gate operates on comment volume; a targeted
     ablation dropping their 4 post-derived features (n_posts_sample,
     mean_post_score, own_post_reply_rate, n_own_posts_with_comments) barely
     moved either (0.653->0.635, 0.637->0.622), so those two channels'
     original numbers look comparatively credible.
   - **A canary check** (fit on ONLY username_char_entropy,
     username_digit_suffix_len, username_is_default_pattern, account_ordinal
     -- features that structurally cannot depend on which rows got sampled)
     gave admin_removal AUC=0.489, i.e. noise -- confirming all of the
     model's real signal sits in the row-content-derived families, exactly
     where the leak would live.
   - **Attempted a full leave-one-out feature rebuild** (recompute every
     contaminated feature per-channel, excluding that channel's own
     label-defining rows from the aggregation) as the more principled fix.
     Hit and fixed a real bug in the exclusion SQL itself first (the same
     `NULL OR FALSE = NULL` three-valued-logic trap already documented for
     is_confirmed_automation_seed in v3_stage0_build.py -- `WHERE NOT
     (meta_removal_type = 'x')` silently drops every row where
     meta_removal_type IS NULL unless wrapped in
     `COALESCE(..., FALSE)`). Even after fixing that, **full LOO produced a
     WORSE leak** (AUC up to 0.98+): thin-history accounts often have ZERO
     rows left after excluding the label row, and "this feature is NaN"
     becomes a near-perfect proxy for the label via XGBoost's native
     missing-value handling. **Abandoned as the fix** -- kept as a
     documented negative result, not silently discarded.
   - **Ruled out one specific sub-hypothesis:** the _meta block's
     text-from-+16s/score-from-+36h snapshot merge (docstring, V3_PLAN.md
     Sec 3) raised the possibility that a removed item's `score` is a
     removal-timing artefact rather than organic reception. Checked
     directly (marginal and within-account, admin-removed vs. kept
     comments) -- admin-removed comments score *higher* on average (mean
     31.5 vs 19.2 within the same accounts), not lower/frozen. This
     mechanism is not what's driving the excess; a broader ablation dropping
     the whole score+reception-adjacent family (score_stddev,
     karma_extremeness, controversiality_rate, is_submitter_rate, mean_depth)
     barely moved admin_removal either (0.896->0.887) -- the signal is
     genuinely broad-based across families, not concentrated in reception.
   - **Also found, Sec 8 item 2:** admin_removal and self_deletion overlap
     39.9% (of admin_removal accounts) -- and self_deletion / comment_
     removed_ambiguous overlap 68.6% (of the latter). The "5 independent
     channels" framing needs this caveat; some of the apparent cross-channel
     structure in Sec 6's transfer matrix may be shared-account overlap, not
     independent transfer.
   - **Resolution adopted:** a `VOLUME_GATE_THRESHOLD`-gated re-evaluation
     (n_comments_sample >= 10) is now computed and reported alongside every
     rung for every channel, labeled explicitly as the corrected number --
     the original un-gated numbers are kept visible, not deleted, but should
     not be cited without this caveat attached.

**Four-rung ladder, adapted for the account-level unit (Sec 8) -- account
features are aggregated over the full ~24 month window, so there is no
literal "month" left at this grain:**
  1. Random 5-fold CV (reference only, optimistic)
  2. Grouped-by-account 5-fold CV -- trivially ~identical to rung 1 since
     each row already IS one account (group size 1); computed anyway, not
     skipped, precisely to show it adds nothing at this grain.
  3. "Month-blocked" substituted with account-chronology-blocked: sort by
     days_since_first_seen descending (train = accounts first seen longer
     ago, test = accounts first seen more recently), 70/30 split, no gap.
     Explicit substitution, not a literal reading of Sec 8.
  4. Grouped + blocked + purged: same chronological order, train = oldest
     60%, purge (dropped) = middle 10%, test = newest 30%.
  + subreddit-blocked (extra, beyond the 4 rungs): primary subreddit per
    account (argmax post+comment count), 36 subs train / 9 subs test,
    fixed seed -- direct confound-detection check given how much of this
    plan is about subreddit/volume confounds specifically.

Feature selection happens INSIDE each fold/split: a quick preliminary
XGBoost fit on the training partition only, keep features contributing to
the top 95% of cumulative gain (or top 30, whichever is smaller), refit on
selected features, evaluate on the (unseen) test partition. Never selected
on the full dataset before splitting (Sec 8's Ambroise & McLachlan citation).

PU learning (Sec 6): Elkan-Noto c-estimate per channel (train g(x) on 80% of
labeled positives vs. rest-as-unlabeled-negative, hold out 20% of positives,
c = mean(g(x)) on that held-out set). Sec 6 explicitly warns this went
degenerate on V2 data (c=0.054) -- computed and reported per channel here,
NOT trusted blindly if it looks degenerate again. Plain supervised
(unlabeled-as-negative) AUC reported alongside as the primary, less
model-dependent number; PU-corrected prevalence (mean(g(x))/c) reported as a
secondary diagnostic. Recall-at-fixed-FPR (1%, 5%) computed against the same
held-out labeled-positive split, using a same-size random unlabeled sample
to set the threshold -- less biased than AUC against the S label per Sec 6.

Interpretation: TreeSHAP, feature_perturbation="interventional",
family-level aggregation (families imported from v3_feature_sanitise.py so
this doesn't re-litigate the grouping).

Reads data/v3/analysis/v3.duckdb read-only throughout (no concurrent writer
expected at this point, but no reason to hold a write lock either -- this
script writes no tables back). Outputs a JSON summary + a static HTML report
(docs/v3-research/eda/stage3.html, linked from the EDA nav)."""
import json
import os
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_feature_sanitise import FAMILIES, PASSTHROUGH_BOOL, REPORTING_ONLY  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3_data.json')
OUT_HTML = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage3.html')

RNG = 42
LEAKAGE_EXCLUDE = ['removal_rate', 'deleted_later_rate', 'removal_rate_nonzero',
                   'deleted_later_rate_nonzero'] + REPORTING_ONLY
SCORE_FAMILY = ['mean_post_score', 'score_stddev', 'karma_extremeness',
                'karma_per_post_extremeness', 'reception_spread', 'worst_sub_mean_score']
KUMAR_LOW, KUMAR_HIGH = 0.65, 0.80
VOLUME_GATE_THRESHOLD = 10  # n_comments_sample >= this; see leakage-audit note in the module docstring

CHANNELS = {
    'admin_removal': dict(
        pop='all',
        sql_pos="""SELECT DISTINCT author FROM commenters_clean WHERE meta_removal_type='removed by reddit'
                   UNION SELECT DISTINCT author FROM posts_clean WHERE meta_removal_type='reddit'"""),
    'self_deletion': dict(
        pop='all',
        sql_pos="""SELECT DISTINCT author FROM commenters_clean WHERE meta_removal_type='deleted' OR meta_was_deleted_later
                   UNION SELECT DISTINCT author FROM posts_clean WHERE meta_removal_type='deleted' OR meta_was_deleted_later"""),
    'comment_removed_ambiguous': dict(
        pop='all',
        sql_pos="SELECT DISTINCT author FROM commenters_clean WHERE meta_removal_type='removed'"),
    'automod_filtered': dict(
        pop='post_authors',
        sql_pos="SELECT DISTINCT author FROM posts_clean WHERE meta_removal_type='automod_filtered'"),
    'moderator_removed': dict(
        pop='post_authors',
        sql_pos="SELECT DISTINCT author FROM posts_clean WHERE meta_removal_type='moderator'"),
}


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


def load_base(con):
    df = con.execute('SELECT * FROM account_features_model').fetchdf()
    post_authors = set(con.execute('SELECT DISTINCT author FROM posts_clean').fetchdf()['author'])
    # primary subreddit per account: argmax(comment count + post count) by sub
    sub_counts = con.execute("""
        SELECT author, sub, count(*) AS n FROM (
            SELECT author, sub FROM commenters_clean
            UNION ALL
            SELECT author, sub FROM posts_clean
        ) GROUP BY author, sub
    """).fetchdf()
    primary_sub = (sub_counts.sort_values('n', ascending=False)
                    .drop_duplicates('author', keep='first')
                    .set_index('author')['sub'])
    df['primary_sub'] = df['author'].map(primary_sub)
    return df, post_authors


def build_label_sets(con):
    labels = {}
    for name, spec in CHANNELS.items():
        pos = set(con.execute(spec['sql_pos']).fetchdf()['author'])
        labels[name] = pos
    return labels


def feature_cols(df, drop_score_family=False):
    exclude = set(LEAKAGE_EXCLUDE) | {'author', 'primary_sub'}
    if drop_score_family:
        exclude |= set(SCORE_FAMILY)
    cols = [c for c in df.columns if c not in exclude]
    return cols


def family_of(feature):
    for fam, cols in FAMILIES.items():
        if feature in cols:
            return fam
    if feature in PASSTHROUGH_BOOL:
        return 'passthrough_bool'
    if feature.endswith('_nonzero') or '_ne_' in feature:
        base = feature.split('_nonzero')[0].split('_ne_')[0]
        return family_of(base) or 'hurdle_indicator'
    return 'other'


def make_xgb(scale_pos_weight, n_estimators=200, max_depth=4):
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
        objective='binary:logistic', eval_metric='auc',
        scale_pos_weight=scale_pos_weight, random_state=RNG, n_jobs=-1,
        missing=np.nan,
    )


def select_features(X_train, y_train, all_features, max_features=30, cum_gain_target=0.95):
    if y_train.sum() < 5 or (len(y_train) - y_train.sum()) < 5:
        return all_features
    spw = max((len(y_train) - y_train.sum()) / max(y_train.sum(), 1), 1.0)
    prelim = make_xgb(spw, n_estimators=100, max_depth=3)
    prelim.fit(X_train, y_train)
    gains = prelim.get_booster().get_score(importance_type='gain')
    if not gains:
        return all_features
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])
    total = sum(v for _, v in ranked)
    keep, running = [], 0.0
    for f, g in ranked:
        keep.append(f)
        running += g
        if running / total >= cum_gain_target or len(keep) >= max_features:
            break
    # xgboost's booster feature names are f0,f1,... when given a numpy array;
    # X_train here is a DataFrame so real column names come through directly
    keep = [f for f in keep if f in all_features]
    return keep if keep else all_features


def fit_eval(df, features, y, train_idx, test_idx):
    X = df[features]
    y = y.astype(int)
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    if y_tr.sum() < 5 or y_te.sum() < 2 or (y_te == 0).sum() < 2:
        return None
    sel = select_features(X_tr, y_tr, features)
    spw = max((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    model = make_xgb(spw)
    model.fit(X_tr[sel], y_tr)
    probs = model.predict_proba(X_te[sel])[:, 1]
    auc = roc_auc_score(y_te, probs)
    return dict(auc=auc, n_train=len(train_idx), n_test=len(test_idx),
                n_pos_train=int(y_tr.sum()), n_pos_test=int(y_te.sum()),
                n_features_selected=len(sel), model=model, selected=sel,
                X_te=X_te[sel], y_te=y_te, probs=probs)


def kfold_auc(df, features, y, groups=None, k=5):
    y_arr = y.values if hasattr(y, 'values') else y
    if groups is None:
        splitter = KFold(n_splits=k, shuffle=True, random_state=RNG)
        splits = splitter.split(df)
    else:
        splitter = GroupKFold(n_splits=k)
        splits = splitter.split(df, groups=groups)
    oof_probs = np.full(len(df), np.nan)
    fold_results = []
    for train_idx, test_idx in splits:
        r = fit_eval(df, features, y, train_idx, test_idx)
        if r is None:
            continue
        oof_probs[test_idx] = r['probs']
        fold_results.append(r['auc'])
    mask = ~np.isnan(oof_probs)
    if mask.sum() == 0 or y_arr[mask].sum() < 2:
        return None
    overall_auc = roc_auc_score(y_arr[mask], oof_probs[mask])
    return dict(auc=overall_auc, fold_aucs=fold_results, n_folds=len(fold_results))


def chrono_split(df, purge_frac=0.0, test_frac=0.30):
    # df is always passed in with a fresh 0..n-1 RangeIndex, so the sorted
    # index values ARE the positions directly -- no need for get_loc.
    order = df['days_since_first_seen'].sort_values(ascending=False).index.to_numpy()  # oldest-first-seen first
    n = len(order)
    n_test = int(n * test_frac)
    n_purge = int(n * purge_frac)
    n_train = n - n_test - n_purge
    train_idx = order[:n_train]
    test_idx = order[n_train + n_purge:]
    return train_idx, test_idx


def subreddit_split(df, seed=RNG, test_frac=0.20):
    subs = sorted(df['primary_sub'].dropna().unique())
    rng = np.random.RandomState(seed)
    subs_shuf = list(subs)
    rng.shuffle(subs_shuf)
    n_test_subs = max(1, int(len(subs_shuf) * test_frac))
    test_subs = set(subs_shuf[:n_test_subs])
    is_test = df['primary_sub'].isin(test_subs).values
    train_idx = np.where(~is_test)[0]
    test_idx = np.where(is_test)[0]
    return train_idx, test_idx, sorted(test_subs)


def volume_only_baseline(df, y, train_idx, test_idx):
    X = np.log1p(df['n_comments_sample'].values).reshape(-1, 1)
    y_arr = y.values.astype(int)
    y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
    if y_tr.sum() < 2 or y_te.sum() < 2:
        return None
    spw = max((len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    model = make_xgb(spw, n_estimators=50, max_depth=1)
    model.fit(X[train_idx], y_tr)
    probs = model.predict_proba(X[test_idx])[:, 1]
    return roc_auc_score(y_te, probs)


def permutation_floor(df, y, features, train_idx, test_idx, n_deciles=10, seed=RNG):
    rng = np.random.RandomState(seed)
    vol = df['n_comments_sample'].values
    deciles = pd.qcut(vol, n_deciles, labels=False, duplicates='drop')
    y_arr = y.values.copy().astype(int)
    y_perm = y_arr.copy()
    for d in np.unique(deciles):
        idx = np.where(deciles == d)[0]
        y_perm[idx] = rng.permutation(y_arr[idx])
    y_perm_s = pd.Series(y_perm, index=df.index)
    r = fit_eval(df, features, y_perm_s, train_idx, test_idx)
    return r['auc'] if r else None


def elkan_noto(df, features, positive_authors, seed=RNG):
    y = df['author'].isin(positive_authors).astype(int).values
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) < 20:
        return dict(c=None, note='too few positives for held-out c estimate')
    rng = np.random.RandomState(seed)
    rng.shuffle(pos_idx)
    n_hold = max(int(len(pos_idx) * 0.2), 5)
    hold_pos_idx = pos_idx[:n_hold]
    train_pos_idx = pos_idx[n_hold:]
    neg_idx = np.where(y == 0)[0]
    train_idx = np.concatenate([train_pos_idx, neg_idx])
    X_all = df[features]
    y_train = np.zeros(len(df), dtype=int)
    y_train[train_pos_idx] = 1
    spw = max((len(neg_idx)) / max(len(train_pos_idx), 1), 1.0)
    g = make_xgb(spw, n_estimators=150, max_depth=4)
    g.fit(X_all.iloc[train_idx], y_train[train_idx])
    g_all = g.predict_proba(X_all)[:, 1]
    c = float(np.mean(g_all[hold_pos_idx]))
    prevalence_naive = float(np.mean(g_all))
    prevalence_pu = min(prevalence_naive / c, 1.0) if c > 1e-6 else None
    # recall at fixed FPR, threshold set on a random unlabeled sample
    unlabeled_sample_idx = rng.choice(neg_idx, size=min(len(neg_idx), 20000), replace=False)
    neg_scores = np.sort(g_all[unlabeled_sample_idx])[::-1]
    pos_scores = g_all[hold_pos_idx]
    recalls = {}
    for fpr_target in (0.01, 0.05):
        k = max(int(len(neg_scores) * fpr_target), 1)
        thresh = neg_scores[k - 1]
        recalls[fpr_target] = float(np.mean(pos_scores >= thresh))
    return dict(c=c, prevalence_naive=prevalence_naive, prevalence_pu_corrected=prevalence_pu,
                recall_at_fpr=recalls, n_holdout_pos=int(len(hold_pos_idx)),
                degenerate=bool(c is not None and c < 0.1))


def construct_validity(cv_frames):
    out = {}
    for ch, cv_df in cv_frames.items():
        s = cv_df['prob']
        tier_score = {}
        for tier in ('high', 'medium', 'low'):
            w = cv_df[f'{tier}_tier_share'].fillna(0)
            mask = w > 0
            if mask.sum() < 10:
                tier_score[tier] = None
                continue
            tier_score[tier] = float(np.average(s[mask], weights=w[mask]))
        out[ch] = tier_score
    return out


def shap_family_importance(model, X, max_rows=5000, seed=RNG, family_fn=None):
    # family_fn: pass a caller-local family_of override (e.g. stage3b/3c's
    # extended version) -- without this, `family_of(f)` below resolves via
    # THIS module's globals regardless of which script imported and called
    # this function, silently bucketing any caller-added feature family
    # under "other". Found in Stage 3b (post_context family mislabeled),
    # fixed here rather than left for a third repeat.
    import shap
    if family_fn is None:
        family_fn = family_of
    X = X.astype('float64')  # mixed bool/float columns produce an object-dtype
    # array inside shap's internal cast otherwise (booleans don't coerce cleanly)
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    explainer = shap.TreeExplainer(model, feature_perturbation='interventional',
                                    data=X.sample(min(500, len(X)), random_state=seed))
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[-1]
    mean_abs = np.abs(sv).mean(axis=0)
    fam_imp = {}
    for f, v in zip(X.columns, mean_abs):
        fam = family_fn(f)
        fam_imp[fam] = fam_imp.get(fam, 0.0) + float(v)
    top_feats = sorted(zip(X.columns, mean_abs), key=lambda kv: -kv[1])[:10]
    return dict(family=dict(sorted(fam_imp.items(), key=lambda kv: -kv[1])),
                top_features=[(f, float(v)) for f, v in top_feats])


T0 = time.time()


def main():
    log('connecting (read-only)')
    con = duckdb.connect(DB_PATH, read_only=True)
    df, post_authors = load_base(con)
    labels = build_label_sets(con)
    con.close()
    log(f'loaded {len(df)} accounts, {len(post_authors)} post authors')

    df['high_tier_share'] = df['high_tier_share'].fillna(0)
    df['medium_tier_share'] = df['medium_tier_share'].fillna(0)
    df['low_tier_share'] = df['low_tier_share'].fillna(0)

    results = {'channels': {}, 'meta': {'n_accounts': len(df), 'n_post_authors': len(post_authors),
                                         'kumar_range': [KUMAR_LOW, KUMAR_HIGH]}}
    probs_for_construct_validity = {}
    models_for_transfer = {}

    for ch_name, spec in CHANNELS.items():
        log(f'=== channel: {ch_name} ===')
        pos_authors = labels[ch_name]
        if spec['pop'] == 'post_authors':
            sub_df = df[df['author'].isin(post_authors)].reset_index(drop=True)
        else:
            sub_df = df.reset_index(drop=True)
        y = sub_df['author'].isin(pos_authors).astype(int)
        n_pos = int(y.sum())
        log(f'  population={len(sub_df)} positives={n_pos} ({100*n_pos/len(sub_df):.2f}%)')

        feats_with_score = feature_cols(sub_df, drop_score_family=False)
        feats_no_score = feature_cols(sub_df, drop_score_family=True)

        ch_result = {'n_population': len(sub_df), 'n_positives': n_pos,
                     'positive_rate': n_pos / len(sub_df), 'population': spec['pop']}

        # rung 1: random 5-fold
        r1 = kfold_auc(sub_df, feats_with_score, y, groups=None)
        log(f'  rung1 random CV AUC={r1["auc"] if r1 else None}')
        # rung 2: grouped by account (trivial)
        r2 = kfold_auc(sub_df, feats_with_score, y, groups=np.arange(len(sub_df)))
        log(f'  rung2 grouped-by-account CV AUC={r2["auc"] if r2 else None}')
        # rung 3: chronology-blocked, no purge
        tr3, te3 = chrono_split(sub_df, purge_frac=0.0, test_frac=0.30)
        r3 = fit_eval(sub_df, feats_with_score, y, tr3, te3)
        log(f'  rung3 chrono-blocked AUC={r3["auc"] if r3 else None}')
        # rung 4: chronology-blocked + purge
        tr4, te4 = chrono_split(sub_df, purge_frac=0.10, test_frac=0.30)
        r4 = fit_eval(sub_df, feats_with_score, y, tr4, te4)
        log(f'  rung4 chrono-blocked+purged AUC={r4["auc"] if r4 else None}')
        # subreddit-blocked
        tr_s, te_s, test_subs = subreddit_split(sub_df)
        rs = fit_eval(sub_df, feats_with_score, y, tr_s, te_s)
        log(f'  subreddit-blocked AUC={rs["auc"] if rs else None} (test subs={len(test_subs)})')

        # without score family, rung 4 only (the number that counts)
        r4_noscore = fit_eval(sub_df, feats_no_score, y, tr4, te4)
        log(f'  rung4 WITHOUT score family AUC={r4_noscore["auc"] if r4_noscore else None}')

        # volume-gated re-evaluation (leakage audit, see module docstring item 5):
        # target-row inclusion means every "behavioral" feature partly IS the
        # label-defining row for a thin-history account. Restricting to
        # n_comments_sample >= VOLUME_GATE_THRESHOLD dilutes that row to at most
        # 1/threshold of each account's aggregate -- this is the corrected number.
        gate_mask = (sub_df['n_comments_sample'] >= VOLUME_GATE_THRESHOLD).values
        sub_gated = sub_df[gate_mask].reset_index(drop=True)
        y_gated = y[gate_mask].reset_index(drop=True)
        r1_gated = r4_gated = None
        if int(y_gated.sum()) >= 20 and int((y_gated == 0).sum()) >= 20:
            r1_gated = kfold_auc(sub_gated, feats_with_score, y_gated, groups=None)
            tr_g, te_g = chrono_split(sub_gated, purge_frac=0.10, test_frac=0.30)
            r4_gated = fit_eval(sub_gated, feats_with_score, y_gated, tr_g, te_g)
        log(f'  volume-gated (n_comments_sample>={VOLUME_GATE_THRESHOLD}): '
            f'pop={len(sub_gated)} pos={int(y_gated.sum())} '
            f'rung1={r1_gated["auc"] if r1_gated else None} rung4={r4_gated["auc"] if r4_gated else None}')

        # mandatory baselines
        vol_auc = volume_only_baseline(sub_df, y, tr4, te4)
        perm_auc = permutation_floor(sub_df, y, feats_with_score, tr4, te4)
        log(f'  volume-only baseline AUC={vol_auc}  permutation floor AUC={perm_auc}')

        # PU
        pu = elkan_noto(sub_df, feats_with_score, pos_authors)
        log(f'  PU c={pu.get("c")} degenerate={pu.get("degenerate")}')

        # SHAP on rung4 model (the number that counts)
        shap_res = None
        if r4 is not None:
            try:
                shap_res = shap_family_importance(r4['model'], r4['X_te'])
            except Exception as e:
                shap_res = {'error': str(e)}
            cv_df = sub_df.iloc[te4][['author', 'high_tier_share', 'medium_tier_share', 'low_tier_share']].copy()
            cv_df['prob'] = r4['probs']
            probs_for_construct_validity[ch_name] = cv_df
            models_for_transfer[ch_name] = dict(model=r4['model'], features=r4['selected'],
                                                 test_subs=test_subs, sub_df=sub_df)

        ch_result.update({
            'rung1_random_cv': {'auc': r1['auc'], 'fold_aucs': r1['fold_aucs']} if r1 else None,
            'rung2_grouped_cv': {'auc': r2['auc'], 'fold_aucs': r2['fold_aucs']} if r2 else None,
            'rung3_chrono_blocked': {'auc': r3['auc'], 'n_test': r3['n_test'], 'n_pos_test': r3['n_pos_test']} if r3 else None,
            'rung4_grouped_blocked_purged': {'auc': r4['auc'], 'n_test': r4['n_test'], 'n_pos_test': r4['n_pos_test'],
                                              'n_features_selected': r4['n_features_selected'], 'selected_features': r4['selected']} if r4 else None,
            'subreddit_blocked': {'auc': rs['auc'], 'n_test': rs['n_test'], 'n_pos_test': rs['n_pos_test'], 'test_subs': test_subs} if rs else None,
            'rung4_without_score_family': {'auc': r4_noscore['auc']} if r4_noscore else None,
            'volume_gated': {
                'threshold': VOLUME_GATE_THRESHOLD,
                'n_population': len(sub_gated), 'n_positives': int(y_gated.sum()),
                'rung1_auc': r1_gated['auc'] if r1_gated else None,
                'rung4_auc': r4_gated['auc'] if r4_gated else None,
            },
            'volume_only_baseline_auc': vol_auc,
            'permutation_floor_auc': perm_auc,
            'pu_learning': {k: v for k, v in pu.items()},
            'shap': shap_res,
            'within_kumar_range': (r4 is not None and KUMAR_LOW <= r4['auc'] <= KUMAR_HIGH),
            'within_kumar_range_gated': (r4_gated is not None and KUMAR_LOW <= r4_gated['auc'] <= KUMAR_HIGH),
        })
        results['channels'][ch_name] = ch_result

    log('construct validity check')
    results['construct_validity'] = construct_validity(probs_for_construct_validity)

    log('cross-channel transfer matrix')
    # common evaluation universe = post-author population (required by 2 of 5 channels anyway)
    common_authors = post_authors
    transfer = {}
    for src, m in models_for_transfer.items():
        transfer[src] = {}
        for dst in models_for_transfer:
            dst_sub_df = df[df['author'].isin(common_authors)].reset_index(drop=True)
            y_dst = dst_sub_df['author'].isin(labels[dst]).astype(int)
            feats = m['features']
            X_dst = dst_sub_df.reindex(columns=feats, fill_value=np.nan)
            try:
                probs = m['model'].predict_proba(X_dst)[:, 1]
                auc = roc_auc_score(y_dst, probs) if y_dst.sum() >= 2 and (y_dst == 0).sum() >= 2 else None
            except Exception as e:
                auc = None
            transfer[src][dst] = auc
    results['cross_channel_transfer'] = transfer
    results['common_transfer_population_n'] = len(common_authors)

    log('serializing results')
    clean = json.loads(json.dumps(results, default=_json_default))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(clean, f, indent=2)
    log(f'wrote {OUT_JSON}')

    html = render_html(clean)
    with open(OUT_HTML, 'w') as f:
        f.write(html)
    log(f'wrote {OUT_HTML}')
    log(f'TOTAL {time.time()-T0:.1f}s')
    return clean


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _fmt(x, nd=3):
    if x is None:
        return '&mdash;'
    try:
        if isinstance(x, float) and np.isnan(x):
            return '&mdash;'
        return f'{x:.{nd}f}'
    except Exception:
        return str(x)


def render_html(r):
    ch_names = list(r['channels'].keys())
    kumar_lo, kumar_hi = r['meta']['kumar_range']

    rows = []
    for ch, c in r['channels'].items():
        r4 = c.get('rung4_grouped_blocked_purged') or {}
        r4ns = c.get('rung4_without_score_family') or {}
        vg = c.get('volume_gated') or {}
        in_range = c.get('within_kumar_range')
        gated_in_range = c.get('within_kumar_range_gated')
        badge = ('<span class="ok">within 0.65-0.80</span>' if in_range
                 else ('<span class="warn">outside 0.65-0.80</span>' if r4.get('auc') is not None else '&mdash;'))
        gated_badge = ('<span class="ok">within</span>' if gated_in_range
                        else ('<span class="warn">outside</span>' if vg.get('rung4_auc') is not None else '&mdash;'))
        rows.append(f"""
        <tr>
          <td><code>{ch}</code><div class="pop">{c['population']} population, n={c['n_population']:,}</div></td>
          <td>{c['n_positives']:,} ({100*c['positive_rate']:.2f}%)</td>
          <td>{_fmt((c.get('rung1_random_cv') or {}).get('auc'))}</td>
          <td>{_fmt((c.get('rung2_grouped_cv') or {}).get('auc'))}</td>
          <td>{_fmt((c.get('rung3_chrono_blocked') or {}).get('auc'))}</td>
          <td><strong>{_fmt(r4.get('auc'))}</strong> {badge}</td>
          <td>{_fmt((c.get('subreddit_blocked') or {}).get('auc'))}</td>
          <td>{_fmt(r4ns.get('auc'))}</td>
          <td>{_fmt(c.get('volume_only_baseline_auc'))}</td>
          <td>{_fmt(c.get('permutation_floor_auc'))}</td>
          <td><strong>{_fmt(vg.get('rung4_auc'))}</strong> {gated_badge}<div class="pop">n={vg.get('n_population', 0):,}, pos={vg.get('n_positives', 0):,}</div></td>
        </tr>""")

    pu_rows = []
    for ch, c in r['channels'].items():
        pu = c.get('pu_learning') or {}
        rec = pu.get('recall_at_fpr') or {}
        deg = ' <span class="warn">DEGENERATE</span>' if pu.get('degenerate') else ''
        pu_rows.append(f"""
        <tr>
          <td><code>{ch}</code></td>
          <td>{_fmt(pu.get('c'))}{deg}</td>
          <td>{_fmt(pu.get('prevalence_naive'))}</td>
          <td>{_fmt(pu.get('prevalence_pu_corrected'))}</td>
          <td>{_fmt(rec.get('0.01'))}</td>
          <td>{_fmt(rec.get('0.05'))}</td>
        </tr>""")

    cv_rows = []
    for ch, tiers in r.get('construct_validity', {}).items():
        ranked_ok = (tiers.get('high') is not None and tiers.get('medium') is not None and tiers.get('low') is not None
                     and tiers['high'] > tiers['medium'] > tiers['low'])
        badge = '<span class="ok">high&gt;med&gt;low holds</span>' if ranked_ok else '<span class="warn">ranking violated</span>'
        cv_rows.append(f"""
        <tr><td><code>{ch}</code></td><td>{_fmt(tiers.get('high'))}</td><td>{_fmt(tiers.get('medium'))}</td>
        <td>{_fmt(tiers.get('low'))}</td><td>{badge}</td></tr>""")

    transfer = r.get('cross_channel_transfer', {})
    transfer_header = ''.join(f'<th>{d}</th>' for d in ch_names)
    transfer_rows = []
    for src in ch_names:
        cells = ''.join(f'<td class="{"diag" if src==dst else ""}">{_fmt(transfer.get(src, {}).get(dst))}</td>' for dst in ch_names)
        transfer_rows.append(f'<tr><td><code>{src}</code></td>{cells}</tr>')

    shap_blocks = []
    for ch, c in r['channels'].items():
        shap_res = c.get('shap')
        if not shap_res or 'error' in shap_res:
            shap_blocks.append(f'<div class="shap-block"><h4>{ch}</h4><p class="muted">SHAP unavailable: {shap_res.get("error") if shap_res else "no rung-4 model"}</p></div>')
            continue
        fam_items = ''.join(f'<li><code>{fam}</code>: {imp:.4f}</li>' for fam, imp in list(shap_res.get('family', {}).items())[:8])
        top_items = ''.join(f'<li><code>{f}</code>: {v:.4f}</li>' for f, v in shap_res.get('top_features', [])[:10])
        shap_blocks.append(f"""
        <div class="shap-block">
          <h4>{ch}</h4>
          <div class="shap-cols">
            <div><strong>By family (mean |SHAP|)</strong><ul>{fam_items}</ul></div>
            <div><strong>Top individual features</strong><ul>{top_items}</ul></div>
          </div>
        </div>""")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>V3 Stage 3 — account model</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; line-height: 1.5; }}
  h1 {{ font-size: 1.5rem; }}
  h2 {{ font-size: 1.15rem; margin-top: 2.5rem; border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
  h4 {{ margin-bottom: .3rem; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .85rem; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: .4rem .5rem; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  code {{ background: #f0f0f0; padding: .1rem .3rem; border-radius: 3px; font-size: .85em; }}
  .ok {{ color: #0a7a2a; font-weight: 600; }}
  .warn {{ color: #b34700; font-weight: 600; }}
  .pop {{ font-size: .75rem; color: #666; font-weight: normal; }}
  .muted {{ color: #888; }}
  .diag {{ background: #fafaf0; }}
  .shap-block {{ margin-bottom: 1.5rem; }}
  .shap-cols {{ display: flex; gap: 2rem; }}
  .shap-cols > div {{ flex: 1; }}
  .shap-cols ul {{ font-size: .8rem; margin: .2rem 0; padding-left: 1.2rem; }}
  .note {{ background: #fff8e6; border: 1px solid #f0d878; border-radius: 4px; padding: .8rem 1rem; font-size: .9rem; }}
  a {{ color: #1a5fb4; }}
</style></head>
<body>
<h1>V3 Stage 3 — account model (XGBoost per label channel)</h1>
<p><a href="index.html">&larr; EDA</a> &middot; <a href="stage2.html">&larr; Stage 2</a></p>

<div class="note">
<strong>Expected ceiling:</strong> Kumar et al. (2017) found per-account sockpuppet detection tops out around
0.65&ndash;0.80 AUC on the same-style features &mdash; this stage is a checkpoint, not the deliverable. The plan's
0.90+ claim lives at Stage 4 (pair-level, not yet built). Channel set below is a data-driven adaptation of
V3_PLAN.md &sect;6's literal 6-channel table (automod vs. moderator is only resolvable for the ~13% of accounts
who authored a sampled post; suspension and confirmed-automation are out of scope here, see script docstring).
</div>

<div class="note">
<strong>2026-08-06 correction &mdash; read this before citing any rung-4 number below.</strong>
The un-gated <code>admin_removal</code> rung-4 AUC (0.896) sat well above the Kumar ceiling and triggered a
dedicated leak audit. <strong>Confirmed cause:</strong> every "behavioral" feature (footprint, reception,
engagement, provenance-rate) is aggregated over ALL of an account's sampled rows, including the row(s) that
define the label &mdash; for a thin-history account (corpus median 2 comments/account) that feature is
substantially just the labelled event's own metadata, not independent past behaviour. A volume-threshold sweep
confirms it: restricting to accounts with &ge;10 sampled comments dilutes the label row to at most 1/10th of the
aggregate and <code>admin_removal</code> drops to 0.743 (inside 0.65&ndash;0.80); <code>comment_removed_ambiguous</code>
drops from 0.805 to 0.641 (crosses <em>below</em> the floor); <code>automod_filtered</code> / <code>moderator_removed</code>
barely move (their label is post-level, not comment-level, so this particular mechanism doesn't apply to them).
A full leave-one-out feature rebuild was attempted as a more principled fix and <strong>made things worse</strong>
(AUC up to 0.98+) &mdash; thin-history accounts often have zero rows left after exclusion, and "this feature is
NaN" becomes a near-perfect label proxy via XGBoost's missing-value handling; abandoned, kept as a documented
negative result (full detail in the module docstring, item 5 of the leakage register). The un-gated numbers below
are kept visible for comparison but should not be cited without the gated column next to them &mdash; see the new
rightmost column in the table below, and the full docstring for the ruled-out sub-hypotheses (score-as-removal-
artefact: not supported; admin_removal&harr;self_deletion account overlap: 39.9%, a real caveat on "5 independent
channels").
</div>

<h2>Validation ladder (AUC) — {r['meta']['n_accounts']:,} accounts, {r['meta']['n_post_authors']:,} post authors</h2>
<table>
<tr><th>channel</th><th>positives</th><th>rung1<br>random CV</th><th>rung2<br>grouped CV</th>
<th>rung3<br>chrono-blocked</th><th>rung4<br>grouped+blocked+purged</th><th>subreddit-<br>blocked</th>
<th>rung4<br>no score family</th><th>volume-only<br>baseline</th><th>permutation<br>floor</th>
<th>rung4<br>volume-gated (n&ge;{VOLUME_GATE_THRESHOLD}) &mdash; corrected</th></tr>
{''.join(rows)}
</table>
<p class="muted">Rung 4 is "the number that counts" per &sect;8. If it doesn't clearly beat the volume-only
baseline, the model is rediscovering account volume, not detecting anything else. The permutation floor is
computed on labels shuffled <em>within</em> volume-decile strata (not fully random), so it already reflects
whatever the real volume/label correlation is &mdash; a rung-4 AUC close to this floor means no signal beyond volume.
The rightmost column is the corrected, defensible number per the note above &mdash; prefer it over the plain rung-4
column for any channel where the two disagree.</p>

<h2>PU learning diagnostics (Elkan&ndash;Noto)</h2>
<table>
<tr><th>channel</th><th>c (P(labeled|positive))</th><th>naive prevalence</th><th>PU-corrected prevalence</th>
<th>recall @ 1% FPR</th><th>recall @ 5% FPR</th></tr>
{''.join(pu_rows)}
</table>
<p class="muted">V3_PLAN.md &sect;6 flags that this method went degenerate (c=0.054) on V2 data &mdash; each c here
is reported, not assumed trustworthy. A c below ~0.1 is flagged DEGENERATE and its PU-corrected prevalence
should not be read as a real estimate.</p>

<h2>Construct validity (free, no labels) — rung-4 score by incentive tier</h2>
<table>
<tr><th>channel</th><th>high-tier accounts</th><th>medium-tier</th><th>low-tier</th><th>ranking</th></tr>
{''.join(cv_rows)}
</table>
<p class="muted">Per &sect;8: the score should rank high &gt; medium &gt; low incentive tier. If it doesn't, the
model is measuring something about subreddit/moderation intensity rather than account behavior.</p>

<h2>Cross-channel transfer matrix</h2>
<p class="muted">Row = trained on; column = evaluated on (held-out labels), on the common post-author population
(n={r.get('common_transfer_population_n', 0):,}). Diagonal = same-channel rung-4 AUC restricted to that
population (may differ slightly from the table above, which used each channel's own native population).
Per &sect;6: if off-diagonal AUC &asymp; 0.5, no single composite "bot score" is defensible.</p>
<table>
<tr><th>train \\ eval</th>{transfer_header}</tr>
{''.join(transfer_rows)}
</table>

<h2>TreeSHAP (rung-4 models, family-level)</h2>
{''.join(shap_blocks)}

</body></html>"""


if __name__ == '__main__':
    main()
