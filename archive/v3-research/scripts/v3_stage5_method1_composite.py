#!/usr/bin/env python3
"""V3 Stage 5b: Method 1 -- hand-built composite score via bivariate correlation
pruning. User-specified methodology:
  1. Check every candidate account-level metric bivariate against every other.
  2. Remove cross-correlated / logically-dependent metrics, keeping one
     representative per correlated cluster (greedy: repeatedly drop whichever
     side of the most-correlated remaining pair has the higher mean |rho|
     against everything else still alive -- the findCorrelation algorithm).
  3. Average the survivors' percentile ranks (direction-flipped so higher =
     more "flagged") into one composite score.
  4. Check the composite's gradient against Target 1 (removal_rate,
     username_char_entropy -- external plausibility, though both are also
     component metrics here, so this is a partial-circularity check, not a
     fully independent one) and Target 2 (AUC / top-decile capture against
     the hand-labeled bot set from Stage 5).

Candidate pool: the same 31 features already screened for the EDA dashboard's
bivariate heatmap (docs/v3-research/eda/index.html) -- includes thin_history_score
and botmarker_composite deliberately, to let the pruning algorithm itself
discover their redundancy (it does -- see PRUNING_LOG in the output) rather than
hard-excluding them again.

Result (see output/v3/method1_results.json): AUC vs the 76 marked bots =
0.474 -- no better than random. An equal-weighted average of independently-
reasonable metrics does not automatically produce a working detector; the 1-2
metrics that actually separate real bots (Stage 6 finds score_stddev and
mean_body_len dominate) get diluted by the other ~15 that don't. Kept as the
honest baseline Method 1 result -- see the whitepaper for the full comparison
against Method 2 (XGBoost, AUC 0.80).
"""
import json
import os
import re

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
EDA_HTML = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'index.html')
OUT = os.path.join(ROOT, 'output', 'v3', 'method1_results.json')
PRUNE_OUT = os.path.join(ROOT, 'output', 'v3', 'method1_pruning.json')

RHO_THRESHOLD = 0.5

# direction: 'high' = higher value is the more-suspicious direction, 'low' = lower is.
# Ten of these are established elsewhere in this project's prior work
# (v3_boundary_discovery.py CANDIDATES, v3_botmarker_composite.py); the rest
# (days_since_first_seen, mean_body_len, mean_depth, median_comment_score,
# n_high_tier, n_low_tier, repeat_engagement_rate) are judgment calls with no
# prior session finding backing them -- flagged, not hidden.
DIRECTIONS = {
    'account_ordinal': 'high', 'comments_per_day_since_first_seen': 'high',
    'controversiality_rate': 'high', 'days_since_first_seen': 'low',
    'karma_extremeness': 'high', 'karma_per_post_extremeness': 'high',
    'mean_body_len': 'low', 'mean_depth': 'low', 'median_comment_score': 'low',
    'n_high_tier': 'low', 'n_low_tier': 'high', 'n_subs_rejected_but_returned': 'high',
    'own_post_reply_rate': 'low', 'posts_per_day_since_first_seen': 'high',
    'removal_rate': 'high', 'repeat_engagement_rate': 'high', 'username_char_entropy': 'low',
    'deleted_later_rate': 'high', 'botmarker_composite': 'high', 'thin_history_score': 'high',
    'reception_spread': 'high', 'score_stddev': 'high', 'karma_per_day_since_first_seen': 'high',
    'subreddit_entropy': 'low', 'n_comments_sample': 'high', 'n_threads_active': 'high',
    'n_posts_sample': 'high', 'n_subs_active': 'high', 'is_submitter_rate': 'low',
    'username_digit_suffix_len': 'high', 'mean_comment_score': 'low',
}


def prune(feats, mat):
    idx = {f: i for i, f in enumerate(feats)}
    alive = list(feats)
    log = []
    while True:
        ii = [idx[f] for f in alive]
        S = mat[np.ix_(ii, ii)]
        absS = np.abs(S.copy())
        np.fill_diagonal(absS, 0)
        if absS.max() <= RHO_THRESHOLD:
            break
        i, j = np.unravel_index(np.argmax(absS), absS.shape)
        fi, fj = alive[i], alive[j]
        n = len(alive)
        mean_i, mean_j = absS[i].sum() / (n - 1), absS[j].sum() / (n - 1)
        drop = fi if mean_i >= mean_j else fj
        log.append({'a': fi, 'b': fj, 'rho': float(S[i, j]), 'dropped': drop})
        alive.remove(drop)
    return alive, log


def main():
    html = open(EDA_HTML).read()
    m = re.search(r'<script id="eda-data" type="application/json">(.*?)</script>', html, re.S)
    d = json.loads(m.group(1))
    feats, mat = d['corr_features'], np.array(d['corr_matrix'])

    alive, log = prune(feats, mat)
    print(f'Step 1-2: bivariate pruning (|rho|>{RHO_THRESHOLD}), {len(feats)} -> {len(alive)} metrics')
    for row in log:
        print(f"  {row['rho']:+.3f}  {row['a']:<38} <-> {row['b']:<38}  drop: {row['dropped']}")
    json.dump({'candidates': feats, 'pruning_log': log, 'final_metrics': sorted(alive)}, open(PRUNE_OUT, 'w'), indent=2)

    con = duckdb.connect(DB_PATH, read_only=True)
    cols_sql = ', '.join(f'"{c}"' for c in alive)
    df = con.execute(f'SELECT author, {cols_sql}, removal_rate AS removal_rate_raw, '
                      f'username_char_entropy AS uce_raw FROM account_features').fetchdf()

    pctl = pd.DataFrame(index=df.index)
    for c in alive:
        x = df[c].astype(float)
        r = x.rank(pct=True, na_option='keep') * 100
        if DIRECTIONS[c] == 'low':
            r = 100 - r
        pctl[c] = r
    n_avail = pctl.notna().sum(axis=1)
    composite = pctl.mean(axis=1, skipna=True)
    composite[n_avail < len(alive) * 0.5] = np.nan
    df['composite'] = composite

    print(f'\nStep 3: composite built from {len(alive)} metrics -- {df.composite.notna().sum():,} scored accounts')

    sub = df.dropna(subset=['composite'])
    r1, _ = spearmanr(sub['composite'], sub['removal_rate_raw'])
    r2, _ = spearmanr(sub['composite'], sub['uce_raw'])
    print(f"Target 1: rho(composite, removal_rate)={r1:.3f}  rho(composite, username_char_entropy)={r2:.3f}")

    bots = set(json.load(open(os.path.join(ROOT, 'output', 'v3', 'confirmed_bots.json'))))
    clean = set(json.load(open(os.path.join(ROOT, 'output', 'v3', 'clean_accounts.json'))))
    lab = df[df['author'].isin(bots | clean)].dropna(subset=['composite']).copy()
    lab['label'] = lab['author'].isin(bots).astype(int)
    auc = roc_auc_score(lab['label'], lab['composite'])
    thresh = df['composite'].quantile(0.90)
    captured = len(set(df.loc[df['composite'] >= thresh, 'author']) & bots)
    print(f"Target 2: AUC vs {lab['label'].sum()} marked bots = {auc:.3f}  "
          f"| top-decile capture = {captured}/{len(bots)}")

    json.dump({
        'final_metrics': sorted(alive), 'direction': {k: DIRECTIONS[k] for k in alive},
        'target1_removal_rate_rho': float(r1), 'target1_username_entropy_rho': float(r2),
        'target2_auc': float(auc), 'target2_n_pos': int(lab['label'].sum()), 'target2_n_neg': int((1 - lab['label']).sum()),
        'target2_top_decile_capture': f'{captured}/{len(bots)}',
    }, open(OUT, 'w'), indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
