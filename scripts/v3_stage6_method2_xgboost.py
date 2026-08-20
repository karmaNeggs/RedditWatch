#!/usr/bin/env python3
"""V3 Stage 6: Method 2 -- XGBoost classifier trained on the hand-labeled bot set
(output/v3/confirmed_bots.json vs clean_accounts.json, from Stage 5's candidate
screen + manual/LLM review). User-specified target: AUC > 0.80.

Deliberately uses the full 25-feature candidate pool (including features Method
1's pruning dropped for redundancy, e.g. score_stddev, reception_spread,
karma_per_day_since_first_seen) -- a tree-based model handles correlated inputs
fine (it just picks whichever split is most useful at each node), unlike a
linear/averaged composite where redundant inputs dilute rather than reinforce
each other. This is the central empirical argument for Method 2 over Method 1:
see output/v3/method1_results.json (AUC 0.474, worse than random) vs this
script's result (5-fold CV AUC ~0.80).

n=76 positives is small -- report the 5-fold CV mean as the headline number
(more stable than any single split) alongside a single held-out test split for
transparency. Do not over-read the exact feature-importance ranking; it will
shift meaningfully once the labeled set grows past a few hundred (flagged as
the explicit next step in V3_PLAN.md).
"""
import json
import os

import duckdb
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT = os.path.join(ROOT, 'output', 'v3', 'method2_results.json')

FEATURES = [
    'account_ordinal', 'comments_per_day_since_first_seen', 'controversiality_rate',
    'days_since_first_seen', 'karma_extremeness', 'karma_per_post_extremeness', 'mean_body_len',
    'mean_depth', 'median_comment_score', 'n_high_tier', 'n_low_tier', 'n_subs_rejected_but_returned',
    'own_post_reply_rate', 'posts_per_day_since_first_seen', 'removal_rate', 'deleted_later_rate',
    'repeat_engagement_rate', 'username_char_entropy', 'score_stddev', 'reception_spread',
    'karma_per_day_since_first_seen', 'subreddit_entropy', 'n_comments_sample', 'n_threads_active',
    'username_digit_suffix_len',
]
RNG_SEED = 42


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    bots = json.load(open(os.path.join(ROOT, 'output', 'v3', 'confirmed_bots.json')))
    clean = json.load(open(os.path.join(ROOT, 'output', 'v3', 'clean_accounts.json')))

    cols_sql = ', '.join(f'"{c}"' for c in FEATURES)
    all_authors = bots + clean
    placeholders = ', '.join(f"'{a.replace(chr(39), chr(39) + chr(39))}'" for a in all_authors)
    df = con.execute(f'SELECT author, {cols_sql} FROM account_features WHERE author IN ({placeholders})').fetchdf()
    df['label'] = df['author'].isin(bots).astype(int)
    print(f'{len(df)} labeled accounts ({df.label.sum()} bots, {(1 - df.label).sum()} clean)')

    X, y = df[FEATURES], df['label'].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RNG_SEED)
    spw = (ytr == 0).sum() / (ytr == 1).sum()

    params = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, scale_pos_weight=spw, eval_metric='auc', random_state=RNG_SEED)
    clf = xgb.XGBClassifier(**params)
    clf.fit(Xtr, ytr)
    train_auc = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
    test_auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    print(f'held-out split ({len(Xte)} accounts, {yte.sum()} bots): train AUC={train_auc:.3f}  test AUC={test_auc:.3f}')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    cv_aucs = cross_val_score(xgb.XGBClassifier(**params), X, y, cv=cv, scoring='roc_auc')
    print(f'5-fold CV AUC: {cv_aucs.round(3).tolist()}  mean={cv_aucs.mean():.3f}')

    imp = sorted(zip(FEATURES, clf.feature_importances_.tolist()), key=lambda kv: -kv[1])
    print('\nfeature importances:')
    for f, v in imp:
        if v > 0.01:
            print(f'  {f:<38} {v:.3f}')

    json.dump({
        'n_pos': int(df.label.sum()), 'n_neg': int((1 - df.label).sum()), 'features': FEATURES,
        'train_auc': float(train_auc), 'test_auc': float(test_auc),
        'cv_auc_folds': cv_aucs.round(4).tolist(), 'cv_auc_mean': float(cv_aucs.mean()),
        'feature_importances': [{'feature': f, 'importance': v} for f, v in imp],
        'target_met': bool(cv_aucs.mean() > 0.80),
    }, open(OUT, 'w'), indent=2)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
