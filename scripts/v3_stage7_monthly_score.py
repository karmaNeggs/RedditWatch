#!/usr/bin/env python3
"""V3 Stage 7: monthly bot-score rollup for the dashboard.

Scores every account in account_features with the Stage 6 XGBoost model (Method
2 -- the one that actually separates the labeled bot set, unlike Method 1's
composite), then aggregates that score to (subreddit, month) using
commenters_dedup's month column, comment-count-weighted.

Important scope note: the account-level score is corpus-wide (fit on
account_features, which aggregates each account's whole 24-month history) --
there is no month-specific feature set. So "monthly" here means "this month's
comment-volume-weighted mix of accounts, each carrying their whole-corpus bot
score," not "this account's bot-likelihood in this specific month." A subreddit
whose most bot-scored accounts happened to be active in a given month will show
higher for that month even though the underlying per-account score didn't
change. Documented, not hidden -- see the whitepaper's limitations section.

Re-run this after any account_features rebuild or Stage 6 model change to keep
the dashboard in sync: `python3 scripts/v3_stage7_monthly_score.py`.
"""
import json
import os

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT = os.path.join(ROOT, 'docs', 'v3-research', 'bot-score-mom.json')
DASHBOARD_TEMPLATE = os.path.join(ROOT, 'scripts', 'v3_bot_dashboard_template.html')
DASHBOARD_OUT = os.path.join(ROOT, 'docs', 'v3-research', 'bot-score-dashboard.html')

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
MIN_ACCOUNTS_PER_SUB_MONTH = 15


def train_full_model(con, bots, clean):
    cols_sql = ', '.join(f'"{c}"' for c in FEATURES)
    all_authors = bots + clean
    placeholders = ', '.join(f"'{a.replace(chr(39), chr(39) + chr(39))}'" for a in all_authors)
    df = con.execute(f'SELECT author, {cols_sql} FROM account_features WHERE author IN ({placeholders})').fetchdf()
    df['label'] = df['author'].isin(bots).astype(int)
    X, y = df[FEATURES], df['label'].values
    spw = (y == 0).sum() / (y == 1).sum()
    clf = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric='auc', random_state=RNG_SEED)
    clf.fit(X, y)
    return clf


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    bots = json.load(open(os.path.join(ROOT, 'output', 'v3', 'confirmed_bots.json')))
    clean = json.load(open(os.path.join(ROOT, 'output', 'v3', 'clean_accounts.json')))
    print(f'training Method 2 model on full labeled set ({len(bots)} bots, {len(clean)} clean)...')
    clf = train_full_model(con, bots, clean)

    print('scoring full account population...')
    cols_sql = ', '.join(f'"{c}"' for c in FEATURES)
    pop = con.execute(f'SELECT author, {cols_sql} FROM account_features').fetchdf()
    pop['bot_score'] = clf.predict_proba(pop[FEATURES])[:, 1]

    print('aggregating to (sub, month), comment-count-weighted...')
    activity = con.execute('SELECT author, sub, month, count(*) AS n FROM commenters_dedup GROUP BY author, sub, month').fetchdf()
    merged = activity.merge(pop[['author', 'bot_score']], on='author', how='inner')

    def wmean(g):
        return pd.Series({'bot_score': np.average(g['bot_score'], weights=g['n']), 'n_accounts': g['author'].nunique(), 'n_comments': g['n'].sum()})

    rollup = merged.groupby(['sub', 'month']).apply(wmean, include_groups=False).reset_index()
    rollup = rollup[rollup['n_accounts'] >= MIN_ACCOUNTS_PER_SUB_MONTH]

    months = sorted(rollup['month'].unique())
    subs = sorted(rollup['sub'].unique())
    series = {}
    for sub in subs:
        sub_rows = rollup[rollup['sub'] == sub].set_index('month')
        series[sub] = {
            'months': months,
            'bot_score': [round(float(sub_rows.loc[m, 'bot_score']), 4) if m in sub_rows.index else None for m in months],
            'n_accounts': [int(sub_rows.loc[m, 'n_accounts']) if m in sub_rows.index else None for m in months],
        }

    out = {
        'meta': {
            'model': 'Method 2 (XGBoost)', 'trained_on': f'{len(bots)} confirmed bots, {len(clean)} clean accounts',
            'min_accounts_per_sub_month': MIN_ACCOUNTS_PER_SUB_MONTH, 'months': months,
        },
        'subreddits': series,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {OUT} ({len(subs)} subs x {len(months)} months)')

    tpl = open(DASHBOARD_TEMPLATE).read()
    with open(DASHBOARD_OUT, 'w') as f:
        f.write(tpl.replace('__MOM_JSON__', json.dumps(out)))
    print(f'wrote {DASHBOARD_OUT}')


if __name__ == '__main__':
    main()
