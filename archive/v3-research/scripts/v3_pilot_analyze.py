#!/usr/bin/env python3
"""V3 pilot analysis -- Stage 1 (univariate multimodality) + Stage 2 (bivariate
separation) on the pilot collection. This is the decision point: does a
separation boundary actually exist in the data, before committing to full
collection?"""
import json, math
import numpy as np
import pandas as pd

SP = '/private/tmp/claude-501/-Users-anupamvashist-Documents-Project-writeups-Analysis-Report--reddit-bot-analysis/e3d807eb-e38c-4468-a2f9-f863e737a2ad/scratchpad/'
OUT = SP + 'pilot/'

posts = pd.DataFrame(json.load(open(OUT + 'posts.json')))
commenters = pd.DataFrame(json.load(open(OUT + 'commenters.json')))
print(f'Loaded: {len(posts)} post rows, {len(commenters)} commenter rows\n')

# ---------------------------------------------------------------------------
# Build account-level table: one row per (sub, account), aggregating across
# all appearances as a first10/top10 commenter in the pilot window.
# ---------------------------------------------------------------------------
c = commenters[~commenters.author.isin([None, '[deleted]', 'AutoModerator'])].copy()

acc = c.groupby(['sub', 'author']).agg(
    n_appearances=('comment_id', 'count'),
    n_posts_seen_in=('post_id', 'nunique'),
    mean_score=('score', 'mean'),
    median_score=('score', 'median'),
    pct_toplevel=('parent_is_post', 'mean'),
    pct_first10=('commenter_tag', lambda s: (s == 'first10').mean()),
    pct_top10=('commenter_tag', lambda s: (s == 'top10').mean()),
    mean_body_len=('body_len', 'mean'),
    pct_top_role=('role', lambda s: (s == 'top').mean()),
    has_fullname=('author_fullname', lambda s: s.notna().any()),
).reset_index()

incentive = posts.drop_duplicates('sub')[['sub', 'incentive_tier']]
acc = acc.merge(incentive, on='sub', how='left')

print(f'Unique (sub, account) rows: {len(acc)}')
print(acc.groupby('sub').size())
print()

# ---------------------------------------------------------------------------
# STAGE 1 -- univariate: is any feature's distribution genuinely multimodal?
# Dip test (Hartigan) via a simple bootstrap-free approximation (no unidip/
# diptest package available) -- use a 2-component GMM vs 1-component BIC
# comparison as the primary test, which is well-supported in stdlib sklearn.
# ---------------------------------------------------------------------------
from sklearn.mixture import GaussianMixture

def bic_multimodality_test(x, name):
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 20:
        return None
    x = x.reshape(-1, 1)
    g1 = GaussianMixture(n_components=1, random_state=0).fit(x)
    g2 = GaussianMixture(n_components=2, random_state=0).fit(x)
    bic1, bic2 = g1.bic(x), g2.bic(x)
    better2 = bic2 < bic1
    # second component's mixture weight = label-free prevalence estimate
    w = sorted(g2.weights_)[0] if better2 else None
    means = sorted(g2.means_.flatten()) if better2 else None
    return {
        'feature': name, 'n': len(x), 'bic_1comp': round(bic1, 1),
        'bic_2comp': round(bic2, 1), 'delta_bic': round(bic1 - bic2, 1),
        'favors_2comp': better2,
        'minority_weight': round(w, 3) if w is not None else None,
        'component_means': [round(m, 2) for m in means] if means else None,
    }

print('=' * 78)
print('STAGE 1 -- univariate multimodality (1-comp vs 2-comp GMM, by BIC)')
print('=' * 78)
univariate_features = ['n_appearances', 'n_posts_seen_in', 'mean_score',
                        'pct_toplevel', 'pct_first10', 'mean_body_len']
stage1_results = []
for feat in univariate_features:
    r = bic_multimodality_test(acc[feat].values, feat)
    if r:
        stage1_results.append(r)
        flag = '  <-- FAVORS 2 COMPONENTS' if r['favors_2comp'] else ''
        print(f"  {feat:<18} n={r['n']:>5}  BIC(1)={r['bic_1comp']:>10}  "
              f"BIC(2)={r['bic_2comp']:>10}  dBIC={r['delta_bic']:>8}{flag}")
        if r['favors_2comp']:
            print(f"      minority component weight={r['minority_weight']}  "
                  f"means={r['component_means']}")

# per incentive tier
print('\n  -- same test, split by incentive tier --')
for tier in ['high', 'medium', 'low']:
    sub_acc = acc[acc.incentive_tier == tier]
    print(f'\n  [{tier}]  n_accounts={len(sub_acc)}')
    for feat in univariate_features:
        r = bic_multimodality_test(sub_acc[feat].values, feat)
        if r and r['favors_2comp']:
            print(f"    {feat:<18} FAVORS 2-COMP  minority_wt={r['minority_weight']}  means={r['component_means']}")

# ---------------------------------------------------------------------------
# STAGE 2 -- bivariate: does incentive tier separate account behaviour?
# No labelled seed set in the pilot (no self-declared-bot regex hits checked
# yet) -- so the primary contrast available is: does the account population
# look different across incentive tiers, and does role (top-post vs matched
# counter-sample) separate anything?
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('STAGE 2a -- does incentive tier separate account-level features?')
print('=' * 78)
from scipy import stats

for feat in univariate_features:
    groups = [acc[acc.incentive_tier == t][feat].dropna().values for t in ['high', 'medium', 'low']]
    if all(len(g) > 5 for g in groups):
        h, p = stats.kruskal(*groups)
        medians = [round(np.median(g), 2) for g in groups]
        print(f'  {feat:<18} medians(high/med/low)={medians}  '
              f'Kruskal-Wallis H={h:.1f} p={p:.2e}')

print('\n' + '=' * 78)
print('STAGE 2b -- does role (top-post vs matched counter-sample) separate posts?')
print('=' * 78)
post_features = ['implied_votes', 'contested_share', 'comment_score_gini',
                  'pct_toplevel', 'removed_comment_rate', 'bot_comment_rate',
                  'submitter_reply_rate', 'n_unique_commenters']
for feat in post_features:
    top = posts[posts.role == 'top'][feat].dropna()
    ctr = posts[posts.role == 'counter'][feat].dropna()
    if len(top) > 5 and len(ctr) > 5:
        u, p = stats.mannwhitneyu(top, ctr, alternative='two-sided')
        print(f'  {feat:<22} top_median={top.median():.3f}  counter_median={ctr.median():.3f}  '
              f'Mann-Whitney p={p:.2e}')

# ---------------------------------------------------------------------------
# Cross-tier account overlap -- do the same accounts show up across subs of
# different incentive tiers? (crude coordination-adjacent signal)
# ---------------------------------------------------------------------------
print('\n' + '=' * 78)
print('STAGE 2c -- cross-sub account overlap (same commenter, different subs)')
print('=' * 78)
by_sub = {sub: set(g.author) for sub, g in c.groupby('sub')}
subs = list(by_sub.keys())
for i in range(len(subs)):
    for j in range(i + 1, len(subs)):
        a, b = subs[i], subs[j]
        overlap = by_sub[a] & by_sub[b]
        print(f'  {a:<20} x {b:<20}  overlap={len(overlap):>4}  '
              f'({len(overlap)/min(len(by_sub[a]),len(by_sub[b]))*100:.1f}% of smaller set)')

# ---------------------------------------------------------------------------
# Save the account table for follow-up
# ---------------------------------------------------------------------------
acc.to_csv(OUT + 'pilot_accounts.csv', index=False)
posts.to_csv(OUT + 'pilot_posts.csv', index=False)
print(f'\nSaved account table -> {OUT}pilot_accounts.csv')
print(f'Saved post table    -> {OUT}pilot_posts.csv')
