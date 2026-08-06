#!/usr/bin/env python3
"""V3 Stage 2 (V3_PLAN.md Sec 7, "Stage 2 -- bivariate"): pairwise density
grid + HDBSCAN on the strongest pairs, plus the three explicit cross-level
segmentation protocols (P4 tercile, P3 tercile, S15 regime). Computes
everything from data/v3/analysis/v3.duckdb (read-only -- Stage-3 feature
sanitisation may be writing to the same file concurrently) and injects it
as JSON into scripts/v3_stage2_template.html, writing
docs/v3-research/eda/stage2.html.

**Deviation from the plan's literal text, stated up front:** Sec 7 says
"pairwise density grid over the Stage-1 shortlist." That shortlist
(v3_stage1_univariate.py) is flagged in V3_PLAN.md Sec 10.4 as unreliable
(18/19 features came back "bimodal", not credible) and is being fixed in a
parallel effort, not here. Rather than block on that fix or silently reuse
an untrustworthy shortlist, candidate pairs for the density grid + HDBSCAN
below are chosen from (a) the Spearman correlation matrix computed fresh in
this script over a theory-motivated candidate set, excluding the 4 pairs
already shown on the main EDA page, and (b) the qualitative findings already
logged in Sec 10.4 (removal_rate, the six botmarker_composite markers,
account_ordinal / days_since_first_seen-derived rates, reception_spread,
karma_extremeness). This substitution is reported on the output page itself,
not hidden.

Re-run after any account_features rebuild. Only summary statistics and a
subsampled scatter (for HDBSCAN visualization) are embedded -- full account
rows never leave DuckDB."""
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
TEMPLATE = os.path.join(ROOT, 'scripts', 'v3_stage2_template.html')
OUT = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'stage2.html')

RNG_SEED = 42
RISK_METRIC = 'removal_rate'

# (feature, transform)  -- mirrors v3_eda_build.py's choices for the same
# features so the two pages agree on how each one is viewed.
CANDIDATE_FEATURES = [
    ('n_comments_sample', 'log1p'),
    ('removal_rate', 'none'),
    ('deleted_later_rate', 'none'),
    # thin_history_score deliberately excluded from this candidate set: it's
    # defined as the inverse of n_comments_sample (see EDA page marker-def),
    # so any pair it forms is a near-mirror of n_comments_sample's own pairs
    # and would just duplicate a panel under a different name.
    ('karma_extremeness', 'log1p'),
    ('karma_per_post_extremeness', 'log1p'),
    ('reception_spread', 'log1p'),
    ('botmarker_composite', 'none'),
    ('account_ordinal', 'none'),
    ('days_since_first_seen', 'none'),
    ('karma_per_day_since_first_seen', 'signed_log'),
    ('controversiality_rate', 'none'),
    ('subreddit_entropy', 'none'),
    ('mean_comment_score', 'signed_log'),
    ('score_stddev', 'log1p'),
    ('interval_entropy', 'none'),          # tier2 only (n>=5 comments)
    ('burstiness_kimjo', 'none'),          # tier2 only
    ('interval_quantization_rate', 'none'),  # tier2 only
    ('username_char_entropy', 'none'),
]
TIER2_FEATURES = {'interval_entropy', 'burstiness_kimjo', 'interval_quantization_rate'}

# already on the main EDA page (scripts/v3_eda_build.py DENSITY_PAIRS) -- skip these
ALREADY_SHOWN = {
    frozenset(('n_comments_sample', 'removal_rate')),
    frozenset(('account_ordinal', 'subreddit_entropy')),
    frozenset(('mean_comment_score', 'controversiality_rate')),
    frozenset(('reception_spread', 'controversiality_rate')),
}

N_DENSITY_PAIRS = 6
N_HDBSCAN_PAIRS = 3
HDBSCAN_SUBSAMPLE = 40_000
SCATTER_POINTS = 3000


def transform(x, kind):
    if kind == 'log1p':
        return np.log1p(x)
    if kind == 'signed_log':
        return np.sign(x) * np.log1p(np.abs(x))
    return x


# ============================================================
# 1. Correlation matrix + candidate pair selection
# ============================================================

def build_correlation(con):
    print('Computing Spearman correlation over candidate set...')
    feats = [f for f, _ in CANDIDATE_FEATURES]
    tx_map = dict(CANDIDATE_FEATURES)
    cols = ', '.join(f'"{f}"' for f in feats)
    df = con.execute(f'SELECT {cols} FROM account_features').fetchdf()
    n_feat = len(feats)
    corr = np.eye(n_feat)
    pair_n = {}
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            sub = df[[feats[i], feats[j]]].dropna()
            pair_n[(i, j)] = len(sub)
            if len(sub) > 100:
                rho, _ = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
                corr[i, j] = corr[j, i] = rho if not np.isnan(rho) else 0.0

    ranked = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            if frozenset((feats[i], feats[j])) in ALREADY_SHOWN:
                continue
            if pair_n[(i, j)] < 1000:
                continue
            rho_abs = abs(corr[i, j])
            # |rho| >= 0.98 on this candidate set is definitional, not
            # behavioral -- e.g. thin_history_score is a direct inverse of
            # n_comments_sample by construction. Excluded, not just
            # deprioritized, so the density grid doesn't waste a panel on it.
            if rho_abs >= 0.98:
                continue
            ranked.append((rho_abs, feats[i], feats[j], corr[i, j], pair_n[(i, j)]))
    ranked.sort(key=lambda r: -r[0])
    return {'features': feats, 'matrix': corr.tolist()}, ranked, tx_map


# ============================================================
# 2. Density panels (same recipe as v3_eda_build.py)
# ============================================================

def density_panel(con, fx, tx_x, fy, tx_y, caption):
    rows = con.execute(
        f'SELECT "{fx}", "{fy}" FROM account_features WHERE "{fx}" IS NOT NULL AND "{fy}" IS NOT NULL'
    ).fetchnumpy()
    x = transform(rows[fx].astype(float), tx_x)
    y = transform(rows[fy].astype(float), tx_y)
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    mask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    x, y = x[mask], y[mask]
    h, xe, ye = np.histogram2d(x, y, bins=24, range=[[xlo, xhi], [ylo, yhi]])
    return {
        'x_feature': fx, 'x_transform': tx_x, 'y_feature': fy, 'y_transform': tx_y,
        'caption': caption, 'n': int(mask.sum()),
        'grid': np.log1p(h.T).tolist(),
        'x_edges': xe.tolist(), 'y_edges': ye.tolist(),
    }


# ============================================================
# 3. HDBSCAN on the strongest pairs
# ============================================================

def hdbscan_pair(con, fx, tx_x, fy, tx_y, rng):
    where = ''
    if fx in TIER2_FEATURES or fy in TIER2_FEATURES:
        where = 'AND has_timing_features'
    rows = con.execute(f"""
        SELECT "{fx}", "{fy}", "{RISK_METRIC}" FROM account_features
        WHERE "{fx}" IS NOT NULL AND "{fy}" IS NOT NULL {where}
    """).fetchnumpy()
    x_raw = rows[fx].astype(float)
    y_raw = rows[fy].astype(float)
    risk = rows[RISK_METRIC].astype(float)
    x = transform(x_raw, tx_x)
    y = transform(y_raw, tx_y)
    xlo, xhi = np.percentile(x, [0.5, 99.5])
    ylo, yhi = np.percentile(y, [0.5, 99.5])
    mask = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
    x, y, risk = x[mask], y[mask], risk[mask]
    n_total = len(x)

    idx = np.arange(n_total)
    if n_total > HDBSCAN_SUBSAMPLE:
        idx = rng.choice(n_total, HDBSCAN_SUBSAMPLE, replace=False)
    xs, ys, risks = x[idx], y[idx], risk[idx]
    n = len(xs)

    xz = (xs - xs.mean()) / (xs.std() + 1e-12)
    yz = (ys - ys.mean()) / (ys.std() + 1e-12)
    pts = np.column_stack([xz, yz])

    min_cluster_size = max(30, int(0.005 * n))
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=max(10, min_cluster_size // 3))
    labels = clusterer.fit_predict(pts)

    uniq = sorted(set(labels.tolist()) - {-1})
    clusters = []
    for lab in uniq:
        sel = labels == lab
        clusters.append({
            'label': int(lab), 'n': int(sel.sum()),
            'x_mean': float(xs[sel].mean()), 'y_mean': float(ys[sel].mean()),
            'risk_mean': float(risks[sel].mean()),
        })
    clusters.sort(key=lambda c: -c['n'])
    noise_share = float((labels == -1).sum() / n)
    pop_risk = float(risks.mean())

    scatter_idx = np.arange(n) if n <= SCATTER_POINTS else rng.choice(n, SCATTER_POINTS, replace=False)
    scatter = {
        'x': xs[scatter_idx].round(4).tolist(),
        'y': ys[scatter_idx].round(4).tolist(),
        'label': labels[scatter_idx].tolist(),
    }

    return {
        'x_feature': fx, 'x_transform': tx_x, 'y_feature': fy, 'y_transform': tx_y,
        'n_clustered': n, 'n_total_available': n_total, 'min_cluster_size': min_cluster_size,
        'n_clusters': len(clusters), 'noise_share': noise_share, 'pop_risk_mean': pop_risk,
        'clusters': clusters, 'scatter': scatter,
    }


# ============================================================
# 4. Segmentation protocol 1 & 2: P4 tercile, P3 tercile
# ============================================================

SEG_COMPARE_FEATURES = [
    'days_since_first_seen', 'karma_per_day_since_first_seen', 'subreddit_entropy',
    'removal_rate', 'controversiality_rate', 'botmarker_composite', 'reception_spread',
    'interval_entropy',  # closest available "timing entropy" proxy; tier2-only, see caveat on page
]
VOLUME_BUCKETS_SQL = """
    CASE WHEN n_comments_sample = 1 THEN '1 (singleton)'
         WHEN n_comments_sample BETWEEN 2 AND 4 THEN '2-4'
         WHEN n_comments_sample BETWEEN 5 AND 9 THEN '5-9'
         ELSE '10+' END
"""


def post_vote_metric_segmentation(con, label, metric_expr, extra_where=''):
    """Generic: tag each account by their mean(metric) over role='top' posts
    they commented on, tercile that, compare account_features means between
    the low and high tercile, both pooled and within volume buckets."""
    account_metric = con.execute(f"""
        WITH pm AS (
            SELECT post_id, {metric_expr} AS m
            FROM posts_clean WHERE role = 'top' {extra_where}
        ), tagged AS (
            SELECT c.author, pm.m
            FROM commenters_clean c JOIN pm USING(post_id)
            WHERE pm.m IS NOT NULL
        )
        SELECT author, avg(m) AS account_m, count(*) AS n_tagged
        FROM tagged GROUP BY author
    """).fetchdf()
    if len(account_metric) < 100:
        return {'label': label, 'skipped': 'too few tagged accounts'}

    t1, t2 = np.percentile(account_metric['account_m'], [33.33, 66.67])
    account_metric['tercile'] = np.where(
        account_metric['account_m'] <= t1, 'low',
        np.where(account_metric['account_m'] >= t2, 'high', 'mid'))

    con.execute("CREATE OR REPLACE TEMP TABLE _seg_tag AS SELECT * FROM account_metric")
    cmp_cols = ', '.join(f'avg("{f}") AS "{f}"' for f in SEG_COMPARE_FEATURES)
    pooled = con.execute(f"""
        SELECT t.tercile, count(*) n, {cmp_cols}
        FROM _seg_tag t JOIN account_features af ON af.author = t.author
        GROUP BY t.tercile ORDER BY t.tercile
    """).fetchdf()

    by_volume = con.execute(f"""
        SELECT t.tercile, {VOLUME_BUCKETS_SQL.replace('n_comments_sample', 'af.n_comments_sample')} AS vol_bucket,
               count(*) n, avg(af.removal_rate) removal_rate, avg(af.controversiality_rate) controversiality_rate
        FROM _seg_tag t JOIN account_features af ON af.author = t.author
        WHERE t.tercile IN ('low', 'high')
        GROUP BY 1, 2 ORDER BY 2, 1
    """).fetchdf()

    return {
        'label': label, 'n_accounts_tagged': int(len(account_metric)),
        'tercile_bounds': [float(t1), float(t2)],
        'pooled': pooled.to_dict('records'),
        'by_volume': by_volume.to_dict('records'),
    }


# ============================================================
# 5. Segmentation protocol 3: S15 regime
# ============================================================

def s15_regime_detection(con):
    df = con.execute("""
        SELECT sub, month, sum(n_comments_observed) AS c
        FROM posts_clean WHERE role = 'top'
        GROUP BY 1, 2 ORDER BY 1, 2
    """).fetchdf()

    regime_rows = []
    sub_summaries = []
    for sub, g in df.groupby('sub'):
        g = g.sort_values('month')
        y = np.log1p(g['c'].values.astype(float)).reshape(-1, 1)
        if len(y) < 12:
            continue
        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=3).fit(y)
        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=5).fit(y)
        bic1, bic2 = gmm1.bic(y), gmm2.bic(y)
        two_component = bic2 < bic1
        if two_component:
            labels = gmm2.predict(y)
            means = gmm2.means_.flatten()
            spike_comp = int(np.argmax(means))
            weights = gmm2.weights_
            # guard: the "spike" component must be a genuine minority (<45%) --
            # otherwise BIC just split a roughly-symmetric series in half, not
            # a baseline-vs-spike regime.
            if weights[spike_comp] >= 0.45:
                two_component = False
        if two_component:
            regime = np.where(labels == spike_comp, 'spike', 'baseline')
        else:
            regime = np.array(['baseline'] * len(g))
        for m, r in zip(g['month'], regime):
            regime_rows.append({'sub': sub, 'month': m, 'regime': r})
        sub_summaries.append({
            'sub': sub, 'two_component': bool(two_component),
            'bic1': float(bic1), 'bic2': float(bic2),
            'n_spike_months': int((regime == 'spike').sum()), 'n_months': int(len(g)),
        })

    n_two_comp = sum(1 for s in sub_summaries if s['two_component'])
    regime_df = pd.DataFrame(regime_rows)
    con.execute("CREATE OR REPLACE TEMP TABLE _regime AS SELECT * FROM regime_df")

    account_metric = con.execute("""
        WITH tagged AS (
            SELECT c.author, r.regime
            FROM commenters_clean c
            JOIN posts_clean p ON p.post_id = c.post_id AND p.role = 'top'
            JOIN _regime r ON r.sub = p.sub AND r.month = p.month
        )
        SELECT author,
               sum(CASE WHEN regime = 'spike' THEN 1 ELSE 0 END) AS n_spike,
               count(*) AS n_tagged
        FROM tagged GROUP BY author
    """).fetchdf()
    account_metric['group'] = np.where(account_metric['n_spike'] > 0, 'spike_exposed', 'baseline_only')

    con.execute("CREATE OR REPLACE TEMP TABLE _seg_tag2 AS SELECT * FROM account_metric")
    cmp_cols = ', '.join(f'avg("{f}") AS "{f}"' for f in SEG_COMPARE_FEATURES)
    pooled = con.execute(f"""
        SELECT t."group", count(*) n, {cmp_cols}
        FROM _seg_tag2 t JOIN account_features af ON af.author = t.author
        GROUP BY t."group" ORDER BY t."group"
    """).fetchdf()
    by_volume = con.execute(f"""
        SELECT t."group", {VOLUME_BUCKETS_SQL.replace('n_comments_sample', 'af.n_comments_sample')} AS vol_bucket,
               count(*) n, avg(af.removal_rate) removal_rate, avg(af.controversiality_rate) controversiality_rate
        FROM _seg_tag2 t JOIN account_features af ON af.author = t.author
        GROUP BY 1, 2 ORDER BY 2, 1
    """).fetchdf()

    return {
        'n_subs_two_component': n_two_comp, 'n_subs_total': len(sub_summaries),
        'sub_summaries': sub_summaries,
        'n_accounts_tagged': int(len(account_metric)),
        'pooled': pooled.to_dict('records'),
        'by_volume': by_volume.to_dict('records'),
    }


# ============================================================
# main
# ============================================================

def main():
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED)
    con = duckdb.connect(DB_PATH, read_only=True)

    corr_data, ranked_pairs, tx_map = build_correlation(con)
    print(f'Top ranked pairs (excluding already-shown):')
    for rho_abs, fi, fj, rho, n in ranked_pairs[:10]:
        print(f'  {fi:<28} x {fj:<28} rho={rho:+.3f} n={n}')

    density_pick = ranked_pairs[:N_DENSITY_PAIRS]
    print('\nBuilding density panels...')
    density_panels = []
    for rho_abs, fi, fj, rho, n in density_pick:
        cap = f'Spearman rho={rho:+.2f} (n={n:,}) -- selected by |rho| rank, substitute for the unreliable Stage-1 shortlist (see page note)'
        if rho_abs >= 0.9:
            cap += ' -- CAUTION: this high a correlation may reflect shared construction (e.g. both features derive from footprint breadth), not an independent behavioral link -- check before treating as a finding'
        density_panels.append(density_panel(con, fi, tx_map[fi], fj, tx_map[fj], cap))

    hdbscan_pick = density_pick[:N_HDBSCAN_PAIRS]
    print('\nRunning HDBSCAN on strongest pairs...')
    hdbscan_results = []
    for rho_abs, fi, fj, rho, n in hdbscan_pick:
        print(f'  {fi} x {fj} ...')
        hdbscan_results.append(hdbscan_pair(con, fi, tx_map[fi], fj, tx_map[fj], rng))

    print('\nSegmentation protocol 1: P4 tercile (votes/comments)...')
    p4_seg = post_vote_metric_segmentation(
        con, 'P4 (votes/comments) tercile',
        'implied_votes / num_comments_reported',
        "AND upvote_ratio >= 0.65 AND implied_votes IS NOT NULL AND num_comments_reported > 0")

    print('Segmentation protocol 2: P3 tercile (contested_share)...')
    p3_seg = post_vote_metric_segmentation(
        con, 'P3 (contested_share) tercile',
        'contested_share',
        "AND contested_share IS NOT NULL")

    print('Segmentation protocol 3: S15 regime (baseline vs spike month)...')
    s15_seg = s15_regime_detection(con)

    con.close()

    data = {
        'correlation': corr_data,
        'ranked_pairs': [
            {'x': fi, 'y': fj, 'rho': rho, 'n': n} for rho_abs, fi, fj, rho, n in ranked_pairs[:20]
        ],
        'density_panels': density_panels,
        'hdbscan': hdbscan_results,
        'p4_segmentation': p4_seg,
        'p3_segmentation': p3_seg,
        's15_segmentation': s15_seg,
    }

    with open(TEMPLATE) as f:
        tpl = f.read()
    out_html = tpl.replace('__STAGE2_JSON__', json.dumps(data))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(out_html)
    print(f'\nWrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB) in {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
