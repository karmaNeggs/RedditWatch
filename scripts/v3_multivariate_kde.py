#!/usr/bin/env python3
"""V3: multivariate KDE boundary discovery + suspected-bot-marker "dye" overlay.

Follow-up to v3_boundary_discovery.py, which tested each behavioral metric
ONE AT A TIME for bimodality. User's explicit redirect: "bimodal was just a
basic suggestion" -- find the real, JOINT shape of the data in higher
dimensions via KDE, not per-metric splits. PCA reduces the 64-feature
candidate space to a tractable number of dimensions (raw 64-D KDE is
statistically meaningless at this dimensionality relative to sample size --
density estimates go sparse fast past ~10 dims). Density-driven clustering
(mean-shift, literally KDE-mode-climbing) finds group boundaries in that
reduced space, cross-checked against HDBSCAN. Discovered groups are then
"dyed" (overlaid) with suspected-bot markers -- including removal_rate,
which is EXCLUDED from construction throughout, exactly as in the prior
round -- to see where those markers concentrate and by how much. Also
checks overlap with the previous round's ~550-account AND-rule group.

Reuses v3_boundary_discovery.py's split (same RNG_SEED=42, identical
stratification) and candidate pool unchanged, so results are directly
comparable and the previous round's flagged accounts can be re-derived
for the overlap check.
"""
import json
import os
import time

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.cluster import HDBSCAN, MeanShift
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
OUT_JSON = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'multivariate_kde_data.json')
OUT_PNG_DIR = os.path.join(ROOT, 'docs', 'v3-research', 'eda')

T0 = time.time()


def log(msg):
    print(f'[{time.time()-T0:7.1f}s] {msg}', flush=True)


RNG_SEED = 42  # identical to v3_boundary_discovery.py -- same split reproduces

LABEL_DERIVED_EXCLUDE = {
    'removal_rate', 'deleted_later_rate', 'removal_rate_nonzero', 'deleted_later_rate_nonzero',
    'removal_rate_pctl', 'deleted_later_rate_pctl', 'thin_history_score',
    'reception_spread_pctl', 'botmarker_composite', 'n_markers_available',
    'post_edit_rate', 'post_edit_rate_nonzero',
}

# identical candidate list + transforms to v3_boundary_discovery.py
CANDIDATES = [
    ('n_comments_sample', 'log1p'), ('n_posts_sample', 'log1p'), ('high_tier_share', 'none'),
    ('medium_tier_share', 'none'), ('low_tier_share', 'none'), ('n_low_tier_subs', 'log1p'),
    ('subreddit_entropy', 'none'), ('score_stddev', 'log1p'), ('controversiality_rate', 'none'),
    ('is_submitter_rate', 'none'), ('mean_depth', 'none'), ('mean_body_len', 'log1p'),
    ('mean_post_score', 'log1p'), ('worst_sub_mean_score', 'signed_log1p'), ('reception_spread', 'log1p'),
    ('account_ordinal', 'log1p'), ('observed_span_days', 'log1p'), ('comments_per_day_observed', 'log1p'),
    ('sample_score_per_day_observed', 'signed_log1p'), ('days_since_first_seen', 'none'),
    ('comments_per_day_since_first_seen', 'log1p'), ('posts_per_day_since_first_seen', 'log1p'),
    ('karma_per_day_since_first_seen', 'signed_log1p'), ('username_char_entropy', 'none'),
    ('username_digit_suffix_len', 'none'), ('interval_entropy', 'none'), ('burstiness_kimjo', 'none'),
    ('interval_quantization_rate', 'none'), ('repeat_engagement_rate', 'none'),
    ('own_post_reply_rate', 'none'), ('n_threads_with_repeat', 'log1p'),
    ('n_own_posts_with_comments', 'log1p'), ('n_subs_rejected_but_returned', 'log1p'),
    ('karma_extremeness', 'log1p'), ('karma_per_post_extremeness', 'log1p'),
    ('n_distinct_posts_ctx', 'log1p'), ('pc_contested_share', 'none'), ('pc_comment_score_gini', 'none'),
    ('pc_reply_reciprocity', 'none'), ('pc_removed_comment_rate', 'none'),
    ('pc_removed_comment_rate_max', 'none'), ('pc_tombstone_rate', 'none'),
    ('pc_tombstone_rate_max', 'none'), ('pc_bot_comment_rate', 'none'), ('pc_bot_comment_rate_max', 'none'),
    ('pc_submitter_reply_rate', 'none'), ('pc_upvote_ratio', 'none'), ('pc_pct_toplevel', 'none'),
    ('pc_mean_depth', 'none'), ('pc_num_crossposts', 'log1p'), ('pc_log_subscribers', 'none'),
    ('pc_n_unique_commenters', 'log1p'), ('pc_n_comments_observed', 'log1p'), ('pc_is_self_rate', 'none'),
    ('pc_over18_rate', 'none'), ('own_repeat_rate', 'none'), ('url_rate', 'none'),
    ('outsider_influx_share', 'none'), ('title_body_ratio', 'log1p'), ('score_per_word', 'signed_log1p'),
    ('sub_month_spike_share', 'none'), ('coappear_degree', 'log1p'), ('coappear_hhi', 'none'),
    ('domain_hhi', 'none'),
]

MARKER_COLS = ['removal_rate', 'deleted_later_rate', 'karma_extremeness', 'thin_history_score',
               'reception_spread', 'shows_silo_mismatch_pattern', 'botmarker_composite']

# previous round's chosen AND-rule (from boundary_discovery_data.json plausibility_check)
PREV_CHOSEN_FEATURES = ['reception_spread', 'domain_hhi', 'account_ordinal',
                         'n_own_posts_with_comments', 'subreddit_entropy']


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def load_data(con):
    log('loading + joining all feature tables on author')
    df = con.execute('SELECT * FROM account_features_model').fetchdf()
    for tbl in ['account_post_context', 'account_tier1_repeat_url', 'account_tier1_post_context',
                'account_tier2_regime', 'account_tier2_coappear', 'account_tier3_domain']:
        t = con.execute(f'SELECT * FROM {tbl}').fetchdf()
        df = df.merge(t, on='author', how='left')
    log(f'joined shape: {df.shape}')
    return df


def stratified_3way_split(df, rng_seed=RNG_SEED):
    d = df.copy()
    tier_bucket = np.select(
        [d['high_tier_share'] >= 0.5, d['medium_tier_share'] >= 0.5],
        ['high', 'medium'], default='low_or_mixed')
    try:
        ordinal_decile = pd.qcut(d['account_ordinal'], 10, labels=False, duplicates='drop')
    except Exception:
        ordinal_decile = pd.Series(0, index=d.index)
    strata = pd.Series(tier_bucket, index=d.index).astype(str) + '_' + ordinal_decile.astype(str)

    rng = np.random.RandomState(rng_seed)
    part_assignment = pd.Series(index=d.index, dtype=object)
    for s, idx in d.groupby(strata).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n = len(idx)
        n1 = n // 3
        n2 = n // 3
        part_assignment.loc[idx[:n1]] = 'part1'
        part_assignment.loc[idx[n1:n1 + n2]] = 'part2'
        part_assignment.loc[idx[n1 + n2:]] = 'part3'
    d['part'] = part_assignment
    return d


def build_feature_matrix(d):
    """Apply the same transforms as v3_boundary_discovery.py, then median-impute
    (fit on part1+2, applied everywhere -- same no-leakage-into-part3 discipline
    as the PCA/clustering fit below) since most accounts are missing post-context
    / tier1-3 columns (gated on distinct-post-count thresholds) and complete-case
    deletion would discard the majority of the population."""
    cols = [c for c, _ in CANDIDATES if c in d.columns]
    log(f'{len(cols)}/{len(CANDIDATES)} candidate columns present')
    X_raw = d[cols].copy()
    transform_map = dict(CANDIDATES)
    for c in cols:
        t = transform_map[c]
        x = X_raw[c].to_numpy(dtype=float)
        if t == 'log1p':
            shift = 0.0
            m = np.nanmin(x)
            if m < 0:
                shift = -m
            X_raw[c] = np.log1p(x + shift)
        elif t == 'signed_log1p':
            X_raw[c] = signed_log1p(x)
        # 'none': leave as-is
    return X_raw, cols


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    df = load_data(con)
    d = stratified_3way_split(df)
    log('split sizes: ' + ', '.join(f'{p}={int((d["part"]==p).sum())}' for p in ['part1', 'part2', 'part3']))

    X_all, cols = build_feature_matrix(d)

    p12_mask = d['part'].isin(['part1', 'part2']).to_numpy()
    median_fill = X_all.loc[p12_mask].median()
    X_filled = X_all.fillna(median_fill)
    n_missing_per_col = X_all.isna().mean().sort_values(ascending=False)
    log('top-5 most-missing candidate columns (median-imputed): ' +
        ', '.join(f'{c}={v:.2%}' for c, v in n_missing_per_col.head(5).items()))

    # ---- Step 1: PCA, fit on Part1+Part2 only ----
    log('=== Step 1: PCA (fit on Part1+Part2) ===')
    scaler = StandardScaler().fit(X_filled.loc[p12_mask])
    Xz_all = pd.DataFrame(scaler.transform(X_filled), index=X_filled.index, columns=cols)

    pca_full = PCA(n_components=min(len(cols), p12_mask.sum())).fit(Xz_all.loc[p12_mask])
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, 0.80) + 1)
    n_components = min(max(n_components, 5), 12)
    log(f'retaining {n_components} components for clustering (cumvar at that point: {cumvar[n_components-1]:.3f})')

    pca = PCA(n_components=n_components, random_state=0).fit(Xz_all.loc[p12_mask])
    Z_all = pca.transform(Xz_all)  # project everyone (part1, part2, part3) using part1+2-fit PCA
    Z_df = pd.DataFrame(Z_all, index=Xz_all.index, columns=[f'PC{i+1}' for i in range(n_components)])

    loadings = pd.DataFrame(pca.components_.T, index=cols,
                             columns=[f'PC{i+1}' for i in range(n_components)])
    loadings_report = {}
    for i in range(min(5, n_components)):
        pc = f'PC{i+1}'
        top = loadings[pc].abs().sort_values(ascending=False).head(6).index.tolist()
        loadings_report[pc] = {
            'explained_variance_ratio': float(pca.explained_variance_ratio_[i]),
            'top_loadings': [{'feature': f, 'loading': float(loadings.loc[f, pc])} for f in top],
        }
        log(f'  {pc} (var={pca.explained_variance_ratio_[i]:.3f}): ' +
            ', '.join(f'{f}={loadings.loc[f,pc]:+.2f}' for f in top))

    # ---- Step 2: density-driven clustering, Part1 vs Part2 independently ----
    log('=== Step 2: mean-shift (KDE-mode) clustering, Part1 vs Part2 independently ===')

    def cluster_part(part_name, sample_cap=40000):
        idx = d.index[d['part'] == part_name]
        Zp = Z_df.loc[idx]
        rng = np.random.RandomState(0)
        if len(Zp) > sample_cap:
            sub_idx = rng.choice(len(Zp), sample_cap, replace=False)
            Zp_fit = Zp.iloc[sub_idx]
        else:
            Zp_fit = Zp
        ms = MeanShift(bandwidth=None, bin_seeding=True, n_jobs=-1).fit(Zp_fit.to_numpy())
        labels_fit = ms.labels_
        centers = ms.cluster_centers_
        # assign full part by nearest center (mean-shift itself only fit on the capped sample)
        from scipy.spatial.distance import cdist
        dmat = cdist(Zp.to_numpy(), centers)
        labels_full = dmat.argmin(axis=1)
        sizes = pd.Series(labels_full).value_counts().sort_index()
        log(f'  {part_name}: mean-shift found {len(centers)} modes, sizes={sizes.to_dict()}')
        return labels_full, centers, idx

    labels1, centers1, idx1 = cluster_part('part1')
    labels2, centers2, idx2 = cluster_part('part2')

    # HDBSCAN cross-check on part1
    log('  cross-check: HDBSCAN on Part1 (capped sample)')
    idx_p1 = d.index[d['part'] == 'part1']
    rng = np.random.RandomState(1)
    cap = 40000
    if len(idx_p1) > cap:
        sub = rng.choice(len(idx_p1), cap, replace=False)
        Z_hdb = Z_df.loc[idx_p1].iloc[sub].to_numpy()
    else:
        Z_hdb = Z_df.loc[idx_p1].to_numpy()
    hdb = HDBSCAN(min_cluster_size=200, n_jobs=-1).fit(Z_hdb)
    hdb_labels, hdb_counts = np.unique(hdb.labels_[hdb.labels_ >= 0], return_counts=True)
    log(f'  HDBSCAN Part1: {len(hdb_labels)} clusters (excl. noise), sizes={dict(zip(hdb_labels.tolist(), hdb_counts.tolist()))}, '
        f'noise={int((hdb.labels_==-1).sum())}/{len(hdb.labels_)}')

    # shape comparison: relative sizes + centroid distances (nearest-center matching)
    from scipy.spatial.distance import cdist
    size1 = pd.Series(labels1).value_counts(normalize=True).sort_index()
    size2 = pd.Series(labels2).value_counts(normalize=True).sort_index()
    cross_dist = cdist(centers1, centers2)
    match2for1 = cross_dist.argmin(axis=1)
    shape_comparison = {
        'n_modes_part1': int(len(centers1)), 'n_modes_part2': int(len(centers2)),
        'part1_relative_sizes': size1.round(4).to_dict(),
        'part2_relative_sizes': size2.round(4).to_dict(),
        'part1_to_part2_nearest_center_dist': cross_dist.min(axis=1).round(3).tolist(),
        'hdbscan_part1_n_clusters': int(len(hdb_labels)),
        'hdbscan_part1_noise_frac': float((hdb.labels_ == -1).mean()),
    }
    log(f'  shape comparison: {shape_comparison["n_modes_part1"]} modes (P1) vs {shape_comparison["n_modes_part2"]} modes (P2); '
        f'nearest-center distances P1->P2: {shape_comparison["part1_to_part2_nearest_center_dist"]}')

    # ---- Step 2b (Part3 validation): project + assign to Part1-fit centers ----
    log('=== Part3 validation: assign to Part1-fit mean-shift centers (no refit) ===')
    idx3 = d.index[d['part'] == 'part3']
    Z3 = Z_df.loc[idx3]
    dmat3 = cdist(Z3.to_numpy(), centers1)
    labels3 = dmat3.argmin(axis=1)
    size3 = pd.Series(labels3).value_counts(normalize=True).sort_index()
    log(f'  Part3 assigned-cluster relative sizes (using Part1 centers): {size3.round(4).to_dict()}')

    # ---- Step 3: dye overlay -- marker enrichment per Part1 cluster ----
    log('=== Step 3: dye overlay (marker enrichment per Part1 cluster, computed AFTER clustering) ===')
    markers_df = con.execute(
        f"SELECT author, {', '.join(MARKER_COLS)} FROM account_features_model").fetchdf()
    d_markers = d.merge(markers_df, on='author', how='left', suffixes=('', '_mk'))

    p1_df = d_markers.loc[idx1].copy()
    p1_df['cluster'] = labels1
    pop_baseline = {m: float(d_markers[m].mean()) for m in MARKER_COLS}
    enrichment = {}
    for cl in sorted(p1_df['cluster'].unique()):
        sub = p1_df[p1_df['cluster'] == cl]
        row = {'n': int(len(sub)), 'pct_of_part1': float(len(sub) / len(p1_df))}
        for m in MARKER_COLS:
            val = float(sub[m].mean())
            row[f'{m}_mean'] = val
            row[f'{m}_ratio_vs_pop'] = val / pop_baseline[m] if pop_baseline[m] else float('nan')
        enrichment[str(cl)] = row
        log(f'  cluster {cl}: n={row["n"]} ({row["pct_of_part1"]:.2%})  ' +
            '  '.join(f'{m}={row[f"{m}_ratio_vs_pop"]:.2f}x' for m in MARKER_COLS))

    # replication check for the strongest Part1 finding: does Part2 -- fit
    # completely independently -- ALSO produce a cluster with strongly
    # elevated removal_rate, or was Part1's 12x a small-n fluke of one run?
    log('=== replication check: does Part2 independently show a similarly removal_rate-enriched cluster? ===')
    p2_df = d_markers.loc[idx2].copy()
    p2_df['cluster'] = labels2
    enrichment2 = {}
    for cl in sorted(p2_df['cluster'].unique()):
        sub = p2_df[p2_df['cluster'] == cl]
        if len(sub) < 30:
            continue
        val = float(sub['removal_rate'].mean())
        enrichment2[str(cl)] = {'n': int(len(sub)), 'removal_rate_ratio_vs_pop': val / pop_baseline['removal_rate']}
    top_p2 = sorted(enrichment2.items(), key=lambda kv: -kv[1]['removal_rate_ratio_vs_pop'])[:5]
    top_p1 = sorted([(k, v) for k, v in enrichment.items() if v['n'] >= 30],
                     key=lambda kv: -kv[1]['removal_rate_ratio_vs_pop'])[:5]
    log(f'  Part1 top-5 removal_rate-enriched clusters (n>=30): ' +
        ', '.join(f'cl{k}(n={v["n"]})={v["removal_rate_ratio_vs_pop"]:.2f}x' for k, v in top_p1))
    log(f'  Part2 top-5 removal_rate-enriched clusters (n>=30): ' +
        ', '.join(f'cl{k}(n={v["n"]})={v["removal_rate_ratio_vs_pop"]:.2f}x' for k, v in top_p2))
    replication_check = {'part1_top5': [{'cluster': k, **v} for k, v in top_p1],
                          'part2_top5': [{'cluster': k, **v} for k, v in top_p2]}

    # overlap check vs previous round's ~550-account AND-rule group
    log('=== overlap check vs previous round\'s AND-rule flagged group ===')
    prev_data = json.load(open(os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'boundary_discovery_data.json')))
    prev_pc = prev_data['plausibility_check']
    log(f'  previous chosen features: {prev_pc["chosen_features"]}, n_flagged_total={prev_pc["n_flagged_total"]}')
    # re-derive the previous rule's Part1 flagged set using the same boundary values
    standout_lookup = {f['feature']: f for f in prev_data['standout_features']}
    cond1_prev = pd.Series(True, index=idx1)
    for feat in prev_pc['chosen_features']:
        info = standout_lookup[feat]
        b1 = info['boundary1']
        col_vals = d.loc[idx1, feat]
        hint = info.get('direction_hint')
        if hint == 'high':
            is_bot = col_vals >= b1
        elif hint == 'low':
            is_bot = col_vals <= b1
        elif info['minority_high1']:
            is_bot = col_vals >= b1
        else:
            is_bot = col_vals <= b1
        cond1_prev &= is_bot.fillna(False)
    prev_flagged_idx = idx1[cond1_prev.to_numpy()]
    log(f'  re-derived previous-round Part1 flagged: n={len(prev_flagged_idx)} (paper number was 550)')
    prev_flagged_clusters = pd.Series(labels1, index=idx1).loc[prev_flagged_idx]
    overlap_dist = prev_flagged_clusters.value_counts(normalize=True).round(4).to_dict()
    log(f'  cluster distribution of previous-round flagged accounts: {overlap_dist}')
    # is this concentrated relative to the cluster's overall population share?
    overall_dist = pd.Series(labels1, index=idx1).value_counts(normalize=True).round(4).to_dict()
    log(f'  (for comparison, overall Part1 cluster shares: {overall_dist})')

    # account-level export for manual verification (not previously persisted --
    # everything above was aggregate-only). Cluster 15 is the ~18x-enriched
    # cluster that overlaps the AND-rule group; dump author identities so a
    # human can pull real threads/comments for both groups and the overlap.
    out_dir = os.path.join(ROOT, 'output', 'v3')
    os.makedirs(out_dir, exist_ok=True)
    part1_export = pd.DataFrame({
        'author': d.loc[idx1, 'author'].to_numpy(),
        'cluster': labels1,
        'and_rule_flagged': pd.Series(idx1, index=idx1).isin(prev_flagged_idx).to_numpy(),
    })
    export_path = os.path.join(out_dir, 'flagged_accounts_part1.csv')
    part1_export.to_csv(export_path, index=False)
    log(f'  wrote {export_path} (n={len(part1_export)}, cluster15={int((part1_export.cluster==15).sum())}, '
        f'and_rule_flagged={int(part1_export.and_rule_flagged.sum())}, '
        f'overlap={int(((part1_export.cluster==15)&part1_export.and_rule_flagged).sum())})')

    # ---- Step 4: visual output data ----
    log('=== Step 4: writing scatter/contour data + PNG sanity plots ===')
    rng = np.random.RandomState(2)
    scatter_records = []
    for part_name, idx_p, labels_p in [('part1', idx1, labels1), ('part2', idx2, labels2), ('part3', idx3, labels3)]:
        n_sample = min(6000, len(idx_p))
        sub_idx = rng.choice(len(idx_p), n_sample, replace=False)
        chosen = np.array(idx_p)[sub_idx]
        sub_z = Z_df.loc[chosen]
        sub_labels = np.array(labels_p)[sub_idx]
        sub_markers = d_markers.loc[chosen, MARKER_COLS]
        for i, row_idx in enumerate(chosen):
            rec = {'part': part_name, 'cluster': int(sub_labels[i]),
                   'pc1': float(sub_z.iloc[i]['PC1']), 'pc2': float(sub_z.iloc[i]['PC2']),
                   'pc3': float(sub_z.iloc[i]['PC3']) if n_components >= 3 else None}
            for m in MARKER_COLS:
                v = sub_markers.iloc[i][m]
                rec[m] = float(v) if pd.notna(v) else None
            scatter_records.append(rec)
    log(f'  scatter sample: {len(scatter_records)} points across 3 parts')

    # KDE density grid on top-2 PCs, Part1 and Part2 separately (visual replication check)
    def kde_grid(idx_p, gridsize=60, cap=20000):
        rng2 = np.random.RandomState(3)
        pts = Z_df.loc[idx_p][['PC1', 'PC2']].to_numpy()
        if len(pts) > cap:
            pts = pts[rng2.choice(len(pts), cap, replace=False)]
        kde = gaussian_kde(pts.T)
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        xs = np.linspace(x_min, x_max, gridsize)
        ys = np.linspace(y_min, y_max, gridsize)
        xx, yy = np.meshgrid(xs, ys)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        return {'x': xs.tolist(), 'y': ys.tolist(), 'z': zz.tolist()}

    log('  computing KDE grids (Part1, Part2) on PC1 x PC2 -- may take a moment')
    kde1 = kde_grid(idx1)
    kde2 = kde_grid(idx2)

    # matplotlib sanity PNGs
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, (part_name, idx_p, labels_p) in zip(
                axes, [('part1', idx1, labels1), ('part2', idx2, labels2), ('part3', idx3, labels3)]):
            n_sample = min(8000, len(idx_p))
            sub_idx = rng.choice(len(idx_p), n_sample, replace=False)
            zp = Z_df.loc[np.array(idx_p)[sub_idx]]
            lp = np.array(labels_p)[sub_idx]
            sc = ax.scatter(zp['PC1'], zp['PC2'], c=lp, cmap='tab10', s=3, alpha=0.4)
            ax.set_title(f'{part_name}: {len(np.unique(labels_p))} clusters')
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        plt.tight_layout()
        png1 = os.path.join(OUT_PNG_DIR, 'multivariate_kde_clusters.png')
        plt.savefig(png1, dpi=110)
        plt.close(fig)
        log(f'  wrote {png1}')

        fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
        for ax, kg, title in zip(axes2, [kde1, kde2], ['Part1 KDE', 'Part2 KDE']):
            ax.contourf(kg['x'], kg['y'], kg['z'], levels=20, cmap='viridis')
            ax.set_title(title); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        plt.tight_layout()
        png2 = os.path.join(OUT_PNG_DIR, 'multivariate_kde_density.png')
        plt.savefig(png2, dpi=110)
        plt.close(fig2)
        log(f'  wrote {png2}')
    except Exception as e:
        log(f'  PNG generation failed (non-fatal): {e}')

    out = {
        'n_candidate_columns': len(cols),
        'n_components_retained': n_components,
        'cumvar_at_retained': float(cumvar[n_components - 1]),
        'pca_fit_on': 'part1+part2 combined',
        'loadings': loadings_report,
        'clustering_method': 'mean-shift (bin_seeding, auto bandwidth) in retained PCA space, capped fit sample 40000, full-part nearest-center assignment',
        'shape_comparison_part1_vs_part2': shape_comparison,
        'part3_validation': {
            'method': 'nearest-center assignment to Part1-fit mean-shift centers, no refit',
            'relative_sizes': {str(k): float(v) for k, v in size3.items()},
        },
        'dye_enrichment_part1_clusters': enrichment,
        'removal_rate_replication_check_part1_vs_part2': replication_check,
        'population_marker_baseline': pop_baseline,
        'prev_round_overlap_check': {
            'prev_chosen_features': prev_pc['chosen_features'],
            'prev_n_flagged_part1_reported': prev_pc['n_flagged_part1'],
            'prev_n_flagged_part1_rederived_here': int(len(prev_flagged_idx)),
            'prev_flagged_cluster_distribution': overlap_dist,
            'overall_part1_cluster_distribution': overall_dist,
        },
        'scatter_sample': scatter_records,
        'kde_grid_part1': kde1,
        'kde_grid_part2': kde2,
    }

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, default=str)
    log(f'wrote {OUT_JSON}')


if __name__ == '__main__':
    main()
