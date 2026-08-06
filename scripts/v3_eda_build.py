#!/usr/bin/env python3
"""Build the V3 account-feature EDA page: histograms (raw + transformed) per
feature, Spearman correlation matrix, 2D density grids for selected pairs.
Computes everything from data/v3/analysis/v3.duckdb and injects it as JSON
into scripts/v3_eda_template.html, writing docs/v3-research/eda/index.html.

Re-run after any account_features rebuild (scripts/v3_account_features.py)
to keep the page in sync. Only summary statistics are embedded -- the
347,886 raw rows never leave DuckDB."""
import json
import os

import duckdb
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')
TEMPLATE = os.path.join(ROOT, 'scripts', 'v3_eda_template.html')
OUT = os.path.join(ROOT, 'docs', 'v3-research', 'eda', 'index.html')

# (feature, family, transform)  transform in {'none','log1p','signed_log'}
FEATURES = [
    ('n_comments_sample', 'activity', 'log1p'),
    ('n_posts_sample', 'activity', 'log1p'),
    ('n_subs_active', 'activity', 'log1p'),
    ('n_threads_active', 'activity', 'log1p'),
    ('removal_rate', 'reception', 'none'),
    ('deleted_later_rate', 'reception', 'none'),
    ('mean_comment_score', 'reception', 'signed_log'),
    ('median_comment_score', 'reception', 'signed_log'),
    ('score_stddev', 'reception', 'log1p'),
    ('controversiality_rate', 'reception', 'none'),
    ('is_submitter_rate', 'reception', 'none'),
    ('mean_depth', 'reception', 'none'),
    ('mean_body_len', 'reception', 'log1p'),
    ('n_high_tier', 'footprint', 'log1p'),
    ('n_low_tier', 'footprint', 'log1p'),
    ('subreddit_entropy', 'footprint', 'none'),
    ('account_ordinal', 'provenance', 'none'),
    ('username_char_entropy', 'provenance', 'none'),
    ('username_digit_suffix_len', 'provenance', 'none'),
    ('interval_entropy', 'timing (n>=5 only)', 'none'),
    ('burstiness_kimjo', 'timing (n>=5 only)', 'none'),
    ('interval_quantization_rate', 'timing (n>=5 only)', 'none'),
    ('days_since_first_seen', 'age (proxy, see note)', 'none'),
    ('comments_per_day_since_first_seen', 'age (proxy, see note)', 'log1p'),
    ('posts_per_day_since_first_seen', 'age (proxy, see note)', 'log1p'),
    ('karma_per_day_since_first_seen', 'age (proxy, see note)', 'signed_log'),
    ('repeat_engagement_rate', 'engagement', 'none'),
    ('own_post_reply_rate', 'engagement', 'none'),
    ('reception_spread', 'engagement', 'log1p'),
    ('n_subs_rejected_but_returned', 'engagement', 'log1p'),
    ('thin_history_score', 'bot-marker (unsupervised, see note)', 'none'),
    ('karma_extremeness', 'bot-marker (unsupervised, see note)', 'log1p'),
    ('karma_per_post_extremeness', 'bot-marker (unsupervised, see note)', 'log1p'),
    ('botmarker_composite', 'bot-marker (unsupervised, see note)', 'none'),
]

DENSITY_PAIRS = [
    ('n_comments_sample', 'log1p', 'removal_rate', 'none',
     'Sample frequency vs. removal rate — does thin history mean higher removal?'),
    ('account_ordinal', 'none', 'subreddit_entropy', 'none',
     'Registration cohort (raw base36 ordinal) vs. footprint breadth'),
    ('mean_comment_score', 'signed_log', 'controversiality_rate', 'none',
     'Typical score received vs. controversiality rate'),
    ('reception_spread', 'log1p', 'controversiality_rate', 'none',
     'Best-sub-minus-worst-sub reception gap vs. controversiality — the silo-mismatch check'),
]


def transform(x, kind):
    if kind == 'log1p':
        return np.log1p(x)
    if kind == 'signed_log':
        return np.sign(x) * np.log1p(np.abs(x))
    return x


RISK_METRIC = 'removal_rate'  # strongest single account-level signal found so far (Sec 10.4)
RISK_MIN_BIN_N = 20  # below this, a bin's mean risk is too noisy to color by -- render neutral


def risk_by_bin(values, risk, edges):
    """Per-bin mean of RISK_METRIC, None where the bin has < RISK_MIN_BIN_N
    accounts (a 2-account bin at 100% removal_rate would paint bright red
    for no statistical reason otherwise)."""
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
    means, counts = [], []
    for b in range(len(edges) - 1):
        sel = risk[idx == b]
        counts.append(int(len(sel)))
        means.append(float(sel.mean()) if len(sel) >= RISK_MIN_BIN_N else None)
    return means, counts


def histogram(con, feature, family, kind, gate_timing):
    where = 'WHERE has_timing_features' if gate_timing else 'WHERE 1=1'
    df = con.execute(
        f'SELECT "{feature}" AS val, "{RISK_METRIC}" AS risk FROM account_features {where} AND "{feature}" IS NOT NULL'
    ).fetchdf()
    raw = df['val'].values.astype(float)
    risk = df['risk'].values.astype(float)
    n = len(raw)
    lo_r, hi_r = np.percentile(raw, [0.5, 99.5])
    clip_mask = (raw >= lo_r) & (raw <= hi_r)
    raw_clip, risk_clip = raw[clip_mask], risk[clip_mask]
    hc, he = np.histogram(raw_clip, bins=28)
    risk_means, risk_ns = risk_by_bin(raw_clip, risk_clip, he)

    out = {
        'feature': feature, 'family': family, 'transform': kind, 'n': int(n),
        'raw': {'counts': hc.tolist(), 'edges': he.tolist(), 'risk_mean': risk_means, 'risk_n': risk_ns},
        'stats': {
            'min': float(raw.min()), 'p10': float(np.percentile(raw, 10)),
            'median': float(np.median(raw)), 'p90': float(np.percentile(raw, 90)),
            'max': float(raw.max()), 'mean': float(raw.mean()),
        },
    }
    if kind != 'none':
        tx = transform(raw, kind)
        lo_t, hi_t = np.percentile(tx, [0.5, 99.5])
        tmask = (tx >= lo_t) & (tx <= hi_t)
        tx_clip, trisk_clip = tx[tmask], risk[tmask]
        tc, te = np.histogram(tx_clip, bins=28)
        trisk_means, trisk_ns = risk_by_bin(tx_clip, trisk_clip, te)
        out['transformed'] = {'counts': tc.tolist(), 'edges': te.tolist(), 'label': kind,
                               'risk_mean': trisk_means, 'risk_n': trisk_ns}

    vals, counts = np.unique(raw, return_counts=True)
    mode_i = np.argmax(counts)
    mode_share = counts[mode_i] / n
    if mode_share >= 0.20:
        out['point_mass'] = {'value': float(vals[mode_i]), 'share': float(mode_share)}
    return out


def build_data(con):
    print('Building histograms...')
    hists = [histogram(con, f, fam, k, fam.startswith('timing')) for f, fam, k in FEATURES]

    print('Building Spearman correlation matrix...')
    corr_features = [f for f, fam, k in FEATURES if not fam.startswith('timing')]
    cols = ', '.join(f'"{f}"' for f in corr_features)
    df = con.execute(f'SELECT {cols} FROM account_features').fetchdf()
    n_feat = len(corr_features)
    corr = np.eye(n_feat)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            sub = df[[corr_features[i], corr_features[j]]].dropna()
            if len(sub) > 100:
                rho, _ = spearmanr(sub.iloc[:, 0], sub.iloc[:, 1])
                corr[i, j] = corr[j, i] = rho if not np.isnan(rho) else 0.0

    print('Building 2D density panels...')
    density_panels = []
    for fx, tx_x, fy, tx_y, caption in DENSITY_PAIRS:
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
        density_panels.append({
            'x_feature': fx, 'x_transform': tx_x, 'y_feature': fy, 'y_transform': tx_y,
            'caption': caption, 'n': int(mask.sum()),
            'grid': np.log1p(h.T).tolist(),
            'x_edges': xe.tolist(), 'y_edges': ye.tolist(),
            'x_max_count': int(h.max()),
        })

    # Scale calibrated to the actual spread of BIN-level means, not raw account-level
    # percentiles -- individual accounts have a much heavier tail (a few near 100%)
    # than bin averages ever reach, so an account-percentile-based max compressed every
    # real per-bin difference into "barely distinguishable pale pink" (found by checking
    # actual bin means after the first version rendered suspiciously flat).
    all_bin_means = []
    for h in hists:
        all_bin_means.extend(m for m in h['raw'].get('risk_mean', []) if m is not None)
        if 'transformed' in h:
            all_bin_means.extend(m for m in h['transformed'].get('risk_mean', []) if m is not None)
    risk_p95 = float(np.percentile(all_bin_means, 95)) if all_bin_means else 0.3
    risk_pop_mean = float(con.execute(f'SELECT avg("{RISK_METRIC}") FROM account_features').fetchone()[0])

    print('Building bot-marker top-1% comparison...')
    marker_rows = []
    pop_removal = risk_pop_mean
    for by in ['botmarker_composite', 'removal_rate', 'reception_spread', 'karma_extremeness', 'karma_per_post_extremeness']:
        r = con.execute(f"""
            SELECT count(*) n, avg(removal_rate) removal_rate, avg(controversiality_rate) controversiality,
                   avg(deleted_later_rate) deleted_later, avg(n_comments_sample) mean_n_comments
            FROM account_features
            WHERE "{by}" >= (SELECT quantile_cont("{by}", 0.99) FROM account_features WHERE "{by}" IS NOT NULL)
        """).fetchone()
        marker_rows.append({
            'marker': by, 'n': int(r[0]), 'removal_rate': float(r[1]) if r[1] is not None else None,
            'controversiality': float(r[2]) if r[2] is not None else None,
            'deleted_later': float(r[3]) if r[3] is not None else None,
            'mean_n_comments': float(r[4]) if r[4] is not None else None,
        })

    return {
        'meta': {
            'n_accounts': int(con.execute('SELECT count(*) FROM account_features').fetchone()[0]),
            'n_timing': int(con.execute('SELECT count(*) FROM account_features WHERE has_timing_features').fetchone()[0]),
            'risk_metric': RISK_METRIC,
            'risk_scale_max': risk_p95,
            'risk_pop_mean': risk_pop_mean,
        },
        'histograms': hists,
        'corr_features': corr_features,
        'corr_matrix': corr.tolist(),
        'density_panels': density_panels,
        'marker_top1pct': marker_rows,
    }


def main():
    con = duckdb.connect(DB_PATH, read_only=True)
    data = build_data(con)
    con.close()

    with open(TEMPLATE) as f:
        tpl = f.read()
    out_html = tpl.replace('__EDA_JSON__', json.dumps(data))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(out_html)
    print(f'\nWrote {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)')


if __name__ == '__main__':
    main()
