#!/usr/bin/env python3
"""V3 Stage 1: univariate screening (V3_PLAN.md Sec 7) over account_features.

Per feature: Hartigan dip test for unimodality (diptest package) + Gaussian
mixture with k selected by BIC (k=1..3). A significant dip + BIC-preferred
k>1 says two populations exist before any label is involved -- but this
needs two guards the first version of this script skipped, both found by
inspecting its own suspicious output (every feature but one came back
"real", several with impossible separation scores in the hundreds):

1. **Zero/point-mass inflation (Sec 5.2).** n_posts_sample, n_high_tier,
   n_low_tier, username_digit_suffix_len all have >=25% of accounts sitting
   on a single value (mostly 0). A GMM fit to that raw distribution finds a
   near-zero-variance spike component every time -- not real structure, a
   divide-by-near-zero artefact (pooled_std -> 0 makes separation_score
   blow up to 693, 4000, etc). Sec 5.2's prescribed fix -- "hurdle indicator
   + magnitude" -- is applied here: point-mass share is reported as its own
   (real, often meaningful) finding, and dip/GMM run only on the remaining
   non-point-mass values.
2. **Census-scale n is a trap for BOTH the dip test AND BIC, not just the
   dip test.** First pass capped only the dip-test subsample and still came
   back with 18/19 features "real" -- not credible on its face given the
   pilot's own 450-account check found real structure in just 3/7. The
   separation_score guard from point (1) doesn't fix this: a GMM will
   always find two "well-separated" Gaussian components to approximate any
   skewed-but-genuinely-unimodal shape (that's literally how a Gaussian
   mixture approximates skew), and BIC's log(n) penalty means even a
   trivial fit improvement from an extra component reads as "significant"
   once n is in the hundreds of thousands. This generalizes Sec 8's "at
   census n, p-values stop discriminating: rank by effect size" to BIC
   model selection, which has the identical large-n pathology. Fix: the
   ENTIRE screen (dip test and GMM/BIC both) runs on a fixed, seeded
   SCREEN_N=20,000 subsample per feature -- large enough for real power,
   inside diptest's validated range, small enough that BIC isn't
   mechanically rewarding infinitesimal fit gains. Component means are
   reported explicitly (not just the separation ratio) so a result can be
   eyeballed the way the pilot eyeballed burstiness's near-identical
   0.273-vs-0.299 means and called it noise."""
import os
import time

import duckdb
import numpy as np
from diptest import diptest
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

SCREEN_N = 20_000
POINT_MASS_THRESHOLD = 0.25
DEGENERATE_SEPARATION_CEILING = 20.0
REAL_SEPARATION_FLOOR = 0.5
RNG = np.random.RandomState(42)

# (feature, log1p?, tier2_only?)
FEATURES = [
    ('n_comments_sample', True, False),
    ('n_posts_sample', True, False),
    ('n_subs_active', True, False),
    ('mean_comment_score', False, False),
    ('median_comment_score', False, False),
    ('score_stddev', True, False),
    ('controversiality_rate', False, False),
    ('is_submitter_rate', False, False),
    ('mean_depth', False, False),
    ('mean_body_len', True, False),
    ('account_ordinal', False, False),
    ('n_high_tier', True, False),
    ('n_low_tier', True, False),
    ('subreddit_entropy', False, False),
    ('username_char_entropy', False, False),
    ('username_digit_suffix_len', False, False),
    ('interval_entropy', False, True),
    ('burstiness_kimjo', False, True),
    ('interval_quantization_rate', False, True),
]

MONTHLY_STABILITY_FEATURES = ['score', 'controversiality', 'depth', 'body_len']


def fit_gmm_bic(x):
    x = x.reshape(-1, 1)
    best = None
    for k in (1, 2, 3):
        gmm = GaussianMixture(n_components=k, random_state=0, n_init=3).fit(x)
        bic = gmm.bic(x)
        if best is None or bic < best[0]:
            best = (bic, k, gmm)
    _, k, gmm = best
    return k, gmm


def separation_score(gmm, n_total):
    """|mean_i - mean_j| / pooled_std for the two heaviest components, with
    a minimum-weight guard so a near-empty spike component can't dominate."""
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_
    valid = [i for i in range(len(weights)) if weights[i] * n_total >= 30]
    if len(valid) < 2:
        return None
    order = sorted(valid, key=lambda i: -weights[i])[:2]
    i, j = order
    pooled = np.sqrt((stds[i] ** 2 + stds[j] ** 2) / 2)
    if pooled < 1e-9:
        return None
    return abs(means[i] - means[j]) / pooled


def screen_feature(con, feature, do_log1p, tier2_only):
    where = 'WHERE has_timing_features' if tier2_only else 'WHERE 1=1'
    raw = con.execute(f'SELECT "{feature}" FROM account_features {where} AND "{feature}" IS NOT NULL').fetchnumpy()[feature]
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if n < 100:
        return {'feature': feature, 'n': n, 'skipped': 'n<100'}

    # point-mass / hurdle check on the RAW (untransformed) values
    vals, counts = np.unique(raw, return_counts=True)
    mode_idx = np.argmax(counts)
    mode_val, mode_share = vals[mode_idx], counts[mode_idx] / n
    point_mass = mode_share >= POINT_MASS_THRESHOLD
    remainder = raw[raw != mode_val] if point_mass else raw

    if len(remainder) < 100:
        return {'feature': feature, 'n': n, 'point_mass': point_mass,
                'mode_val': mode_val, 'mode_share': mode_share,
                'skipped': 'remainder<100 after removing point mass'}

    x = remainder
    if do_log1p:
        x = np.log1p(x - x.min()) if x.min() < 0 else np.log1p(x)

    # fixed, seeded, validated-size subsample -- used for BOTH the dip test
    # and the GMM/BIC fit, not just the dip test (see module docstring)
    x_screen = x if len(x) <= SCREEN_N else RNG.choice(x, SCREEN_N, replace=False)

    dip_stat, dip_p = diptest(x_screen)
    k, gmm = fit_gmm_bic(x_screen)
    sep = separation_score(gmm, len(x_screen)) if k > 1 else None
    minority_mass, means_str = None, None
    if k > 1 and sep is not None:
        minority_mass = float(np.min(gmm.weights_))
        means_str = ','.join(f'{m:.2f}' for m in sorted(gmm.means_.flatten()))

    return {
        'feature': feature, 'n': n, 'n_screen': len(x_screen), 'point_mass': point_mass,
        'mode_val': mode_val, 'mode_share': mode_share,
        'n_remainder': len(remainder), 'log1p': do_log1p, 'tier2_only': tier2_only,
        'dip_stat': dip_stat, 'dip_p': dip_p,
        'bic_k': k, 'minority_mass': minority_mass, 'separation_score': sep,
        'component_means': means_str,
        'skipped': None,
    }


def verdict_for(r):
    if r.get('skipped'):
        return f'skipped ({r["skipped"]})'
    if r['bic_k'] == 1:
        return 'unimodal (after removing point mass, if any)'
    if r['separation_score'] is None:
        return 'k>1 but all extra components near-empty -- discard'
    if r['separation_score'] > DEGENERATE_SEPARATION_CEILING:
        return f'DEGENERATE (sep={r["separation_score"]:.0f}, near-zero-variance component) -- discard, not a real finding'
    if r['dip_p'] > 0.05:
        return 'BIC k>1 but dip test not significant -- likely noise'
    if r['separation_score'] < REAL_SEPARATION_FLOOR:
        return 'k>1 + dip significant BUT weak separation -- pilot-style BIC overfit, treat as unimodal'
    return '*** REAL CANDIDATE: dip-significant + well-separated (0.5-20) k>1 ***'


def monthly_stability(con, feature):
    df = con.execute(f"""
        SELECT month, avg("{feature}") AS m, count(*) AS n
        FROM commenters_clean
        GROUP BY month ORDER BY month
    """).fetchdf()
    if len(df) < 6:
        return None
    x = np.arange(len(df))
    y = df['m'].values
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pct_range = (y.max() - y.min()) / abs(y.mean()) if y.mean() != 0 else None
    return {'feature': feature, 'n_months': len(df), 'slope': slope, 'r2': r2,
            'range_pct_of_mean': pct_range}


def main():
    t0 = time.time()
    con = duckdb.connect(DB_PATH, read_only=True)

    print(f'=== STAGE 1: UNIVARIATE SCREENING (point-mass-aware, n_screen<={SCREEN_N}) ===\n')
    print(f'{"feature":<28} {"n":>8} {"n_scr":>6} {"pt-mass":>8} {"dip_p":>8} {"k":>3} {"minor%":>8} {"sep":>8}  component_means  verdict')
    real_candidates = []
    for feature, do_log1p, tier2_only in FEATURES:
        r = screen_feature(con, feature, do_log1p, tier2_only)
        v = verdict_for(r)
        if r.get('skipped'):
            print(f'{feature:<28} {r["n"]:>8}  -- {v}')
            continue
        pm = f'{100*r["mode_share"]:.0f}%@{r["mode_val"]:g}' if r['point_mass'] else '--'
        dip_p = f'{r["dip_p"]:.4f}'
        minority_pct = f'{100*r["minority_mass"]:.1f}%' if r['minority_mass'] is not None else '--'
        sep = f'{r["separation_score"]:.2f}' if r['separation_score'] is not None else '--'
        means = r.get('component_means') or '--'
        print(f'{feature:<28} {r["n"]:>8} {r["n_screen"]:>6} {pm:>8} {dip_p:>8} {r["bic_k"]:>3} {minority_pct:>8} {sep:>8}  {means:<20} {v}')
        if v.startswith('*** REAL'):
            real_candidates.append(feature)

    print(f'\nReal candidates surviving all guards: {real_candidates or "(none)"}')

    print('\n=== MONTHLY DISTRIBUTIONAL STABILITY (comment-level features) ===\n')
    print(f'{"feature":<15} {"months":>7} {"slope":>12} {"R2":>7} {"range/mean":>11}  verdict')
    for feature in MONTHLY_STABILITY_FEATURES:
        m = monthly_stability(con, feature)
        if m is None:
            continue
        drift_flag = 'DRIFTING (V2-style spurious trend risk)' if (m['r2'] > 0.5 and m['range_pct_of_mean'] and m['range_pct_of_mean'] > 0.2) else 'stable'
        rp = f'{m["range_pct_of_mean"]:.2f}' if m['range_pct_of_mean'] is not None else '--'
        print(f'{m["feature"]:<15} {m["n_months"]:>7} {m["slope"]:>12.5f} {m["r2"]:>7.2f} {rp:>11}  {drift_flag}')

    con.close()
    print(f'\nDONE in {time.time()-t0:.1f}s.')


if __name__ == '__main__':
    main()
