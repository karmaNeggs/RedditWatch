#!/usr/bin/env python3
"""V3 Stage 1: univariate screening (V3_PLAN.md Sec 7) over account_features.

Per feature: Hartigan dip test for unimodality (diptest package) + Gaussian
mixture with k selected by BIC (k=1..3). A significant dip + BIC-preferred
k>1 says two populations exist before any label is involved -- but this
needs guards, found by inspecting the script's own suspicious output at each
stage (first pass: every feature but one came back "real"; second pass,
after fix (1) below: still 18/19 "bimodal", not credible against the
pilot's 3/7):

1. **Zero/point-mass inflation (Sec 5.2).** n_posts_sample, n_high_tier,
   n_low_tier, username_digit_suffix_len all have >=25% of accounts sitting
   on a single value (mostly 0). A GMM fit to that raw distribution finds a
   near-zero-variance spike component every time -- not real structure, a
   divide-by-near-zero artefact (pooled_std -> 0 makes separation_score
   blow up to 693, 4000, etc). Sec 5.2's prescribed fix -- "hurdle indicator
   + magnitude" -- is applied: point-mass share is reported as its own
   (real, often meaningful) finding, and dip/GMM run only on the remainder.
2. **Census-scale n is a trap for BOTH the dip test AND BIC, not just the
   dip test.** Fix: the ENTIRE screen (dip test and GMM/BIC both) runs on a
   fixed, seeded SCREEN_N=20,000 subsample per feature -- large enough for
   real power, inside diptest's validated range, small enough that BIC
   isn't mechanically rewarding infinitesimal fit gains from census n.
3. **Single-mode point-mass stripping wasn't enough (found 2026-08-06,
   session 2).** Stripping only the single largest tied value still left
   18/19 features "bimodal". Root cause, shared with the bot-marker
   percentile ranks (scripts/v3_botmarker_composite.py): small-denominator
   RATE features (median 2 comments/account) can only take values like
   {0, 0.5, 1} or {0, 1/6, 1/3, ..., 1} -- several distinct fractions each
   carry real mass (not just the mode), and those secondary clumps survive
   a single-mode hurdle and still fool GMM into "finding" extra components
   at the clump locations. Fix: `strip_point_masses` generalizes the hurdle
   to iteratively strip EVERY value carrying >=1% of n (not just the
   largest), before dip/GMM ever see the data. This only fires on
   genuinely discrete-valued features -- a continuous float column has
   ~zero chance of 1% of 20,000+ rows landing on the exact same value by
   chance, so this is a no-op on truly continuous features.
4. **Still needed even after (3): a GMM will always find 2-3 offset
   Gaussians to approximate a skewed-but-genuinely-unimodal shape** --
   that's mechanically how a Gaussian mixture approximates skew, and no
   amount of point-mass stripping fixes it, because there's no discrete
   clump to strip, just smooth monotonic decay (example: mean_comment_score
   raw histogram is one peak near 0-20 decaying smoothly to 4625). Two
   guards: (a) `signed_log1p` transform for signed heavy-tailed features
   (mean_comment_score, median_comment_score, and other score-derived
   features weren't log-transformed before because they can be negative --
   plain log1p breaks on negatives, signed_log1p = sign(x)*log1p(|x|)
   fixes that); (b) `kde_valley_ratio` -- an actual KDE fit to the
   (transformed) data, checked for a real density DIP between the two
   heaviest GMM component means, not just dip-test-p + BIC-k agreement.
   Ratio close to 1 means no visible valley (GMM decorating a monotonic
   slope); a verdict only reaches "REAL CANDIDATE" if the valley is
   genuinely there. This operationalizes the plan's own prescription
   ("needs an actual KDE-valley visual check, not more automated threshold
   tuning") as a checkable number rather than eyeballing every feature."""
import os
import time

import duckdb
import numpy as np
from diptest import diptest
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT, 'data', 'v3', 'analysis', 'v3.duckdb')

SCREEN_N = 20_000
POINT_MASS_THRESHOLD = 0.25          # single-mode reporting threshold (unchanged)
MULTI_MASS_THRESHOLD = 0.01          # generalized stripper threshold, fix (3)
MAX_MASSES_TO_STRIP = 8
MAX_TOTAL_STRIP_SHARE = 0.5          # never strip more than half the data --
                                      # a wide-range count feature (e.g.
                                      # n_posts_sample, cardinality in the
                                      # hundreds) has many small integers
                                      # each individually >=1% of n, and an
                                      # unbounded strip ends up testing
                                      # structure only in a thin tail
                                      # instead of the feature as a whole
DEGENERATE_SEPARATION_CEILING = 20.0
REAL_SEPARATION_FLOOR = 0.5
KDE_VALLEY_CEILING = 0.9             # valley density must be <90% of the lower peak's density
RNG = np.random.RandomState(42)

# (feature, transform: 'log1p' | 'signed_log1p' | 'none', tier2_only?)
FEATURES = [
    ('n_comments_sample', 'log1p', False),
    ('n_posts_sample', 'log1p', False),
    ('n_subs_active', 'log1p', False),
    ('n_threads_active', 'log1p', False),
    ('mean_post_score', 'log1p', False),
    ('mean_comment_score', 'signed_log1p', False),
    ('median_comment_score', 'signed_log1p', False),
    ('score_stddev', 'log1p', False),
    ('controversiality_rate', 'none', False),
    ('is_submitter_rate', 'none', False),
    ('mean_depth', 'none', False),
    ('mean_body_len', 'log1p', False),
    ('account_ordinal', 'log1p', False),
    ('n_high_tier', 'log1p', False),
    ('n_low_tier', 'log1p', False),
    ('subreddit_entropy', 'none', False),
    ('username_char_entropy', 'none', False),
    ('username_digit_suffix_len', 'none', False),
    ('interval_entropy', 'none', True),
    ('burstiness_kimjo', 'none', True),
    ('interval_quantization_rate', 'none', True),
    ('removal_rate', 'none', False),
    ('deleted_later_rate', 'none', False),
    ('observed_span_days', 'log1p', False),
    ('comments_per_day_observed', 'log1p', False),
    ('sample_score_per_day_observed', 'signed_log1p', False),
    ('days_since_first_seen', 'none', False),
    ('comments_per_day_since_first_seen', 'log1p', False),
    ('posts_per_day_since_first_seen', 'log1p', False),
    ('karma_per_day_since_first_seen', 'signed_log1p', False),
    ('repeat_engagement_rate', 'none', False),
    ('own_post_reply_rate', 'none', False),
    ('best_sub_mean_score', 'signed_log1p', False),
    ('worst_sub_mean_score', 'signed_log1p', False),
    ('reception_spread', 'log1p', False),
    ('n_threads_with_repeat', 'log1p', False),
    ('n_own_posts_with_comments', 'log1p', False),
]

MONTHLY_STABILITY_FEATURES = ['score', 'controversiality', 'depth', 'body_len']


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def strip_point_masses(raw, threshold=MULTI_MASS_THRESHOLD, max_strip=MAX_MASSES_TO_STRIP,
                        max_total_share=MAX_TOTAL_STRIP_SHARE):
    """Iteratively strip every value carrying >=threshold share of n, not
    just the single largest mode -- see module docstring fix (3). Capped at
    max_strip distinct values and max_total_share of n so this stays
    targeted at a few dominant discrete masses (what the plan's diagnosis
    actually described -- 0, 0.5, 1, 1/6) rather than decimating a
    wide-range count feature's entire distribution one small integer at a
    time."""
    n = len(raw)
    vals, counts = np.unique(raw, return_counts=True)
    order = np.argsort(-counts)
    stripped = []
    running_share = 0.0
    for idx in order:
        share = counts[idx] / n
        if share < threshold or len(stripped) >= max_strip or running_share + share > max_total_share:
            break
        stripped.append((float(vals[idx]), share))
        running_share += share
    if not stripped:
        return raw, stripped, 0.0
    stripped_vals = np.array([v for v, _ in stripped])
    remainder = raw[~np.isin(raw, stripped_vals)]
    return remainder, stripped, sum(s for _, s in stripped)


def kde_valley_ratio(x, means):
    """Ratio of the KDE density at its lowest point between the two
    component means to the lower of the two means' own densities. <<1
    means a real dip; ~1 means the GMM is decorating a monotonic slope
    with no visible valley (module docstring fix (4))."""
    lo, hi = float(min(means)), float(max(means))
    if hi - lo < 1e-9:
        return None
    try:
        kde = gaussian_kde(x)
    except Exception:
        return None
    grid = np.linspace(lo, hi, 200)
    density = kde(grid)
    d_lo, d_hi = kde(np.array([lo]))[0], kde(np.array([hi]))[0]
    peak_ref = min(d_lo, d_hi)
    if peak_ref < 1e-12:
        return None
    return float(density.min() / peak_ref)


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


def screen_feature(con, feature, transform='log1p', tier2_only=False):
    if isinstance(transform, bool):  # backward-compat for old bool call sites
        transform = 'log1p' if transform else 'none'
    where = 'WHERE has_timing_features' if tier2_only else 'WHERE 1=1'
    raw = con.execute(f'SELECT "{feature}" FROM account_features {where} AND "{feature}" IS NOT NULL').fetchnumpy()[feature]
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if n < 100:
        return {'feature': feature, 'n': n, 'skipped': 'n<100'}

    # legacy single-mode report (unchanged meaning -- the printed pt-mass column)
    vals, counts = np.unique(raw, return_counts=True)
    mode_idx = np.argmax(counts)
    mode_val, mode_share = vals[mode_idx], counts[mode_idx] / n
    point_mass = mode_share >= POINT_MASS_THRESHOLD

    # generalized multi-mass strip actually fed to dip/GMM -- fix (3)
    remainder, stripped_masses, total_stripped_share = strip_point_masses(raw)

    if len(remainder) < 100:
        return {'feature': feature, 'n': n, 'point_mass': point_mass,
                'mode_val': mode_val, 'mode_share': mode_share,
                'stripped_masses': stripped_masses,
                'skipped': 'remainder<100 after stripping point masses'}

    x = remainder
    if transform == 'log1p':
        x = np.log1p(x - x.min()) if x.min() < 0 else np.log1p(x)
    elif transform == 'signed_log1p':
        x = signed_log1p(x)

    # fixed, seeded, validated-size subsample -- used for BOTH the dip test
    # and the GMM/BIC fit, not just the dip test (see module docstring)
    x_screen = x if len(x) <= SCREEN_N else RNG.choice(x, SCREEN_N, replace=False)

    dip_stat, dip_p = diptest(x_screen)
    k, gmm = fit_gmm_bic(x_screen)
    sep = separation_score(gmm, len(x_screen)) if k > 1 else None
    minority_mass, means_str, valley_ratio = None, None, None
    if k > 1 and sep is not None:
        minority_mass = float(np.min(gmm.weights_))
        means_str = ','.join(f'{m:.2f}' for m in sorted(gmm.means_.flatten()))
        heavy = sorted(range(len(gmm.weights_)), key=lambda i: -gmm.weights_[i])[:2]
        two_means = sorted(gmm.means_.flatten()[heavy])
        valley_ratio = kde_valley_ratio(x_screen, two_means)

    return {
        'feature': feature, 'n': n, 'n_screen': len(x_screen), 'point_mass': point_mass,
        'mode_val': mode_val, 'mode_share': mode_share,
        'stripped_masses': stripped_masses, 'total_stripped_share': total_stripped_share,
        'n_remainder': len(remainder), 'transform': transform, 'tier2_only': tier2_only,
        'dip_stat': dip_stat, 'dip_p': dip_p,
        'bic_k': k, 'minority_mass': minority_mass, 'separation_score': sep,
        'component_means': means_str, 'kde_valley_ratio': valley_ratio,
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
    if r['kde_valley_ratio'] is None:
        return 'k>1 + dip + separated BUT KDE valley check did not run -- unverified, treat as not real'
    if r['kde_valley_ratio'] > KDE_VALLEY_CEILING:
        return f'k>1 + dip + separated BUT no visible KDE dip (valley/peak={r["kde_valley_ratio"]:.2f}) -- GMM decorating skew, not a real second mode -- discard'
    return f'*** REAL CANDIDATE: dip-significant + well-separated + KDE valley confirmed (valley/peak={r["kde_valley_ratio"]:.2f}) ***'


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

    print(f'=== STAGE 1: UNIVARIATE SCREENING (multi-mass hurdle + KDE valley check, n_screen<={SCREEN_N}) ===\n')
    print(f'{"feature":<32} {"n":>8} {"n_scr":>6} {"strip%":>7} {"dip_p":>8} {"k":>3} {"minor%":>8} {"sep":>8} {"valley":>7}  verdict')
    real_candidates = []
    for feature, transform, tier2_only in FEATURES:
        r = screen_feature(con, feature, transform, tier2_only)
        v = verdict_for(r)
        if r.get('skipped'):
            print(f'{feature:<32} {r["n"]:>8}  -- {v}')
            continue
        strip_pct = f'{100*r["total_stripped_share"]:.0f}%'
        dip_p = f'{r["dip_p"]:.4f}'
        minority_pct = f'{100*r["minority_mass"]:.1f}%' if r['minority_mass'] is not None else '--'
        sep = f'{r["separation_score"]:.2f}' if r['separation_score'] is not None else '--'
        valley = f'{r["kde_valley_ratio"]:.2f}' if r['kde_valley_ratio'] is not None else '--'
        print(f'{feature:<32} {r["n"]:>8} {r["n_screen"]:>6} {strip_pct:>7} {dip_p:>8} {r["bic_k"]:>3} {minority_pct:>8} {sep:>8} {valley:>7}  {v}')
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
