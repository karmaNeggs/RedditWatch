#!/usr/bin/env python3
"""
Reddit Bot Analysis V2 — Referee: pick the weight scheme against real evidence

Every weighting scheme in this pipeline so far (CV-calibrated, PCA) was derived
from the signals' own internal statistics — never checked against anything
outside the scoring system itself. scale_weak_labels.py produces the first
external check: an activity-weighted per-subreddit "gone rate" (current
suspended/deleted status of a random account sample) — a weak, noisy, but real
proxy for "this subreddit has more bad accounts than that one."

This script recombines each subreddit's already-computed raw component scores
(account/ring/engagement/temporal/distribution/network — stored per month in
output/v2/analysis_*.json) under each candidate weight vector in
reports/findings.json, averages each subreddit's final_score across all
available months, and Spearman-correlates that against the gone-rate. Whichever
candidate correlates best becomes `live_weights` — what analyze_data_v2.py
actually uses (see _load_weights). Thin subs (<5 sampled accounts) are excluded
from the primary comparison per scale_weak_labels.py's own caveat.

Usage:
  python3 scripts/referee_weights.py
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT        = Path(__file__).parent.parent
OUTPUT_DIR  = ROOT / 'output' / 'v2'
FINDINGS    = ROOT / 'reports' / 'findings.json'

COMPONENT_SCORE_KEY = {
    'account':      'account_score',
    'ring':         'ring_score',
    'engagement':   'engagement_score',
    'temporal':     'temporal_score',
    'distribution': 'distribution_score',
    'network':      'network_score',
}
ANALYSIS_KEY = {
    'account':      'account_analysis',
    'ring':         'ring_analysis',
    'engagement':   'engagement_analysis',
    'temporal':     'temporal_analysis',
    'distribution': 'distribution_analysis',
    'network':      'network_analysis',
}


def load_gone_rates():
    path = OUTPUT_DIR / 'subreddit_gone_rates.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} not found — run scale_weak_labels.py first')
    with open(path) as f:
        d = json.load(f)
    rates = {r['subreddit']: r['gone_rate'] for r in d['subreddit_gone_rates']}
    thin  = set(d.get('thin_subs_low_confidence', []))
    return rates, thin


def load_latest_per_month():
    """Latest analysis_<month>_<ts>.json per month (skips analysis_latest.json)."""
    files = sorted(glob.glob(str(OUTPUT_DIR / 'analysis_*.json')))
    by_month = {}
    for f in files:
        if 'latest' in Path(f).name:
            continue
        with open(f) as fh:
            raw = json.load(fh)
        month = raw.get('month')
        if not month:
            continue
        # keep newest run per month (filenames sort by timestamp suffix)
        by_month[month] = raw
    return by_month


def score_under_weights(by_month, weights):
    """
    Per-sub average final_score across all months, using `weights` on stored raw
    components. Defensive on missing keys (older analysis_*.json files scored before
    a component existed, e.g. network_score) — skips that sub-month rather than
    crashing, so a stale file degrades the average instead of aborting the referee.
    Callers should still re-score every month with current code before running this,
    so skips should be rare/zero in practice.
    """
    active = [c for c in weights if c in COMPONENT_SCORE_KEY]
    per_sub_scores = {}
    skipped = 0
    for month, raw in by_month.items():
        analyses = {c: raw.get(ANALYSIS_KEY[c], {}) for c in active}
        subs = set.intersection(*[set(analyses[c].keys()) for c in active]) if active else set()
        for sub in subs:
            try:
                final = sum(analyses[c][sub][COMPONENT_SCORE_KEY[c]] * weights[c] for c in active)
            except (KeyError, TypeError):
                skipped += 1
                continue
            per_sub_scores.setdefault(sub, []).append(final)
    if skipped:
        print(f'    (skipped {skipped} sub-months missing a required component score — '
              f'stale analysis file? re-score all months before running the referee)')
    return {sub: float(np.mean(vals)) for sub, vals in per_sub_scores.items()}


def evaluate(name, weights, by_month, gone_rates, thin):
    if not weights:
        return None
    scores = score_under_weights(by_month, weights)
    subs = [s for s in scores if s in gone_rates and s not in thin]
    if len(subs) < 8:
        return {'name': name, 'weights': weights, 'n_subs': len(subs), 'spearman_r': None, 'p_value': None}
    x = [scores[s] for s in subs]
    y = [gone_rates[s] for s in subs]
    r, p = spearmanr(x, y)
    return {'name': name, 'weights': weights, 'n_subs': len(subs),
            'spearman_r': round(float(r), 3), 'p_value': round(float(p), 3)}


def main():
    print('\n' + '=' * 70)
    print('REFEREE: CHOOSING A WEIGHT SCHEME AGAINST THE WEAK-LABEL GROUND TRUTH')
    print('=' * 70)

    gone_rates, thin = load_gone_rates()
    print(f'\nGone-rate ground truth: {len(gone_rates)} subs ({len(thin)} flagged thin, excluded)')

    by_month = load_latest_per_month()
    print(f'Loaded component scores for {len(by_month)} months')

    with open(FINDINGS) as f:
        findings = json.load(f)

    candidates = [
        ('calibrated_weights (CV, 5-comp, current live default)', findings.get('calibrated_weights', {})),
        ('pca_weights (PCA, 5-comp)',                              findings.get('pca_weights', {})),
        ('calibrated_weights_6comp (CV, 6-comp w/ network)',       findings.get('calibrated_weights_6comp', {})),
        ('pca_weights_6comp (PCA, 6-comp w/ network)',             findings.get('pca_weights_6comp', {})),
    ]

    results = []
    print('\nEvaluating candidates (Spearman r vs. subreddit gone-rate):')
    for name, w in candidates:
        res = evaluate(name, w, by_month, gone_rates, thin)
        if res is None:
            print(f'  {name:<55} — missing from findings.json, skipped')
            continue
        results.append(res)
        r_str = f"r={res['spearman_r']:+.3f} (p={res['p_value']:.3f}, n={res['n_subs']})" if res['spearman_r'] is not None else 'insufficient subs'
        print(f'  {name:<55} {r_str}')

    valid = [r for r in results if r['spearman_r'] is not None]
    if not valid:
        print('\nNo candidate could be evaluated (insufficient overlapping subs). Aborting — live_weights not set.')
        return

    winner = max(valid, key=lambda r: r['spearman_r'])
    print(f"\nWinner: {winner['name']}  (r={winner['spearman_r']:+.3f})")

    findings['live_weights'] = winner['weights']
    findings['live_weights_selection'] = {
        'method': 'Spearman correlation of per-subreddit avg final_score (under each candidate '
                  'weight scheme) against an activity-weighted account gone-rate from a random '
                  f"{json.load(open(OUTPUT_DIR / 'subreddit_gone_rates.json'))['n_sampled']}-account sample "
                  '(scale_weak_labels.py). Highest correlation wins.',
        'winner': winner['name'],
        'winner_spearman_r': winner['spearman_r'],
        'winner_p_value': winner['p_value'],
        'all_candidates': [
            {'name': r['name'], 'spearman_r': r['spearman_r'], 'p_value': r['p_value'], 'n_subs': r['n_subs']}
            for r in results
        ],
        'caveat': 'n≈25 subreddits, weak/noisy label, single point-in-time snapshot — this is real '
                  'evidence, not proof. p-values here are exploratory, not confirmatory.',
    }

    with open(FINDINGS, 'w') as f:
        json.dump(findings, f, indent=2, default=str)

    print(f'\nWrote live_weights to {FINDINGS}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
