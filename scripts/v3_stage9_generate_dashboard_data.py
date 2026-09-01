"""
V3 Stage 9 — generate docs/data_v3/*.json for the Bot & Spam Compass dashboard
(docs/bot-spam-compass.html), from output/v3/subreddit_bot_prevalence_mom.csv.

Mirrors V2's generate_site.py conventions exactly (same severity-band percentile
method: moderate=P50, high=P80, critical=P95 of the observed rollup distribution;
same history.json + per-month {month}.json split) so the two dashboards share one
visual and data language.

Run via scripts/v3_stage8_monthly_refresh.py (calls this automatically), or
standalone after re-running that script.
"""
import datetime as dt
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'output/v3/subreddit_bot_prevalence_mom.csv'
DATA_DIR = ROOT / 'docs/data_v3'
COMPASS = ROOT / 'docs/bot-spam-compass.html'
INDEX = ROOT / 'docs/index.html'
BASELINE = ROOT / 'output/v3/severity_baseline.json'
# Bump ONLY for a deliberate methodology change. A bump starts a new vintage directory and
# leaves the previous series in place; it never edits already-published month files.
SERIES_VERSION = '1.1.0'
VINTAGE_DIR = DATA_DIR / f'v{SERIES_VERSION}'


def embed_into_index(history):
    """Embed the ecosystem-wide monthly trend (history only, no per-subreddit detail)
    into index.html so the Report page's trend chart works from file://, a local
    server, or GitHub Pages alike -- same reasoning as embed_into_compass."""
    payload = json.dumps(history)
    html = INDEX.read_text()
    html, n = re.subn(r'<script id="embedded-trend" type="application/json">.*?</script>',
                       lambda _: f'<script id="embedded-trend" type="application/json">{payload}</script>',
                       html, flags=re.S)
    if n:
        INDEX.write_text(html)
        print(f'Embedded trend data into {INDEX.relative_to(ROOT)}.')


def embed_into_compass(history, month_docs_full):
    """Embed all history+month data directly into bot-spam-compass.html so the
    page works from file://, a local server, or GitHub Pages alike -- no fetch()
    of external JSON, which silently fails (blank page) when opened via file://."""
    payload = json.dumps({'history': history, 'months': month_docs_full})
    html = COMPASS.read_text()
    html = re.sub(r'<script id="embedded-data" type="application/json">.*?</script>',
                  lambda _: f'<script id="embedded-data" type="application/json">{payload}</script>',
                  html, flags=re.S)
    COMPASS.write_text(html)
    print(f'Embedded data directly into {COMPASS.relative_to(ROOT)}.')


def severity_bands(dist):
    if len(dist) < 20:
        return {'moderate': 20.0, 'high': 40.0, 'critical': 70.0}
    p50, p80, p95 = (float(np.percentile(dist, q)) for q in (50, 80, 95))
    return {'moderate': round(p50, 2), 'high': round(p80, 2), 'critical': round(p95, 2)}


def load_or_freeze_bands(df, roles):
    """Severity bands are FROZEN to a declared baseline, not recomputed each refresh.

    Recomputing them meant two things, both bad. (1) Drift: a subreddit's published severity
    could change with no change in its own number -- 3.4% of published labels flipped on the
    2026-08-22 refresh from band movement alone. (2) Worse, self-normalization: bands defined as
    P50/P80/P95 of the observed distribution mean exactly 50%/20%/5% of subreddit-months land in
    moderate/high/critical *by construction, forever*. If ecosystem-wide bot prevalence doubled,
    the bands would double with it and the dashboard would look identical -- a relative ranking
    presented to viewers as an absolute severity scale.

    First run writes the baseline from the current distribution and records which months defined
    it; every later run loads it verbatim. Delete the file (or edit it deliberately) to
    re-baseline -- that is a methodology change and should be a visible commit, not a side effect
    of a monthly run."""
    if BASELINE.exists():
        payload = json.loads(BASELINE.read_text())
        print(f"Loaded frozen severity baseline {payload['baseline_version']} "
              f"({payload['baseline_window']}, n={payload['n_subreddit_months']}).")
        return payload['bands_by_role'], payload

    bands_by_role = {}
    for role in roles:
        col = f'{role}_pct_high_risk'
        dist = df[col].dropna().values if col in df.columns else df['pct_high_risk_of_scored'].values
        bands_by_role[role] = severity_bands(dist)
    months = sorted(df['month'].unique())
    payload = {
        'baseline_version': '1.1.0',
        'baseline_window': f'{months[0]}..{months[-1]}',
        'n_subreddit_months': int(len(df)),
        'method': 'P50/P80/P95 of the pct_high_risk distribution over the baseline window, '
                  'computed once and frozen. Not recomputed per refresh.',
        'bands_by_role': bands_by_role,
    }
    BASELINE.write_text(json.dumps(payload, indent=2))
    print(f"Froze severity baseline 1.1.0 from {payload['baseline_window']} "
          f"(n={payload['n_subreddit_months']}) -> {BASELINE.relative_to(ROOT)}")
    return bands_by_role, payload


def get_severity(score, bands):
    if score >= bands['critical']: return 'critical'
    if score >= bands['high']: return 'high'
    if score >= bands['moderate']: return 'moderate'
    return 'low'


ROLES = ['combined', 'poster', 'commenter']


def role_block(row, role, bands_by_role):
    pct = row.get(f'{role}_pct_high_risk')
    if pd.isna(pct):
        return {'pct_high_risk': None, 'mean_bot_score': None, 'coverage_pct': None, 'n': 0, 'severity': None}
    mean_score = row.get(f'{role}_mean_bot_score')
    coverage = row.get(f'{role}_coverage_pct')
    n = row.get(f'{role}_n')
    return {
        'pct_high_risk': round(float(pct), 3),
        'mean_bot_score': round(float(mean_score), 4) if pd.notna(mean_score) else None,
        'coverage_pct': round(float(coverage), 2) if pd.notna(coverage) else None,
        'n': int(n) if pd.notna(n) else 0,
        'severity': get_severity(float(pct), bands_by_role[role]),
    }


def main():
    df = pd.read_csv(SRC)
    df = df.dropna(subset=['pct_high_risk_of_scored'])
    months = sorted(df['month'].unique())

    # separate severity bands per role -- posters and commenters have genuinely different
    # baseline risk distributions (posters run higher, see V3_PLAN.md 2026-08-22 diagnostic),
    # so one shared band set would misrepresent severity for whichever role sits off-center.
    # Frozen to a declared baseline rather than recomputed -- see load_or_freeze_bands.
    bands_by_role, baseline_meta = load_or_freeze_bands(df, ROLES)
    bands = bands_by_role['combined']
    for role in ROLES:
        print(f'severity bands ({role}, P50/P80/P95):', bands_by_role[role])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    month_docs = {}
    for m in months:
        mdf = df[df['month'] == m].set_index('sub')
        subs = {}
        for sub, row in mdf.iterrows():
            roles = {role: role_block(row, role, bands_by_role) for role in ROLES}
            csdr = row.get('comment_self_del_rate')
            subgrowth = row.get('subscriber_growth_pct')
            subs[sub] = {
                'pct_high_risk': roles['combined']['pct_high_risk'],
                'mean_bot_score': roles['combined']['mean_bot_score'],
                'coverage_pct': roles['combined']['coverage_pct'],
                'n_influencers': roles['combined']['n'],
                'severity': roles['combined']['severity'],
                'comment_self_del_rate': round(float(csdr), 2) if pd.notna(csdr) else None,
                'subscriber_growth_pct': round(float(subgrowth), 2) if pd.notna(subgrowth) else None,
                'roles': roles,
            }
        month_docs[m] = subs

    def build_narrative(m, subs, role):
        prev_idx = months.index(m) - 1
        prev_subs = month_docs.get(months[prev_idx]) if prev_idx >= 0 else None
        rb = bands_by_role[role]

        def val(d, key='pct_high_risk'):
            return d['roles'][role][key]

        scored_subs = {s: d for s, d in subs.items() if val(d) is not None}
        scores = [val(d) for d in scored_subs.values()]
        avg = sum(scores) / len(scores) if scores else 0
        pct_mod = round(100 * sum(1 for s in scores if s >= rb['moderate']) / len(scores)) if scores else 0
        pct_high = round(100 * sum(1 for s in scores if s >= rb['high']) / len(scores)) if scores else 0

        movers = []
        if prev_subs:
            for sub, d in scored_subs.items():
                if sub in prev_subs and val(prev_subs[sub]) is not None:
                    delta = val(d) - val(prev_subs[sub])
                    movers.append({'subreddit': sub, 'delta': round(delta, 2), 'final_score': val(d)})
        movers.sort(key=lambda x: x['delta'])
        fallers = [x for x in movers if x['delta'] < 0][:3]
        risers = [x for x in movers[::-1] if x['delta'] > 0][:3]

        avg_prev = None
        if prev_subs:
            prev_scores = [val(d) for d in prev_subs.values() if val(d) is not None]
            avg_prev = sum(prev_scores) / len(prev_scores) if prev_scores else None

        toppers = sorted(({'subreddit': s, 'final_score': val(d)} for s, d in scored_subs.items()),
                          key=lambda x: -x['final_score'])[:5]

        findings = []
        if risers:
            findings.append(f"r/{risers[0]['subreddit']} saw the largest increase in {role} bot/spam prevalence "
                             f"this month (+{risers[0]['delta']:.1f}pp), now at {risers[0]['final_score']:.1f}%.")
        crit_subs = [s for s, d in scored_subs.items() if d['roles'][role]['severity'] == 'critical']
        if crit_subs:
            findings.append(f"{len(crit_subs)} subreddit(s) in the Critical band this month ({role}): "
                             + ', '.join(f'r/{s}' for s in crit_subs[:5]) + '.')
        low_cov = [s for s, d in scored_subs.items() if d['roles'][role]['coverage_pct'] is not None and d['roles'][role]['coverage_pct'] < 40]
        if low_cov:
            findings.append(f"{len(low_cov)} subreddit(s) have low {role}-scoring coverage (<40%) this month — "
                             "those readings skew toward accounts too new/thin-history to score; read cautiously.")

        return {
            'headline': {
                'avg_score': round(avg, 2),
                'avg_score_delta': round(avg - avg_prev, 2) if avg_prev is not None else None,
                'pct_moderate_plus': pct_mod, 'pct_high_plus': pct_high,
                'biggest_riser': risers[0] if risers else None,
                'biggest_faller': {**fallers[0], 'delta': fallers[0]['delta']} if fallers else None,
            },
            'toppers': toppers, 'risers': risers, 'fallers': fallers,
            'curated_findings': findings,
        }

    # ---- append-only publishing ----------------------------------------------------------
    # A month file, once written into the vintage directory, is never rewritten. Each refresh
    # recomputes every month (cheap, and we want the new month), but only *new* months are
    # persisted; for months already published we keep the frozen file and discard the
    # recomputed values. That is what makes the viewer's trend stable: measured on the
    # 2026-08-22 refresh, an ordinary monthly run silently restated 94.5% of published
    # subreddit-months and flipped a quarter of published severity labels.
    #
    # A deliberate methodology change does NOT edit these files -- it bumps SERIES_VERSION and
    # publishes a new vintage alongside the old one, so a restatement is always visible as a
    # new series rather than a number that quietly changed.
    VINTAGE_DIR.mkdir(parents=True, exist_ok=True)
    today = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d')
    manifest_path = DATA_DIR / 'manifest.json'
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {'series_version': SERIES_VERSION, 'first_published': {}})
    if manifest.get('series_version') != SERIES_VERSION:
        # New vintage: start a fresh manifest rather than inheriting another series' history.
        manifest = {'series_version': SERIES_VERSION, 'first_published': {}}

    full_docs, frozen, appended = {}, [], []
    for m in months:
        path = VINTAGE_DIR / f'{m}.json'
        if path.exists():
            full_docs[m] = json.loads(path.read_text())
            frozen.append(m)
            continue
        subs = month_docs[m]
        doc = {
            'month': m,
            'series_version': SERIES_VERSION,
            'first_published': today,
            'subreddits': subs,
            'severity_bands': bands,
            'severity_bands_by_role': bands_by_role,
            # Provenance for the bands: which window defined them and when. Without this a
            # viewer can't tell whether "critical" means the same thing it meant last month.
            'severity_baseline': {k: baseline_meta[k] for k in
                                  ('baseline_version', 'baseline_window', 'n_subreddit_months',
                                   'method')},
            'narrative': {role: build_narrative(m, subs, role) for role in ROLES},
        }
        full_docs[m] = doc
        path.write_text(json.dumps(doc))
        manifest['first_published'][m] = today
        appended.append(m)

    published_months = sorted(full_docs)

    def role_avg(subs, role):
        vals = [d['roles'][role]['pct_high_risk'] for d in subs.values() if d['roles'][role]['pct_high_risk'] is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    # Built from the PUBLISHED docs, not the fresh recompute -- otherwise the per-month files
    # would be frozen while the trend chart above them still moved every refresh.
    history = {
        'series_version': SERIES_VERSION,
        'months': [
            {'month': m, 'aggregate': {
                'avg_score': round(sum(d['pct_high_risk'] for d in full_docs[m]['subreddits'].values())
                                   / len(full_docs[m]['subreddits']), 2),
                'poster_avg_score': role_avg(full_docs[m]['subreddits'], 'poster'),
                'commenter_avg_score': role_avg(full_docs[m]['subreddits'], 'commenter'),
            }}
            for m in published_months
        ],
    }
    (VINTAGE_DIR / 'history.json').write_text(json.dumps(history))
    manifest['months'] = published_months
    manifest['updated'] = today
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Mirror the active vintage at the flat paths the existing consumers already use, so
    # nothing downstream has to learn the vintage layout to read the current series.
    for m in published_months:
        (DATA_DIR / f'{m}.json').write_text(json.dumps(full_docs[m]))
    (DATA_DIR / 'history.json').write_text(json.dumps(history))

    print(f'Series v{SERIES_VERSION}: {len(appended)} month(s) appended '
          f'({", ".join(appended) if appended else "none"}), '
          f'{len(frozen)} kept frozen. -> {VINTAGE_DIR.relative_to(ROOT)}')
    embed_into_compass(history, full_docs)
    embed_into_index(history)


if __name__ == '__main__':
    main()
