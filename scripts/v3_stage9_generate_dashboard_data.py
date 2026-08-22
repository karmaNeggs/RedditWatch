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
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'output/v3/subreddit_bot_prevalence_mom.csv'
DATA_DIR = ROOT / 'docs/data_v3'
COMPASS = ROOT / 'docs/bot-spam-compass.html'


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


def get_severity(score, bands):
    if score >= bands['critical']: return 'critical'
    if score >= bands['high']: return 'high'
    if score >= bands['moderate']: return 'moderate'
    return 'low'


def main():
    df = pd.read_csv(SRC)
    df = df.dropna(subset=['pct_high_risk_of_scored'])
    months = sorted(df['month'].unique())

    bands = severity_bands(df['pct_high_risk_of_scored'].values)
    print('severity bands (P50/P80/P95 of observed pct_high_risk_of_scored):', bands)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    month_docs = {}
    for m in months:
        mdf = df[df['month'] == m].set_index('sub')
        subs = {}
        for sub, row in mdf.iterrows():
            subs[sub] = {
                'pct_high_risk': round(float(row['pct_high_risk_of_scored']), 3),
                'mean_bot_score': round(float(row['mean_bot_score_scored']), 4) if pd.notna(row['mean_bot_score_scored']) else None,
                'coverage_pct': round(float(row['coverage_pct']), 2) if pd.notna(row['coverage_pct']) else None,
                'n_influencers': int(row['n_influencers']),
                'severity': get_severity(float(row['pct_high_risk_of_scored']), bands),
            }
        month_docs[m] = subs

    def build_narrative(m, subs):
        prev_idx = months.index(m) - 1
        prev_subs = month_docs.get(months[prev_idx]) if prev_idx >= 0 else None
        scores = [d['pct_high_risk'] for d in subs.values()]
        avg = sum(scores) / len(scores) if scores else 0
        pct_mod = round(100 * sum(1 for s in scores if s >= bands['moderate']) / len(scores)) if scores else 0
        pct_high = round(100 * sum(1 for s in scores if s >= bands['high']) / len(scores)) if scores else 0

        movers = []
        if prev_subs:
            for sub, d in subs.items():
                if sub in prev_subs:
                    delta = d['pct_high_risk'] - prev_subs[sub]['pct_high_risk']
                    movers.append({'subreddit': sub, 'delta': round(delta, 2), 'final_score': d['pct_high_risk']})
        movers.sort(key=lambda x: x['delta'])
        fallers = [x for x in movers if x['delta'] < 0][:3]
        risers = [x for x in movers[::-1] if x['delta'] > 0][:3]

        avg_prev = None
        if prev_subs:
            prev_scores = [d['pct_high_risk'] for d in prev_subs.values()]
            avg_prev = sum(prev_scores) / len(prev_scores) if prev_scores else None

        toppers = sorted(({'subreddit': s, 'final_score': d['pct_high_risk']} for s, d in subs.items()),
                          key=lambda x: -x['final_score'])[:5]

        findings = []
        if risers:
            findings.append(f"r/{risers[0]['subreddit']} saw the largest increase in bot/spam-influencer "
                             f"prevalence this month (+{risers[0]['delta']:.1f}pp), now at {risers[0]['final_score']:.1f}%.")
        crit_subs = [s for s, d in subs.items() if d['severity'] == 'critical']
        if crit_subs:
            findings.append(f"{len(crit_subs)} subreddit(s) in the Critical band this month: "
                             + ', '.join(f'r/{s}' for s in crit_subs[:5]) + '.')
        low_cov = [s for s, d in subs.items() if d['coverage_pct'] is not None and d['coverage_pct'] < 40]
        if low_cov:
            findings.append(f"{len(low_cov)} subreddit(s) have low scoring coverage (<40%) this month — "
                             "their top-30 influencer set skews toward accounts too new/thin-history to score; "
                             "read those readings cautiously.")

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

    full_docs = {}
    for m in months:
        subs = month_docs[m]
        doc = {
            'month': m,
            'subreddits': subs,
            'severity_bands': bands,
            'narrative': build_narrative(m, subs),
        }
        full_docs[m] = doc
        (DATA_DIR / f'{m}.json').write_text(json.dumps(doc))

    history = {
        'months': [
            {'month': m, 'aggregate': {'avg_score': round(sum(d['pct_high_risk'] for d in month_docs[m].values()) / len(month_docs[m]), 2)}}
            for m in months
        ],
    }
    (DATA_DIR / 'history.json').write_text(json.dumps(history))
    print(f'Wrote {len(months)} month files + history.json to {DATA_DIR.relative_to(ROOT)}')
    embed_into_compass(history, full_docs)


if __name__ == '__main__':
    main()
