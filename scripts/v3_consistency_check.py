#!/usr/bin/env python3
"""Cross-page consistency check. Run after every monthly run, before pushing.

Recomputes, from the published month files only, everything the Report page and the Compass
show (headline average and delta, risers/fallers, top 5, Moderate+/High+ shares, history trend,
severity labels) and compares it with what the pages carry. Exit code 1 on any mismatch.
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / 'docs'
DATA = DOCS / 'data_v3'
ROLES = ['combined', 'poster', 'commenter']
HKEY = {'combined': 'avg_score', 'poster': 'poster_avg_score', 'commenter': 'commenter_avg_score'}
errs = []


def err(*a):
    errs.append(' '.join(str(x) for x in a))


def sev(v, b):
    return 'critical' if v >= b['critical'] else 'high' if v >= b['high'] else 'moderate' if v >= b['moderate'] else 'low'


man = json.loads((DATA / 'manifest.json').read_text())
months = man['months']
D = {m: json.loads((DATA / f'{m}.json').read_text()) for m in months}
hist = {x['month']: x['aggregate'] for x in json.loads((DATA / 'history.json').read_text())['months']}


def vals(m, role):
    return {s: d['roles'][role]['pct_high_risk'] for s, d in D[m]['subreddits'].items()
            if d['roles'][role]['pct_high_risk'] is not None}


for i, m in enumerate(months):
    bands = D[m]['severity_bands_by_role']
    for s, d in D[m]['subreddits'].items():
        for r in ROLES:
            x = d['roles'][r]
            # values are stored rounded, so a value within 0.006 of a band edge may sit either side
            if x['pct_high_risk'] is not None and x['severity'] not in {
                    sev(x['pct_high_risk'] + e, bands[r]) for e in (-0.006, 0, 0.006)}:
                err(m, s, r, 'severity label does not match value', x)
    for r in ROLES:
        n = D[m]['narrative'][r]
        h = n['headline']
        v = vals(m, r)
        avg = sum(v.values()) / len(v)
        if abs(avg - h['avg_score']) > 0.011: err(m, r, 'headline avg', h['avg_score'], 'vs files', round(avg, 2))
        if abs(hist[m][HKEY[r]] - h['avg_score']) > 0.011: err(m, r, 'history trend', hist[m][HKEY[r]], 'vs headline', h['avg_score'])
        b = bands[r]
        for key, thr in (('pct_moderate_plus', b['moderate']), ('pct_high_plus', b['high'])):
            want = round(100 * sum(1 for x in v.values() if x >= thr) / len(v))
            if want != h[key]: err(m, r, key, h[key], 'vs files', want)
        top = sorted(v.items(), key=lambda kv: -kv[1])[:5]
        if [t['subreddit'] for t in n['toppers']] != [t[0] for t in top]: err(m, r, 'top 5 differs from files')
        if i:
            p = months[i - 1]
            pv = vals(m=p, role=r)
            d_avg = avg - sum(pv.values()) / len(pv)
            if abs(d_avg - (h['avg_score_delta'] or 0)) > 0.011: err(m, r, 'headline delta', h['avg_score_delta'], 'vs published prior month', round(d_avg, 2))
            mv = {s: v[s] - pv[s] for s in v if s in pv}
            up, dn = max(mv, key=mv.get), min(mv, key=mv.get)
            if h['biggest_riser'] and (h['biggest_riser']['subreddit'] != up or abs(h['biggest_riser']['delta'] - mv[up]) > 0.011):
                err(m, r, 'riser', h['biggest_riser'], 'vs files', up, round(mv[up], 2))
            if h['biggest_faller'] and (h['biggest_faller']['subreddit'] != dn or abs(h['biggest_faller']['delta'] - mv[dn]) > 0.011):
                err(m, r, 'faller', h['biggest_faller'], 'vs files', dn, round(mv[dn], 2))

# Compass: embedded data must equal the published files
comp = (DOCS / 'bot-spam-compass.html').read_text()
a = comp.index('<script id="embedded-data" type="application/json">') + len('<script id="embedded-data" type="application/json">')
emb = json.loads(comp[a:comp.index('</script>', a)])
if emb['months'] != D: err('Compass embedded month data differs from docs/data_v3')
if {x['month']: x['aggregate'] for x in emb['history']['months']} != hist: err('Compass embedded history differs from history.json')

# Report page: tiles, top 5 and findings must equal the latest combined narrative
idx = (DOCS / 'index.html').read_text()
last = months[-1]
n = D[last]['narrative']['combined']
h = n['headline']
need = [f'This month — {last}', f"{h['avg_score']:.1f}%", f"{h['biggest_riser']['subreddit']}", f"{h['biggest_faller']['subreddit']}",
        f"{h['pct_moderate_plus']}%", f"{h['pct_high_plus']}%", f"{len(months)} months"]
need += [f"r/{t['subreddit']}</div><div class=\"top-score\">{t['final_score']:.1f}%" for t in n['toppers']]
need += [f"{abs(h['avg_score_delta']):.1f}pp vs last month"]
for t in need:
    if t not in idx: err('Report page missing/stale:', t)
for t in n['curated_findings']:
    if t.startswith('r/') and t not in idx: err('Report finding not on page:', t)
if f"{len(months)} months" not in idx and f"{len(months)} months" not in idx: err('trend month count')

print(f'checked {len(months)} months x {len(ROLES)} roles; {len(errs)} problem(s)')
for e in errs: print(' -', e)
sys.exit(1 if errs else 0)
