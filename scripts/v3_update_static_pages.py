#!/usr/bin/env python3
"""Refresh the hand-written parts of the published pages from the latest published month.

Stage 9 embeds the trend/dashboard JSON, but several statements are plain text that used to be
edited by hand each month: the Report page's "This month" tiles, top-5, findings and month
counts (docs/index.html), the README header line, and the corpus size / date range in the
whitepaper. This script derives all of them from docs/data_v3/v{SERIES}/ and the corpus, so a
monthly run leaves no stale text. Run after stage 9 and before the whitepaper HTML build.
Idempotent: running it twice changes nothing.
"""
import html
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'docs/data_v3'
MINUS = '−'


def pp(x, up_sign=True):
    return f"{MINUS}{abs(x):.1f}pp" if x < 0 else f"{'+' if up_sign else ''}{x:.1f}pp"


def sub_in(text, pattern, repl, label, flags=re.S):
    new, n = re.subn(pattern, lambda _: repl, text, count=1, flags=flags)
    if n != 1:
        raise SystemExit(f"static-page update: pattern not found for {label}")
    return new


def main():
    manifest = json.loads((DATA / 'manifest.json').read_text())
    series, months = manifest['series_version'], manifest['months']
    latest, first, n_months = months[-1], months[0], len(months)
    doc = json.loads((DATA / f'v{series}/{latest}.json').read_text())
    n = doc['narrative']['combined']
    h = n['headline']

    # ---- docs/index.html ----------------------------------------------------------------
    p = ROOT / 'docs/index.html'
    s = p.read_text()
    s = sub_in(s, r'This month — \d{4}-\d{2}', f'This month — {latest}', 'index month label')

    d = h['avg_score_delta']
    avg_delta = ''
    if d is not None:
        cls = 'delta-down' if d < 0 else 'delta-up'
        avg_delta = f'<div class="stat-delta {cls}">{pp(d)} vs last month</div>'

    def mover(m, cls):
        if not m:
            return ('<div class="stat-value" style="font-size:16px">n/a</div>', '')
        return (f'<div class="stat-value" style="font-size:16px">r/{html.escape(m["subreddit"])}</div>',
                f'<div class="stat-delta {cls}">{pp(m["delta"])}</div>')

    riser, faller = mover(h['biggest_riser'], 'delta-up'), mover(h['biggest_faller'], 'delta-down')
    grid = (
        '<div class="headline-grid">\n'
        f'      <div class="stat-tile"><div class="stat-label">Ecosystem avg prevalence</div><div class="stat-value">{h["avg_score"]:.1f}%</div>{avg_delta}</div>\n'
        f'      <div class="stat-tile"><div class="stat-label">Moderate+ subreddits</div><div class="stat-value">{h["pct_moderate_plus"]}%</div></div>\n'
        f'      <div class="stat-tile"><div class="stat-label">High+ subreddits</div><div class="stat-value">{h["pct_high_plus"]}%</div></div>\n'
        f'      <div class="stat-tile"><div class="stat-label">Biggest riser</div>{riser[0]}{riser[1]}</div>\n'
        f'      <div class="stat-tile"><div class="stat-label">Biggest faller</div>{faller[0]}{faller[1]}</div>\n'
        '    </div>')
    s = sub_in(s, r'<div class="headline-grid">.*?</div>\n    </div>', grid, 'index headline grid')

    tops = ''.join(
        f'\n      <div class="top-tile"><div class="top-rank">#{i}</div><div class="top-sub">r/{html.escape(t["subreddit"])}</div>'
        f'<div class="top-score">{t["final_score"]:.1f}%</div></div>' for i, t in enumerate(n['toppers'], 1))
    s = sub_in(s, r'<div class="top-grid">.*?\n    </div>', f'<div class="top-grid">{tops}\n    </div>', 'index top-5')

    # Findings: rebuilt from the same numbers the Compass narrative uses, in the Report's wording.
    subs = doc['subreddits']
    scored = {k: v for k, v in subs.items() if v['roles']['combined']['pct_high_risk'] is not None}
    cards = []
    if n['risers']:
        r = n['risers'][0]
        cards.append(f'r/{r["subreddit"]} saw the largest increase in combined bot/spam prevalence this month '
                     f'(+{r["delta"]:.1f}pp), now at {r["final_score"]:.1f}%.')
    crit = [k for k, v in scored.items() if v['roles']['combined']['severity'] == 'critical']
    if crit:
        cards.append(f'{len(crit)} subreddit{"s" if len(crit) != 1 else ""} in the Critical band this month: '
                     + ', '.join(f'r/{k}' for k in crit[:5]) + '.')
    low = [k for k, v in scored.items()
           if v['roles']['combined']['coverage_pct'] is not None and v['roles']['combined']['coverage_pct'] < 40]
    if low:
        cards.append(f'{len(low)} subreddit{"s" if len(low) != 1 else ""} could only score fewer than 40% of their top '
                     "accounts this month — for the rest, there just wasn't enough posting history yet to make a "
                     'call, so those particular readings are worth treating cautiously.')
    body = ''.join(f'\n      <div class="finding-card">{html.escape(c, quote=False)}</div>' for c in cards)
    s = sub_in(s, r'<div class="findings-list">.*?\n    </div>', f'<div class="findings-list">{body}\n    </div>', 'index findings')

    s = sub_in(s, r'Ecosystem trend, \d+ months', f'Ecosystem trend, {n_months} months', 'index trend title')
    s = sub_in(s, r'plus the full \d+-month trend', f'plus the full {n_months}-month trend', 'index leaderboard sentence')
    p.write_text(s)

    # ---- README.md header ---------------------------------------------------------------
    p = ROOT / 'README.md'
    s = p.read_text()
    s = sub_in(s, r'(\d+) months \(\d{4}-\d{2} → \d{4}-\d{2}\)',
               f'{n_months} months ({first} → {latest})', 'README header', flags=0)
    p.write_text(s)

    # ---- whitepaper corpus size and range -----------------------------------------------
    con = duckdb.connect(str(ROOT / 'data/v3/analysis/v3.duckdb'), read_only=True)
    n_comments = con.sql('select count(*) from commenters_dedup').fetchone()[0]
    n_accounts = con.sql('select count(distinct author) from commenters_dedup').fetchone()[0]
    mom = pd.read_csv(ROOT / 'output/v3/subreddit_bot_prevalence_mom.csv')
    n_rows, n_subs = len(mom), mom['sub'].nunique()
    p = ROOT / 'docs/v3-research/whitepaper.md'
    s = p.read_text()
    s = sub_in(s, r'\*\*[\d,]+ comments\*\*', f'**{n_comments:,} comments**', 'whitepaper comments')
    s = sub_in(s, r'\*\*[\d,]+ distinct commenting accounts\*\*, \d+ India-focused subreddits, \d{4}-\d{2} through \d{4}-\d{2} \(\d+\s+months\)',
               f'**{n_accounts:,} distinct commenting accounts**, {n_subs} India-focused subreddits, {first} through {latest} ({n_months}\n  months)',
               'whitepaper accounts')
    s = sub_in(s, r'Full \d+-month history \(\d{4}-\d{2}→\d{4}-\d{2}\)\. [\d,]+ subreddit-month rows, \d+ subreddits\.',
               f'Full {n_months}-month history ({first}→{latest}). {n_rows:,} subreddit-month rows, {n_subs} subreddits.',
               'whitepaper history')
    p.write_text(s)
    print(f'Static pages refreshed for {latest}: {n_months} months, {n_rows:,} sub-month rows, '
          f'{n_comments:,} comments, {n_accounts:,} accounts.')


if __name__ == '__main__':
    main()
