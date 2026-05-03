#!/usr/bin/env python3
"""
Generate a plain-text newsletter report — copy-paste directly into Substack.
Usage: python3 scripts/text_report.py
"""

import json, os
from datetime import datetime
from pathlib import Path

ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / 'output'

def load():
    p = OUTPUT_DIR / 'analysis_latest.json'
    if not p.exists():
        raise FileNotFoundError("Run analyze_data.py first.")
    with open(p) as f:
        return json.load(f)

def severity(s):
    if s >= 70: return 'Critical'
    if s >= 50: return 'High'
    if s >= 30: return 'Moderate'
    return 'Low'

def icon(s):
    if s >= 70: return '🔴'
    if s >= 50: return '🟠'
    if s >= 30: return '🟡'
    return '🟢'

def summary(subs):
    total   = len(subs)
    high    = [s for s in subs if s[1] >= 50]
    low     = [s for s in subs if s[1] < 30]
    avg     = sum(s[1] for s in subs) / total
    top     = subs[0]
    cleanest= subs[-1]

    scores  = {'user_score':top[2],'engagement_score':top[3],'temporal_score':top[4],'distribution_score':top[5]}
    labels  = {'user_score':'suspicious account patterns','engagement_score':'low comment engagement vs upvotes',
               'temporal_score':'concentrated posting windows','distribution_score':'uniform vote distribution'}
    dominant = max(scores, key=scores.get)

    lines = [
        f"This month's scan covered {total} Indian subreddits, with an average bot activity score "
        f"of {avg:.1f}/100. {len(high)} communities scored High or above (≥50).",
        "",
        f"r/{top[0]} ranked highest at {top[1]:.1f}, driven primarily by {labels[dominant]}. "
        + (f"Political and entertainment subs — {', '.join('r/'+s[0] for s in high[:3])} — "
           f"show the strongest signals, consistent with agenda-driven coordination."
           if len(high) >= 3 else ""),
        "",
        f"r/{cleanest[0]} was the cleanest community at {cleanest[1]:.1f}"
        + (f", along with {len(low)-1} other subreddits in the low-risk zone — mostly hobby "
           f"and niche interest spaces." if len(low) > 1 else "."),
    ]
    return "\n".join(lines)

def main():
    data    = load()
    dt      = datetime.fromisoformat(data['analysis_date'])
    month   = dt.strftime('%B %Y')
    scores  = data['unified_scores']

    subs = sorted(
        [(sub,
          round(v['final_score'],1),
          round(v['user_score'],1),
          round(v['engagement_score'],1),
          round(v['temporal_score'],1),
          round(v['distribution_score'],1))
         for sub, v in scores.items()],
        key=lambda x: -x[1]
    )

    W = 62
    div = '─' * W

    lines = []
    lines += [
        "",
        "━" * W,
        f"  REDDIT BOT WATCH · {month.upper()}",
        f"  Monthly Bot Activity Report — Indian Subreddits",
        "━" * W,
        "",
        "SUMMARY",
        div,
        summary(subs),
        "",
        "RANKINGS",
        div,
        f"{'#':<4} {'Subreddit':<26} {'Score':>6}   {'Severity':<10}  Components (U·E·T·D)",
        div,
    ]

    for i, (sub, final, u, e, t, d) in enumerate(subs, 1):
        bar_len = int(final / 100 * 20)
        bar     = '█' * bar_len + '░' * (20 - bar_len)
        ic      = icon(final)
        sev     = severity(final)
        lines.append(
            f"{ic} {i:<2}  r/{sub:<24} {final:>5.1f}   {sev:<10}  {u:.0f}·{e:.0f}·{t:.0f}·{d:.0f}"
        )

    lines += [
        div,
        "   Components: U=User Accounts (35%)  E=Engagement (30%)",
        "               T=Temporal (20%)        D=Distribution (15%)",
        "",
        "HOW WE SCORE THIS",
        div,
        "Each subreddit is scored 0–100 across four signals:",
        "",
        "🔵 User Accounts (35%)    Flags accounts with >200 karma/day or new",
        "                          accounts (<90 days) with elevated growth.",
        "",
        "🔴 Engagement (30%)       High upvote-to-comment ratio signals bots",
        "                          voting without joining discussion.",
        "",
        "🟠 Temporal (20%)         Automated systems post in tight windows.",
        "                          Low entropy = scheduled, not organic.",
        "",
        "🟣 Distribution (15%)     Organic communities have high vote variance.",
        "                          Uniform scores across all posts = suspicious.",
        "",
        "Score interpretation:  0–30 Low · 30–50 Moderate · 50–70 High · 70+ Critical",
        "",
        div,
        f"Data: Top 30 posts · Last 30 days · Collected {dt.strftime('%d %b %Y')}",
        "Source: https://github.com/karmaNeggs/RedditWatch",
        "━" * W,
        "",
    ]

    report = "\n".join(lines)
    print(report)

    # Also save to file
    out = ROOT / f"report_{dt.strftime('%Y%m')}.txt"
    out.write_text(report)
    print(f"[Saved to {out.name}]")

if __name__ == '__main__':
    main()
