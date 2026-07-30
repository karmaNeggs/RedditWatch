#!/usr/bin/env python3
"""
Generate static site data from analysis outputs.
V1: reads output/analysis_*.json        → docs/data/
V2: reads output/v2/analysis_*.json     → docs/data_v2/

V2 also generates a narrative layer (headline/toppers/movers/curated findings)
on top of the per-sub numbers — see build_narrative(). This replaced the old
approach of just re-projecting every computed field and letting a static,
ever-growing F1-F10 findings list sit on the homepage: the dashboard's job is
"here's this month, here's the trend, here's who moved and why, here's what's
actually worth knowing" — not a raw data dump.

Usage:
  python3 scripts/generate_site.py           # V1
  python3 scripts/generate_site.py --v2      # V2
"""

import argparse
import json
import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--v2', action='store_true', help='Generate V2 site data')
    p.add_argument('--month', default=None, help='Limit to specific YYYY-MM month')
    p.add_argument('--min-sub-coverage', type=float, default=0.6,
                   help='Warn (not block) if a month covers fewer than this fraction of tracked subreddits')
    return p.parse_args()


def expected_sub_count() -> int:
    subs_file = ROOT / 'subreddits.txt'
    if not subs_file.exists():
        return 0
    return len([l for l in subs_file.read_text().splitlines()
                if l.strip() and not l.startswith('#')])


def validate_month(month, doc, min_coverage, expected_subs):
    """
    Collection/scoring bugs upstream (thin scrape, a broken analysis run) should be
    loud at publish time, not silently baked into the public dashboard. Non-blocking —
    warnings are surfaced in the JSON itself and printed, but the month still publishes,
    since a genuinely thin in-progress current month is a legitimate case, not a bug.
    """
    warnings = []
    subs = doc.get('subreddits', {})

    if expected_subs and len(subs) < expected_subs * min_coverage:
        warnings.append(f'only {len(subs)}/{expected_subs} tracked subreddits present '
                         f'(<{min_coverage*100:.0f}% coverage) — check for a partial/failed collection run')

    bad_scores = [sub for sub, d in subs.items()
                  if d.get('final_score') is None or not isinstance(d.get('final_score'), (int, float))]
    if bad_scores:
        warnings.append(f'{len(bad_scores)} subreddit(s) missing a valid final_score: {bad_scores[:5]}')

    if warnings:
        print(f"  ⚠ VALIDATION WARNINGS for {month}:")
        for w in warnings:
            print(f"      - {w}")
    return warnings


def _dirs(v2: bool):
    if v2:
        return ROOT / 'output' / 'v2', ROOT / 'docs' / 'data_v2'
    return ROOT / 'output', ROOT / 'docs' / 'data'


OUTPUT_DIR    = ROOT / 'output'        # default; overridden in main()
DOCS_DATA_DIR = ROOT / 'docs' / 'data' # default; overridden in main()


def load_severity_bands() -> dict:
    """Same bands score_accounts.py calibrated and analyze_data_v2.py scores
    against — read from findings.json so this file never has its own
    hardcoded copy to drift out of sync (it did, silently, until this
    rewrite: this function used to hardcode 20/40/70, the pre-Phase-1
    fixed thresholds, which no longer mean anything on the 0-100
    pct-high-risk-activity scale)."""
    findings_path = ROOT / 'reports' / 'findings.json'
    defaults = {'moderate': 20.0, 'high': 40.0, 'critical': 70.0}
    if findings_path.exists():
        try:
            with open(findings_path) as f:
                bands = json.load(f).get('severity_bands', {})
            if all(k in bands for k in ('moderate', 'high', 'critical')):
                return bands
        except Exception:
            pass
    return defaults


def get_severity(score, bands=None):
    bands = bands or load_severity_bands()
    if score >= bands['critical']: return 'critical'
    if score >= bands['high']:     return 'high'
    if score >= bands['moderate']: return 'moderate'
    return 'low'


def load_analysis_files(output_dir: Path, month_filter=None):
    files = sorted(glob.glob(str(output_dir / 'analysis_*.json')))
    results = []
    for f in files:
        if 'latest' in os.path.basename(f):
            continue
        try:
            with open(f) as fh:
                raw = json.load(fh)
            # V2 files store month explicitly; V1 derives from analysis_date
            if 'month' in raw:
                month = raw['month']
                dt = datetime.fromisoformat(raw['analysis_date'])
            else:
                dt = datetime.fromisoformat(raw['analysis_date'])
                month = dt.strftime('%Y-%m')
            if month_filter and month != month_filter:
                continue
            results.append((month, dt, raw))
        except Exception as e:
            print(f"  Skipping {os.path.basename(f)}: {e}")
    return results


def build_month_doc(month, dt, raw):
    scores     = raw.get('unified_scores', {})
    user_a     = raw.get('user_analysis', {})
    post_a     = raw.get('post_analysis', {})
    temporal_a = raw.get('temporal_analysis', {})
    dist_a     = raw.get('distribution_analysis', {})
    network_a  = raw.get('network_analysis', {})
    regular_a  = raw.get('post_regularity', {})
    eng_str_a  = raw.get('engagement_structure', {})
    astro_a    = raw.get('astroturf_density', {})

    subreddits = {}
    for sub, sc in scores.items():
        u  = user_a.get(sub, {})
        p  = post_a.get(sub, {})
        t  = temporal_a.get(sub, {})
        d  = dist_a.get(sub, {})
        n  = network_a.get(sub, {})
        r  = regular_a.get(sub, {})
        es = eng_str_a.get(sub, {})
        at = astro_a.get(sub, {})

        subreddits[sub] = {
            'final_score':        round(sc.get('final_score', 0), 1),
            'user_score':         round(sc.get('user_score', 0), 1),
            'engagement_score':   round(sc.get('engagement_score', 0), 1),
            'temporal_score':     round(sc.get('temporal_score', 0), 1),
            'distribution_score': round(sc.get('distribution_score', 0), 1),
            'details': {
                'posts_analyzed':       p.get('posts_analyzed', 0),
                'users_analyzed':       u.get('users_analyzed', 0),
                'avg_account_age_days': round(u.get('avg_account_age_days', 0)),
                'avg_karma_per_day':    round(u.get('avg_karma_per_day', 0), 1),
                'suspicious_pct':       round(u.get('suspicious_accounts_pct', 0), 1),
                'avg_upvote_ratio':     round(p.get('avg_upvote_ratio', 0), 3),
                'avg_score':            round(p.get('avg_score', 0)),
                'avg_comments':         round(p.get('avg_comments', 0), 1),
                'ucr':                  round(p.get('ucr', 0), 1),
                'peak_hour_utc':        t.get('peak_hour_utc', 0),
                'top_3_concentration':  round(t.get('top_3_hours_concentration', 0), 1),
                'entropy':              round(t.get('entropy', 0), 2),
                'score_cv':             round(d.get('score_cv', 0), 3),
                'comments_cv':          round(d.get('comments_cv', 0), 3),
                # new signals
                'cross_sub_author_pct':    n.get('cross_sub_author_pct'),
                'multi_sub_author_pct':    n.get('multi_sub_author_pct'),
                'interval_cv':             r.get('interval_cv'),
                'mean_interval_hours':     r.get('mean_interval_hours'),
                'score_comment_corr':      es.get('score_comment_corr'),
                'upvote_ratio_std':        es.get('upvote_ratio_std'),
                'fully_coordinated_pct':   at.get('fully_coordinated_pct'),
                'poster_only_susp_pct':    at.get('poster_only_susp_pct'),
                'commenter_only_susp_pct': at.get('commenter_only_susp_pct'),
            },
        }

    net_summary = network_a.get('_network', {})
    return {
        'month':         month,
        'analysis_date': dt.isoformat(),
        'network': {
            'total_unique_authors':     net_summary.get('total_unique_authors', 0),
            'cross_sub_authors_2plus':  net_summary.get('cross_sub_2plus', 0),
            'cross_sub_authors_3plus':  net_summary.get('cross_sub_3plus', 0),
            'top_cross_sub_authors':    net_summary.get('top_cross_sub_authors', {}),
        },
        'subreddits': subreddits,
    }


def build_month_doc_v2(month, dt, raw, bands):
    """Build month doc from a V2 analysis JSON. final_score is the validated
    account-risk rollup (score_accounts.py) — everything else here is
    diagnostic detail, kept per-sub so the narrative layer (build_narrative)
    can explain *why* a score moved, not just that it did."""
    scores  = raw.get('unified_scores', {})
    acct_a  = raw.get('account_analysis', {})
    ring_a  = raw.get('ring_analysis', {})
    eng_a   = raw.get('engagement_analysis', {})
    temp_a  = raw.get('temporal_analysis', {})
    dist_a  = raw.get('distribution_analysis', {})
    net_a   = raw.get('network_analysis', {})
    coord_a = raw.get('coordination_analysis', {})
    cooc_a  = raw.get('cooccurrence_analysis', {})
    base_a  = raw.get('baseline_comparison', {})

    subreddits = {}
    for sub, sc in scores.items():
        ac = acct_a.get(sub, {})
        ri = ring_a.get(sub, {})
        en = eng_a.get(sub, {})
        te = temp_a.get(sub, {})
        di = dist_a.get(sub, {})
        ne = net_a.get(sub, {})
        co = coord_a.get(sub, {})
        cc = cooc_a.get(sub, {})
        ba = base_a.get(sub, {})

        subreddits[sub] = {
            'final_score':             round(sc.get('final_score', 0), 1),
            'severity':                get_severity(sc.get('final_score', 0), bands),
            'pct_high_risk_activity':  sc.get('pct_high_risk_activity'),
            'n_activity_rows':         sc.get('n_activity_rows'),
            # diagnostic component scores — not weighted into final_score, kept
            # for the "why did this move" narrative and the Explore/detail view
            'account_score':      round(sc.get('account_score', 0), 1),
            'ring_score':          round(sc.get('ring_score', 0), 1),
            'engagement_score':    round(sc.get('engagement_score', 0), 1),
            'temporal_score':      round(sc.get('temporal_score', 0), 1),
            'distribution_score':  round(sc.get('distribution_score', 0), 1),
            'pct_posts_fully_coordinated':      co.get('pct_posts_fully_coordinated'),
            'pct_posts_commenter_only_risk':    co.get('pct_posts_commenter_only_risk'),
            'pct_posts_any_high_risk_commenter': co.get('pct_posts_any_high_risk_commenter'),
            'repeat_pair_rate':                 cc.get('repeat_pair_rate'),
            'top_vs_baseline_risk_ratio':        ba.get('top_vs_baseline_risk_ratio'),
            'details': {
                'new_poster_pct':      ac.get('new_poster_pct'),
                'new_commenter_pct':   ac.get('new_commenter_pct'),
                'unverified_comm_pct': ac.get('unverified_comm_pct'),
                'burst_score':         ri.get('burst_score'),
                'fast_ttfc_pct':       ri.get('fast_ttfc_pct'),
                'avg_ttfc_minutes':    ri.get('avg_ttfc_minutes'),
                'recurrence_rate':     ri.get('recurrence_rate'),
                'self_amp_rate':       ri.get('self_amp_rate'),
                'overlap_rate':        ri.get('overlap_rate'),
                'score_comment_corr':  en.get('score_comment_corr'),
                'upvote_ratio_std':    en.get('upvote_ratio_std'),
                'ucr':                 en.get('ucr'),
                'avg_awards':          en.get('avg_awards'),
                'simulacra_rate':      en.get('simulacra_rate'),
                'interval_cv':         te.get('interval_cv'),
                'top3_concentration':  te.get('top3_concentration'),
                'entropy':             te.get('entropy'),
                'peak_hour_utc':       te.get('peak_hour_utc'),
                'decay_slope':         di.get('decay_slope'),
                'decay_r2':            di.get('decay_r2'),
                'comments_cv':         di.get('comments_cv'),
                'score_cv':            di.get('score_cv'),
                'avg_comment_depth':   di.get('avg_comment_depth'),
                'near_dupe_rate':      ne.get('near_dupe_rate'),
                'cross_sub_rate':      ne.get('cross_sub_rate'),
                'gini_score':          ne.get('gini_score'),
                'n_repeat_pairs':      cc.get('n_repeat_pairs'),
                'max_pair_cooccurrence': cc.get('max_pair_cooccurrence'),
            },
        }

    return {
        'version':       2,
        'month':         month,
        'analysis_date': dt.isoformat(),
        'severity_bands': bands,
        'subreddits':    subreddits,
    }


# ── Narrative generation ────────────────────────────────────────────────────
# Turns the per-sub numbers into "here's this month, here's who moved and
# why, here's what's actually worth knowing" — regenerated fresh every run,
# not accumulated like the old static F1-F10 findings list.

DRIVER_FIELDS = {
    'account_score':                    'account-risk signals',
    'ring_score':                       'comment-ring timing',
    'engagement_score':                 'engagement structure (votes vs. discussion)',
    'temporal_score':                   'posting-time patterns',
    'distribution_score':               'score/comment distribution shape',
    'pct_posts_fully_coordinated':      'fully-coordinated posts',
    'pct_posts_commenter_only_risk':    'suspicious-commenter support on clean-looking posts',
    'repeat_pair_rate':                 'repeat-commenter-pair coordination',
}


def _biggest_secondary_shift(sub, this_doc, prev_doc):
    """Which diagnostic signal moved the most for this sub between two
    months? NOT a causal claim — final_score is the account-risk rollup
    alone (see score_accounts.py); none of DRIVER_FIELDS are summed into it
    post-Phase-1, so a diagnostic can move opposite to final_score with no
    contradiction. This reports "what else shifted", not "what caused it" —
    keep the wording in any caller honest about that distinction. Returns
    (field_label, delta) or None if no prior month."""
    if prev_doc is None:
        return None
    this_row = this_doc['subreddits'].get(sub, {})
    prev_row = prev_doc['subreddits'].get(sub, {})
    if not this_row or not prev_row:
        return None

    best = None
    for field, label in DRIVER_FIELDS.items():
        a, b = this_row.get(field), prev_row.get(field)
        if a is None or b is None:
            continue
        delta = round(a - b, 1)
        if best is None or abs(delta) > abs(best[1]):
            best = (label, delta)
    return best


def build_narrative(month, this_doc, prev_doc, months_back_doc):
    """
    months_back_doc: the doc from ~3 months prior, for a longer-baseline
    trend line ("rising over the last quarter" vs. "up from last month" can
    tell different stories on noisy monthly data).
    """
    subs = this_doc['subreddits']
    all_scores = [d['final_score'] for d in subs.values()]
    if not all_scores:
        return {'headline': {}, 'toppers': [], 'risers': [], 'fallers': [], 'curated_findings': []}

    avg_score = round(sum(all_scores) / len(all_scores), 1)
    pct_mod   = round(sum(1 for s in all_scores if s >= this_doc['severity_bands']['moderate']) / len(all_scores) * 100)
    pct_high  = round(sum(1 for s in all_scores if s >= this_doc['severity_bands']['high']) / len(all_scores) * 100)

    prev_avg = None
    deltas = {}
    if prev_doc:
        prev_scores = {s: d['final_score'] for s, d in prev_doc['subreddits'].items()}
        prev_all = list(prev_scores.values())
        if prev_all:
            prev_avg = round(sum(prev_all) / len(prev_all), 1)
        for sub, d in subs.items():
            if sub in prev_scores:
                deltas[sub] = round(d['final_score'] - prev_scores[sub], 1)

    toppers = sorted(subs.items(), key=lambda kv: -kv[1]['final_score'])[:5]
    toppers = [{'subreddit': s, 'final_score': d['final_score']} for s, d in toppers]

    risers  = sorted(deltas.items(), key=lambda kv: -kv[1])[:5]
    fallers = sorted(deltas.items(), key=lambda kv: kv[1])[:5]

    def _mover_entry(sub, delta):
        shift = _biggest_secondary_shift(sub, this_doc, prev_doc)
        entry = {'subreddit': sub, 'final_score': subs[sub]['final_score'], 'delta': delta}
        if shift:
            entry['secondary_signal_label'], entry['secondary_signal_delta'] = shift
        return entry

    risers_out  = [_mover_entry(s, d) for s, d in risers  if d > 0]
    fallers_out = [_mover_entry(s, d) for s, d in fallers if d < 0]

    biggest_riser  = risers_out[0]  if risers_out  else None
    biggest_faller = fallers_out[0] if fallers_out else None

    headline = {
        'avg_score': avg_score,
        'avg_score_delta': round(avg_score - prev_avg, 1) if prev_avg is not None else None,
        'pct_moderate_plus': pct_mod,
        'pct_high_plus': pct_high,
        'sub_count': len(all_scores),
        'biggest_riser': biggest_riser,
        'biggest_faller': biggest_faller,
    }

    curated = []

    if biggest_riser:
        line = f"r/{biggest_riser['subreddit']} rose the most this month ({biggest_riser['delta']:+.1f} pts, now {biggest_riser['final_score']:.1f})"
        if 'secondary_signal_label' in biggest_riser:
            line += (f" — also notable: {biggest_riser['secondary_signal_label']} shifted "
                      f"{biggest_riser['secondary_signal_delta']:+.1f} pts this month.")
        else:
            line += "."
        curated.append(line)

    if biggest_faller:
        line = f"r/{biggest_faller['subreddit']} fell the most this month ({biggest_faller['delta']:+.1f} pts, now {biggest_faller['final_score']:.1f})"
        if 'secondary_signal_label' in biggest_faller:
            line += (f" — also notable: {biggest_faller['secondary_signal_label']} shifted "
                      f"{biggest_faller['secondary_signal_delta']:+.1f} pts this month.")
        else:
            line += "."
        curated.append(line)

    commenter_only = [(s, d.get('pct_posts_commenter_only_risk') or 0) for s, d in subs.items()]
    commenter_only.sort(key=lambda kv: -kv[1])
    if commenter_only and commenter_only[0][1] > 0:
        sub, pct = commenter_only[0]
        curated.append(f"r/{sub} shows the clearest 'clean post, suspicious support' pattern this month "
                        f"({pct:.0f}% of sampled posts had a low-risk poster but majority-high-risk top commenters).")

    repeat_pairs = [(s, d.get('repeat_pair_rate') or 0) for s, d in subs.items()]
    repeat_pairs.sort(key=lambda kv: -kv[1])
    if repeat_pairs and repeat_pairs[0][1] > 0:
        sub, pct = repeat_pairs[0]
        curated.append(f"r/{sub} has the strongest repeat-commenter-pair signal this month "
                        f"({pct:.0f}% of sampled posts had a commenter pair that also appeared together elsewhere).")

    if months_back_doc:
        mb_scores = [d['final_score'] for d in months_back_doc['subreddits'].values()]
        if mb_scores:
            mb_avg = round(sum(mb_scores) / len(mb_scores), 1)
            direction = 'risen' if avg_score > mb_avg + 1 else 'fallen' if avg_score < mb_avg - 1 else 'held steady'
            curated.append(f"Ecosystem average has {direction} over the last quarter "
                            f"({mb_avg:.1f} → {avg_score:.1f}).")

    return {
        'headline': headline,
        'toppers': toppers,
        'risers': risers_out,
        'fallers': fallers_out,
        'curated_findings': curated[:5],
    }


def main():
    args = parse_args()
    v2   = args.v2

    output_dir, docs_dir = _dirs(v2)

    print("\n" + "=" * 70)
    print(f"GENERATE SITE DATA  (V{'2' if v2 else '1'})")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    docs_dir.mkdir(parents=True, exist_ok=True)

    entries = load_analysis_files(output_dir, month_filter=args.month)
    if not entries:
        print(f"No analysis files found in {output_dir}. Run analyze_data{'_v2' if v2 else ''}.py first.")
        return

    # Keep only the latest run per calendar month
    by_month = {}
    for month, dt, raw in entries:
        if month not in by_month or dt > by_month[month][0]:
            by_month[month] = (dt, raw)

    bands = load_severity_bands() if v2 else None
    expected_subs = expected_sub_count()
    history_months = []
    docs_by_month = {}  # month -> built doc, for narrative's prev/months-back lookups

    sorted_months = sorted(by_month)
    for i, month in enumerate(sorted_months):
        dt, raw = by_month[month]
        doc = build_month_doc_v2(month, dt, raw, bands) if v2 else build_month_doc(month, dt, raw)
        docs_by_month[month] = doc

        warnings = validate_month(month, doc, args.min_sub_coverage, expected_subs)
        if warnings:
            doc['validation_warnings'] = warnings

        if v2:
            prev_doc = docs_by_month.get(sorted_months[i - 1]) if i > 0 else None
            months_back_doc = docs_by_month.get(sorted_months[i - 3]) if i >= 3 else None
            doc['narrative'] = build_narrative(month, doc, prev_doc, months_back_doc)

        out_path = docs_dir / f'{month}.json'
        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"  Wrote {out_path.name}")

        all_scores = [d['final_score'] for d in doc['subreddits'].values()]
        sev_bands = bands if v2 else {'moderate': 20, 'high': 40}
        history_entry = {
            'month':         month,
            'analysis_date': dt.isoformat(),
            'data_file':     f'{month}.json',
            'aggregate': {
                'avg_score':        round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
                'pct_moderate_plus': round(sum(1 for s in all_scores if s >= sev_bands['moderate']) / len(all_scores) * 100) if all_scores else 0,
                'pct_high_plus':     round(sum(1 for s in all_scores if s >= sev_bands['high']) / len(all_scores) * 100) if all_scores else 0,
                'sub_count':    len(all_scores),
                'max_score':    round(max(all_scores), 1) if all_scores else 0,
                'min_score':    round(min(all_scores), 1) if all_scores else 0,
            },
            'summary': {
                sub: {'final_score': d['final_score'], 'severity': get_severity(d['final_score'], bands) if v2 else get_severity(d['final_score'])}
                for sub, d in doc['subreddits'].items()
            },
        }
        if v2:
            history_entry['headline'] = doc['narrative']['headline']
        history_months.append(history_entry)

    history   = {'version': 2 if v2 else 1, 'months': history_months}
    hist_path = docs_dir / 'history.json'
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  Wrote {hist_path.name}")

    if v2:
        findings_src = ROOT / 'reports' / 'findings.json'
        if findings_src.exists():
            shutil.copy(findings_src, docs_dir / 'findings.json')
            print(f"  Copied findings.json")

    print(f"\nDone. {len(by_month)} month(s) written to {docs_dir.relative_to(ROOT)}/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
