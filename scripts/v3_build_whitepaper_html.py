"""
Renders docs/v3-research/whitepaper.md into docs/v3-research/whitepaper.html --
a styled, navigable page matching the site's design system (paired visually
with charts/model_analysis.html), instead of an unstyled raw markdown file.
Also injects four interactive Chart.js charts at <!--CHART:x--> markers in the
source markdown, built from the same output/v3/*.json data model_analysis.html
uses, so the whitepaper doesn't just describe the validation, it shows it.

whitepaper.md remains the source of truth; this script is the only thing that
should touch whitepaper.html. Run standalone after editing the .md, or it runs
automatically at the end of scripts/v3_stage8_monthly_refresh.py.
"""
import json
import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'docs/v3-research/whitepaper.md'
OUT = ROOT / 'docs/v3-research/whitepaper.html'
OUTDIR = ROOT / 'output/v3'

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RedditWatch — Whitepaper</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --accent: #2a78d6; --accent2: #d95926; --accent3: #199e70;
    --hdr-bg: rgba(249,249,247,0.94);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --accent: #3987e5; --accent2: #d95926; --accent3: #2bc48a;
      --hdr-bg: rgba(13,13,13,0.94);
    }}
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ background: var(--page); color: var(--ink); margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.6; font-size: 15px; }}
  a {{ color: var(--accent); }}
  .mono {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace; font-size: 0.92em; }}
  code {{ font-family: ui-monospace, "SF Mono", "Cascadia Code", Consolas, monospace; font-size: 0.9em;
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px; padding: 0.1em 0.4em; }}
  pre code {{ display: block; padding: 12px 14px; overflow-x: auto; }}

  header {{ position: sticky; top: 0; z-index: 10; background: var(--hdr-bg); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }}
  .hdr-inner {{ max-width: 1000px; margin: 0 auto; padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
  .brand {{ font-weight: 800; font-size: 15px; color: var(--ink); text-decoration: none; }}
  .nav-tab {{ color: var(--muted); font-size: 13px; font-weight: 600; padding: 6px 10px; border-radius: 6px; text-decoration: none; }}
  .nav-tab.active {{ color: var(--ink); background: var(--surface); }}

  .layout {{ max-width: 1000px; margin: 0 auto; padding: 32px 24px 80px; display: grid; grid-template-columns: 200px minmax(0,1fr); gap: 40px; }}
  @media (max-width: 820px) {{ .layout {{ grid-template-columns: 1fr; }} }}

  .toc {{ position: sticky; top: 72px; align-self: start; font-size: 12.5px; max-height: calc(100vh - 100px); overflow-y: auto; }}
  .toc-title {{ font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 700; margin-bottom: 10px; }}
  .toc ul {{ list-style: none; margin: 0; padding: 0; }}
  .toc > ul > li {{ margin-bottom: 3px; }}
  .toc ul ul {{ padding-left: 12px; }}
  .toc a {{ color: var(--ink-2); text-decoration: none; display: block; padding: 2px 0; border-left: 2px solid transparent; padding-left: 8px; margin-left: -8px; }}
  .toc a:hover {{ color: var(--accent); }}
  @media (max-width: 820px) {{ .toc {{ position: static; max-height: none; margin-bottom: 20px; border-bottom: 1px solid var(--grid); padding-bottom: 16px; }} }}

  .content {{ min-width: 0; max-width: 74ch; }}
  .content h1 {{ font-size: 25px; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 6px; text-wrap: balance; }}
  .content > p:first-of-type {{ font-size: 14px; color: var(--ink-2); margin-bottom: 22px; }}
  .content h2 {{ font-size: 18px; font-weight: 700; margin: 40px 0 12px; padding-top: 8px; border-top: 1px solid var(--grid); scroll-margin-top: 76px; }}
  .content h2:first-of-type {{ border-top: none; margin-top: 28px; }}
  .content h3 {{ font-size: 15px; font-weight: 700; margin: 24px 0 8px; color: var(--ink-2); scroll-margin-top: 76px; }}
  .content p {{ margin: 0 0 14px; }}
  .content ul, .content ol {{ margin: 0 0 14px; padding-left: 22px; }}
  .content li {{ margin-bottom: 4px; }}
  .content hr {{ border: none; border-top: 1px solid var(--grid); margin: 32px 0; }}
  .content strong {{ color: var(--ink); }}
  .content blockquote {{ margin: 0 0 14px; padding: 4px 16px; border-left: 3px solid var(--accent); color: var(--ink-2); background: var(--surface); border-radius: 0 6px 6px 0; }}

  table {{ border-collapse: collapse; width: 100%; margin: 4px 0 18px; font-size: 13px; }}
  th, td {{ padding: 6px 12px; border-bottom: 1px solid var(--grid); text-align: left; }}
  thead th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.03em; border-bottom: 1px solid var(--baseline); }}
  tbody tr:hover {{ background: var(--surface); }}
  .table-wrap {{ overflow-x: auto; }}

  .chart-embed {{ margin: 6px 0 20px; padding: 16px 18px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; }}
  .chart-embed-title {{ font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); font-weight: 700; margin-bottom: 10px; }}
  .chart-wrap {{ position: relative; }}

  footer {{ max-width: 1000px; margin: 0 auto; padding: 20px 24px 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<header>
  <div class="hdr-inner">
    <a class="brand" href="../index.html">RedditWatch</a>
    <nav style="display:flex;gap:6px">
      <a class="nav-tab" href="../index.html">Report</a>
      <a class="nav-tab" href="../methodology.html">Methodology</a>
      <a class="nav-tab" href="../bot-spam-compass.html">Bot &amp; Spam Compass</a>
      <a class="nav-tab active" href="whitepaper.html">Whitepaper</a>
    </nav>
  </div>
</header>

<div class="layout">
  <nav class="toc">
    <div class="toc-title">On this page</div>
    {toc}
  </nav>
  <article class="content">
    {body}
  </article>
</div>

<footer>
  Source: <a href="whitepaper.md">whitepaper.md</a> (this page is generated from it — edit the source, not this file). Full validation charts: <a href="charts/model_analysis.html">model_analysis.html</a>. Repo: <a href="https://github.com/karmaNeggs/RedditWatch">GitHub</a>.
</footer>
<script id="chart-data" type="application/json">{chart_data}</script>
<script>
  document.querySelectorAll('table').forEach(t => {{
    const wrap = document.createElement('div'); wrap.className = 'table-wrap';
    t.parentNode.insertBefore(wrap, t); wrap.appendChild(t);
  }});

  const DATA = JSON.parse(document.getElementById('chart-data').textContent);
  function cssVar(name) {{ return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }}

  // -------- target/feature comparison (grouped bars) --------
  (function() {{
    const el = document.getElementById('chart-target');
    if (!el) return;
    const rows = DATA.config;
    new Chart(el.getContext('2d'), {{
      type: 'bar',
      data: {{
        labels: rows.map(r => r.target.split(' (')[0] + (r.features.includes('with') ? ' + age' : '')),
        datasets: [{{ data: rows.map(r => r.mean), backgroundColor: rows.map(r => r.target.startsWith('clubbed') ? cssVar('--accent') : cssVar('--accent2')), borderRadius: 4 }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ label: c => `AUC = ${{c.raw.toFixed(3)}} ± ${{rows[c.dataIndex].std.toFixed(3)}}` }} }} }},
        scales: {{
          y: {{ min: 0.5, max: 0.85, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted') }} }},
          x: {{ grid: {{ display: false }}, ticks: {{ color: cssVar('--ink-2'), font: {{ size: 10.5 }} }} }},
        }},
      }},
    }});
  }})();

  // -------- feature-count elimination curve --------
  (function() {{
    const el = document.getElementById('chart-elimination');
    if (!el) return;
    const rows = DATA.elimination.slice().reverse();
    new Chart(el.getContext('2d'), {{
      type: 'line',
      data: {{
        labels: rows.map(r => r.n_features),
        datasets: [{{ label: 'CV AUC', data: rows.map(r => r.mean), borderColor: cssVar('--accent2'),
          backgroundColor: cssVar('--accent2') + '18', borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25, fill: true }}],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{ title: c => `${{c[0].label}} features`, label: c => `AUC = ${{c.formattedValue}}` }} }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'features remaining', color: cssVar('--muted') }}, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted'), font: {{ size: 10 }}, maxTicksLimit: 10 }} }},
          y: {{ min: 0.6, max: 0.85, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted') }} }},
        }},
      }},
    }});
  }})();

  // -------- ROC curve --------
  (function() {{
    const el = document.getElementById('chart-roc');
    if (!el) return;
    const d = DATA.roc;
    new Chart(el.getContext('2d'), {{
      type: 'scatter',
      data: {{
        datasets: [
          {{ label: 'Model', data: d.fpr.map((f,i) => ({{x: f, y: d.tpr[i]}})), showLine: true, borderColor: cssVar('--accent'),
            backgroundColor: cssVar('--accent') + '18', borderWidth: 2.5, pointRadius: 0, fill: true, tension: 0.1 }},
          {{ label: 'Random guess', data: [{{x:0,y:0}},{{x:1,y:1}}], showLine: true, borderColor: cssVar('--muted'),
            borderWidth: 1, borderDash: [4,3], pointRadius: 0 }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false, aspectRatio: 1,
        plugins: {{ legend: {{ display: false }},
          tooltip: {{ callbacks: {{ label: c => c.datasetIndex === 0 ? `catches ${{(c.parsed.y*100).toFixed(0)}}% of removed accounts at a ${{(c.parsed.x*100).toFixed(0)}}% false-alarm rate` : '' }} }} }},
        scales: {{
          x: {{ min: 0, max: 1, title: {{ display: true, text: '% of active accounts wrongly flagged', color: cssVar('--muted') }}, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted'), callback: v => (v*100)+'%' }} }},
          y: {{ min: 0, max: 1, title: {{ display: true, text: '% of removed accounts correctly caught', color: cssVar('--muted') }}, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted'), callback: v => (v*100)+'%' }} }},
        }},
      }},
    }});
    const lbl = document.getElementById('roc-auc-label');
    if (lbl) lbl.textContent = d.auc.toFixed(3);
  }})();

  // -------- confirmed-rate-by-bucket (100% stacked bars) --------
  (function() {{
    const el = document.getElementById('chart-bucket');
    if (!el) return;
    const rows = DATA.bucket;
    new Chart(el.getContext('2d'), {{
      type: 'bar',
      data: {{
        labels: rows.map(r => `${{r.lo.toFixed(1)}}–${{r.hi.toFixed(1)}}`),
        datasets: [
          {{ label: 'Still active', data: rows.map(r => r.pct_ok), backgroundColor: cssVar('--accent'), stack: 's' }},
          {{ label: 'Banned', data: rows.map(r => r.pct_banned), backgroundColor: cssVar('--accent2'), stack: 's' }},
          {{ label: 'Deleted', data: rows.map(r => r.pct_deleted), backgroundColor: cssVar('--accent3'), stack: 's' }},
        ],
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ position: 'bottom', labels: {{ color: cssVar('--ink-2'), boxWidth: 10, font: {{ size: 11 }} }} }},
          tooltip: {{ callbacks: {{ afterTitle: c => `n = ${{rows[c[0].dataIndex].n}}`, label: c => `${{c.dataset.label}}: ${{c.raw.toFixed(1)}}%` }} }} }},
        scales: {{
          x: {{ stacked: true, title: {{ display: true, text: 'predicted removal probability', color: cssVar('--muted') }}, grid: {{ display: false }}, ticks: {{ color: cssVar('--muted'), font: {{ size: 10 }} }} }},
          y: {{ stacked: true, max: 100, grid: {{ color: cssVar('--grid') }}, ticks: {{ color: cssVar('--muted'), callback: v => v+'%' }} }},
        }},
      }},
    }});
  }})();
</script>
</body>
</html>
"""

CHART_HTML = {
    'target': '''<div class="chart-embed">
  <div class="chart-embed-title">Chart — target choice: clubbed vs. banned-only vs. deleted-only</div>
  <div class="chart-wrap" style="height:260px"><canvas id="chart-target"></canvas></div>
</div>''',
    'elimination': '''<div class="chart-embed">
  <div class="chart-embed-title">Chart — accuracy vs. number of features kept</div>
  <div class="chart-wrap" style="height:260px"><canvas id="chart-elimination"></canvas></div>
</div>''',
    'roc': '''<div class="chart-embed">
  <div class="chart-embed-title">Chart — ROC curve (AUC = <span id="roc-auc-label">…</span>)</div>
  <div class="chart-wrap" style="height:320px;max-width:420px;margin:0 auto"><canvas id="chart-roc"></canvas></div>
</div>''',
    'bucket': '''<div class="chart-embed">
  <div class="chart-embed-title">Chart — real outcome mix by score bucket</div>
  <div class="chart-wrap" style="height:280px"><canvas id="chart-bucket"></canvas></div>
</div>''',
}


def load_chart_data():
    config = json.load(open(OUTDIR / 'config_summary.json'))
    elimination = json.load(open(OUTDIR / 'backward_elim_history.json'))
    roc = json.load(open(OUTDIR / 'final_roc_data.json'))
    bucket = json.load(open(OUTDIR / 'score_bucket_composition.json'))
    return {'config': config, 'elimination': elimination, 'roc': roc, 'bucket': bucket}


def main():
    md_text = SRC.read_text()
    md = markdown.Markdown(extensions=['extra', 'toc', 'sane_lists'],
                            extension_configs={'toc': {'permalink': False, 'anchorlink': False}})
    body_html = md.convert(md_text)

    for marker, snippet in CHART_HTML.items():
        body_html = body_html.replace(f'<!--CHART:{marker}-->', snippet)

    chart_data = json.dumps(load_chart_data())
    html = TEMPLATE.format(toc=md.toc, body=body_html, chart_data=chart_data)
    OUT.write_text(html)
    print(f'Rendered {SRC.relative_to(ROOT)} -> {OUT.relative_to(ROOT)} ({len(html)} bytes, 4 charts embedded)')


if __name__ == '__main__':
    main()
