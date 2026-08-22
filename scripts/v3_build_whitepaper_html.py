"""
Renders docs/v3-research/whitepaper.md into docs/v3-research/whitepaper.html --
a styled, navigable page matching the site's design system (paired visually
with charts/model_analysis.html), instead of an unstyled raw markdown file.

whitepaper.md remains the source of truth; this script is the only thing that
should touch whitepaper.html. Run standalone after editing the .md, or it runs
automatically at the end of scripts/v3_stage8_monthly_refresh.py.
"""
import markdown
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'docs/v3-research/whitepaper.md'
OUT = ROOT / 'docs/v3-research/whitepaper.html'

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RedditWatch — Whitepaper</title>
<style>
  :root {{
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --accent: #2a78d6; --accent2: #d95926;
    --hdr-bg: rgba(249,249,247,0.94);
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --accent: #3987e5; --accent2: #d95926;
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
<script>
  document.querySelectorAll('table').forEach(t => {{
    const wrap = document.createElement('div'); wrap.className = 'table-wrap';
    t.parentNode.insertBefore(wrap, t); wrap.appendChild(t);
  }});
</script>
</body>
</html>
"""


def main():
    md_text = SRC.read_text()
    md = markdown.Markdown(extensions=['extra', 'toc', 'sane_lists'],
                            extension_configs={'toc': {'permalink': False, 'anchorlink': False}})
    body_html = md.convert(md_text)
    html = TEMPLATE.format(toc=md.toc, body=body_html)
    OUT.write_text(html)
    print(f'Rendered {SRC.relative_to(ROOT)} -> {OUT.relative_to(ROOT)} ({len(html)} bytes)')


if __name__ == '__main__':
    main()
