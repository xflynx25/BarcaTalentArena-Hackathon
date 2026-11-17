"""The team site's landing page and the header bar on every page.

The page: the task, entry cards (Match viewer, dashboard), where we stand, the lessons (headlines
that fold out), then the story: one compact row per report page, newest first, opening to its chapter.

Text comes from experiments/reports.json: `overview` (lede, lessons, what's next) and each page row's
`story` (why / did / found / surprise / next, plus the scores measured in that session). Agents write
a story whenever they add a page row (CLAUDE.md). The page computes each chapter's change against
the best score before it, and a scoreboard table of every step.

The dashboard's card shows the best SkillCorner score so far and its history, read from the
dashboard page itself (`card`) and kept, encrypted, in cards.json on the site branch. External data only.
"""

from __future__ import annotations

import html as htmlesc
import json
import os
import re

_DATA = re.compile(r'<script id="report-data" type="application/json">(.*?)</script>', re.S)
SECTIONS = (("why", "Why this next"), ("did", "What we did"), ("found", "What we found"),
            ("surprise", "Surprise"), ("next", "What it led to"))


def _esc(text) -> str:
    """Escape, then show `code` spans as code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", htmlesc.escape(str(text)))


def _data(page: str) -> dict | None:
    match = _DATA.search(page)
    if not match:
        return None
    try:
        return json.loads(match.group(1).replace("<\\/", "</").replace("<\\!--", "<!--"))
    except ValueError:
        return None


def card(page: str) -> dict | None:
    """The dashboard's score history (best SkillCorner BAR-ATH-like spellF1 over time), else None."""
    data = _data(page)
    if not isinstance(data, dict) or "groups" not in data:
        return None
    runs = sorted((run["time"], run["spellF1"]) for group in data["groups"] if group.get("primary")
                  for run in group["runs"] if run.get("spellF1") is not None)
    best, history = None, []
    for time, score in runs:
        best = score if best is None else max(best, score)
        history.append([time[:10], round(best, 4)])
    return {"kind": "history", "history": history, "runs": data.get("n_runs")} if history else None


def history_svg(history: list) -> str:
    width, height, pad, left, right = 300, 56, 6, 40, 44
    values = [v for _, v in history]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 0.05
    xs = [left + (width - left - right) * (i / max(1, len(values) - 1)) for i in range(len(values))]
    ys = [height - pad - (height - 2 * pad) * (v - lo) / span for v in values]
    line = " ".join("{:.1f},{:.1f}".format(x, y) for x, y in zip(xs, ys))
    tip = "Best SkillCorner spellF1: {:.3f} on {} to {:.3f} on {}".format(
        values[0], history[0][0], values[-1], history[-1][0])
    return ('<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{t}"><title>{t}</title>'
            '<polyline class="line" points="{line}"/>'
            '<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4"/>'
            '<text class="val" x="{tx:.1f}" y="{ty:.1f}">{v:.3f}</text>'
            '<text class="lab" x="{sx:.1f}" y="{sy:.1f}" text-anchor="end">{s:.3f}</text></svg>').format(
        w=width, h=height, t=htmlesc.escape(tip), line=line, x=xs[-1], y=ys[-1], tx=xs[-1] + 8,
        ty=ys[-1] + 4, v=values[-1], sx=xs[0] - 6, sy=ys[0] + 4, s=values[0])


# -- site header (every page) ---------------------------------------------------------------------
VIEWER = "viewer.html"  # the Match viewer: a full-screen app, so it gets the header but no story block
DASHBOARD = "dashboard.html"
HEADER_HEIGHT = 44  # px; the viewer's desktop layout subtracts it from the screen height
# Scoped to .mt-bar (plus the body margin and the viewer's height) so report pages keep their styles.
HEADER_STYLE = """<style>
  body { margin: 0; }
  .mt-bar { --mt-bg: #ffffffee; --mt-ink: #17201a; --mt-ink-2: #5b665f; --mt-rule: #dce2da; --mt-accent: #1d6b45; --mt-wash: #e8f1eb;
    position: sticky; top: 0; z-index: 25; height: __H__px; box-sizing: border-box; display: flex; align-items: center;
    gap: 1rem; padding: 0 max(1rem, env(safe-area-inset-left, 0px)); background: var(--mt-bg); border-bottom: 1px solid var(--mt-rule);
    font: 500 14px/1 system-ui, -apple-system, sans-serif; backdrop-filter: saturate(1.4) blur(8px); -webkit-backdrop-filter: saturate(1.4) blur(8px); }
  @media (prefers-color-scheme: dark) {
    .mt-bar { --mt-bg: #141a16ee; --mt-ink: #e7ece8; --mt-ink-2: #a6b0aa; --mt-rule: #2c352f; --mt-accent: #4fb985; --mt-wash: #1d2b23; }
  }
  .mt-bar a { color: var(--mt-ink-2); text-decoration: none; white-space: nowrap; }
  .mt-bar .mt-brand { color: var(--mt-ink); font-weight: 750; letter-spacing: .02em; margin-right: auto; }
  .mt-bar .mt-brand span { color: var(--mt-accent); }
  .mt-bar nav { display: flex; gap: .25rem; overflow-x: auto; }
  .mt-bar nav a { padding: .45rem .7rem; border-radius: 8px; }
  .mt-bar nav a:hover { color: var(--mt-ink); background: var(--mt-wash); }
  .mt-bar nav a.on { color: var(--mt-accent); background: var(--mt-wash); font-weight: 650; }
  @media (max-width: 30rem) { .mt-bar { gap: .5rem; } .mt-bar .mt-brand small { display: none; } .mt-bar nav a { padding: .45rem .5rem; } }
  @media (min-width: 901px) and (orientation: landscape) { html body .mv-match { height: calc(100dvh - __H__px); } }
</style>""".replace("__H__", str(HEADER_HEIGHT))


def header(active: str = "", names: set[str] | None = None) -> str:
    """The bar on top of every site page. active: the page's site name; names: pages on the site."""
    links = [(VIEWER, "Watch matches"), ("index.html#model", "How it works"), ("index.html#story", "Story"),
             (DASHBOARD, "All runs")]
    items = []
    for href, label in links:
        target = href.split("#")[0]
        if names is not None and target not in names:
            continue
        on = target == active and target != "index.html"
        items.append('<a href="{}"{}>{}</a>'.format(href, ' class="on" aria-current="page"' if on else "", label))
    return ('{}\n<header class="mt-bar"><a class="mt-brand" href="index.html">MTAH <span>&middot;</span> '
            '<small>who has the ball?</small></a><nav>{}</nav></header>').format(HEADER_STYLE, "".join(items))


# -- story ----------------------------------------------------------------------------------------

def chapters(reports: dict) -> list[dict]:
    """Report rows oldest first, numbered, with each score's change against the best before it."""
    best = {"skillcorner": None, "mean": None}
    out = []
    for number, row in enumerate(reports.get("reports", []), 1):
        row = dict(row, number=number, name=os.path.basename(row.get("file", "")),
                   anchor="ch{}".format(number))
        scores = (row.get("story") or {}).get("scores") or {}
        row["changes"] = {}
        for key in best:
            score = scores.get(key)
            if score is None:
                continue
            before = best[key]
            row["changes"][key] = {"score": score, "before": before,
                                   "new_best": before is None or score > before + 1e-9}
            best[key] = score if before is None else max(before, score)
        out.append(row)
    return out


def _change(label: str, change: dict) -> str:
    score, before = change["score"], change["before"]
    if before is None:
        note = "starting point"
    elif change["new_best"]:
        note = "+{:.3f}, new best".format(score - before)
    elif abs(score - before) < 5e-4:
        note = "ties the best"
    else:
        note = "{:+.3f} vs best {:.3f}".format(score - before, before).replace("-", "&minus;", 1)
    return '<span class="score{}"><b>{}</b> {:.3f} <small>({})</small></span>'.format(
        " best" if change["new_best"] and before is not None else "", label, score, note)


def _link(row: dict, names: set[str]) -> tuple[str, bool]:
    on_site = row["name"] in names
    return (row["name"] if on_site else row.get("url")) or "", on_site


def _scores(row: dict) -> str:
    scores = "".join(_change(label, row["changes"][key]) for key, label in
                     (("skillcorner", "SkillCorner"), ("mean", "Average of 3")) if key in row["changes"])
    return scores or '<span class="score"><small>A measurement, not a new model</small></span>'


def _sections(row: dict) -> str:
    story = row.get("story") or {}
    body = []
    for key, label in SECTIONS:
        if story.get(key):
            body.append('<div class="sec{}"><h4>{}</h4><p>{}</p></div>'.format(
                " surprise" if key == "surprise" else "", label, _esc(story[key])))
    if not body and row.get("summary"):
        body.append("<p>{}</p>".format(_esc(row["summary"])))
    return "\n    ".join(body)


def _open_link(link: str, on_site: bool) -> str:
    return '<a class="open" href="{}">{} &rarr;</a>'.format(
        htmlesc.escape(link), "Open the page" if on_site else "Open in Claude (owner only)") if link else ""


def _chapter(row: dict, names: set[str]) -> str:
    """The full chapter card (tops each report page)."""
    link, on_site = _link(row, names)
    return """<article class="chapter" id="{anchor}">
  <div class="num">{number}</div>
  <div class="content">
    <h3><a href="{link}">{title}</a></h3>
    <div class="meta"><span class="chip">{task}</span><span class="date">{date}</span></div>
    <div class="scores">{scores}</div>
    {body}
    {open}
  </div>
</article>""".format(
        anchor=row["anchor"], number=row["number"], link=htmlesc.escape(link),
        title=htmlesc.escape(row.get("title") or row["name"]), task=htmlesc.escape(row.get("task", "")),
        date=htmlesc.escape(row.get("updated", "")), scores=_scores(row), body=_sections(row),
        open=_open_link(link, on_site))


def _cell(change: dict | None) -> str:
    if not change:
        return '<span class="v none">&ndash;</span>'
    best = change["new_best"] and change["before"] is not None
    return '<span class="v{}">{:.3f}</span>'.format(" best" if best else "", change["score"])


def _step(row: dict, names: set[str]) -> str:
    """One compact, expandable row of the story on the landing page."""
    link, on_site = _link(row, names)
    title = row.get("title") or row["name"]
    task = row.get("task", "")
    if task and title.startswith(task + " "):
        title = title[len(task) + 1:]
    return """<details class="step" id="{anchor}">
  <summary><span class="n">{number}</span><span class="t"><b>{title}</b><small>{task}{date}</small></span>{sc}{avg}</summary>
  <div class="body">
    <div class="scores">{scores}</div>
    {body}
    {open}
  </div>
</details>""".format(
        anchor=row["anchor"], number=row["number"], title=htmlesc.escape(title),
        task=htmlesc.escape(task) + " &middot; " if task else "", date=htmlesc.escape(row.get("updated", "")),
        sc=_cell(row["changes"].get("skillcorner")), avg=_cell(row["changes"].get("mean")),
        scores=_scores(row), body=_sections(row), open=_open_link(link, on_site))


def story_block(name: str, reports: dict) -> str:
    """The page's own chapter, folded at the top of the page on the site (empty if it has none)."""
    if name == VIEWER:
        return ""
    row = next((r for r in chapters(reports) if r["name"] == name and r.get("story")), None)
    if not row:
        return ""
    chapter = _chapter(row, {name}).replace('<a class="open"', '<a hidden class="open"')
    return ('{}\n<details class="story-top"><summary><span class="kicker">Step {} of the story</span> '
            '<span class="hint">Why we did this and what we found</span></summary>\n{}\n'
            '<p class="ctx"><a href="index.html#{}">See it in the whole story &rarr;</a></p>\n</details>\n').format(
               STORY_TOP_STYLE, row["number"], chapter, row["anchor"])


# Scoped to .story-top so it can't touch the report page's own styles (which reuse names like .chip).
STORY_TOP_STYLE = """<style>
  .story-top { --st-surface: #fff; --st-ink: #17201a; --st-ink-2: #4b5650; --st-ink-3: #737e77; --st-rule: #dce2da;
    --st-accent: #1d6b45; --st-wash: #e8f1eb; --st-warm: #fff6e5; --st-warm-rule: #f0d9a8;
    display: block; max-width: 46rem; margin: 1rem auto 0; padding: 0 1rem; color: var(--st-ink); font: 15px/1.55 system-ui, -apple-system, sans-serif; }
  @media (prefers-color-scheme: dark) {
    .story-top { --st-surface: #1a211d; --st-ink: #e7ece8; --st-ink-2: #a6b0aa; --st-ink-3: #87918b; --st-rule: #2c352f;
      --st-accent: #4fb985; --st-wash: #1d2b23; --st-warm: #2a2416; --st-warm-rule: #5a4a26; }
  }
  .story-top > summary { cursor: pointer; list-style: none; display: flex; flex-wrap: wrap; gap: .25rem .6rem; align-items: baseline;
    background: var(--st-wash); border-radius: 10px; padding: .55rem .9rem; }
  .story-top > summary::-webkit-details-marker { display: none; }
  .story-top > summary::after { content: "\\25BE"; color: var(--st-accent); margin-left: auto; }
  .story-top[open] > summary::after { content: "\\25B4"; }
  .story-top .kicker { font-weight: 650; color: var(--st-accent); }
  .story-top .hint { color: var(--st-ink-2); font-size: .9rem; }
  .story-top .ctx { margin: 0 0 .5rem; font-size: .88rem; }
  .story-top a { color: var(--st-accent); }
  .story-top code { font-size: .88em; background: var(--st-wash); padding: 0 .25em; border-radius: 4px; }
  .story-top .chapter { display: grid; grid-template-columns: 2.2rem 1fr; gap: .75rem; background: var(--st-surface);
    border: 1px solid var(--st-rule); border-radius: 12px; padding: 1rem; margin: .6rem 0; }
  .story-top .num { width: 2rem; height: 2rem; border-radius: 50%; background: var(--st-wash); color: var(--st-accent);
    font-weight: 700; display: flex; align-items: center; justify-content: center; }
  .story-top h3 { margin: .1rem 0 .2rem; font-size: 1.1rem; }
  .story-top h3 a { text-decoration: none; }
  .story-top .meta { display: flex; gap: .5rem; align-items: center; margin: 0 0 .5rem; }
  .story-top .chip { font-size: .72rem; font-weight: 600; color: var(--st-ink-2); border: 1px solid var(--st-rule); border-radius: 999px; padding: 0 .45rem; }
  .story-top .date { font-size: .78rem; color: var(--st-ink-3); }
  .story-top .scores { display: flex; flex-wrap: wrap; gap: .4rem .9rem; margin: 0 0 .6rem; font-size: .9rem; }
  .story-top .score small { color: var(--st-ink-2); }
  .story-top .score.best b::before { content: "\\2605\\00a0"; color: var(--st-accent); }
  .story-top .sec h4 { margin: .5rem 0 .1rem; font-size: .75rem; text-transform: uppercase; letter-spacing: .04em; color: var(--st-ink-3); }
  .story-top .sec p { margin: 0; }
  .story-top .sec.surprise { background: var(--st-warm); border: 1px solid var(--st-warm-rule); border-radius: 8px; padding: .1rem .7rem .5rem; margin-top: .6rem; }
  @media (max-width: 30rem) { .story-top .chapter { grid-template-columns: 1fr; } .story-top .num { display: none; } }
</style>"""


# -- landing page ---------------------------------------------------------------------------------

def _dashboard(reports: dict, names: set[str], cards: dict) -> str:
    dash = reports.get("dashboard") or {}
    name = os.path.basename(dash.get("file", ""))
    if name not in names:
        return ""
    data = cards.get(name) or {}
    hero = ""
    if data.get("kind") == "history":
        hero = ('<div class="hero">{:.3f}<small>best SkillCorner spellF1 so far{}</small></div>{}').format(
            data["history"][-1][1], " &middot; {} runs".format(data["runs"]) if data.get("runs") else "",
            history_svg(data["history"]))
    return ('<a class="card" href="{}"><span class="kick">All runs</span><span class="title">Dashboard</span>'
            '{}<span class="go">Every run, per-match scores and open tasks &rarr;</span></a>').format(
        htmlesc.escape(name), hero)


def _viewer(reports: dict, names: set[str]) -> str:
    if VIEWER not in names:
        return ""
    row = next((r for r in reports.get("reports", []) if os.path.basename(r.get("file", "")) == VIEWER), {})
    text = row.get("summary") or "Watch the external matches with the truth and our predictions on top."
    return ('<a class="card primary" href="{}"><span class="kick">Watch matches</span>'
            '<span class="title">Match viewer</span><p>{}</p>'
            '<span class="go">Open the viewer &rarr;</span></a>').format(VIEWER, _esc(text))


def _lessons(lessons: list) -> str:
    """Each lesson's first sentence as a headline; the rest folds out."""
    items = []
    for text in lessons:
        text = str(text)
        cut = re.search(r"[.:!?](\s|$)", text)
        head, rest = (text[:cut.end()].strip(), text[cut.end():].strip()) if cut else (text, "")
        items.append('<li><details><summary>{}</summary><p>{}</p></details></li>'.format(_esc(head), _esc(rest))
                     if rest else "<li><span>{}</span></li>".format(_esc(head)))
    return '<ol class="lessons">{}</ol>'.format("".join(items))


VERDICTS = (("worked", "Worked", "in the model"), ("didnt", "Didn't work", "tested, left out"),
            ("maybe", "Maybe", "something there, not proven"))


def _model(model: dict) -> str:
    """reports.json `model`: how the current model works, for non-technical readers. The steps are
    shown; the clues, the training, the major versions and the ideas grid fold out."""
    if not model:
        return ""
    out = ['<section class="part model" id="model">' + _part_head(1, "The model", model.get("title", "How our model works")),
           '<div class="panel">']
    if model.get("intro"):
        out.append('<p>{}</p>'.format(_esc(model["intro"])))
    if model.get("steps"):
        out.append('<ol class="pipeline">{}</ol>'.format("".join(
            '<li><b>{}</b><span>{}</span></li>'.format(_esc(st["name"]), _esc(st["plain"])) for st in model["steps"])))
    if model.get("features"):
        out.append('<details class="fold"><summary>What it looks at: the clues</summary><dl class="clues">{}</dl>'
                   '</details>'.format("".join('<dt>{}</dt><dd>{}</dd>'.format(_esc(f["group"]), _esc(f["plain"]))
                                               for f in model["features"])))
    if model.get("training"):
        out.append('<details class="fold"><summary>How it learns, and how we choose what ships</summary>'
                   '<p>{}</p></details>'.format(_esc(model["training"])))
    versions = model.get("versions") or []
    if versions:
        rows = []
        for i, v in enumerate(versions):
            now = i == len(versions) - 1
            rows.append('<div class="ver{}"><span class="vn"><b>{}</b><small>{}</small></span><span class="vc">{}</span>'
                        '<span class="vs">{}</span><span class="vs">{}</span></div>'.format(
                            " now" if now else "", _esc(v["name"]), _esc(v.get("date", "")), _esc(v["change"]),
                            "{:.3f}".format(v["skillcorner"]) if v.get("skillcorner") is not None else "&ndash;",
                            "{:.3f}".format(v["mean"]) if v.get("mean") is not None else "&ndash;"))
        out.append('<details class="fold"><summary>The major versions, {} to {}</summary>'
                   '<p class="note">{} The full step-by-step story, every wave and every idea, is '
                   '<a href="#story">further down</a>.</p><div class="vers"><div class="ver head"><span>Version</span>'
                   '<span>What changed</span><span class="vs">SkillCorner</span><span class="vs">Avg of 3</span></div>{}'
                   '</div></details>'.format(_esc(versions[0]["name"].split()[0]), _esc(versions[-1]["name"].split()[0]),
                                             _esc(model.get("scores_note", "")), "".join(rows)))
    ideas = model.get("ideas") or []
    if ideas:
        cols = []
        for key, label, sub in VERDICTS:
            items = [i for i in ideas if i.get("verdict") == key]
            if not items:
                continue
            cols.append('<div class="col {k}"><h3>{l} <small>{n} &middot; {s}</small></h3><ul>{items}</ul></div>'.format(
                k=key, l=label, n=len(items), s=sub, items="".join(
                    '<li><b>{}</b><span>{}</span></li>'.format(_esc(i["idea"]), _esc(i.get("why", ""))) for i in items)))
        out.append('<details class="fold"><summary>What we tried: worked, didn\'t, maybe</summary>'
                   '<p class="note">"Maybe" means a small gain we couldn\'t tell apart from luck, or one that '
                   'depends on something we can\'t check on our match, so it stays out.</p>'
                   '<div class="ideas">{}</div></details>'.format("".join(cols)))
    out.append("</div></section>")
    return "\n".join(out)


def _part_head(num: int, kick: str, title: str, anchor: str = "") -> str:
    """The numbered heading that opens each big section of the landing page, so the eye can find
    where one part ends and the next begins."""
    return '<div class="part-head"><span class="part-kick">Part {} &middot; {}</span><h2{}>{}</h2></div>'.format(
        num, _esc(kick), ' id="{}"'.format(anchor) if anchor else "", _esc(title))


# Opens the story step a link points at (#ch12), on load and when the hash changes.
OPEN_TARGET = """<script>
  (function () {
    function open() { var el = location.hash && document.getElementById(location.hash.slice(1));
      if (el && el.tagName === "DETAILS") { el.open = true; el.scrollIntoView({block: "start"}); } }
    window.addEventListener("hashchange", open); open();
  })();
</script>"""


def page(reports: dict, names: set[str], cards: dict) -> str:
    """reports: experiments/reports.json; names: pages on the site; cards: per-page chart data."""
    overview = reports.get("overview") or {}
    rows = chapters(reports)
    listed = {row["name"] for row in rows} | {os.path.basename((reports.get("dashboard") or {}).get("file", ""))}
    extra = sorted(names - listed - {"index.html"})
    parts = ["<title>MTAH: who has the ball?</title>", STYLE, header("index.html", names), "<main>",
             "<h1>Who has the ball?</h1>"]
    if overview.get("lede"):
        parts.append('<p class="lede">{}</p>'.format(_esc(overview["lede"])))
    cards_html = _viewer(reports, names) + _dashboard(reports, names, cards)
    if cards_html:
        parts.append('<div class="cards">{}</div>'.format(cards_html))
    if overview.get("now"):
        parts.append('<section class="now"><h2>Where we stand</h2><p>{}</p></section>'.format(_esc(overview["now"])))
    model = _model(reports.get("model") or {})
    parts.append(model)
    num = 2 if model else 1
    if overview.get("lessons"):
        parts.append('<section class="part" id="lessons">' + _part_head(num, "Lessons", "What we've learned"))
        num += 1
        parts.append('<p class="note">{} lessons, in the order we learned them. Tap one to read more.</p>'.format(
            len(overview["lessons"])))
        parts.append(_lessons(overview["lessons"]))
        parts.append("</section>")
    if rows:
        parts.append('<section class="part">' + _part_head(num, "The story", "The story, step by step", "story"))
        parts.append('<p class="note">Newest first. Tap a step for why we did it, what we found and what '
                     'it led to. Scores are spellF1 on the external matches; &#9733; marks a new best. {}</p>'.format(
                         _esc(overview.get("scores_note", ""))))
        parts.append('<div class="steps"><div class="steps-head"><span></span><span>Step</span>'
                     '<span>SkillCorner</span><span>Avg of 3</span></div>')
        parts.extend(_step(row, names) for row in reversed(rows))
        parts.append("</div></section>")
    if extra:
        parts.append("<h2>Other pages</h2>")
        parts.append('<ul class="other">{}</ul>'.format("".join(
            '<li><a href="{0}">{0}</a></li>'.format(htmlesc.escape(n)) for n in extra)))
    parts.append("</main>")
    parts.append(OPEN_TARGET)
    return "\n".join(parts)


STYLE = """<style>
  :root { --bg: #f5f6f2; --surface: #fff; --ink: #17201a; --ink-2: #4b5650; --ink-3: #737e77;
    --rule: #dce2da; --accent: #1d6b45; --wash: #e8f1eb; --warm: #fff6e5; --warm-rule: #f0d9a8; --panel: #ebeee7; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #111613; --surface: #1a211d; --ink: #e7ece8; --ink-2: #a6b0aa; --ink-3: #87918b;
      --rule: #2c352f; --accent: #4fb985; --wash: #1d2b23; --warm: #2a2416; --warm-rule: #5a4a26; --panel: #151b17; }
  }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.55 system-ui, -apple-system, sans-serif; }
  main { max-width: 48rem; margin: 0 auto; padding: 2rem 1rem 4rem; }
  h1 { font-size: 1.9rem; line-height: 1.15; margin: 0 0 .6rem; letter-spacing: -.01em; }
  h2 { font-size: 1.15rem; margin: 2.5rem 0 .4rem; }
  a { color: var(--accent); }
  code { font-size: .88em; background: var(--wash); padding: 0 .25em; border-radius: 4px; }
  summary { cursor: pointer; }
  .lede { margin: 0 0 1.5rem; color: var(--ink-2); max-width: 40rem; }
  .note { color: var(--ink-3); font-size: .85rem; margin: 0 0 .8rem; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: .9rem; }
  .card { display: flex; flex-direction: column; gap: .3rem; background: var(--surface); border: 1px solid var(--rule);
    border-radius: 14px; padding: 1.1rem 1.2rem; text-decoration: none; color: inherit; transition: border-color .15s, transform .15s; }
  .card:hover { border-color: var(--accent); transform: translateY(-1px); }
  .card.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  .card .kick { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3); }
  .card.primary .kick { color: #ffffffb3; }
  .card .title { font-size: 1.25rem; font-weight: 700; }
  .card p { margin: 0; font-size: .92rem; color: var(--ink-2); }
  .card.primary p { color: #ffffffd9; }
  .card .go { margin-top: auto; padding-top: .5rem; font-size: .9rem; font-weight: 650; color: var(--accent); }
  .card.primary .go { color: #fff; }
  @media (prefers-color-scheme: dark) { .card.primary { color: #0d1a12; } .card.primary .kick { color: #0d1a12b3; }
    .card.primary p { color: #0d1a12d9; } .card.primary .go { color: #0d1a12; } }
  .hero { font-size: 2rem; font-weight: 750; line-height: 1.1; font-variant-numeric: tabular-nums; }
  .hero small { font-size: .8rem; font-weight: 500; color: var(--ink-2); margin-left: .4rem; }
  .chart { width: 100%; max-width: 26rem; height: auto; display: block; }
  .chart .lab { fill: var(--ink-2); font-size: 11px; }
  .chart .val { fill: var(--ink); font-size: 11px; font-weight: 600; }
  .chart .line { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .chart .dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
  .now { background: var(--wash); border-radius: 14px; padding: .2rem 1.2rem 1rem; margin-top: 1.5rem; }
  .now h2 { margin: .9rem 0 .3rem; }
  .now p { margin: 0; }
  .lessons { margin: 0; padding: 0; list-style: none; counter-reset: l; background: var(--surface);
    border: 1px solid var(--rule); border-radius: 12px; }
  .lessons li { counter-increment: l; border-bottom: 1px solid var(--rule); }
  .lessons li:last-child { border-bottom: 0; }
  .lessons summary, .lessons li > span { display: flex; gap: .7rem; padding: .6rem 1rem; font-weight: 550; list-style: none; }
  .lessons summary::-webkit-details-marker { display: none; }
  .lessons summary::before, .lessons li > span::before { content: counter(l); flex: none; width: 1.4rem; color: var(--ink-3);
    font-weight: 600; font-variant-numeric: tabular-nums; }
  .lessons details[open] summary { color: var(--accent); }
  .lessons p { margin: 0; padding: 0 1rem .8rem 3.1rem; color: var(--ink-2); font-size: .95rem; }
  .steps { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; overflow: hidden; }
  .steps-head, .step > summary { display: grid; grid-template-columns: 2.2rem 1fr 5.5rem 5.5rem; gap: .5rem; align-items: center; padding: .55rem 1rem; }
  .steps-head { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-3);
    border-bottom: 1px solid var(--rule); padding-block: .45rem; }
  .steps-head span:nth-child(n+3), .v { text-align: right; font-variant-numeric: tabular-nums; }
  .step { border-bottom: 1px solid var(--rule); }
  .step:last-child { border-bottom: 0; }
  .step > summary { list-style: none; }
  .step > summary::-webkit-details-marker { display: none; }
  .step > summary:hover { background: var(--wash); }
  .step[open] > summary { background: var(--wash); }
  .step .n { color: var(--ink-3); font-weight: 600; font-variant-numeric: tabular-nums; }
  .step .t b { display: block; font-weight: 600; line-height: 1.3; }
  .step .t small { color: var(--ink-3); font-size: .78rem; }
  .v.none { color: var(--ink-3); }
  .v.best { font-weight: 700; color: var(--accent); }
  .v.best::after { content: " \\2605"; font-size: .8em; }
  .step .body { padding: .3rem 1rem 1rem 3.7rem; }
  .scores { display: flex; flex-wrap: wrap; gap: .4rem .9rem; margin: .4rem 0 .4rem; font-size: .88rem; font-variant-numeric: tabular-nums; }
  .score small { color: var(--ink-2); }
  .score.best b::before { content: "\\2605\\00a0"; color: var(--accent); }
  .sec h4 { margin: .6rem 0 .1rem; font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-3); }
  .sec p { margin: 0; }
  .sec.surprise { background: var(--warm); border: 1px solid var(--warm-rule); border-radius: 8px; padding: .1rem .7rem .5rem; margin-top: .7rem; }
  .open { display: inline-block; margin-top: .8rem; font-size: .9rem; font-weight: 650; text-decoration: none; }
  .part { margin-top: 3rem; padding-top: 1.6rem; border-top: 2px solid var(--rule); }
  .part-head { margin: 0 0 .7rem; }
  .part-kick { display: block; font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--accent); }
  .part-head h2 { font-size: 1.45rem; line-height: 1.2; margin: .15rem 0 0; letter-spacing: -.01em; scroll-margin-top: 4rem; }
  .part { scroll-margin-top: 3.5rem; }
  .panel { background: var(--panel); border: 1px solid var(--rule); border-radius: 16px; padding: 1.1rem 1.1rem .5rem; }
  .panel > p { margin: 0 0 .9rem; color: var(--ink-2); }
  .pipeline { list-style: none; counter-reset: p; margin: 0 0 .8rem; padding: 0; display: grid; gap: .6rem;
    grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); }
  .pipeline li { counter-increment: p; background: var(--surface); border: 1px solid var(--rule); border-radius: 12px;
    padding: .8rem 1rem; display: flex; flex-direction: column; gap: .25rem; }
  .pipeline li b::before { content: counter(p); display: inline-flex; align-items: center; justify-content: center;
    width: 1.5rem; height: 1.5rem; margin-right: .5rem; border-radius: 50%; background: var(--wash); color: var(--accent);
    font-size: .85rem; }
  .pipeline li span { color: var(--ink-2); font-size: .93rem; }
  .fold { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; margin: .6rem 0; }
  .fold > summary { list-style: none; padding: .7rem 1rem; font-weight: 650; display: flex; align-items: center; gap: .5rem; }
  .fold > summary::-webkit-details-marker { display: none; }
  .fold > summary::after { content: "\\25BE"; color: var(--accent); margin-left: auto; }
  .fold[open] > summary { border-bottom: 1px solid var(--rule); }
  .fold[open] > summary::after { content: "\\25B4"; }
  .fold > p, .fold > .note { margin: .7rem 1rem; }
  .fold > p { color: var(--ink-2); }
  .clues { margin: .4rem 1rem .9rem; }
  .clues dt { font-weight: 650; margin-top: .6rem; }
  .clues dd { margin: .1rem 0 0; color: var(--ink-2); font-size: .95rem; }
  .vers { margin: 0 0 .3rem; }
  .ver { display: grid; grid-template-columns: 7rem 1fr 5.5rem 5.5rem; gap: .6rem; align-items: baseline;
    padding: .55rem 1rem; border-top: 1px solid var(--rule); }
  .ver.head { font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--ink-3); border-top: 0; }
  .ver .vn b { display: block; }
  .ver .vn small { color: var(--ink-3); font-size: .78rem; }
  .ver .vc { color: var(--ink-2); font-size: .93rem; }
  .ver .vs { text-align: right; font-variant-numeric: tabular-nums; }
  .ver.now { background: var(--wash); }
  .ver.now .vn b, .ver.now .vs { color: var(--accent); font-weight: 700; }
  .ideas { display: grid; grid-template-columns: repeat(3, 1fr); gap: .7rem; padding: 0 1rem 1rem; }
  .ideas .col { border-radius: 10px; padding: .2rem .8rem .6rem; border: 1px solid var(--rule); }
  .ideas .col.worked { background: var(--wash); }
  .ideas .col.maybe { background: var(--warm); border-color: var(--warm-rule); }
  .ideas h3 { font-size: .95rem; margin: .6rem 0 .4rem; }
  .ideas h3 small { display: block; font-weight: 500; color: var(--ink-3); font-size: .75rem; }
  .ideas .col.worked h3 { color: var(--accent); }
  .ideas ul { list-style: none; margin: 0; padding: 0; }
  .ideas li { padding: .45rem 0; border-top: 1px solid var(--rule); font-size: .9rem; }
  .ideas li b { display: block; font-weight: 600; line-height: 1.3; }
  .ideas li span { color: var(--ink-2); font-size: .85rem; }
  @media (max-width: 44rem) { .ideas { grid-template-columns: 1fr; } }
  .other { padding-left: 1.2rem; }
  @media (max-width: 34rem) {
    main { padding-top: 1.4rem; }
    .panel { padding: .8rem .7rem .3rem; }
    .steps-head, .step > summary { grid-template-columns: 1.8rem 1fr 3.6rem 3.6rem; padding-inline: .7rem; gap: .4rem; }
    .steps-head span:nth-child(3), .steps-head span:nth-child(4) { font-size: .62rem; }
    .step .body { padding-left: .9rem; padding-right: .9rem; }
    .v.best::after { content: ""; }
    .ver { grid-template-columns: 1fr 4.6rem 3.8rem; padding-inline: .7rem; }
    .ver.head { font-size: .6rem; }
    .ver .vn { grid-column: 1; }
    .ver .vc { grid-column: 1 / -1; grid-row: 2; }
    .ver.head span:nth-child(2) { display: none; }
  }
</style>"""
