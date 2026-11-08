#!/usr/bin/env python3
"""Build the project notebook as a portable, link-checked static HTML site."""

import argparse
from collections import OrderedDict
from datetime import date
from html import escape, unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import quote, unquote, urljoin, urlsplit

try:
    import markdown
    from markdown.extensions import Extension
    from markdown.preprocessors import Preprocessor
except ImportError as error:
    raise SystemExit(
        "Install documentation dependencies: python -m pip install -r docs/requirements.txt"
    ) from error


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "build" / "docs-site"
JOURNAL_SECTIONS = (
    "Outcome", "Reasoning", "Changes", "Verification", "Open questions", "Next step"
)


class MathPreserver(Preprocessor):
    """Stash math before Markdown consumes its backslashes or underscores."""

    def run(self, lines):
        text = "\n".join(lines)
        # Inline code and fenced-code placeholders must stay literal.
        pattern = r"`+[^`\n]*`+|\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]"

        def preserve(match):
            value = match.group()
            if value.startswith("`"):
                return value
            return self.md.htmlStash.store(
                '<span class="site-math">' + escape(value) + "</span>"
            )

        return re.sub(pattern, preserve, text).split("\n")


class MathExtension(Extension):
    def extendMarkdown(self, md):
        md.preprocessors.register(MathPreserver(md), "preserve_math", 24)


def render_markdown(text):
    converter = markdown.Markdown(
        extensions=["meta", "tables", "fenced_code", "toc", "sane_lists", MathExtension()],
        extension_configs={"toc": {"permalink": "¶", "toc_depth": "2-3"}},
        output_format="html",
    )
    rendered = converter.convert(text)
    rendered = re.sub(
        r"<li>\[([ xX])\] ",
        lambda match: '<li class="site-task"><input type="checkbox" disabled'
        + (" checked" if match[1].lower() == "x" else "")
        + ' aria-label="Task status"> ',
        rendered,
    )
    metadata = {key: " ".join(value) for key, value in converter.Meta.items()}
    return rendered, converter.toc, metadata


def git_revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def read_catalog():
    catalog = json.loads((ROOT / "docs/site.json").read_text())
    pages = catalog["pages"]
    for page in pages:
        for field in ("source", "url", "title", "group", "summary"):
            if not page.get(field):
                raise ValueError(f"Catalog entry missing {field}: {page}")
        source = (ROOT / page["source"]).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError(f"Invalid source: {page['source']}")
        route = Path(page["url"])
        if route.is_absolute() or ".." in route.parts or route.suffix != ".html":
            raise ValueError(f"Invalid HTML route: {route}")
        if page["url"] in ("index.html", "journal.html") or route.parts[0] in (
            "journal", "assets", "vendor"
        ):
            raise ValueError(f"Reserved generated route: {route}")
    sources = [page["source"] for page in pages]
    routes = [page["url"] for page in pages]
    if len(set(sources)) != len(sources) or len(set(routes)) != len(routes):
        raise ValueError("Duplicate source or URL in the catalog")
    discovered = {
        str(path.relative_to(ROOT))
        for pattern in ("*.md", "*.html")
        for path in (ROOT / "docs").rglob(pattern)
        if path.relative_to(ROOT / "docs").parts[0]
        not in ("journal", "vendor", "site-assets")
    } | {"README.md"}
    if discovered != set(sources):
        raise ValueError(f"Uncataloged/missing documents: {discovered ^ set(sources)}")
    return catalog


def read_journal():
    entries = []
    for source in sorted((ROOT / "docs/journal").glob("*.md"), reverse=True):
        if source.name.startswith("_"):
            continue
        body, toc, metadata = render_markdown(source.read_text())
        for field in ("title", "date", "summary", "phase", "status"):
            if not metadata.get(field):
                raise ValueError(f"{source.name}: missing {field}")
        entry_date = date.fromisoformat(metadata["date"])
        if source.stem != entry_date.isoformat():
            raise ValueError(f"{source.name}: filename and date disagree")
        if metadata["status"] not in ("draft", "published"):
            raise ValueError(f"{source.name}: status must be draft or published")
        if metadata["status"] == "draft":
            continue
        headings = set(re.findall(r"^## (.+)$", source.read_text(), re.MULTILINE))
        if not set(JOURNAL_SECTIONS).issubset(headings):
            raise ValueError(f"{source.name}: missing required journal sections")
        entries.append({
            **metadata,
            "source": str(source.relative_to(ROOT)),
            "url": f"journal/{source.stem}.html",
            "group": "Daily journal",
            "body": body,
            "toc": toc,
        })
    return entries


def relative_url(current, target):
    return Path(os.path.relpath(target, Path(current).parent)).as_posix()


def rewrite_url(value, source, current, source_routes, repository, revision):
    parsed = urlsplit(unescape(value))
    if parsed.scheme or parsed.netloc or not parsed.path:
        return value
    target = ((ROOT / source).parent / unquote(parsed.path)).resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError(f"Link leaves repository: {source}: {value}")
    source_path = target.relative_to(ROOT).as_posix()
    if source_path in source_routes:
        result = relative_url(current, source_routes[source_path])
    elif target.is_file():
        result = f"{repository}/blob/{revision}/{quote(source_path)}"
    else:
        raise ValueError(f"Missing link target in {source}: {value}")
    if parsed.query:
        result += "?" + parsed.query
    if parsed.fragment:
        result += "#" + parsed.fragment
    return escape(result, quote=True)


def rewrite_links(body, source, current, source_routes, repository, revision):
    # Markdown-generated attributes are double quoted; script text is not touched.
    return re.sub(
        r'(?P<prefix>\bhref=")(?P<url>[^"]*)(?P<end>")',
        lambda match: match["prefix"] + rewrite_url(
            match["url"], source, current, source_routes, repository, revision
        ) + match["end"],
        body,
    )


def navigation(pages, current):
    groups = OrderedDict([
        ("Start here", [("index.html", "Overview")]),
        ("Daily journal", [("journal.html", "All entries")]),
    ])
    for page in pages:
        groups.setdefault(page["group"], []).append((page["url"], page["title"]))
    output = []
    for group, links in groups.items():
        output.append(f'<div class="site-nav-group"><p>{escape(group)}</p>')
        for url, title in links:
            active = current == url or (url == "journal.html" and current.startswith("journal/"))
            output.append(
                f'<a href="{relative_url(current, url)}"'
                + (' aria-current="page"' if active else "")
                + f'>{escape(title)}</a>'
            )
        output.append("</div>")
    return "".join(output)


def shell(page, content, pages, revision, repository, toc="", legacy_head="", public_base_url=""):
    current = page["url"]
    prefix = relative_url(current, "assets/").rstrip("/")
    home = relative_url(current, "index.html")
    source = page.get("source")
    source_link = (
        f'<a href="{repository}/blob/{revision}/{quote(source)}">View source ↗</a>'
        if source else f'<a href="{repository}">Repository ↗</a>'
    )
    math_head = "" if legacy_head else f"""
<link rel="stylesheet" href="{relative_url(current, 'vendor/katex/katex.min.css')}">
<script defer src="{relative_url(current, 'vendor/katex/katex.min.js')}"></script>
<script defer src="{relative_url(current, 'vendor/katex/contrib/auto-render.min.js')}"></script>
<script defer src="{relative_url(current, 'math-rendering.js')}"></script>"""
    # The GPU explainer has no math loader; its own formulas are SVG/Unicode.
    title = escape(page["title"])
    description = escape(page["summary"], quote=True)
    article_toc = (
        f'<aside class="site-on-this-page" aria-label="On this page"><p>On this page</p>{toc}</aside>'
        if toc else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="{description}">
<link rel="canonical" href="{escape(urljoin(public_base_url, current), quote=True)}">
<title>{title} · ntt-gpu</title>
{legacy_head}
{math_head}
<link rel="stylesheet" href="{prefix}/site.css">
<script defer src="{relative_url(current, 'search-index.js')}"></script>
<script defer src="{prefix}/site.js"></script>
</head>
<body class="site-body" data-search-index="{relative_url(current, 'search-index.json')}">
<a class="site-skip" href="#site-main">Skip to content</a>
<aside class="site-sidebar" id="site-sidebar">
  <a class="site-brand" href="{home}"><span class="site-mark" aria-hidden="true">N<span>×</span></span>
    <span>ntt-gpu<small>THE ENGINEERING NOTEBOOK</small></span></a>
  <div class="site-nav-label">DOCUMENTATION <span>INDEX</span></div>
  <nav class="site-navigation" aria-label="Documentation">{navigation(pages, current)}</nav>
  <div class="site-sidebar-bottom"><span class="site-status-dot"></span> Building in the open
    <a href="{repository}">GitHub repository ↗</a></div>
</aside>
<div class="site-workspace">
  <header class="site-topbar">
    <button class="site-menu-button" type="button" aria-controls="site-sidebar" aria-expanded="false">☰ <span>Menu</span></button>
    <div class="site-breadcrumb"><a href="{home}">Notebook</a><span>/</span>{escape(page["group"])}</div>
    <button class="site-search-button" type="button" aria-haspopup="dialog"><span>⌕</span> Search documentation <kbd>⌘ K</kbd></button>
  </header>
  <main id="site-main" class="site-main">
    <div class="site-page-meta"><span>{escape(page["group"])}</span>{source_link}</div>
    <div class="site-reading-layout{' site-wide' if not toc else ''}">
      <div class="site-page-content">{content}</div>{article_toc}
    </div>
  </main>
  <footer class="site-footer"><span>ntt-gpu · CKKS / CUDA / OpenFHE</span>
    <span>Source revision <a href="{repository}/tree/{revision}">{revision[:7]}</a> · See journal for changes in this build</span></footer>
</div>
<dialog class="site-search-dialog" aria-labelledby="site-search-title">
  <div class="site-search-heading"><h2 id="site-search-title">Search the notebook</h2>
    <button type="button" class="site-search-close" aria-label="Close search">Esc</button></div>
  <label class="site-visually-hidden" for="site-search-input">Search documentation</label>
  <input id="site-search-input" type="search" placeholder="Try “twiddle”, “Hybrid”, or “2026-11-07”…" autocomplete="off">
  <p class="site-search-status" role="status">Search articles, plans, and daily entries.</p>
  <div class="site-search-results"></div>
</dialog>
</body>
</html>
"""


def checkpoint():
    text = (ROOT / "docs/openfhe-integration-plan.md").read_text()
    result = {}
    for key in ("Working day", "Active phase", "Active task", "Next gate"):
        match = re.search(rf"^\| {key} \| (.*?) \|$", text, re.MULTILINE)
        if not match:
            raise ValueError(f"Integration plan missing tracker field: {key}")
        result[key] = match[1].replace("**", "")
    return result


def entry_cards(entries, current, limit=None):
    cards = []
    for entry in entries[:limit]:
        formatted_date = date.fromisoformat(entry["date"]).strftime("%d %b %Y")
        cards.append(f"""
<a class="site-entry-card" href="{relative_url(current, entry['url'])}">
  <div class="site-entry-meta"><time datetime="{entry['date']}">{formatted_date}</time><span>{escape(entry['phase'])}</span></div>
  <h3>{escape(entry['title'])}</h3><p>{escape(entry['summary'])}</p>
  <span class="site-entry-link">Read the entry <span aria-hidden="true">↗</span></span>
</a>""")
    return "".join(cards)


def page_directory(pages, entries, public_base_url=None):
    public_base_url = public_base_url or read_catalog()["public_base_url"]
    directory_pages = list(pages) + [{
        "url": "journal.html", "title": "All journal entries",
        "group": "Daily journal", "summary": "The complete daily progress timeline.",
    }] + list(entries)
    groups = []
    for group in ("Start here", "Learn", "Research", "Plans", "Daily journal", "Contribute"):
        links = []
        for page in directory_pages:
            if page["group"] != group:
                continue
            title = (page["date"] + " · " if page.get("date") else "") + page["title"]
            links.append(
                f'<a class="site-directory-link" href="{escape(page["url"], quote=True)}">'
                f'<strong>{escape(title)} <span aria-hidden="true">↗</span></strong>'
                f'<p>{escape(page["summary"])}</p>'
                f'<code class="site-page-url" aria-label="Public URL">{escape(urljoin(public_base_url, page["url"]))}</code></a>'
            )
        groups.append(
            f'<div class="site-directory-group"><h3>{escape(group)}</h3>'
            + "".join(links) + "</div>"
        )
    return '<div class="site-directory">' + "".join(groups) + "</div>"


def home_content(pages, entries):
    state = checkpoint()
    latest = entries[0]["date"] if entries else "No published entries"
    collections = []
    for number, group in enumerate(("Learn", "Research", "Plans"), 1):
        links = "".join(
            f'<a href="{page["url"]}">{escape(page["title"])} <span>↗</span></a>'
            for page in pages if page["group"] == group
        )
        descriptions = {
            "Learn": "Follow the mathematics into the code.",
            "Research": "Read the evidence behind our choices.",
            "Plans": "See the gates between here and integration.",
        }
        collections.append(f'<div class="site-collection"><span class="site-section-number">0{number}</span><h3>{group}</h3><p>{descriptions[group]}</p>{links}</div>')
    return f"""
<div class="site-home-hero">
  <div><p class="site-eyebrow"><span class="site-status-dot"></span> AN OPEN ENGINEERING PROJECT</p>
    <h1>Building the GPU<br>path for <em>CKKS.</em></h1>
    <p class="site-lede">From one butterfly to OpenFHE integration.<br>Follow the code, the evidence, and the decisions along the way.</p>
    <div class="site-hero-actions"><a class="site-primary-link" href="#all-pages">Browse all pages <span>↓</span></a><a class="site-secondary-link" href="journal.html">Follow our progress →</a></div>
    <div class="site-specs"><span>RTX 5090</span><span>u64 / u32</span><span>N = 4,096–65,536</span></div>
  </div>
  <div class="site-checkpoint"><div class="site-checkpoint-label">CURRENT CHECKPOINT <span>{escape(state['Working day'])}</span></div>
    <div class="site-checkpoint-symbol" aria-hidden="true">⌁</div>
    <h2>{escape(state['Active phase'])}</h2>
    <p>{escape(state['Active task'])}.</p>
    <a href="openfhe-integration-plan.html">Explore the working-day plan <span>↗</span></a>
  </div>
</div>
<div class="site-evidence-strip">
  <div><span class="site-evidence-label">VERIFIED BASELINE</span><strong><span class="site-status-dot"></span> CPU + OpenFHE exactness</strong><p>Array equality and schedule simulation.</p></div>
  <div><span class="site-evidence-label">GPU EXECUTION</span><strong><span class="site-pending-dot"></span> Hardware gate pending</strong><p>CUDA source exists; RTX validation is next.</p></div>
  <div><span class="site-evidence-label">CURRENT RECORD</span><strong>{escape(latest)}</strong><p>Read dated evidence in the journal.</p></div>
</div>
<div class="site-section-heading" id="all-pages"><div><p class="site-eyebrow">THE MAIN DIRECTORY</p><h2>All pages, one place.</h2></div></div>
<p class="site-directory-note">Open any page below. The public URLs shown here are ready to bookmark or share.</p>
{page_directory(pages, entries)}
<div class="site-section-heading"><div><p class="site-eyebrow">THE DIRECTION</p><h2>One pipeline. Explicit gates.</h2></div><a href="openfhe-integration-plan.html">Full roadmap →</a></div>
<div class="site-pipeline" aria-label="Implementation sequence">
  <div class="site-pipeline-active"><span>01 / IN PROGRESS</span><strong>GPU NTT</strong><small>Exactness → hardware → tuning</small></div>
  <div><span>02 / PLANNED</span><strong>Base conversion</strong><small>Q ↔ QP</small></div>
  <div><span>03 / PLANNED</span><strong>Hybrid key switch</strong><small>Decompose → MAC → ModDown</small></div>
  <div><span>04 / PLANNED</span><strong>OpenFHE</strong><small>Integration + CPU fallback</small></div>
  <div><span>05 / AFTER INTEGRATION</span><strong>Cipher inference</strong><small>A separate delivery phase</small></div>
</div>
<div class="site-section-heading"><div><p class="site-eyebrow">THE DAILY JOURNAL</p><h2>What changed. What we learned.</h2></div><a href="journal.html">All entries →</a></div>
<div class="site-entry-grid">{entry_cards(entries, 'index.html', 3)}</div>
<div class="site-section-heading"><div><p class="site-eyebrow">EXPLORE THE NOTEBOOK</p><h2>Choose a thread to follow.</h2></div></div>
<div class="site-collections">{''.join(collections)}</div>
<div class="site-contribute-note"><strong>New here?</strong> Run the <a href="vscode-debugging.html">N=8 example</a>, or learn <a href="contributing-docs.html">how to add the next journal entry</a>.</div>
"""


class PageLinks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.references = []
        self.ids = set()
        self.text = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if attrs.get(key):
                self.references.append(attrs[key])

    def handle_data(self, data):
        self.text.append(data)


def check_links(output):
    output = output.resolve()
    parsed_pages = {}
    for path in output.rglob("*.html"):
        parser = PageLinks()
        parser.feed(path.read_text())
        parsed_pages[path.resolve()] = parser
    errors = []
    for page, parser in parsed_pages.items():
        for reference in parser.references:
            parsed = urlsplit(reference)
            if parsed.scheme or parsed.netloc:
                continue
            target = (page.parent / unquote(parsed.path)).resolve() if parsed.path else page
            if not target.is_relative_to(output.resolve()) or not target.is_file():
                errors.append(f"{page.relative_to(output)}: missing target {reference}")
            elif parsed.fragment and target in parsed_pages:
                if unquote(parsed.fragment) not in parsed_pages[target].ids:
                    errors.append(f"{page.relative_to(output)}: missing fragment {reference}")
    if errors:
        raise ValueError("\n".join(errors))
    return len(parsed_pages)


def build():
    catalog = read_catalog()
    pages = catalog["pages"]
    entries = read_journal()
    revision = git_revision()
    repository = catalog["repository"]
    routes = {page["source"]: page["url"] for page in pages + entries}
    search = []
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="docs-site-", dir=OUTPUT.parent) as staging:
        output = Path(staging)
        shutil.copytree(ROOT / "docs/site-assets", output / "assets")
        shutil.copytree(ROOT / "docs/vendor", output / "vendor")
        shutil.copy2(ROOT / "docs/math-rendering.js", output / "math-rendering.js")

        def write_page(page, body, toc="", legacy_head=""):
            destination = output / page["url"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(shell(page, body, pages, revision, repository, toc, legacy_head, catalog["public_base_url"]))
            parser = PageLinks()
            # Strip scripts/styles from indexed text, keeping the article itself.
            indexed = re.sub(r"<(script|style)\b[^>]*>[\s\S]*?</\1>", "", body)
            parser.feed(indexed)
            search.append({
                "title": page["title"], "summary": page["summary"],
                "group": page["group"], "url": page["url"],
                "text": " ".join(" ".join(parser.text).split()),
            })

        for page in pages + entries:
            text = (ROOT / page["source"]).read_text()
            if page["source"].endswith(".html"):
                head = re.search(r"<head>([\s\S]*?)</head>", text)[1]
                head = re.sub(r"<title>[\s\S]*?</title>|<meta[^>]*>", "", head)
                body = re.search(r"<body[^>]*>([\s\S]*?)</body>", text)[1]
                body = body.replace("<main>", '<div class="legacy-main">').replace("</main>", "</div>")
                # HTML explainers have root-relative sibling assets already.
                body = rewrite_links(body, page["source"], page["url"], routes, repository, revision)
                write_page(page, '<div class="site-legacy">' + body + "</div>", legacy_head=head)
            else:
                body, toc, _ = render_markdown(text)
                body = rewrite_links(body, page["source"], page["url"], routes, repository, revision)
                content = '<article class="site-article">' + body + "</article>"
                if page["group"] == "Daily journal":
                    content = f'<div class="site-journal-banner"><time datetime="{page["date"]}">{page["date"]}</time><span>{escape(page["phase"])}</span><a href="../journal.html">← All entries</a></div>' + content
                write_page(page, content, toc)

        home = {"url": "index.html", "title": "Building the GPU path for CKKS",
                "summary": "The open engineering notebook for ntt-gpu: progress, reasoning, and exactness.",
                "group": "Start here"}
        write_page(home, home_content(pages, entries))
        journal = {"url": "journal.html", "title": "Daily engineering journal",
                   "summary": "Dated outcomes, reasoning, verification, and next steps.",
                   "group": "Daily journal"}
        write_page(journal, f"""
<div class="site-journal-intro"><p class="site-eyebrow">THE DAILY JOURNAL</p>
<h1>The work, as it happens.</h1><p>Outcomes, decisions, and the evidence behind them.
Follow the project one entry at a time.</p></div>
<div class="site-journal-explainer">Calendar entries track the work. Roadmap working days track deliverables.
Entries reconstructed from history are labeled in their articles.</div>
<div class="site-journal-list">{entry_cards(entries, 'journal.html')}</div>""")
        (output / "search-index.json").write_text(json.dumps(search, ensure_ascii=False))
        (output / "search-index.js").write_text(
            "window.nttNotebookSearchIndex = " + json.dumps(search, ensure_ascii=False) + ";\n"
        )
        (output / ".nojekyll").touch()
        count = check_links(output)
        if OUTPUT.is_symlink():
            raise ValueError("Refusing to replace a symlink at the documentation output")
        if OUTPUT.exists():
            shutil.rmtree(OUTPUT)
        shutil.copytree(output, OUTPUT)
    print(f"Built {count} HTML pages, including {len(entries)} journal entries.")
    print(f"Validated local links and fragments. Output: {OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build and validate the site (validation is always enabled).")
    parser.parse_args()
    try:
        build()
    except (ValueError, OSError) as error:
        raise SystemExit(f"Documentation build failed: {error}") from error


if __name__ == "__main__":
    main()
