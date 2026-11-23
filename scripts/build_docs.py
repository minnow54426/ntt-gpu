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
PAGE_GROUPS = OrderedDict([
    ("Start here", "Build the project and choose a reading path."),
    ("Fundamentals", "Learn the transform, then follow it onto GPU threads."),
    ("CUDA runtime", "Understand arithmetic, buffers and resident operations."),
    ("OpenFHE", "Match polynomial formats, basis conversion and key switching."),
    ("Results", "Read measured results together with their scope and limits."),
    ("Research", "Compare implementation ideas and hardware choices."),
    ("Procedures", "Follow repeatable steps for optimization and GPU sessions."),
    ("Plans", "Check the current milestone and prepare the next deliverable."),
    ("Daily journal", "Browse dated work records; use the roadmap for current status."),
    ("Contribute", "Edit, verify and publish the notebook."),
])


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
        extensions=["meta", "tables", "fenced_code", "toc", "sane_lists", "md_in_html", MathExtension()],
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
        takeaways = page.get("takeaways")
        if not isinstance(takeaways, list) or not 1 <= len(takeaways) <= 3 or not all(
            isinstance(point, str) and point.strip() for point in takeaways
        ):
            raise ValueError(f"Expected one to three takeaways: {page['source']}")
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
    titles = {page["url"]: page["title"] for page in pages}
    for page in pages:
        if page.get("next_page") not in titles:
            raise ValueError(f"Unknown next page: {page['source']}")
        page["next_title"] = titles[page["next_page"]]
        if page["group"] not in PAGE_GROUPS:
            raise ValueError(f"Unknown page group: {page['source']}")
        if not page.get("prerequisite"):
            raise ValueError(f"Missing reading prerequisite: {page['source']}")
        before = page.get("before_page")
        if before and (before not in titles or before == page["url"]):
            raise ValueError(f"Unknown or self-referencing prerequisite: {page['source']}")
        page["before_title"] = titles.get(before, "")
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
    groups = OrderedDict((group, []) for group in PAGE_GROUPS)
    groups["Start here"].append(("index.html", "Overview"))
    groups["Daily journal"].append(("journal.html", "All entries"))
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


class SectionOutline(HTMLParser):
    """Read real headings without treating JavaScript strings as HTML."""

    def __init__(self, content):
        super().__init__(convert_charrefs=True)
        self.offsets = [0]
        for line in content.splitlines(keepends=True):
            self.offsets.append(self.offsets[-1] + len(line))
        self.ids = set()
        self.headings = []
        self.heading = None
        self.feed(content)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag == "h2":
            line, column = self.getpos()
            self.heading = {"id": attrs.get("id"), "text": [],
                            "offset": self.offsets[line - 1] + column + 3}

    def handle_data(self, data):
        if self.heading is not None:
            self.heading["text"].append(data)

    def handle_endtag(self, tag):
        if tag == "h2" and self.heading is not None:
            self.headings.append(self.heading)
            self.heading = None


def section_outline(content):
    outline = SectionOutline(content)
    links, insertions = [], []
    for heading in outline.headings:
        label = " ".join("".join(heading["text"]).replace("¶", "").split())
        if not label:
            continue
        anchor = heading["id"]
        if not anchor:
            base = re.sub(r"[^\w-]+", "-", label.lower()).strip("-") or "section"
            anchor, suffix = base, 2
            while anchor in outline.ids:
                anchor = f"{base}-{suffix}"
                suffix += 1
            outline.ids.add(anchor)
            insertions.append((heading["offset"], f' id="{escape(anchor, quote=True)}"'))
        links.append(f'<li><a href="#{quote(anchor)}">{escape(label)}</a></li>')
    for offset, value in reversed(insertions):
        content = content[:offset] + value + content[offset:]
    return content, '<ol>' + "".join(links) + '</ol>' if links else ""


def shell(page, content, pages, revision, repository, toc="", legacy_head="", public_base_url=""):
    content, toc = section_outline(content)
    current = page["url"]
    prefix = relative_url(current, "assets/").rstrip("/")
    home = relative_url(current, "index.html")
    source = page.get("source")
    source_link = (
        f'<a href="{repository}/blob/{revision}/{quote(source)}">View source</a>'
        if source else f'<a href="{repository}">Repository</a>'
    )
    math_head = "" if legacy_head else f"""
<link rel="stylesheet" href="{relative_url(current, 'vendor/katex/katex.min.css')}">
<script defer src="{relative_url(current, 'vendor/katex/katex.min.js')}"></script>
<script defer src="{relative_url(current, 'vendor/katex/contrib/auto-render.min.js')}"></script>
<script defer src="{relative_url(current, 'math-rendering.js')}"></script>"""
    title = escape(page["title"])
    description = escape(page["summary"], quote=True)
    article_toc = (
        '<aside class="site-on-this-page" id="site-section-nav" aria-label="On this page">'
        '<div class="site-contents-heading"><p>On this page</p>'
        '<button type="button" class="site-contents-close" aria-label="Close sections">×</button></div>'
        f'<nav class="site-section-links" aria-label="Page sections">{toc}</nav></aside>'
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
<link rel="stylesheet" href="{prefix}/sunlit-paper.css">
<script defer src="{relative_url(current, 'search-index.js')}"></script>
<script defer src="{prefix}/site.js"></script>
</head>
<body class="site-body" data-search-index="{relative_url(current, 'search-index.json')}">
<a class="site-skip" href="#site-main">Skip to content</a>
<aside class="site-sidebar" id="site-sidebar">
  <a class="site-brand" href="{home}"><span class="site-mark" aria-hidden="true">N<span>×</span></span>
    <span>ntt-gpu<small>Engineering notebook</small></span></a>
  <div class="site-nav-label">Documentation <span>Index</span></div>
  <nav class="site-navigation" aria-label="Documentation">{navigation(pages, current)}</nav>
  <div class="site-sidebar-bottom"><span class="site-status-dot"></span> Building in the open
    <a href="{repository}">GitHub repository</a></div>
</aside>
<div class="site-workspace">
  <header class="site-topbar">
    <button class="site-menu-button" type="button" aria-controls="site-sidebar" aria-expanded="false">☰ <span>Menu</span></button>
    <div class="site-breadcrumb"><a href="{home}">Notebook</a><span>/</span>{escape(page["group"])}</div>
{'    <button class="site-contents-button" type="button" aria-controls="site-section-nav" aria-expanded="false">Sections</button>' if toc else ''}
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
  <span class="site-entry-link">Read the entry</span>
</a>""")
    return "".join(cards)


def page_directory(pages, entries):
    directory_pages = list(pages) + [{
        "url": "journal.html", "title": "All journal entries",
        "group": "Daily journal", "summary": "The complete daily progress timeline.",
    }] + list(entries)
    groups = []
    for group, description in PAGE_GROUPS.items():
        links = []
        for page in directory_pages:
            if page["group"] != group:
                continue
            title = (page["date"] + " · " if page.get("date") else "") + page["title"]
            links.append(
                f'<a class="site-directory-link" href="{escape(page["url"], quote=True)}">'
                f'{escape(title)}</a>'
            )
        groups.append(
            f'<details class="site-directory-group"{" open" if group == "Start here" else ""}>'
            f'<summary>{escape(group)}</summary><p>{escape(description)}</p>'
            + "".join(links) + "</details>"
        )
    return '<div class="site-directory">' + "".join(groups) + "</div>"


def home_content(pages, entries):
    state = checkpoint()
    latest = entries[0]["date"] if entries else "No published entries"
    return f"""
<div class="site-home-hero">
  <div><p class="site-eyebrow"><span class="site-status-dot"></span> An open engineering notebook</p>
    <h1>GPU NTT for CKKS</h1>
    <p class="site-lede">Learn how the NTT helps multiply polynomials, see the GPU test results, and follow the work toward OpenFHE integration.</p>
    <div class="site-hero-actions"><a class="site-primary-link" href="paired-ntt-native-analysis.html">Read the latest GPU analysis</a><a class="site-secondary-link" href="#all-pages">Browse all pages</a></div>
    <div class="site-specs"><span>CKKS · Hybrid</span><span>N = 4,096–65,536</span><span>Target: RTX 5090</span></div>
  </div>
  <div class="site-checkpoint"><div class="site-checkpoint-label">Working day <span>{escape(state['Working day'])}</span></div>
    <div class="site-checkpoint-symbol" aria-hidden="true">⌁</div>
    <h2>Current checkpoint</h2><p><strong>{escape(state['Active phase'])}</strong></p>
    <p>{escape(state['Active task']).rstrip('.')}.</p>
    <a href="openfhe-integration-plan.html">Explore the working-day plan</a>
  </div>
</div>
<div class="site-evidence-strip">
  <div><span class="site-evidence-label">Verified</span><strong><span class="site-status-dot"></span> Temporary GPU baseline passed</strong><p>Recorded answer checks, memory/threading checks and timings.</p></div>
  <div><span class="site-evidence-label">Still pending</span><strong><span class="site-pending-dot"></span> RTX 5090 validation</strong><p>Validate the prepared resident conversion chain, then measure its costs.</p></div>
  <div><span class="site-evidence-label">Current record</span><strong>{escape(latest)}</strong><p>Latest published journal entry.</p></div>
</div>
<section class="site-article" aria-labelledby="reading-path">
<h2 id="reading-path">New to the project?</h2>
<p>CKKS is an encryption scheme that allows calculations on encrypted data. It represents data using polynomials. OpenFHE is the cryptography library we aim to integrate with. The Number Theoretic Transform (NTT) changes how a polynomial is represented so multiplication becomes easier. This project first aims to run that transform efficiently on a GPU.</p>
<ol>
<li><a href="butterfly-explained.html">Learn the NTT and butterflies</a> with small numerical examples.</li>
<li><a href="gpu-ntt-explained.html">See how the GPU divides the work</a>, then trace one thread in the <a href="ntt-65536-work-distribution.html">65536-point explorer</a>.</li>
<li><a href="batch40-native-analysis.html">Read the measured results</a> or <a href="project.html">build the project</a>.</li>
</ol>
<h2 id="next-step">What comes next?</h2>
<p>The paired Ada analysis is complete. Native ModUp/ModDown checks, timings and tagged profiles are prepared as one RTX 5090 batch for later. Offline work continues on builds, CPU checks and profiling preparation; no rental is needed for those tasks.</p>
<p>Use the <a href="next-gpu-rental.html">current GPU session queue</a> for commands and the <a href="openfhe-integration-plan.html#live-tracker">live tracker</a> for acceptance requirements. D11/P2 stays open until its required evidence is recorded.</p>
<details><summary>Useful words before you start</summary>
<ul>
<li><strong>Modulo q:</strong> keep the remainder after division by q. For example, 20 modulo 17 is 3.</li>
<li><strong>Coefficient:</strong> a number multiplying a power of x in a polynomial. An evaluation is the value obtained by substituting a point for x.</li>
<li><strong>Tower:</strong> one polynomial represented under one prime. Several towers represent its coefficients under several primes.</li>
<li><strong>Host / device:</strong> the CPU and its memory / the GPU and its memory. An upload copies host data to the device; a download copies it back.</li>
<li><strong>Kernel / stream:</strong> a function executed by GPU threads / an ordered queue of GPU work.</li>
<li><strong>Fixture / oracle:</strong> repeatable test inputs / a separate calculation used to check the answers.</li>
<li><strong>Native check:</strong> a test that actually runs on the target hardware. Compiling CUDA alone does not run a GPU test.</li>
<li><strong>Gate:</strong> a required check before work can be called complete. D labels a working milestone and P labels a phase.</li>
</ul></details>
</section>
<div class="site-section-heading" id="all-pages"><h2>All pages</h2></div>
<p class="site-directory-note">Choose a lesson, reference, plan, or dated journal entry.</p>
{page_directory(pages, entries)}
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


def concise_article(page, full_content):
    journal = page["group"] == "Daily journal"
    lesson = page["source"].endswith(".html")
    label = "Dated work record" if journal else "Interactive lesson" if lesson else "Detailed guide"
    title = escape(page["title"])
    points = page.get("takeaways", [page["summary"]])
    brief = (
        f'<p>{escape(points[0])}</p>' if len(points) == 1 else
        '<ul>' + ''.join(f'<li>{escape(point)}</li>' for point in points) + '</ul>'
    )
    if journal:
        next_url = relative_url(page["url"], "openfhe-integration-plan.html")
        next_label = "Current roadmap"
        brief += '<p>This is a dated record of work on its calendar date. Its test counts, design and next steps describe that time; use the current roadmap for today’s status.</p>'
        before = '<p class="site-prerequisite"><strong>Before you start:</strong> No earlier journal entry is required. For current priorities, read the roadmap linked below.</p>'
    else:
        next_url = relative_url(page["url"], page["next_page"])
        next_label = "Next: " + page["next_title"]
        before = '<p class="site-prerequisite"><strong>Before you start:</strong> ' + escape(page["prerequisite"])
        if page.get("before_page"):
            before += ' Read <a href="' + relative_url(page["url"], page["before_page"]) + '">' + escape(page["before_title"]) + '</a> first.'
        before += '</p>'
    following = f'<nav class="site-article site-read-next" aria-label="Continue reading"><h2 id="site-continue">Continue reading</h2><a href="{escape(next_url, quote=True)}">{escape(next_label)}</a></nav>'
    return (
        f'<section class="site-article site-brief"><h1>{title}</h1><p class="site-summary">{escape(page["summary"])}</p>'
        f'<h2 id="site-overview">At a glance</h2>{brief}{before}</section>'
        f'<details class="site-full-notes"{"" if journal else " open"}><summary>{label}</summary>{full_content}</details>'
        f'{following}'
    )


def preserve_title_anchor(body):
    """The shared header supplies h1; retain the source title's bookmark."""
    return re.sub(r"<h1\b([^>]*)>[\s\S]*?</h1>",
                  r'<div\1 class="site-title-anchor" aria-hidden="true"></div>', body, count=1)


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
        shutil.copy2(ROOT / "docs/lesson-examples.js", output / "lesson-examples.js")

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
                head = head.replace('href="site-assets/', 'href="assets/')
                body = re.search(r"<body[^>]*>([\s\S]*?)</body>", text)[1]
                body = body.replace("<main>", '<div class="legacy-main">').replace("</main>", "</div>")
                body = preserve_title_anchor(body)
                # HTML explainers have root-relative sibling assets already.
                body = rewrite_links(body, page["source"], page["url"], routes, repository, revision)
                write_page(page, concise_article(page, '<div class="site-legacy">' + body + "</div>"),
                           legacy_head=head)
            else:
                body, toc, _ = render_markdown(text)
                body = rewrite_links(body, page["source"], page["url"], routes, repository, revision)
                body = preserve_title_anchor(body)
                content = concise_article(page, '<article class="site-article">' + body + "</article>")
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
<div class="site-journal-intro"><h1>Daily journal</h1><p>Read what changed, why it changed and which checks ran on each calendar date. An older entry may describe work that was unfinished then. Its results remain tied to the recorded source version and machine. Use the <a href="openfhe-integration-plan.html">current roadmap</a> to find today’s milestone; a roadmap working day is not a calendar date.</p></div>
<section><h2>Entries</h2><div class="site-journal-list">{entry_cards(entries, 'journal.html')}</div></section>""")
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
