"""Regression checks for published routes, journal integrity, and Markdown."""
import importlib.util
from html import escape
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


site = load_script("build_docs")
journal = load_script("new_journal")
with patch.dict(sys.modules, {"build_docs": site}):
    website = load_script("export_website")


class MarkdownTests(unittest.TestCase):
    def test_section_outline_preserves_anchors_scripts_and_duplicate_headings(self):
        source = '''<section id="example"><h2>Example</h2>
<h2 id="existing">Existing &amp; exact <a class="headerlink">¶</a></h2>
<h2>Example</h2><script>const label = "<h2>Not a heading</h2>";</script></section>'''
        body, outline = site.section_outline(source)
        self.assertIn('<h2 id="example-2">', body)
        self.assertIn('<h2 id="example-3">', body)
        self.assertIn('href="#existing">Existing &amp; exact</a>', outline)
        self.assertIn('const label = "<h2>Not a heading</h2>";', body)
        self.assertNotIn('Not a heading', outline)
        self.assertEqual(outline.count('<li>'), 3)

    def test_collapsed_markdown_keeps_links_and_math_readable(self):
        body, _, _ = site.render_markdown('''## Evidence

<details markdown="1">
<summary>Sources</summary>

[Reference](https://example.com) and \\(a_i\\).

| Size | State |
|---|---|
| 65536 | checked |

</details>
''')
        self.assertIn('<details>', body)
        self.assertIn('<a href="https://example.com">Reference</a>', body)
        self.assertIn('<table>', body)
        self.assertIn('class="site-math"', body)

    def test_tables_tasks_code_and_anchors(self):
        source = """# Page

## Exactness

| Size | State |
|---|---|
| 65536 | checked |

- [x] CPU gate
- [ ] GPU gate

```cpp
values[i] = a < b;
```
"""
        html, toc, _ = site.render_markdown(source)
        self.assertIn("<table>", html)
        self.assertIn('id="exactness"', html)
        self.assertIn('href="#exactness"', toc)
        self.assertIn("checked", html)
        self.assertIn("a &lt; b", html)

    def test_math_survives_but_code_is_literal(self):
        source = r"""# Formula

Inline \(a_i = \psi^{2k+1}\).

\[\sum_{i=0}^{N-1} a_i\]

Write `\(a_i\)` as source.

```text
\(literal_code\)
```
"""
        html, _, _ = site.render_markdown(source)
        self.assertIn(r"\(a_i = \psi^{2k+1}\)", html)
        self.assertIn(r"\[\sum_{i=0}^{N-1} a_i\]", html)
        self.assertIn(r"<code>\(a_i\)</code>", html)
        self.assertIn(r"\(literal_code\)", html)
        self.assertEqual(html.count('class="site-math"'), 2)


class RoutingTests(unittest.TestCase):
    def test_visual_guides_cover_articles_and_keep_3d_optional(self):
        pages = site.read_catalog()['pages']
        for page in pages:
            self.assertIn(page['visual_guide'], site.VISUAL_GUIDES)
            body = site.concise_article(page, '<h2>Original explanation</h2>')
            self.assertIn(f'<h1>{escape(page["title"])}</h1>', body)
            html = site.shell(page, body, pages, 'abc123', 'https://example.test/repo')
            self.assertIn('Beginner visual guides', html)
            if page.get('visual_demo'):
                self.assertIn(f'id="visual-{page["visual_demo"]}"', body)
                self.assertIn('data-visual-svg', body)
                self.assertIn('data-visual-explanation', body)
                self.assertIn('data-visual-controls hidden', body)
                self.assertIn('visual-lessons.mjs', html)
            else:
                self.assertIn('Need a picture first?', body)
                self.assertNotIn('visual-lessons.mjs', html)
            self.assertNotIn('three.module.min.js', html)
            self.assertIn('Original explanation', body)

    def test_visual_home_destinations_exist_and_unknown_model_is_rejected(self):
        pages = site.read_catalog()['pages']
        home = site.home_content(pages, site.read_journal())
        self.assertIn('id="visual-guides"', home)
        for kind, (_, route, _) in site.VISUAL_GUIDES.items():
            page = next(p for p in pages if p['url'] == route)
            self.assertEqual(page['visual_demo'], kind)
            self.assertIn(f'href="{route}#visual-{kind}"', home)
        catalog = site.read_catalog()
        catalog['pages'][0]['visual_demo'] = 'unknown-scene'
        with patch.object(site.json, 'loads', return_value=catalog):
            with self.assertRaisesRegex(ValueError, 'Unknown visual'):
                site.read_catalog()

    def test_main_directory_links_every_article_and_published_entry(self):
        pages = site.read_catalog()["pages"]
        entries = site.read_journal()
        directory = site.page_directory(pages, entries)
        parser = site.PageLinks()
        parser.feed(directory)
        expected = {page["url"] for page in pages + entries} | {"journal.html"}
        self.assertEqual(set(parser.references), expected)
        self.assertNotIn('class="site-page-url"', directory)
        self.assertIn('<summary>Procedures</summary>', directory)
        self.assertEqual(directory.count('<summary>'), len({page['group'] for page in pages} | {'Daily journal'}))
        self.assertEqual(directory.count(' open>'), 1)
        self.assertNotIn('https://minnow54426.github.io/', directory)
        self.assertNotIn("127.0.0.1", directory)
        self.assertNotIn("localhost", directory)

    def test_every_source_page_gets_a_brief_and_preserves_full_content(self):
        for page in site.read_catalog()["pages"] + site.read_journal():
            with self.subTest(page=page["url"]):
                original = '<h2 id="evidence">Evidence</h2><script>const test = 1;</script>'
                body = site.concise_article(page, original)
                summary, full = body.split('<details class="site-full-notes"', 1)
                self.assertIn('site-brief', summary)
                self.assertIn(original, full)
                self.assertNotIn(original, summary)
                self.assertLessEqual(summary.count("<li>"), 3)
                self.assertIn('id="site-overview"', summary)
                self.assertIn('Before you start:', summary)
                self.assertEqual(full.startswith('>'), page['group'] == 'Daily journal')
                label = "Interactive lesson" if page["source"].endswith(".html") else "Dated work record" if page['group'] == 'Daily journal' else "Detailed guide"
                self.assertIn(label, full)

    def test_articles_keep_section_navigation_and_one_title(self):
        page = site.read_catalog()['pages'][0]
        source = '<h1 id="old-title">Original title</h1><h2 id="example">Example</h2>'
        body = site.concise_article(page, site.preserve_title_anchor(source))
        html = site.shell(page, body, site.read_catalog()['pages'], 'abc123', 'https://example.com')
        self.assertEqual(html.count('<h1>'), 1)
        self.assertIn('id="old-title"', html)
        self.assertIn('aria-label="Page sections"', html)
        self.assertIn('href="#example"', html)
        self.assertIn('href="#site-overview"', html)
        self.assertIn('href="#site-continue"', html)

    def test_prerequisite_links_work_from_nested_pages(self):
        page = next(p for p in site.read_catalog()['pages'] if p['url'] == 'procedures/gpu-benchmark-session.html')
        body = site.concise_article(page, '<p>Procedure</p>')
        self.assertIn('href="../offline-gpu-preparation.html"', body)

    def test_unknown_prerequisite_is_rejected(self):
        catalog = site.read_catalog()
        catalog['pages'][0]['before_page'] = 'missing.html'
        with patch.object(site.json, 'loads', return_value=catalog):
            with self.assertRaisesRegex(ValueError, 'prerequisite'):
                site.read_catalog()

    def test_home_shows_current_evidence_without_old_pending_message(self):
        home = site.home_content(site.read_catalog()['pages'], site.read_journal())
        self.assertIn('Temporary GPU baseline passed', home)
        self.assertIn('RTX 5090 validation', home)
        self.assertIn('batch40-native-analysis.html', home)
        self.assertIn('id="all-pages"', home)
        self.assertNotIn('CUDA source exists', home)
        self.assertNotIn('Implementation sequence', home)

    def test_reading_path_and_dated_records_have_useful_destinations(self):
        pages = site.read_catalog()["pages"]
        lesson = next(page for page in pages if page["url"] == "butterfly-explained.html")
        body = site.concise_article(lesson, "<p>Full derivation</p>")
        self.assertIn('href="gpu-ntt-explained.html"', body)
        self.assertIn('Next: From CPU to CUDA', body)
        entry = site.read_journal()[0]
        body = site.concise_article(entry, "<p>Original record</p>")
        self.assertIn('href="../openfhe-integration-plan.html"', body)
        self.assertIn('This is a dated record', body)
        self.assertIn('Original record', body)

    def test_unknown_next_article_is_rejected(self):
        catalog = site.read_catalog()
        catalog["pages"][0]["next_page"] = "missing.html"
        with patch.object(site.json, "loads", return_value=catalog):
            with self.assertRaisesRegex(ValueError, "Unknown next page"):
                site.read_catalog()

    def test_nested_journal_document_link_keeps_fragment(self):
        url = site.rewrite_url(
            "../openfhe-integration-plan.md#live-tracker",
            "docs/journal/2026-11-08.md", "journal/2026-11-08.html",
            {"docs/openfhe-integration-plan.md": "openfhe-integration-plan.html"},
            "https://example.test/repo", "abc123",
        )
        self.assertEqual(url, "../openfhe-integration-plan.html#live-tracker")

    def test_code_reference_is_pinned_to_source_revision(self):
        url = site.rewrite_url(
            "../../src/cuda/ntt_kernels.cu", "docs/journal/2026-11-08.md",
            "journal/2026-11-08.html", {}, "https://example.test/repo", "abc123",
        )
        self.assertEqual(url, "https://example.test/repo/blob/abc123/src/cuda/ntt_kernels.cu")

    def test_urls_are_relative_for_project_subpaths(self):
        self.assertEqual(site.relative_url("journal/2026-11-08.html", "assets/site.css"), "../assets/site.css")
        self.assertEqual(site.relative_url("index.html", "journal/2026-11-08.html"), "journal/2026-11-08.html")

    def test_missing_source_link_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing link target"):
            site.rewrite_url("not-present.md", "docs/README.md", "contributing-docs.html", {}, "", "")

    def test_missing_local_fragment_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "index.html").write_text('<a href="page.html#missing">Link</a>')
            (output / "page.html").write_text('<h1 id="present">Page</h1>')
            with self.assertRaisesRegex(ValueError, "missing fragment"):
                site.check_links(output)


class WebsiteExportTests(unittest.TestCase):
    def test_beginner_export_has_only_the_reading_path_and_search_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            personal = root / "website"
            personal.mkdir()
            (personal / "index.html").write_text('<section id="code">Systems</section>')
            output = root / "beginner"
            with patch.object(site, "BEGINNER_OUTPUT", output), patch.object(website, "OUTPUT", output):
                destination = website.export(personal)
            routes = [page["url"] for page in site.beginner_pages(site.read_catalog())]
            expected = {"index.html", *routes}
            self.assertEqual({file.relative_to(destination).as_posix() for file in destination.rglob("*.html")},
                             expected)
            search = site.json.loads((destination / "search-index.json").read_text())
            self.assertEqual({page["url"] for page in search}, expected)
            home = (destination / "index.html").read_text()
            self.assertIn("/docs/journal", home)
            self.assertNotIn("journal.html", home)
            for route in routes[:-1]:
                self.assertIn('data-visual-svg', (destination / route).read_text())

    def test_nested_pages_return_to_personal_website_and_refresh_is_managed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            personal = root / "website"
            personal.mkdir()
            (personal / "index.html").write_text('<section id="code">Systems</section>')
            generated = root / "generated"
            (generated / "journal").mkdir(parents=True)
            body = '<div class="site-sidebar-bottom">Status</div>'
            (generated / "index.html").write_text(body)
            (generated / "journal/day.html").write_text(body)
            with patch.object(website, "OUTPUT", generated), patch.object(website, "build"):
                destination = website.export(personal)
                self.assertIn('../../index.html#code', (destination / "index.html").read_text())
                self.assertIn('../../../index.html#code', (destination / "journal/day.html").read_text())
                (destination / "obsolete.html").write_text("Old generated output")
                website.export(personal)
                self.assertFalse((destination / "obsolete.html").exists())

    def test_unmanaged_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            personal = Path(temporary)
            (personal / "index.html").write_text("Personal website")
            destination = personal / "systems/ntt-gpu"
            destination.mkdir(parents=True)
            original = destination / "index.html"
            original.write_text("Handwritten page")
            with patch.object(website, "build") as build:
                with self.assertRaisesRegex(ValueError, "unmanaged"):
                    website.export(personal)
                build.assert_not_called()
            self.assertEqual(original.read_text(), "Handwritten page")

    def test_export_does_not_follow_destination_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            personal = root / "website"
            personal.mkdir()
            (personal / "index.html").write_text("Personal website")
            external = root / "external"
            external.mkdir()
            (personal / "systems").symlink_to(external, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                website.export(personal)


class JournalTests(unittest.TestCase):
    def test_published_entries_are_complete_and_sorted(self):
        entries = site.read_journal()
        self.assertGreaterEqual(len(entries), 4)
        self.assertEqual([item["date"] for item in entries], sorted((item["date"] for item in entries), reverse=True))
        self.assertTrue(all(item["status"] == "published" for item in entries))

    def test_helper_never_overwrites_and_drafts_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "docs/journal"
            directory.mkdir(parents=True)
            (directory / "_template.md").write_text((ROOT / "docs/journal/_template.md").read_text())
            with patch.object(journal, "ROOT", root), patch.object(site, "ROOT", root):
                entry = journal.create_entry("2026-11-09", "CUDA indices")
                original = entry.read_text()
                self.assertIn("status: draft", original)
                self.assertEqual(site.read_journal(), [])
                with self.assertRaises(FileExistsError):
                    journal.create_entry("2026-11-09", "Overwrite")
                self.assertEqual(entry.read_text(), original)

    def test_date_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "docs/journal"
            directory.mkdir(parents=True)
            (directory / "2026-11-09.md").write_text(
                "title: Test\ndate: 2026-11-10\nsummary: Test\nphase: P1\nstatus: published\n"
            )
            with patch.object(site, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "filename and date"):
                    site.read_journal()

    def test_catalog_covers_all_existing_articles(self):
        self.assertGreaterEqual(len(site.read_catalog()["pages"]), 9)


if __name__ == "__main__":
    unittest.main()
