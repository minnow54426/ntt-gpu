"""Regression checks for published routes, journal integrity, and Markdown."""
import importlib.util
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
    def test_main_directory_links_every_article_and_published_entry(self):
        pages = site.read_catalog()["pages"]
        entries = site.read_journal()
        directory = site.page_directory(pages, entries)
        parser = site.PageLinks()
        parser.feed(directory)
        expected = {page["url"] for page in pages + entries} | {"journal.html"}
        self.assertEqual(set(parser.references), expected)
        self.assertEqual(directory.count('class="site-page-url"'), len(expected))
        for url in expected:
            self.assertIn("https://minnow54426.github.io/systems/ntt-gpu/" + url, directory)
        self.assertNotIn("127.0.0.1", directory)
        self.assertNotIn("localhost", directory)

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
