# Writing and publishing the project notebook

The documentation is a static HTML site for developers following the project.
It combines learning pages, research, implementation plans, and a dated journal.
Markdown remains the editable source; the build converts it to HTML.
The interactive HTML explainers keep their controls and local math renderer.

## Preview locally

The documentation dependency is separate from the C++/CUDA build:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/python scripts/build_docs.py
python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs-site
```

Open `http://127.0.0.1:8000/`. Rebuild after editing, then refresh.
Generated HTML lives in the ignored `build/docs-site/` directory.
The original interactive HTML files can still be opened independently.

## Read the site

- **Start here:** purpose, current checkpoint, and setup.
- **Learn:** mathematics, GPU execution, and debugging.
- **Research:** surveys, evidence, and candidate optimizations.
- **Plans:** working-day milestones and benchmark gates.
- **Daily journal:** changes, reasoning, verification, and next actions.

Calendar dates describe when work was recorded. Roadmap labels such as D1
describe deliverables; they do not advance automatically at midnight.
Publishing an entry does not mark an implementation gate complete.

## Add a daily entry

```sh
python3 scripts/new_journal.py --date 2026-11-09 --title "CUDA thread and butterfly mapping"
```

The helper creates `docs/journal/YYYY-MM-DD.md` from the checked-in template
and refuses to overwrite an existing entry. For an existing day, append new
observations. Entries are not created or committed automatically.

Fill in all six sections before publishing:

1. **Outcome:** what actually changed.
2. **Reasoning:** the question, evidence, alternatives, and decision.
3. **Changes:** links to implementation and related articles.
4. **Verification:** commands, results, hardware, and limitations.
5. **Open questions:** unresolved issues and evidence needed.
6. **Next step:** one concrete deliverable.

Keep the metadata at the top: `title`, `date`, `summary`, `phase`, and `status`.
Use `status: draft` while writing; drafts are omitted from the published site.
Use `status: published` when ready. Published entries appear automatically.
Historical entries reconstructed from git must say so and cite their commits.
Do not invent past reasoning, experiments, or benchmark results.

When a gate changes, update the [integration plan](openfhe-integration-plan.md)
in the same change. The homepage reads its day and phase directly from that plan.

## Add or edit an article

Edit Markdown or HTML in `docs/`. Register each new article's source, HTML URL,
group, title, and summary in `docs/site.json`. The build fails if a first-party
Markdown or HTML document is missing from the catalog.

Use relative Markdown links to local documents and code. The builder converts
document links to HTML and code links to the revision on GitHub. Tables,
task lists, fenced code, heading anchors, and per-page contents are supported.
Write math as `\(...\)` inline or `\[...\]` on its own line; the build preserves
it for KaTeX. Code fences remain code.

Search covers titles, summaries, and article text using a local generated index.
There is no external search service or account requirement.

## Validate before publishing

```sh
.venv/bin/python scripts/build_docs.py --check
.venv/bin/python -m unittest discover -s tests/docs -v
node --test tests/docs/work_distribution.test.cjs
```

The build validates catalog coverage, journal metadata, and local links,
assets, and heading fragments. The site supports project subpaths.
The Node test runs the work-distribution page's actual inline script using a
small DOM stub and checks N=65536, stage controls, boundary selections,
and complete per-stage index coverage. It requires Node.js and no packages;
it does not replace browser inspection or CUDA execution.

For the exhaustive forward-column index/twiddle proof check, run:

```sh
cmake -S . -B build
cmake --build build --target ntt_column_mapping_tests ntt_device_sim
ctest --test-dir build -R 'column_mapping_check|device_schedule_check' --output-on-failure
```

The column check executes the real shared stage helper with tagged power
tables at N=65536. It probes the input pair, ladder multiplier,
and twiddle selection for every column butterfly. This is CPU verification,
not a native CUDA build or a GPU race check.

## Publish with GitHub Pages

### Add the notebook to the personal website

The personal website can host this notebook as a topic under
**Systems from Scratch**, at `/systems/ntt-gpu/index.html`. From this repository, run:

```sh
.venv/bin/python scripts/export_website.py --website-root /path/to/minnow54426.github.io
python3 -m http.server 8001 --bind 127.0.0.1 --directory /path/to/minnow54426.github.io
```

Open `http://127.0.0.1:8001/#code`, then follow **NTT-GPU · CKKS From Scratch**.
The export includes every article, journal entry, asset, and search index, with
a return link to the website. Its main page lists the full URLs of all pages
using the public base URL in `docs/site.json`, even during local previews.
Navigation remains relative so the same HTML also works locally. Update source documents here and repeat the
export when publishing new work. The exporter refuses to replace unmanaged files.

The actual HTML entry file is `systems/ntt-gpu/index.html`, not the containing
folder. You can open that file directly in a browser: navigation, search, and
math rendering use local assets. Search is loaded from a script rather than
requiring a JSON fetch from a local-file URL.

### Standalone project Pages deployment

`.github/workflows/docs.yml` builds and validates pull requests and pushes to
`main`, and uploads a preview artifact. The personal website hosts the public
notebook. Optional standalone deployment is available through a manual workflow
run with `deploy_standalone` enabled.

For the first deployment, a repository administrator must set **Settings →
Pages → Build and deployment → Source → GitHub Actions**. A public website
exists only after that setting is enabled and a deployment succeeds.

The expected URL is
[minnow54426.github.io/ntt-gpu](https://minnow54426.github.io/ntt-gpu/).

The workflow follows the
[GitHub Pages documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
Prose is converted with [Python-Markdown](https://python-markdown.github.io/).
