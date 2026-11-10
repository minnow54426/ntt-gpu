# Writing and publishing the project notebook

Edit Markdown and the three interactive HTML lessons in `docs/`. Build the site
locally, then export it to the personal website. Generated HTML is not source.

## Preview locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/python scripts/build_docs.py --check
python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs-site
```

Open `http://127.0.0.1:8000/`. Rebuild and refresh after editing.
The interactive source HTML also works independently.

## Read the site

The left sidebar switches pages. **On this page** lists the current page's
sections and highlights your reading position. On smaller screens, use
**Menu** for pages and **Sections** for headings. Search opens from the top bar
or `Ctrl+K` / `Cmd+K`.

Learn pages explain the mathematics and code; Research records evidence;
Plans track gates; the Daily journal records dated results. Calendar dates and
roadmap working days are separate: publishing a note does not complete a gate.

## Add a daily entry

```sh
python3 scripts/new_journal.py --date 2026-11-10 --title "Describe the outcome"
```

The helper creates a draft without overwriting an existing entry. Append to an
existing day. Keep the `title`, `date`, `summary`, `phase`, and `status` metadata.
Fill all six required sections:

| Section | Record |
|---|---|
| Outcome | What changed |
| Reasoning | Evidence behind the decision |
| Changes | Links to code and documents |
| Verification | Observed checks, hardware, and limits |
| Open questions | What remains unresolved |
| Next step | The next deliverable |

Use `status: published` when complete; drafts stay off the site. Label
reconstructed history and cite commits. Update the [integration plan](openfhe-integration-plan.md)
when a gate changes. Its tracker supplies the homepage checkpoint.

## Add or edit an article

Register new documents in `docs/site.json` with source, URL, title, group, and
summary. Use relative links to documents and code; the builder converts them
to HTML routes or revision-pinned GitHub links.

Use `##` headings for main sections and `###` for subsections. Main headings
populate the section sidebar automatically, including headings in HTML lessons.
Existing anchors are preserved. Tables scroll inside the article.

Write inline math as `\(...\)` and display math as `\[...\]`. KaTeX is bundled
locally. Put long reference material in `<details markdown="1">` with a clear
`<summary>`; keep the main conclusion visible. Search includes collapsed text.

## Validate before publishing

```sh
.venv/bin/python -m unittest discover -s tests/docs -v
.venv/bin/python scripts/build_docs.py --check
node tests/docs/work_distribution.test.cjs
```

The build validates catalog coverage, journal metadata, and local links,
assets, and heading fragments. The site supports project subpaths.
The Node test runs the work-distribution page's actual inline script using a
small DOM stub and checks N=65536 in both directions, stage controls, boundary selections,
and complete per-stage index coverage. It requires Node.js and no packages;
it does not replace browser inspection or CUDA execution.

The separate C++ column proof executes the production stage helper on CPU:

```sh
cmake -S . -B build
cmake --build build --target ntt_column_mapping_tests ntt_device_sim
ctest --test-dir build -R 'column_mapping_check|inverse_column_check|device_schedule_check' --output-on-failure
```

The column check executes the real shared stage helper with tagged power
tables at N=65536. It probes the input pair, ladder multiplier,
and twiddle selection for every column butterfly. This is CPU verification,
not a native CUDA build or a GPU race check.
The inverse gate additionally checks reciprocal-table selection, the factor
of two from one forward/inverse stage pair, and full N-inverse normalization
with valid roots for two primes at N=65536.

For D4, run:

```sh
cmake --build build --target ntt_tile_mapping_tests
ctest --test-dir build -R tile_mapping_check --output-on-failure
```

It probes every row butterfly at
N=65536 in both directions and checks the two-pass stage lists
and global endpoint coverage at N=65536 and small-size
boundaries. Run a full build and `ctest --test-dir build --output-on-failure`
to include the D3 and D4 gates together.

The Python documentation suite also includes D5 source-contract checks for
the allocation/copy/API map. These inspect C++ source text and pin byte-count
arithmetic; they do not compile CUDA, inject failures, or validate runtime
resource cleanup. See [CUDA buffers and dataflow](cuda-buffer-dataflow.md).

## Publish with GitHub Pages

### Add the notebook to the personal website

From this repository:

```sh
.venv/bin/python scripts/export_website.py --website-root /path/to/minnow54426.github.io
```

Review, commit, and push the generated `systems/ntt-gpu/` files in the website
repository. Its Pages workflow publishes the site. Open
[the notebook](https://minnow54426.github.io/systems/ntt-gpu/index.html).

The exporter refreshes managed files only and adds a return link to the personal
website. Navigation, search, and math also work in local previews.

### Standalone project Pages deployment

Ordinary pushes to `ntt-gpu` build a preview artifact; they do not publish the
personal website. For an optional standalone site, set repository **Settings →
Pages → Source → GitHub Actions**, then manually run the Documentation workflow
with `deploy_standalone` enabled. Its URL is
[minnow54426.github.io/ntt-gpu](https://minnow54426.github.io/ntt-gpu/).
