<div id="writing-and-publishing-the-project-notebook" aria-hidden="true"></div>

# Write and publish the notebook

Edit Markdown or the three interactive HTML lessons in this repository.
Build and check the notebook, then export it to the personal website.
The HTML in the website repository is generated from these files. Make
content changes here so the next export preserves them.

## Preview locally

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r docs/requirements.txt
.venv/bin/python scripts/build_docs.py --check
python3 -m http.server 8000 --bind 127.0.0.1 --directory build/docs-site
```

Open `http://127.0.0.1:8000/`. Rebuild after each edit. The interactive source
HTML also works as a standalone page.

## Read the site

Use the left menu to change pages and search with `Ctrl+K` / `Cmd+K`.
Each page starts with its main idea. Expand the notes for explanations,
commands, evidence or an interactive lesson. A link to a specific section
opens any collapsed notes automatically.

Learn explains concepts; Research reports evidence; Plans lists remaining
work. The journal records what happened on each calendar date. Its dates
are separate from the roadmap's working-day milestones.

## Add a daily entry

```sh
python3 scripts/new_journal.py --date 2026-11-17 --title "Describe the outcome"
```

The helper creates a draft and never overwrites an existing entry. Append new
work to the existing day. Keep the metadata fields `title`, `date`, `summary`,
`phase` and `status`, and complete these six sections:

| Section | What to write |
|---|---|
| Outcome | The result |
| Reasoning | Why the change was needed |
| Changes | What changed, with source links |
| Verification | Checks actually run, their environment and limits |
| Open questions | What is unresolved |
| Next step | The next deliverable |

Set `status: published` when ready. Drafts stay off the website. Label
reconstructed history and cite commits. Update the [roadmap](openfhe-integration-plan.md)
only when a milestone's acceptance checks pass.

## Add or edit an article

Add an entry to `docs/site.json` for each new article. Include its source file,
web address, title, group, summary, one to three takeaways, and the address of
the next page. Keep existing page addresses and section bookmarks stable so
old links continue to work. The builder turns relative links into notebook
pages or GitHub links to the exact source revision.

Lead with the page's purpose and main result. Define unfamiliar terms before
using abbreviations. Put commands beside their expected outcome; keep long
source excerpts and historical evidence in labelled expandable sections:

```html
<details markdown="1">
<summary>Source and verification details</summary>

Reference material goes here.

</details>
```

Use `##` for main headings and `###` for subsections. Write inline math as
`\(...\)` and display math as `\[...\]`; KaTeX is bundled locally. Search
includes collapsed text. Keep wide tables scrollable inside the page.

For OpenFHE behavior, link the function name to the actual implementation at
the pinned commit, with GitHub line anchors. Verify the cited lines and retain
the BSD license for literal excerpts. Distinguish existing upstream code from
our proposed GPU interfaces.

Preserve the Sunlit Paper theme: warm paper, orange/green accents, Anton
headings and Space Grotesk body text. Its source is
`docs/site-assets/sunlit-paper.css`; layout and lesson contrast rules stay in
their existing stylesheets.

<div id="openfhe-representation-gate" aria-hidden="true"></div>

<div id="device-backend-contract-gate" aria-hidden="true"></div>

## Validate before publishing

```sh
.venv/bin/python -m unittest discover -s tests/docs -v
.venv/bin/python scripts/build_docs.py --check
node --test tests/docs/*.test.cjs
```

The build checks that every article is listed, required description fields
exist, and local page and section links point to real targets. The Node
tests execute the lesson scripts and require Node.js 22+.
Inspect changed pages in a browser at desktop and phone widths, including
expanded notes, math, controls, search and long tables.

Documentation checks do not establish CUDA correctness. For implementation
checks, use the relevant guide:

| Area | Instructions and scope |
|---|---|
| Stage indices and twiddles | [Work-distribution proofs](ntt-65536-work-distribution.html): CPU execution of shared stage helpers |
| Modular arithmetic | [Arithmetic contract](modular-arithmetic-contract.md): bigint and full-size CPU checks |
| OpenFHE arrays and formats | [Representation contract](openfhe-representation-contract.md): pinned core-library comparisons |
| Hybrid key switching | [Operation graph](openfhe-hybrid-operation-graph.md): intermediate CPU arrays and actual keys |
| Runtime ownership | [Resident-chain prototype](resident-operation-chain.md): host lifetime/failure tests; native CUDA pending |
| Native CUDA session | [Preparation guide](offline-gpu-preparation.md): actual GPU exactness, sanitizers and measurements |

<div id="add-the-notebook-to-the-personal-website" aria-hidden="true"></div>

<div id="standalone-project-pages-deployment" aria-hidden="true"></div>

## Publish with GitHub Pages

From this repository:

```sh
.venv/bin/python scripts/export_website.py --website-root /path/to/minnow54426.github.io
```

Review, commit and push the generated `systems/ntt-gpu/` files in the website
repository. Its Pages workflow publishes [the notebook](https://minnow54426.github.io/systems/ntt-gpu/index.html).
The exporter refreshes only its managed files and retains the return link to
the personal website.

Ordinary pushes to `ntt-gpu` create a documentation preview artifact; they do
not publish the personal website. For a separate project Pages site, choose
**Settings → Pages → Source → GitHub Actions**, then run the Documentation
workflow with `deploy_standalone` enabled. That optional site lives at
[minnow54426.github.io/ntt-gpu](https://minnow54426.github.io/ntt-gpu/).
