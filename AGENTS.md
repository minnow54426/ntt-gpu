# Project working agreements

## Keep the engineering notebook current

For substantive implementation, research, design, or documentation work, update
the current calendar day's entry in `docs/journal/` in the same change.
Use the user's Asia/Shanghai date. Append to an existing entry rather than
replacing earlier observations. Brief explanations and git-only operations
do not require an otherwise empty daily entry.

Entries record outcomes, reasoning, changes, verification, open questions, and
the next deliverable. Report only observed results. Label reconstructed history,
estimates, and unverified hypotheses. CPU simulation is not a real GPU run.

When a roadmap gate changes, also update `docs/openfhe-integration-plan.md`.
Calendar days do not automatically advance the plan's working-day counter.

## Documentation workflow

Markdown and the interactive HTML files in `docs/` are source documents.
Register new articles in `docs/site.json`. Published daily entries are
discovered automatically. `build/docs-site/` is generated output.

For documentation changes, install `docs/requirements.txt` in an isolated
environment, run `python -m unittest discover -s tests/docs -v`, and run
`python scripts/build_docs.py --check`. For layout or interaction changes,
also inspect the relevant pages in a browser at desktop and mobile widths.

Preserve exact OpenFHE compatibility and existing interactive examples.

## Next rental: check profiling access first

Before the long rented-GPU batch, complete the profiling-access TODO in
`docs/next-gpu-rental.md`. Verify real CUDA/NVTX tracing and SM/L2/DRAM counter
capture on that exact instance, and retain the versions, reports and errors.
Installed tools or container root access do not establish counter permission.
If counters are blocked, record the limitation and continue independent work;
do not mark the hardware profiling gate complete.
