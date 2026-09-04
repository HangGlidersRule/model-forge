# Contributing to Model Forge

Model Forge is the public canonical framework and catalog. Darkstar is the product
brand used for model families produced with the framework.

## Before contributing

- Use an issue for a bounded proposal or bug. Use
  [GitHub private vulnerability reporting](https://github.com/HangGlidersRule/model-forge/security/advisories/new)
  for security issues.
- Never submit secrets, credentials, private infrastructure metadata, internal
  operational details, model weights, generated artifacts, or raw run dumps.
- Never submit raw evaluation questions, prompts containing restricted benchmark
  content, answer keys, or per-question responses. Use synthetic fixtures and curated
  aggregates.
- Keep changes reviewable. Maintainers may close generated spam, abusive automation,
  or attempts to consume external services or review budgets.

Public pull requests run deterministic checks only. Opening an issue or pull request,
adding text to it, using labels, or mentioning a tool never authorizes paid review,
code execution, publication, or access to secrets. Maintainers make those decisions
outside contributor-controlled content. Any private AI review requires an explicit
allowlisted-maintainer request bound to the current immutable head SHA; a changed
head invalidates the request, and every result is advisory. See
[`contracts/ai-review/`](contracts/ai-review/) and
[ADR-0003](docs/decisions/0003-maintainer-gated-advisory-ai-review.md).

## Developer setup

Use Python 3.11 or newer:

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
```

Before requesting review, run the checks relevant to the change and report the exact
commands and results. The normal baseline is:

```sh
pytest -q
ruff check .
mypy src
git diff --check
```

Recipe, schema, documentation-link, export, or release-contract changes need their
focused validators too. Tests must be deterministic, offline where practical, and
must not depend on private services or paid APIs.

## Recipes and evidence

A recipe contribution must:

- pin source and dataset revisions to immutable full identifiers;
- declare lineage, transform, quantization, calibration, runtime, and protected-tensor
  assumptions needed to reproduce it;
- pass the repository recipe and schema validators; and
- include synthetic tests or sanitized, reviewable evidence for any new behavior.

Evidence contributions must identify the exact artifact and recipe, protocol version,
runtime settings, numerator, fixed denominator, completion coverage, exclusions,
errors, seeds, and relevant caveats. Do not change a denominator after observing
results, omit failures from it, combine unmatched checkpoints or protocols, or
backfill a matrix cell from a different run. Comparisons require the same protocol and
denominator unless the mismatch is explicit and the results are kept separate.

Only curated aggregate evidence and approved artifact identity metadata belong in
Git. Raw evidence remains outside the public repository.

## AI-assisted contributions

Disclose material AI assistance in the pull request, including what was generated and
what you verified. Do not paste untrusted issue text into an agent with execution or
secret access. The human contributor remains responsible for provenance, licensing,
correctness, security, tests, and every submitted line. Maintainers may require
changes or decline contributions whose provenance cannot be reviewed.

## Developer Certificate of Origin

This project uses the
[Developer Certificate of Origin 1.1](https://developercertificate.org/). The DCO
certifies that you have the right to submit the contribution; it does not replace the
project's [MIT License](LICENSE). Sign every commit:

```text
Signed-off-by: Your Name <your-public-or-noreply-address>
```

Create the signoff with `git commit -s`. By contributing, you agree that your
contribution is licensed under the repository's MIT License and that the DCO
certification is accurate.

## Review and merge

Changes are reviewed under [GOVERNANCE.md](GOVERNANCE.md). Code-owner approval is
required for protected areas. Maintainers may ask for additional evidence, split a
change, or reject it when the security, reproducibility, maintenance, or publication
risk is not justified.
