# Model Forge governance

## Project identity and boundary

Model Forge is the public canonical source for the reusable framework, catalog,
release contracts, reproducible recipes, generic containers, sanitized aggregate
evidence, and publication metadata. Darkstar is the product brand for model families
produced with Model Forge; it is not a separate governance authority.

The public repository intentionally excludes private operator archives, raw evaluation
evidence, restricted prompts and answer keys, credentials, model weights, unpublished
artifacts, private infrastructure inventory, and private release operations. Public
records may describe reproducible inputs and outcomes, but must not reveal private
locations or access details. The boundary is detailed in
[ADR-0001](docs/decisions/0001-private-archive-public-root-separation.md).

## Roles

### Maintainers

`@HangGlidersRule` is the current maintainer and repository owner. Maintainers:

- set project direction and contribution policy;
- review and merge changes;
- administer repository settings and security reports;
- appoint or remove maintainers and code owners;
- approve release and publication records; and
- enforce the public/private boundary.

Maintainer changes are recorded by pull request in this file and `.github/CODEOWNERS`.
A maintainer must have demonstrated sustained, trustworthy contributions and accept
the security and release responsibilities above.

### Contributors

Contributors propose changes and provide the evidence required by
[CONTRIBUTING.md](CONTRIBUTING.md). Contribution does not grant merge, release,
publication, secret, infrastructure, or paid-service authority.

## Decisions and review

Routine changes use pull-request consensus. The responsible code owner approves
protected areas; the repository owner resolves deadlock. Significant architecture,
security-boundary, governance, or release-authority changes require an accepted
architecture decision record under `docs/decisions/`.

Review is risk-based. Branch protection should require pull requests, passing
deterministic checks, resolved conversations, code-owner approval for owned paths, and
no direct or force pushes to the default branch. Administrative bypass should be
limited to documented emergencies and followed by review. Repository and automation
permissions use least privilege: read-only by default, short-lived credentials where
available, protected release environments, and separation between untrusted
contributor input and privileged operations.

## Release and publication authority

Only maintainers may authorize a Model Forge release or publish a Darkstar artifact.
A merged recipe, passing test, issue label, comment, or contributor request is not
publication authorization. Releases must be traceable to reviewed source, immutable
inputs, reproducible recipes, required evidence, and recorded approval. Release tags
and attestations should be cryptographically signed using repository-supported
identity mechanisms, and consumers must be able to verify artifact digests and
reproduce or independently validate the build where the toolchain permits.

Publication authority, withdrawal, and correction rules are defined in
[ADR-0002](docs/decisions/0002-release-publication-authority.md). Publication
automation is intentionally outside this governance change.

## Conduct and security

Project participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities privately as described
in [SECURITY.md](SECURITY.md). Maintainers may restrict participation that threatens
people, project integrity, private data, or finite review resources.
