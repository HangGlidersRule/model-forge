# Pull request path risk

Public CI classifies changed paths to provide consistent review guidance. The
classifier is deterministic, local, read-only, and does not use secrets or external
services. It writes JSON and a short job summary; it does not label, comment on,
approve, merge, or dispatch work from a pull request.

## Semantics

The highest matching path risk becomes the pull request risk:

- **Low:** governance files, documentation, issue templates, and tests.
- **Medium:** framework source, recipes, contracts, containers, and configuration.
  Ordinary repository paths that do not match an explicit rule also default to
  medium.
- **High:** GitHub workflows, security policy or scanning configuration, public
  export tooling, and release tooling.

Rules are ordered and explicit. For example, `SECURITY.md` is high even though other
governance documents are low, and public-export source is high even though other
framework source is medium. A high result is not a CI failure and is not an automatic
merge decision. It indicates that reviewers should apply the relevant ownership,
security, or release checks.

The classifier fails when it cannot produce a trustworthy result: malformed or
unbounded paths, Unicode or case-folding collisions, more than 1,000 changed files,
an unsupported configuration, an invalid event, or an unavailable Git comparison.
Those failures are distinct from a valid high-risk classification.

## Branch protection

For the default branch, enable pull request reviews, resolved conversations, and
code-owner review for owned paths. Require the deterministic CI checks for Python
3.11 and 3.12, Gitleaks, and PR risk classification. Requiring the classifier check
is safe because a valid high-risk result exits successfully; only an invalid or
unbounded classification fails.

Keep force pushes and direct pushes disabled. Limit administrative bypass to
documented emergencies. Release and publication authorization remains a maintainer
decision under [GOVERNANCE.md](../GOVERNANCE.md), not a consequence of a risk score.
