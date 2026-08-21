# ADR-0003: Gate paid AI review behind an exact maintainer request

- Status: Accepted
- Date: 2026-08-19

## Context

Public pull requests contain contributor-controlled titles, bodies, comments,
patches, and files. Treating any of that text as trusted workflow instructions
would let an untrusted contributor trigger spending, select a provider, expose
credentials, or obtain repository write authority.

## Decision

Public CI never invokes paid AI and never turns contributor text into a dispatch
event. It remains secretless, read-only, and deterministic.

A paid review may start only in the private operator archive from a versioned
request that a separately authenticated allowlisted maintainer explicitly
created. The request binds the repository, pull-request number, provider, model,
limits, and exact immutable PR head SHA. The private dispatcher independently
reads the current head identity and rejects a stale SHA. Provider/model
allowlists, per-run token and cost ceilings, a daily budget, deduplication, and
an operator kill switch apply before any provider boundary is reached.

Contributor-controlled text is always untrusted data. It cannot alter dispatch
configuration, policy, credentials, tools, or repository state. Review output
is advisory: it cannot approve, label, comment, merge, release, or otherwise
write to GitHub. Any publication of a result is a separate maintainer action.

The public request and result formats are
`contracts/ai-review/v1/request.schema.json` and
`contracts/ai-review/v1/result.schema.json`.

## Consequences

- A maintainer must deliberately request each paid review for the current SHA.
- Updating the PR invalidates the request instead of silently reviewing new code.
- Replay and budget state remain private and local to the dispatcher.
- AI findings are evidence for human review, never a required approval or an
  authority decision.
