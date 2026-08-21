# ADR-0002: Reserve release and publication authority to maintainers

- Status: Accepted
- Date: 2026-08-19

## Context

Model Forge records reproducible build inputs and evidence for Darkstar products.
Merging code is not equivalent to authorizing a framework release, publishing model
weights, or changing a public artifact. Publication can be costly, irreversible, and
security-sensitive.

## Decision

Only a listed Model Forge maintainer may approve releases and publication. Approval
must bind the exact reviewed source revision, recipe and contract versions, artifact
digests, evidence, destination, and immutable release identity. Contributor-authored
text, issue commands, labels, passing checks, or merged recipes never confer this
authority.

Release and publication credentials are least-privilege, short-lived where supported,
and available only in protected environments after maintainer approval. Untrusted pull
request code does not run with those credentials.

Releases must be reproducible or independently verifiable from pinned inputs. Tags,
attestations, and release metadata should be cryptographically signed using supported
repository identity mechanisms. Publication completes only after upload, clean
redownload, digest verification, applicable smoke tests, and synchronized public
catalog and ledger state.

Published revisions are not silently overwritten. A defective revision is marked
withdrawn or deprecated, an incident record is retained, and a corrected release uses
a new immutable identity. Security response may temporarily restrict access without
erasing the audit trail.

## Consequences

- Contribution and review remain open while privileged publication remains narrow.
- Automation may prepare or verify a release but cannot infer authorization from
  contributor-controlled input.
- Publication requires more explicit evidence and approval, reducing speed in favor
  of provenance, rollback clarity, and supply-chain integrity.
- Workflow and dispatcher implementation is deferred to later changes; this record
  defines policy only.
