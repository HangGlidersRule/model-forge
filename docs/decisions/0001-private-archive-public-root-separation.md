# ADR-0001: Separate the private archive from the public root

- Status: Accepted
- Date: 2026-08-19

## Context

Model Forge needs a canonical public framework and catalog, but its development
history and operator environment can contain material that is neither needed nor safe
for public collaboration. Git history cannot reliably be sanitized by deleting files
from only the latest tree.

## Decision

The public Model Forge repository is created from a fail-closed, deterministic export
into a fresh Git root. It contains reusable framework code, contracts, recipes,
generic containers, sanitized aggregate evidence, governance, tests, and publication
metadata. It is the canonical public source for those materials.

A separately access-controlled private archive retains private planning and operations,
raw evidence, restricted evaluation content, credentials, infrastructure inventory,
unpublished artifacts, model weights, and any history containing private metadata.
The public repository must not name or encode private storage locations, network
identifiers, or access details.

Every tracked source file must have an explicit export disposition. Public output is
verified against the committed source revision and manifest, scanned for prohibited
metadata and credentials, and compared deterministically before promotion. Public
changes flow through the public repository after cutover; the archive is not a second
public source of truth.

## Consequences

- Public history begins from a reviewed export rather than inherited private history.
- Some useful operational evidence remains unavailable publicly; curated aggregates
  must carry enough protocol and provenance to be independently assessed.
- Moving material across the boundary requires explicit classification, review, and
  validation.
- A leak in the public tree is handled as a security incident; sensitive history is
  rebuilt from a corrected export when deletion cannot provide reliable removal.
