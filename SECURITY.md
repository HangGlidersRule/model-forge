# Security policy

## Supported versions

Model Forge is pre-1.0. Security fixes are applied to the default branch and, when a
release exists, to the latest release line. Older releases, development snapshots,
model weights, and third-party runtimes are not supported unless a release notice
explicitly says otherwise.

| Version | Supported |
| --- | --- |
| Default branch | Yes |
| Latest release line | Yes |
| Older releases | No |

## Reporting a vulnerability

Do not open a public issue, discussion, or pull request for a suspected vulnerability.
Use [GitHub private vulnerability reporting](https://github.com/HangGlidersRule/model-forge/security/advisories/new)
to create a private security advisory. This is the only designated security contact;
the project does not publish or infer an email address for vulnerability reports.

Include the affected revision, impact, minimal reproduction, and any suggested
mitigation. Do not include real credentials, private infrastructure details, model
weights, raw benchmark questions, or answer keys. Use synthetic data where possible.

Maintainers target:

- acknowledgement within 3 business days;
- initial triage within 7 business days; and
- a status update at least every 14 days while remediation is active.

These are good-faith targets, not guarantees. Complexity, maintainer availability, and
coordinated disclosure needs can change the timeline. Maintainers will coordinate a
disclosure date with the reporter, request a CVE when appropriate, and publish a
sanitized advisory after a fix or mitigation is available. Please do not disclose the
issue publicly before that coordination is complete.

## Scope

Reports are in scope when they affect Model Forge framework code, public export
integrity, release contracts, recipes, containers, or publication metadata. Security
issues in upstream models or dependencies should normally be reported upstream as
well. Model behavior disagreements without a security impact belong in the ordinary
support process described in [SUPPORT.md](SUPPORT.md).
