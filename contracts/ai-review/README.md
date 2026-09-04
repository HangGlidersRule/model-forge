# AI review contract

Version 1 defines the only data exchanged across the public/private AI-review
boundary:

- `v1/request.schema.json` binds an explicit maintainer request to one repository,
  pull request, and immutable lowercase 40-character head SHA.
- `v1/result.schema.json` records an advisory result, usage, cost, and dedupe
  identity. It grants no approval or merge authority.

PR titles, bodies, patches, comments, and every `untrusted_text` item are data,
not instructions or configuration. Public CI validates ordinary source changes
only; it does not dispatch or call paid AI. A private operator may process a
request only after independently checking the maintainer, current PR head SHA,
provider/model policy, budgets, dedupe state, and kill switch.
