# Public file classification

`public-files.yaml` is the fail-closed inventory for the future public export. It
classifies every tracked source as `copy`, `transform`, or `exclude`; it does
not perform an export or change existing release behavior.

## Rule contract

Each rule has:

- `id`: stable rule identifier.
- `source`: a canonical POSIX repository-relative exact path or bounded glob
  with a literal directory prefix. Drive/UNC and absolute paths, backslashes,
  empty components, `.`, `..`, repeated separators, trailing separators, and
  normalization-changing spellings are rejected before glob compilation.
- `disposition`: `copy`, `transform`, or `exclude`.
- `public_destination`: repository-relative destination, normally `{source}`;
  excluded rules use `null`. Literal destinations must be canonical POSIX paths:
  no platform separators, drive/UNC or absolute paths, empty components, `.`,
  `..`, repeated separators, or trailing separators.
- `reason`: why the classification is safe and necessary.
- `transformation`: future deterministic transform name for `transform`;
  otherwise `null`.
- `owner`: the responsible CODEOWNER class. These classes are policy labels
  until public CODEOWNERS is introduced by its planned governance change.
- `max_size_bytes`: upper bound for each matched source.
- `generated`: whether the source is machine-generated.
- `regeneration_check`: required when `generated` is true.
- `precedence`: deterministic integer priority.

Optional `resolves` names every lower-precedence rule intentionally shadowed by
a rule. An overlap is rejected unless there is a unique higher-precedence
winner and it explicitly resolves all other matching rules. `allow_empty` is
reserved for private-boundary guards such as `.hermes/**` and
`private_archive/**`, which have nothing to match inside a generated public root;
ordinary rules must match at least one tracked file.

The supported glob grammar uses `/` as the only separator. Literal characters,
`?` (one non-`/` character), `*` (zero or more non-`/` characters), `**` (zero
or more characters, including `/`), and character classes such as `[abc]`,
`[a-z]`, and `[!abc]`/`[^abc]` are supported. A `**/` sequence matches zero or
more complete directory levels. Character classes do not match `/`; ranges use
Unicode code-point order. Backslash escaping, empty or unterminated classes, and
reversed ranges are unsupported.

Glob sources must have a literal directory prefix. Root-wide catchalls (`*`,
`**`, and `**/*`), absolute paths, and parent traversal are invalid. Validation
computes whether every pair of bounded rule languages intersects, independent
of tracked files. An intersection requires distinct precedence and an explicit
`resolves` override on the winner. Glob rules preserve the source path with
`{source}` so they cannot silently flatten files into colliding destinations.
Every concrete resolved destination is validated as a canonical POSIX
repository-relative path, including destinations produced by `{source}`.
Adjacent equivalent star runs are normalized before automata are built.
Validation rejects manifests over 1 MiB, more than 256 rules, source patterns
over 1,024 characters or 256 normalized tokens, and more than 1,000,000 glob
product-transition checks. These fixed budgets make hostile policy input fail
with deterministic errors instead of consuming unbounded CI resources.

The YAML loader rejects duplicate keys. The manifest and each rule accept only
the documented fields; booleans and integers use exact YAML/Python types, and
numeric policy values are positive and bounded. Tracked members are checked
component by component: missing paths, broken or leaf symlinks, and symlinked
ancestors are rejected, and the fully resolved member must remain beneath the
fully resolved repository root.

## Classification policy

- Framework source, contracts, and deterministic tests copy after later export
  scans.
- Recipes, Qwen scripts, public-facing model documents, and private-default
  container inputs require named validation/sanitization transforms.
- Model data and plans remain private.
- Model results are private by default. Only named curated aggregate evidence
  and artifact identity manifests override that default.
- Existing `.github` workflows are excluded for replacement by hardened public
  workflows.
- `.hermes/**` is an empty-allowed exclusion guard and must never be tracked.
- `private_archive/**` is tracked privately, excluded, and empty-allowed because a
  generated public root contains no such file.

Manifest classification tests run in two contexts. The private source must track
and exclude every private path the guards protect. A generated public root is
identified only by its verifier-owned `PUBLIC_EXPORT_MANIFEST.json` attestation,
which is export output rather than classified source. Its canonical, unique
`output_path`/`source_id` pairs are the only admissible source-rule provenance:
path-form source IDs must equal their output path, while `rule:<id>` records must
name an existing manifest rule. The Git tracked inventory must equal exactly the
attested output paths plus the attestation itself; missing, extra, malformed,
duplicate, ignored, or untracked members fail closed before classification.
Because a public root contains no excluded file by construction, exclusion rules
may match nothing in that context, while any private path introduced into a public
root is still classified as excluded and rejected.

Transform names are implemented by the public exporter and are validated when
the manifest is loaded. The exact `recipes/README.md` rule intentionally
overrides `recipes/**`: Markdown is sanitized as Markdown, while recipe YAML is
parsed and semantically validated.

## Export contract

`model-forge public-export` accepts a clean Git source tree, its exact `HEAD`
SHA, the tracked manifest selector, and an output directory. Cleanliness is an
operator check, not a trust boundary. The exporter obtains paths, blob IDs, and
executable modes from `git ls-tree -rz --full-tree <SHA>`, reads those blobs
through `git cat-file --batch`, and parses the manifest from its committed blob.
Transforms therefore consume committed bytes despite `assume-unchanged`,
`skip-worktree`, or mutation after the initial cleanliness check. Git replace
refs are disabled. Symlinks, submodules, unsupported modes or object types,
noncanonical paths, and case/Unicode portable path collisions fail closed. The
exporter classifies every committed file, applies the winning
explicit-precedence rule, scans emitted payloads, and atomically promotes the
result. Output paths are resolved before any write, so safe system
temporary-directory symlink ancestors are supported while outputs inside or
containing the source tree and symlink output leaves remain rejected.
The promoted root and every payload directory use mode `0755`; executable files
use `0755` and all other files use `0644`, independent of the process umask.

Blob reads are batched so the reader can never fill a pipe and wait on itself.
Every batch requests at most 256 object IDs and at most 16 KiB of request bytes,
and each invocation feeds standard input from one thread while other threads
drain standard output and error, so no request is written without a concurrent
reader. A `git cat-file --batch-check` pass declares every object's type and
size first, which enforces the 100,000 tracked-file and 1 GiB total-source
ceilings before any content is buffered. Content batches then cap declared
payload at 32 MiB, bound the accepted response to its exact expected length, and
require the exact response count, order, object ID, `blob` type, and declared
size for every request. Object IDs are validated as 40 hexadecimal characters
before any process starts. Each batch has its own timeout inside a total read
budget, timeouts kill the reader's whole process group, and any malformed,
truncated, overlong, or surplus output fails the export rather than accepting a
partial tree.

The CLI cannot assert that a credential scan passed. The exporter locates
`gitleaks` only on its fixed system path and runs `gitleaks git --redact` itself
with JSON reporting, an embedded exporter-owned configuration extending the
default rules, a fixed clean environment, bounded process/report output, a
timeout, and Git history ending at the requested source SHA. Repository
`.gitleaks.toml` and `.gitleaksignore` files are ignored in favor of the trusted
configuration and an exporter-owned empty ignore file. The scanner runs in a
new process group, which is killed as a group on timeout. Its report is opened
without following symlinks, checked with descriptor and path metadata, and read
through the descriptor with a fixed bound. Missing tools, timeouts, nonzero
status, malformed reports, and any finding fail closed.
Tests may inject the trusted runner interface; there is no corresponding CLI
flag. The public attestation records only the tool version, report SHA-256,
trusted configuration SHA-256, normalized fixed flags, fixed scan scope, and
bound source SHA.

## Independent verification contract

`model-forge public-verify` requires the exported root, asserted source SHA,
trusted source repository, committed manifest selector, and a trusted local
wheelhouse. It uses the exporter's pure deterministic planner to reconstruct
every public destination and transformed byte from committed Git objects, the
committed manifest, and explicitly trusted transform context. It requires the
exact path set, output bytes, `source_id`, input/output SHA-256, transform ID,
mode, semantic recipe linkage, and payload tree digest. It also reruns fixed
full-history Gitleaks and requires its version, report digest, source, scope,
configuration digest, and flags to match the attestation.

The wheelhouse may contain only regular, single-link `.whl` files. Every wheel
is opened without following symlinks, bounded, SHA-256 hashed, and returned as
verification evidence. Fresh private build and runtime environments install
declared requirements with `--find-links <wheelhouse> --no-index`; the verifier
checks build requirement versions, imports the backend, builds, runs
`pip check`, and invokes installed CLIs. Parent `site-packages`, `PYTHONPATH`,
user package configuration, and indexes are not used.

The canonical wrapper accepts
`EXPORT_ROOT SOURCE_REPO EXPECTED_SOURCE_SHA [TRUSTED_WHEELHOUSE]`. When no
wheelhouse is supplied, it creates a private temporary wheelhouse and SHA-256
lock from the exact build/runtime dependency closure already installed in the
source checkout's `.venv`. The bootstrap uses only local distribution files,
works when that environment has no `pip`, and never contacts an index. A
local Python 3.11+ with `packaging` (or a `uv`-located Python) runs the
bootstrap. Missing or version-incompatible local distributions fail with the
unsatisfied requirement chain. Supplied wheelhouses require a sibling
`WHEELHOUSE.sha256` lock (or `MODEL_FORGE_WHEELHOUSE_LOCK`). The verifier
checks that lock before package smoke installation; generated wheels remain
temporary and are never added to the checkout.

`MODEL_FORGE_PUBLIC_CONTACT` supplies the trusted transform contact for the
wrapper and defaults to `security@hangglidersrule.com`. It must match the
contact used for export. `MODEL_FORGE_ENVIRONMENT_PYTHON` may select a local
installed distribution environment other than `SOURCE_REPO/.venv/bin/python`.

`scripts/stage_public_root.sh OUTPUT_ROOT SUMMARY_JSON [SOURCE_REPO
[SOURCE_SHA [TRUSTED_WHEELHOUSE]]]` performs the canonical export and independent
verification, then writes a deterministic JSON staging summary containing the
source SHA, payload digest, and exported file count. The summary is outside the
export root so verification retains an exact inventory. The generated root and
summary are reproducible staging artifacts, not source files committed back into
the private archive. A reviewed root can later be initialized as a new repository
with one clean root commit.

`payload_tree_sha256` is the digest of the emitted payload records before the
attestation is added. Each record contributes its public output path,
deterministic mode, and output SHA-256 in sorted path order. Attestation source
metadata is restricted to a public output-relative identifier (or a manifest
rule identifier when source and output differ). `public_contact` must be a
conventional explicitly public email; localhost and reserved private/internal
domains are rejected.

## Detector contract

The content detectors are bounded policy primitives, not an exporter and not a
transformation layer. Each file has fixed byte, finding-count, and work-unit
limits. Callers scanning a repository must share one `ScanWorkBudget` across
`scan_file` calls to enforce a repository-wide ceiling. Reaching either the
per-file or shared ceiling returns the stable redacted `scan.work-limit`
finding; detected values are never included in messages. Match line numbers
and UTF-8 byte offsets are materialized in one pass over decoded text.

Python is parsed before findings are finalized. String and byte literals inside
`test_*` functions are treated as executable test-fixture syntax, while
module-level runtime constants remain scanned. Regex pattern literals passed to
`compile` are excluded only from path findings, and network literals passed to
`ip_network` are excluded only from network findings. Syntax errors fall back
to ordinary text scanning. This contextual parsing prevents detector
implementations and tests from finding their own synthetic patterns without
creating a path, network, identity, or credential allowlist.

Absolute-path detection targets private operator filesystem contexts
(`/Users/<name>`, `/home/<name>`, and `/Volumes/<name>`), Windows drive
absolute paths, and UNC shares. Root-relative links, API routes, JSON pointers,
and regex fragments are intentionally outside that policy.

Secret-like filename detection is case-insensitive, uses only the path
basename, and fails closed. It covers credential families such as
`credentials`, `service-account`/`service_account`, `client-secret`/
`client_secret`, `application_default_credentials`, `secret`/`secrets`, and
`token`/`tokens` in `.json`, `.toml`, `.yaml`, and `.yml` form, private SSH and
post-quantum key basenames such as `id_rsa` and `id_ed25519` with any suffix
other than `.pub`, and backup or editor copies of any recognized name. The
supported backup and editor suffixes are `~`, `.backup`, `.bak`, `.copy`,
`.old`, `.orig`, `.save`, `.swp`, and `.tmp`, and at most two stacked suffixes
are peeled, so peeling cost stays bounded per basename.

A key basename is classified after that peeling: the normalized name is public
only when it is exactly a recognized key basename plus `.pub`. So
`id_rsa.pub~`, `id_ed25519.pub.bak`, and `id_rsa.pub~.orig` are public, while
`id_rsa~`, `id_rsa.old`, and stacked private backups such as `id_ed25519.bak~`
remain secret-like. Suffixes outside the documented set are never peeled and
never made public, so `id_rsa.pub.enc` and chains deeper than two stacked
suffixes such as `id_ed25519.pub~.orig.bak` fail closed as secret-like.

The only exemption is the documented safe template form: the basename must end
with `.example`, `.sample`, or `.template` immediately followed by a final
`.json`, `.tfvars`, `.toml`, `.txt`, `.yaml`, or `.yml` extension, or match
`<name>.{example,sample,template}.auto.tfvars`. Because the exemption is
suffix-constrained, appending anything after the documented form (for example
`secrets.template.json.bak`) removes the exemption, and template markers inside
a name are ignored when matching credential families.

## Allowlist contract

An allowlist entry is an exact path plus exact rule exception with a
justification and a timezone-aware expiry, rechecked when it is applied and not
only when it is constructed. Rules that make `scan_file` return before the
content detectors run on scannable text are never allowlistable:
`path.escape`, `path.forbidden-directory`, `scan.work-limit`, and any
`credential.*` rule including `credential.gitleaks-required`. Suppressing one of
those findings would report an empty result for a file whose content was never
examined, so a forbidden path such as `.git/config` or `results/raw/**` must be
excluded from the export rather than excepted.

Export detector suppressions are a separate, narrower manifest policy. Only
`benchmark.raw-key` may currently be suppressed, and only on unchanged copies
classified as `trusted-source-code` under `src/` or `tests/` with a Python
suffix, or on an exact JSON path under `tests/public_export/fixtures/`
classified as `trusted-detector-fixture`. Network, path, identity,
secret-filename, binary, oversized, work-limit, and credential/Gitleaks
findings cannot be suppressed. Prompt-bearing framework data is explicitly
excluded; curated evidence and raw result paths do not inherit source-code
policy.

`file.oversized` and `file.binary` stay allowlistable because their content
cannot be scanned within the policy byte limits or is not text at all, so the
exception records reviewed non-text or over-limit content instead of hiding a
skipped text scan. The manifest `max_size_bytes` bound remains the primary
control for those files.

Inputs must be canonical UTF-8 before scanning. If an ingest boundary accepts
URL encoding, base64, archive members, or another encoded representation, that
boundary must perform the format-specific canonical decoding and validation
before invoking these detectors. The detector does not attempt arbitrary URL
decoding or other transforms because ambiguous repeated decoding can change the
security boundary.

## Updating the inventory

1. Add the file and the narrowest suitable manifest rule in the same change.
2. Prefer an exact path. Use a bounded glob only for a coherent ownership and
   policy boundary.
3. If a rule overlaps, assign a unique precedence and list every intentionally
   shadowed rule in `resolves`.
4. Set a realistic size ceiling. Mark generated files accurately and provide a
   deterministic regeneration check.
5. Run:

   ```sh
   pytest -q tests/public_export/test_public_file_manifest.py
   ```

The test obtains the inventory from `git ls-files` and adds only the three PR-A
contract files when they exist but have not yet been staged. It deliberately
does not include arbitrary untracked state, so local files cannot weaken or
expand the tracked export boundary. A newly tracked, unclassified file fails
the test.
