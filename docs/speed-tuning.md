# Speed tuning

Run the speed-tuning sweep independently for every artifact produced by the
four-stage model workflow:

1. base BF16
2. base NVFP4
3. abliterated BF16
4. abliterated NVFP4

Quantization and abliteration can change the best speculative-decoding depth,
so a winner from one stage must not be carried into another without measuring
it again.

## Running a sweep

For each stage, point `--artifact-dir` at that stage's completed artifact:

```console
model-forge tune \
  --artifact-dir artifacts/qwen-base-bf16 \
  --served-name qwen-base-bf16 \
  --image local/vllm-dflash2:runtime \
  --results-dir results/tune/base-bf16 \
  --mtp-min 1 \
  --mtp-max 12
```

The default remote is your `~/.ssh` key against the WSL host (see
`--remote-host`/`--remote-user`/`--ssh-key`). Override it with `--remote-host`,
`--remote-user`, and `--ssh-key`. If the artifact has a different location on
the serving host, pass its Windows path with
`--ssh-wsl-remote-artifact ${PUBLIC_ARTIFACT_PATH}<name>`.

Candidate measurements are cached under
`<results-dir>/<artifact-sha>/<sweep-config-sha>/<candidate>.json`. The sweep
config identity covers lanes, runs, warmups, max tokens, and temperature. An
interrupted sweep resumes with only the missing candidates. `--force` reruns
all candidates, while `--dry-run` writes the proposed matrix without contacting
the remote.

## Winner semantics

Every candidate is measured at the configured context lanes. Runs that stop
before 80% of `max_tokens` are excluded as EOS-truncated samples. The winner
is the highest traffic-weighted mean throughput, using these default lane
weights:

- 4K: 60%
- 16K: 30%
- 48K: 10%

Custom lanes use uniform weights unless `--lane-weights` supplies explicit
`lane:weight` pairs. Equal weighted scores resolve to the lowest candidate key.

The JSON report and a Markdown winner table are written beside the cached
candidate files. Boot failures are recorded in the report and do not discard
successful or previously cached measurements.

## Candidate guidance

Sweep native MTP from 1 through 12 unless measured resource constraints justify
a different upper bound. Do not cap the search at MTP6: deployed models can
win at MTP10 or MTP11.

DFlash, DFlash2, and DSpark require architecture-compatible drafter or spark
heads and an image that implements the selected method. They are available to
programmatic callers through `TuneMatrix.drafters`; omit unsupported methods
for an architecture. A drafter entry supplies `(method, model, tokens)`.
