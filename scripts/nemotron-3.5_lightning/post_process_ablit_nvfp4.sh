#!/bin/bash
# Post-process abliterated Lightning NVFP4 export for vLLM serving.
# 1. Strip ModelOpt sentinel suffixes from hf_quant_config.json
# 2. Restore original BF16 config.json (structural fields corrupted by ModelOpt)
# 3. Change quant_algo W4A16_NVFP4 -> NVFP4 in hf_quant_config.json
# 4. Write quantization_config (modelopt_mixed) into config.json for vLLM
set -e
EXPORT=${1:-/mnt/d/model-forge/cache/lightning-ablit-export}
SRC_BF16=${2:-/mnt/d/model-forge/ablit/runs/apply_abliteration}
SRC_BF16_CFG=${3:-/mnt/d/model-forge/lightning-bf16}

echo "=== 1. strip sentinels ==="
python3 ${PUBLIC_ARTIFACT_PATH} "$EXPORT/hf_quant_config.json" 2>/dev/null || \
  python3 - "$EXPORT/hf_quant_config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
s = json.dumps(d)
n = s.count("sentinel")
s = s.replace("\u0000backbone.pt_name_sentinel", "").replace("\\u0000...pt_name_sentinel", "")
s = s.replace("\\u0000backbone.pt_name_sentinel", "").replace("\\u0000...pt_name_sentinel", "")
json.loads(s)  # validate
open(p, "w").write(s)
print("sentinel occurrences:", n)
PY

echo "=== 2. restore BF16 config.json ==="
cp -f "$SRC_BF16_CFG/config.json" "$EXPORT/config.json"
python3 - "$EXPORT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
# ModelOpt serializes Infinity weirdly; normalize to the float form vLLM expects
if isinstance(d.get("time_step_limit"), list) and any(isinstance(x, dict) for x in d["time_step_limit"]):
    d["time_step_limit"] = [0.0, float("inf")]
open(p, "w").write(json.dumps(d, indent=2))
print("config restored; layers_block_type:", d.get("layers_block_type", "?"))
PY

echo "=== 3. quant_algo W4A16_NVFP4 -> NVFP4 ==="
python3 - "$EXPORT/hf_quant_config.json" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
q = d["quantization"]
old = q["quant_algo"]
q["quant_algo"] = "NVFP4"
open(p, "w").write(json.dumps(d, indent=2))
print("quant_algo:", old, "->", q["quant_algo"])
PY

echo "=== 4. write vLLM quantization_config (modelopt_mixed) ==="
cd ${PUBLIC_ARTIFACT_PATH}
# Use the committed writer when available, else inline fallback
python3 - <<PY
import sys
sys.path.insert(0, "${PUBLIC_ARTIFACT_PATH}")
from model_forge.modelopt.write_vllm_quant_config import write_quantization_config
from pathlib import Path
write_quantization_config(Path("$EXPORT"), Path("$SRC_BF16_CFG"))
PY
echo POST_PROCESS_DONE
