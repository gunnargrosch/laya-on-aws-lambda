#!/usr/bin/env bash
set -euo pipefail

# Exports the Laya checkpoint to ONNX for every variant. One-time, slow step.
# Everything that can change is pinned here so "update the model" is a single edit.

CHECKPOINT="convaiinnovations/laya"
REVISION="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"   # pinned commit SHA on the HF hub
LAYA_REF="main"                                       # laya repo ref for the export scripts
PYTHON="${PYTHON:-python3}"                           # override if torch has no wheel for the default python

REPO_DIR=".laya-repo"
CKPT_DIR=".checkpoint"
OUT_TS="plain/typescript/model"
OUT_PY="plain/python/model"

cd "$(dirname "$0")/.."

echo "==> Creating virtualenv and installing export tooling"
if command -v uv >/dev/null 2>&1; then
  uv venv --python "$PYTHON" .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  # CPU torch + the onnx stack. onnxscript is required by torch.onnx.export.
  uv pip install "laya[onnx]" "torch" --torch-backend=cpu
else
  "$PYTHON" -m venv .venv
  # shellcheck disable=SC1091
  . .venv/bin/activate
  python -m pip install --upgrade pip
  # CPU torch + the onnx stack. onnxscript is required by torch.onnx.export.
  python -m pip install "laya[onnx]" "torch" --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Cloning the laya repo for its export scripts"
rm -rf "$REPO_DIR"
git clone --depth 1 https://github.com/NandhaKishorM/laya.git "$REPO_DIR"
git -C "$REPO_DIR" checkout "$LAYA_REF" 2>/dev/null || true

# laya-ts is not published to npm, so build it from source and vendor package.json + dist
# into each TypeScript variant (referenced as "laya-ts": "file:./laya-ts").
echo "==> Building laya-ts and vendoring it into the TypeScript variants"
(
  cd "$REPO_DIR/laya-ts"
  npm install --no-save --omit=optional "typescript@^5.6.0" "@types/node@^22"
  npm run build
)
rm -rf plain/typescript/laya-ts
mkdir -p plain/typescript/laya-ts
cp "$REPO_DIR/laya-ts/package.json" plain/typescript/laya-ts/
cp -r "$REPO_DIR/laya-ts/dist" plain/typescript/laya-ts/dist

echo "==> Downloading checkpoint $CHECKPOINT@$REVISION"
rm -rf "$CKPT_DIR"
python - "$CHECKPOINT" "$REVISION" "$CKPT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

checkpoint, revision, local_dir = sys.argv[1], sys.argv[2], sys.argv[3]
snapshot_download(
    checkpoint,
    revision=revision,
    local_dir=local_dir,
    allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*", "model.safetensors"],
)
print(f"downloaded to {local_dir}")
PY

echo "==> TypeScript: split ONNX via laya-ts export"
mkdir -p "$OUT_TS"
python "$REPO_DIR/laya-ts/scripts/export_onnx.py" --model-dir "$CKPT_DIR" --out-dir "$OUT_TS"

# Note: INT8 quantization of the split models is intentionally NOT done. It loads and predicts
# correctly on a workstation, but on Lambda's x86 CPUs the ONNX Runtime INT8 kernels returned
# uniform (zero) logits, so the sample ships fp32. The model is loaded lazily at first invoke
# (see the handlers) so the large read happens in the invoke phase, not the 10s-capped init phase.

echo "==> Python: checkpoint for the PyTorch path"
# The laya ONNX export (scripts/export_onnx.py) errors at runtime for multi-question batches
# ("Add node ... broadcast 51 by 153") on onnxruntime 1.22 and 1.30, so the Python variant uses
# the reference PyTorch runtime (laya.load) and bakes the checkpoint instead.
mkdir -p "$OUT_PY"
cp "$CKPT_DIR/model.safetensors" "$OUT_PY/"
cp "$CKPT_DIR/rl_agent_config.json" "$OUT_PY/"
cp -r "$CKPT_DIR/encoder" "$OUT_PY/encoder"
cp -r "$CKPT_DIR/tokenizer" "$OUT_PY/tokenizer"

echo "==> Done. Model files in $OUT_TS and $OUT_PY"
