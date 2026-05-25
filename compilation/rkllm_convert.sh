#!/bin/bash
# 19.6b — Convert merged HF weights to RKLLM W8A8 .rkllm.
#
# Wraps Rockchip's RKLLM-Toolkit. Requires:
#   - Linux x86_64 build host (RKLLM-Toolkit isn't packaged for arm64 build hosts)
#   - CUDA 11.8+ + GPU (toolkit's quantization step uses it for >3B models)
#   - rkllm-toolkit pip package installed (see pin_rkllm_toolkit.md)
#
# Usage:
#   bash compilation/rkllm_convert.sh <merged_hf_dir> <output_dir>
#
# Output:
#   <output_dir>/qwen2.5-3b-instruct-rk3588-w8a8.rkllm
#   <output_dir>/rkllm_convert.log
#
# Exits non-zero on any failure so CI can gate.

set -euo pipefail

MERGED_DIR="${1:?merged_hf_dir is required}"
OUTPUT_DIR="${2:?output_dir is required}"

if [ ! -d "$MERGED_DIR" ]; then
    echo "ERROR: merged_hf_dir not found: $MERGED_DIR" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/rkllm_convert.log"
OUTPUT_RKLLM="$OUTPUT_DIR/qwen2.5-3b-instruct-rk3588-w8a8.rkllm"

# Check toolkit availability
if ! python3 -c 'from rkllm.api import RKLLM' 2>/dev/null; then
    cat >&2 <<EOM
ERROR: rkllm-toolkit not importable on this host.
Install per compilation/pin_rkllm_toolkit.md:
  1. Sign up at https://www.rockchip.com/ (developer account)
  2. Download RKLLM-Toolkit matching pin_rkllm_toolkit.md
  3. pip install <downloaded-wheel>
EOM
    exit 3
fi

echo "=== RKLLM convert: $MERGED_DIR -> $OUTPUT_RKLLM ===" | tee "$LOG"
echo "started: $(date -u +%FT%TZ)" | tee -a "$LOG"

# Drive the toolkit via the inline Python snippet. Keep the conversion
# parameters here in source control (single source of truth) so they
# don't drift between cycles.
python3 - "$MERGED_DIR" "$OUTPUT_RKLLM" 2>&1 | tee -a "$LOG" <<'PYEOF'
import os
import sys
from rkllm.api import RKLLM

merged_dir, output_path = sys.argv[1], sys.argv[2]

llm = RKLLM()
ret = llm.load_huggingface(model=merged_dir)
if ret != 0:
    print(f"load_huggingface returned {ret}", file=sys.stderr)
    sys.exit(4)

ret = llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
    target_platform="rk3588",
    num_npu_core=3,
    extra_qparams=None,
    dataset=None,
    hybrid_rate=0.5,
)
if ret != 0:
    print(f"build returned {ret}", file=sys.stderr)
    sys.exit(5)

ret = llm.export_rkllm(output_path)
if ret != 0:
    print(f"export_rkllm returned {ret}", file=sys.stderr)
    sys.exit(6)

print(f"OK: wrote {output_path}")
PYEOF

if [ ! -f "$OUTPUT_RKLLM" ]; then
    echo "ERROR: expected output file missing: $OUTPUT_RKLLM" >&2
    exit 7
fi

SIZE=$(stat -c%s "$OUTPUT_RKLLM")
SHA=$(sha256sum "$OUTPUT_RKLLM" | awk '{print $1}')

cat >> "$LOG" <<EOM

=== summary ===
file:     $OUTPUT_RKLLM
size:     $SIZE bytes
sha256:   $SHA
finished: $(date -u +%FT%TZ)
EOM

echo "OK: $OUTPUT_RKLLM ($SIZE bytes, sha256=$SHA)"
