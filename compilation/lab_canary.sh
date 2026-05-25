#!/bin/bash
# 19.7 — A/B canary on the lab RK3588 device.
#
# Drops the new .rkllm on pi@192.168.68.107 + restarts the blox-ai
# container + runs eval_held_out.py against the running container.
# Captures evidence to E:/fxblox/evidence/phase-19-canary/<date>/.
#
# Usage:
#   bash compilation/lab_canary.sh <new.rkllm>
#
# Exit codes:
#   0  canary passed (eval gate green; safe to publish)
#   1  canary failed (operator MUST review the eval report before deciding)
#   2  setup error (lab unreachable, file missing, etc.)

set -uo pipefail

NEW_RKLLM="${1:?Usage: bash lab_canary.sh <new.rkllm>}"

if [ ! -f "$NEW_RKLLM" ]; then
    echo "ERROR: file not found: $NEW_RKLLM" >&2
    exit 2
fi

LAB_HOST="${LAB_HOST:-pi@192.168.68.107}"
LAB_PASS="${LAB_PASS:-fxblox}"
LAB_MODEL_DIR=/uniondrive/blox-ai-canary/model
EVIDENCE_BASE="${EVIDENCE_BASE:-E:/fxblox/evidence/phase-19-canary}"
DATE_STAMP=$(date -u +%Y%m%d-%H%M%S)
EVIDENCE_DIR="$EVIDENCE_BASE/$DATE_STAMP"

mkdir -p "$EVIDENCE_DIR"
cp "$NEW_RKLLM" "$EVIDENCE_DIR/candidate.rkllm.info" 2>/dev/null || true  # symlink ok
echo "candidate model: $NEW_RKLLM" >> "$EVIDENCE_DIR/run.log"
echo "lab host: $LAB_HOST" >> "$EVIDENCE_DIR/run.log"
sha256sum "$NEW_RKLLM" >> "$EVIDENCE_DIR/run.log"

# 1. scp the new model to the lab (uses sshpass / plink in PowerShell; the
#    operator must have one of those configured for $LAB_HOST).
echo "uploading new model to $LAB_HOST:$LAB_MODEL_DIR/" | tee -a "$EVIDENCE_DIR/run.log"
sshpass -p "$LAB_PASS" scp -o StrictHostKeyChecking=no \
    "$NEW_RKLLM" "$LAB_HOST:$LAB_MODEL_DIR/candidate.rkllm" || {
        echo "ERROR: scp failed" >&2
        exit 2
    }

# 2. restart blox-ai container with BLOX_AI_MODEL_PATH pointing at the candidate
echo "restarting blox-ai container with candidate" | tee -a "$EVIDENCE_DIR/run.log"
sshpass -p "$LAB_PASS" ssh -o StrictHostKeyChecking=no "$LAB_HOST" <<'EOF' 2>&1 | tee -a "$EVIDENCE_DIR/run.log"
sudo docker rm -f blox-ai-canary 2>/dev/null
# Operator: this assumes the canary run script is at /tmp/canary_run.sh on the lab.
# See compilation/lab_canary_run_template.sh for the template.
BLOX_AI_MODEL_PATH=/uniondrive/blox-ai-canary/model/candidate.rkllm \
    sudo bash /tmp/canary_run.sh
sleep 45  # NPU + model load
curl -s http://127.0.0.1:18083/status
EOF

# 3. Run the held-out eval against the running container
echo "running eval gate against http://${LAB_HOST#*@}:18083" | tee -a "$EVIDENCE_DIR/run.log"
LAB_IP=$(echo "$LAB_HOST" | awk -F@ '{print $2}')

REPORT="$EVIDENCE_DIR/eval_report.json"
python3 -m training.eval_held_out \
    --test-set corpus/test_set \
    --whitelist ../fula-ota/docker/fxsupport/linux/plugins/blox-ai/action_whitelist.json \
    --backend http \
    --http-url "http://$LAB_IP:18083" \
    --output "$REPORT" || EVAL_RC=$?
EVAL_RC=${EVAL_RC:-0}

if [ "$EVAL_RC" -eq 0 ]; then
    echo "CANARY PASSED — eval gate green" | tee -a "$EVIDENCE_DIR/run.log"
    echo "Report: $REPORT"
    exit 0
else
    echo "CANARY FAILED — eval gate red (rc=$EVAL_RC)" | tee -a "$EVIDENCE_DIR/run.log"
    echo "Report: $REPORT — review before publishing"
    exit 1
fi
