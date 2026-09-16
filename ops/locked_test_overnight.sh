#!/bin/bash
# Unattended Locked Test chain on the instance: full collection (runbook §3),
# then scoring (§4) if and only if collection exits 0.  Never deletes, reruns
# or cleans anything; a non-zero exit stops the chain and leaves everything in
# place for the fail-closed recovery in the runbook.  Stage wall times are
# appended to $RUNSTATE/stage-times.jsonl for the §5 cost receipt.
set -uo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
CHECKOUT="$WORKSPACE/locked-test-checkout"
PY=/venv/main/bin/python
RUNSTATE="$WORKSPACE/runstate"
ART="$WORKSPACE/research-artifacts"
MANIFEST="$RUNSTATE/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json"
AUTHORITY="$RUNSTATE/locked-test-authority.json"
LOG="$WORKSPACE/run-logs/locked-test/overnight.log"
mkdir -p "$(dirname "$LOG")" "$ART/raw" "$ART/scores" "$ART/locked-test-features"
export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
stage() { echo "{\"stage\":\"$1\",\"start\":\"$2\",\"end\":\"$(ts)\",\"wall_seconds\":$3,\"exit\":$4}" >> "$RUNSTATE/stage-times.jsonl"; }

echo "[$(ts)] overnight chain start" | tee -a "$LOG"
c0=$(date +%s); cs=$(ts)
$PY "$CHECKOUT/ops/locked_test_supervisor.py" \
    --repo-root "$CHECKOUT" \
    --environment-lock "$CHECKOUT/environment.lock" \
    --manifest "$MANIFEST" \
    --authority "$AUTHORITY" \
    --cache-dir "$WORKSPACE/hf-cache" \
    --artifact-root "$ART/raw" \
    --cell-script "$CHECKOUT/ops/locked_test_cell.py" \
    --log-dir "$WORKSPACE/run-logs/locked-test" \
    --completion-receipt "$RUNSTATE/locked-test-complete.json" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
stage collection "$cs" $(( $(date +%s)-c0 )) "$rc"
if [ "$rc" -ne 0 ]; then echo "[$(ts)] COLLECTION_FAILED exit=$rc — chain stopped, nothing cleaned" | tee -a "$LOG"; exit "$rc"; fi
echo "[$(ts)] COLLECTION_COMPLETE" | tee -a "$LOG"

s0=$(date +%s); ss=$(ts)
$PY "$CHECKOUT/ops/locked_test_score.py" \
    --repo-root "$CHECKOUT" \
    --environment-lock "$CHECKOUT/environment.lock" \
    --cache-dir "$WORKSPACE/hf-cache" \
    --manifest "$MANIFEST" \
    --authority "$AUTHORITY" \
    --raw-root "$ART/raw" \
    --calibration-freeze "$CHECKOUT/locks/calibration_frozen.json" \
    --bound-probe "$CHECKOUT/artifacts/calibration-analysis-rescore-001/bound-probe/e94269a149491d30a8ba52e8d66c816c87ce489e2d37f0b4179b9f4ead5a1146" \
    --calibration-feature-reference "$CHECKOUT/artifacts/calibration-features-rescore-001/reference/4441c760eb1bd4acb9ff43dceb70986a0848f96c77ddfff19f836022b2b39da1" \
    --calibration-predictor-metadata "$CHECKOUT/artifacts/calibration-features-rescore-001/predictors.json" \
    --calibration-predictor-bundle "$CHECKOUT/artifacts/calibration-features-rescore-001/predictors.pkl" \
    --score-root "$ART/scores" \
    --feature-root "$ART/locked-test-features" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
stage scoring "$ss" $(( $(date +%s)-s0 )) "$rc"
if [ "$rc" -ne 0 ]; then echo "[$(ts)] SCORING_FAILED exit=$rc — chain stopped" | tee -a "$LOG"; exit "$rc"; fi
echo "[$(ts)] SCORING_COMPLETE — chain finished" | tee -a "$LOG"
