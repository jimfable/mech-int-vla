# Locked Test runbook (approved 2026-08-25; hardened 2026-09-01)

Applies the approved amendments (AMENDMENTS.md 2026-08-25). Nothing in this
runbook reads Locked Test data; every evaluation step is pre-fixed here and in
PREREG.md §11 order. Any deviation from the freeze or from this runbook stops
the run: instance stopped, deviation reported, no continuation.

Expected GPU cost (Calibration precedent, log.md 2026-08-24): **rollouts
~5–10 GPU h, scoring ~11–18 GPU h → plan 16–36 GPU h (one to two days) on the
1× RTX 5090.** Budget control: after rollouts and after scoring, log elapsed
GPU-hours; if scoring would exceed 24 GPU h wall budget, stop and report
before finalizing.

---

## 0. Freeze verification (local, CPU)

```bash
cd /Users/fynnvanriessen/Developer/research/mech-int-vla
git rev-parse HEAD                                  # == tag commit of calibration-locked-v1
git rev-list -n 1 tags/calibration-locked-v1        # must equal HEAD
git status --porcelain                              # must be empty
# per-artifact byte hashes: locks/calibration_frozen.json -> artifact_hashes
PYTHONPATH=src uv run --isolated --no-project --python 3.12 \
  --with 'numpy==2.2.6' --with 'PyYAML>=6,<7' \
  python ops/prepare_locked_test_artifacts.py \
    --repo-root /Users/fynnvanriessen/Developer/research/mech-int-vla \
    --environment-lock /Users/fynnvanriessen/Developer/research/mech-int-vla/environment.lock
# prints 160-episode manifest sha256 1fd8c818… and writes
#   artifacts/manifests/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json
#   artifacts/locked-test-authority.json            (locked_test_accessed: True)
# ANY digest mismatch -> STOP + report; do not start the instance.
```

Before proceeding to section 1, run the mandatory pre-access capability gate
in section 5.0. It must pass after verifying the content-addressed natural
Calibration activation reference. Its two authorized limitation records (the
unavailable 9a position trace and unavailable supporting-layer coefficients)
are required report semantics, not blockers and not permission to omit
selected-layer patching. This ordering prevents spending collection or scoring
GPU time while any unapproved capability blocker remains.

## 1. Instance start + environment verification

1. `vastai start instance <ID>` (single RTX 5090, disk-preserved).
2. `vastai execute <ID> "nvidia-smi"` → expect 1× RTX 5090, 0% util.
3. Verify the instance layout used by the freeze: `/venv/main/bin/python`,
   `/workspace/hf-cache`, `/workspace/runstate`, and
   `/workspace/research-artifacts`. Verify
   `environment-gpu.freeze` has SHA-256
   `d738fb679db3682292481dfb74154b2d1d22da37630fd0156092c281ff31f821`
   and verify the package/GPU values in that file (including LeRobot v0.6.0,
   `hf-libero==0.1.4`, CUDA 13.0, and MuJoCo EGL). If any value differs → STOP
   + report, no rollout.
4. Copy the current repo (the frozen HEAD from step 0) to
   `/workspace/locked-test-checkout`; `cd` there and verify
   `git rev-parse HEAD` == local HEAD and `calibration-locked-v1` == HEAD and
   clean tree. Copy these exact files to `/workspace/runstate/`:
   `locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json`
   and `locked-test-authority.json`. Create and verify writable
   `/workspace/research-artifacts`, `/workspace/run-logs`, and
   `/workspace/runstate` before the preflight.

## 2. First three collection cells + CPU-safe preflight (fail-closed)

Run the CPU-safe supervisor plan before opening the collection and require
`resume_episodes: 0`. Only after that passes, indices 0, 1, and 2 are the first
three **real Locked Test collection cells**, not disposable diagnostics. Launch
each index exactly once, sequentially. Each cell runs the full frozen chain
(snapshot, instrumentation, rollout, atomic raw artifact publication,
provenance). Expect per-cell wall time in the Calibration envelope (≈6–8 min or
less on this GPU); expect valid artifacts with policy_revision `31d453f7…` and
code_commit `18d64941…`. Then run the same plan again and require exactly three
resume-validated artifacts.

```bash
set -euo pipefail
supervisor=(
    /venv/main/bin/python
    /workspace/locked-test-checkout/ops/locked_test_supervisor.py
    --repo-root /workspace/locked-test-checkout
    --environment-lock /workspace/locked-test-checkout/environment.lock
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json
    --authority /workspace/runstate/locked-test-authority.json
    --cache-dir /workspace/hf-cache
    --artifact-root /workspace/research-artifacts/raw
    --cell-script /workspace/locked-test-checkout/ops/locked_test_cell.py
    --log-dir /workspace/run-logs/locked-test
    --completion-receipt /workspace/runstate/locked-test-complete.json
)
"${supervisor[@]}" --plan-only

for index in 0 1 2; do
    /venv/main/bin/python /workspace/locked-test-checkout/ops/locked_test_cell.py \
        --index "$index" \
        --repo-root /workspace/locked-test-checkout \
        --environment-lock /workspace/locked-test-checkout/environment.lock \
        --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
        --cache-dir /workspace/hf-cache \
        --artifact-root /workspace/research-artifacts/raw
done
"${supervisor[@]}" --plan-only
```

The cell sets and verifies EGL, offline Hugging Face/Transformers access, and a
single visible GPU before loading runtime code. The plans must report
`{"kind":"locked_test_plan_validated","episodes":160,"resume_episodes":0,…}`
before collection access and the same event with `resume_episodes:3` afterwards;
it resolves and hashes local snapshots and verifies paths, writable output
parents, Python/cell-script executability, disk capacity, the global supervisor
lock, and every existing artifact without loading a GPU model or simulator. Its
`remote_runtime` object must machine-readably report: Python 3.12.13 under
`/venv/main`; the exact tracked `environment-gpu.freeze` digest
`d738fb679db3682292481dfb74154b2d1d22da37630fd0156092c281ff31f821`;
all required distribution versions; one physical/visible
`NVIDIA GeForce RTX 5090` at index 0 and compute capability 12.0; the observed
host-driver version (recorded, not frozen); torch 2.11.0+cu130 with CUDA 13.0;
EGL/offline environment values; exact
offline policy/base-VLM snapshot paths; and `free_disk_bytes`. Any absent or
mismatched field is a hard stop.
Indices 0–2 remain immutable resume artifacts. **Never delete, overwrite, or
rerun them.** Any failed launch, corrupt artifact, wrong resume count, or freeze
deviation → STOP + preserve everything + report; do not relaunch an index
without an explicit protocol-deviation authorization.

## 3. Full collection (160 rollouts)

```bash
/venv/main/bin/python /workspace/locked-test-checkout/ops/locked_test_supervisor.py \
    --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --authority /workspace/runstate/locked-test-authority.json \
    --cache-dir /workspace/hf-cache \
    --artifact-root /workspace/research-artifacts/raw \
    --cell-script /workspace/locked-test-checkout/ops/locked_test_cell.py \
    --log-dir /workspace/run-logs/locked-test \
    --completion-receipt /workspace/runstate/locked-test-complete.json
```

Supervision: events `locked_test_episode_starting`,
`locked_test_resume_validated`, and `locked_test_episode_completed` cover the
160 episode ids
(`libero_10-task5-locked-test-…`); stop on `unexplained staging`,
`non-manifest entries`, `authority mismatch`, child failure, or any
`RuntimeError` — then STOP and report. The supervisor holds one non-blocking
global lock derived from the canonical raw-artifact root, so a second
supervisor cannot collect concurrently. Normal continuation uses exactly the
same command and resume-validates every published artifact. Expected wall time
≈5–10 GPU h.

**Fail-closed staging recovery:** never remove or rename a
`.libero_10-task5-locked-test-….tmp-*` directory ad hoc. Stop all collection
processes; preserve and inventory/hash the staging directory and matching log;
verify the final episode directory is absent; record the failure cause and
report it. Resume is forbidden while staging or any other non-manifest entry is
present. Cleanup/relaunch requires explicit written deviation authorization;
the supervisor never performs it automatically.

## 4. Scoring (frozen predictor/probe applied to Locked Test)

Frozen scoring tooling is `ops/locked_test_score.py` (Locked Test bound, tag
`locked-test-score-v1`, exact probe/reference/predictor bytes from the freeze).
`--raw-root` is always the parent before the split directory; the scorer adds
`locked_test` exactly once and rejects a split-root argument.

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_score.py \
    --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --cache-dir /workspace/hf-cache \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --authority /workspace/runstate/locked-test-authority.json \
    --raw-root /workspace/research-artifacts/raw \
    --calibration-freeze /workspace/locked-test-checkout/locks/calibration_frozen.json \
    --bound-probe /workspace/locked-test-checkout/artifacts/calibration-analysis-rescore-001/bound-probe/e94269a149491d30a8ba52e8d66c816c87ce489e2d37f0b4179b9f4ead5a1146 \
    --calibration-feature-reference /workspace/locked-test-checkout/artifacts/calibration-features-rescore-001/reference/4441c760eb1bd4acb9ff43dceb70986a0848f96c77ddfff19f836022b2b39da1 \
    --calibration-predictor-metadata /workspace/locked-test-checkout/artifacts/calibration-features-rescore-001/predictors.json \
    --calibration-predictor-bundle /workspace/locked-test-checkout/artifacts/calibration-features-rescore-001/predictors.pkl \
    --score-root /workspace/research-artifacts/scores \
    --feature-root /workspace/research-artifacts/locked-test-features
```

The final JSON event names the immutable score allocation, feature cohort,
prediction receipt and summary paths plus all digests. Copy those values exactly
into section 5; do not infer a newest directory. Predictions are created by
applying the all-Calibration models without a label argument. Expected wall time
≈11–18 GPU h. Budget gate: beyond 24 h → stop and report.

## 5. Evaluation — FIXED template (no improvisation; PREREG §11 order)

### 5.0 Mandatory post-scoring capability gate (run before Locked Test collection)

The final evaluator requires causal, sensitivity, and cost receipts; it never
accepts manually entered scientific results. Before starting section 1, run the
CPU-only frozen-capability check:

```bash
cd /Users/fynnvanriessen/Developer/research/mech-int-vla
../tiny-vla-interp/.venv/bin/python ops/locked_test_postscore.py capabilities \
    --bound-probe artifacts/calibration-analysis-rescore-001/bound-probe/e94269a149491d30a8ba52e8d66c816c87ce489e2d37f0b4179b9f4ead5a1146/bound_probe.json \
    --bound-probe-sha256 e94269a149491d30a8ba52e8d66c816c87ce489e2d37f0b4179b9f4ead5a1146 \
    --calibration-reference artifacts/calibration-features-rescore-001/reference/4441c760eb1bd4acb9ff43dceb70986a0848f96c77ddfff19f836022b2b39da1 \
    --calibration-reference-sha256 4441c760eb1bd4acb9ff43dceb70986a0848f96c77ddfff19f836022b2b39da1 \
    --calibration-activation-reference artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --calibration-activation-reference-sha256 cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c
```

This command must pass. It validates the immutable full-Calibration natural
activation reference (160 episodes, 9,455 rows, width 720, fixed natural 5-NN
geometry) and reports two prospectively authorized limitations. First, 9a is
reported exactly as `unavailable_preaccess_missing_position_trace`, reason
`frozen_position_decoder_and_all_object_trace_absent`; no orientation,
coverage, or other proxy number is permitted. Second, missing frozen
non-selected-layer coefficients are reported as
`multi_layer_support_available=false`, reason
`frozen_supporting_layer_coefficients_absent`. Selected-layer patching remains
mandatory and reportable, while the positive confirmatory multi-layer causal
claim is deterministically unsupported/false. These two limitation markers do
not block sections 1–9. A missing/malformed activation reference, a hash
mismatch, label use, refitting, or any additional blocker does.

After a future valid freeze and completed scoring, the full content-binding
preflight is:

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_postscore.py preflight \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --manifest-sha256 1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49 \
    --predictions "$PREDICTION_RECEIPT_PATH" \
    --predictions-sha256 "$PREDICTION_RECEIPT_SHA256" \
    --raw-root /workspace/research-artifacts/raw \
    --score-root /workspace/research-artifacts/scores \
    --cohort "$FEATURE_COHORT_PATH" \
    --cohort-sha256 "$FEATURE_COHORT_SHA256" \
    --bound-probe "$BOUND_PROBE_PATH" \
    --bound-probe-sha256 "$BOUND_PROBE_SHA256" \
    --calibration-reference "$CALIBRATION_REFERENCE_PATH" \
    --calibration-reference-sha256 "$CALIBRATION_REFERENCE_SHA256" \
    --calibration-activation-reference /workspace/locked-test-checkout/artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --calibration-activation-reference-sha256 cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --calibration-freeze /workspace/locked-test-checkout/locks/calibration_frozen.json \
    --calibration-freeze-sha256 eb39e6952ad8864c8f9ae88a07f382b0efcbe18fd36a1221b67fe9f59106bed9
```

The variables are copied byte-for-byte from the canonical Locked Test scoring
receipt; unset variables are a hard stop. This checks the manifest/prediction,
all 160 Raw directories, every valid score sidecar, feature cohort, bound probe,
Calibration feature reference, final Calibration freeze, and the full natural
activation reference content links before intervention work. The final freeze
must bind the activation-reference `metadata.json` and `arrays.npz`; the
reference's predecessor-freeze field is provenance and must not be substituted
for the final freeze hash.

Run the selected-layer GPU producer. The command is resumable: it verifies the
pair plan and every completed content-addressed pair checkpoint before
continuing. Pair selection consumes frozen states and source hashes, never
outcomes or prediction labels.

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_causal.py causal \
    --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --cache-dir /workspace/hf-cache \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --manifest-sha256 1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49 \
    --predictions "$PREDICTION_RECEIPT_PATH" \
    --predictions-sha256 "$PREDICTION_RECEIPT_SHA256" \
    --raw-root /workspace/research-artifacts/raw \
    --score-root /workspace/research-artifacts/scores \
    --feature-cohort "$FEATURE_COHORT_PATH" \
    --feature-cohort-sha256 "$FEATURE_COHORT_SHA256" \
    --bound-probe "$BOUND_PROBE_PATH" \
    --bound-probe-sha256 "$BOUND_PROBE_SHA256" \
    --calibration-feature-reference "$CALIBRATION_REFERENCE_PATH" \
    --calibration-feature-reference-sha256 "$CALIBRATION_REFERENCE_SHA256" \
    --calibration-activation-reference /workspace/locked-test-checkout/artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --calibration-activation-reference-sha256 cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --output-root /workspace/research-artifacts/postscore/causal
```

Copy `receipt_path` and `receipt_sha256` from the final
`locked_test_causal_complete` event into `CAUSAL_RECEIPT_PATH` and
`CAUSAL_RECEIPT_SHA256`. The receipt must contain 60 hash-addressed Pair
Evidence files. Every valid pair contains selected alpha 0.25, a strict <5°
matched control, Calibration-reference 5-NN evidence and controls indexed
exactly 0..999. Missing supporting-layer coefficients do not skip this run;
they force the confirmatory result to `unsupported` and `succeeds=false`.
If an outcome-blind seed has fewer than 20 eligible confirmatory donors, its
remaining deterministic slots carry
`invalid_reason=no_eligible_confirmatory_donor` and null donor/orientation;
missing <5° controls carry `no_eligible_matched_donor`. Such slots contain no
scientific numbers and count toward the 60 attempts, not the valid estimand.

Then run the two-dose sensitivity producer against those exact causal bytes:

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_causal.py sensitivity \
    --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --cache-dir /workspace/hf-cache \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --manifest-sha256 1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49 \
    --predictions "$PREDICTION_RECEIPT_PATH" \
    --predictions-sha256 "$PREDICTION_RECEIPT_SHA256" \
    --raw-root /workspace/research-artifacts/raw \
    --score-root /workspace/research-artifacts/scores \
    --feature-cohort "$FEATURE_COHORT_PATH" \
    --feature-cohort-sha256 "$FEATURE_COHORT_SHA256" \
    --bound-probe "$BOUND_PROBE_PATH" \
    --bound-probe-sha256 "$BOUND_PROBE_SHA256" \
    --calibration-feature-reference "$CALIBRATION_REFERENCE_PATH" \
    --calibration-feature-reference-sha256 "$CALIBRATION_REFERENCE_SHA256" \
    --calibration-activation-reference /workspace/locked-test-checkout/artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --calibration-activation-reference-sha256 cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c \
    --causal-receipt "$CAUSAL_RECEIPT_PATH" \
    --causal-receipt-sha256 "$CAUSAL_RECEIPT_SHA256" \
    --output-root /workspace/research-artifacts/postscore/sensitivity
```

Copy the final `locked_test_sensitivity_complete` path/digest into
`SENSITIVITY_RECEIPT_PATH`/`SENSITIVITY_RECEIPT_SHA256`. This run must publish
60 dose-evidence files and the numerical 0.5/1.0 × cells 0–7 grid. It also
publishes the exact authorized 9a unavailable marker; it never fabricates a
nearest-object/error-distance diagnostic.
If a cell has zero valid pairs, the row remains mandatory and auditably reports
its attempted `pair_indices`, `valid_pairs=0`, zero sign count, null rates and
medians, `unavailable_no_valid_pairs`, and `specificity_passes=false`.

Cost evidence is the only post-score receipt that may be operator supplied.
Publish it after recording all five stages, in this exact order and unit format:

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_postscore.py cost \
    --manifest-sha256 1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49 \
    --predictions-sha256 "$PREDICTION_RECEIPT_SHA256" \
    --stage "collection,$COLLECTION_WALL_SECONDS,$COLLECTION_GPU_HOURS,$COLLECTION_CHARGES" \
    --stage "scoring,$SCORING_WALL_SECONDS,$SCORING_GPU_HOURS,$SCORING_CHARGES" \
    --stage "evaluation,$EVALUATION_WALL_SECONDS,$EVALUATION_GPU_HOURS,$EVALUATION_CHARGES" \
    --stage "causal_patching,$CAUSAL_WALL_SECONDS,$CAUSAL_GPU_HOURS,$CAUSAL_CHARGES" \
    --stage "sensitivity,$SENSITIVITY_WALL_SECONDS,$SENSITIVITY_GPU_HOURS,$SENSITIVITY_CHARGES" \
    --output-root /workspace/research-artifacts/postscore/cost
```

Add one `--budget-gate-stop "<description>"` per actual stop. Zero values are
valid evidence; omitted stages, reordered stages, negatives, NaN/Infinity, or
duplicate stop descriptions fail closed. Publication is canonical,
content-addressed, immutable, and resume-verified.

Output is a single report with sections in this exact order:

1. **Data-integrity checks:** artifact count == 160; per-episode provenance
   (split, task 5, policy revision, code commit); validity envelope rates per
   condition cell (reject_nan, workspace, speed, penetration, phase
   displacement) ≤10% per cell, else the cell is reported invalid per protocol.
2. **Primary estimand — paired log loss:** M2 vs M1 relative log-loss lift
   `Delta_LL = mean(LL_M2 − LL_M1)` on episodes with total sample weight one,
   with the Locked Test reference bundle fitted on Calibration only; report
   cluster-bootstrapped 90% interval (init-ID clusters, seed 260803). Bar:
   preregistered ≥3% lift.
3. **Brier / AUROC** per model (M0, M1, M2), 90% cluster intervals.
4. **M2 vs M0** (substitution ceiling per start.md §12 last row).
5. **Lead time** at 10% FPR (median paired lead_M2 − lead_M1; bar ≥5 steps).
6. **Condition rankings** (AUROC/log loss per cell, tabulated).
7. **Causal patching** at FROZEN alpha 0.25: 60 pairs, 3 pairing seeds; target
   sign (>50% with 90% cluster interval above 50%), specificity (off-target
   ratio ≤0.25 median), 1000 norm-matched random 2-D subspaces, off-manifold
   check, matched-donor control (<5°). <30 valid pairs → selected-layer
   inconclusive. Because no frozen supporting-layer coefficients exist,
   `multi_layer_support_available=false` and the positive confirmatory causal
   claim is always `unsupported`/false; selected-layer evidence is still fully
   reported.
8. **Cost accounting:** GPU hours, instance charges, per-stage wall times,
   any budget-gate stops.
9. **Sensitivity (amendment items 1–3, separate section):**
   9a. exact status `unavailable_preaccess_missing_position_trace`, reason
       `frozen_position_decoder_and_all_object_trace_absent`; no proxy metrics;
   9b. patching dose × difficulty: alphas {0.5, 1.0} × 8 cells (sign +
       specificity, same definitions);
   9c. exact status `unavailable`, reason
       `patched_closed_loop_outcome_not_defined`; no outcome proxy.
10. **Decision-table mapping (start.md §12), pre-stated expectations:**
    toy evidence predicts M2 ≈ M1 (redundancy: internals substitute
    privileged state) with M2 ≫ M0 — reported exactly as measured, whatever
    the outcome; "weder Lift noch Spezifität" → Negativbefund-Publizierpfad.

After the two commands above have created the immutable causal and sensitivity
receipts, run the final evaluator exactly once:

```bash
/venv/main/bin/python \
    /workspace/locked-test-checkout/ops/locked_test_evaluate.py \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49.json \
    --manifest-sha256 1fd8c8184bb7028ad89ef42e05ef4a12939ce11be733d4c59848cc407bc15a49 \
    --raw-root /workspace/research-artifacts/raw \
    --predictions "$PREDICTION_RECEIPT_PATH" \
    --predictions-sha256 "$PREDICTION_RECEIPT_SHA256" \
    --calibration-freeze /workspace/locked-test-checkout/locks/calibration_frozen.json \
    --calibration-freeze-sha256 eb39e6952ad8864c8f9ae88a07f382b0efcbe18fd36a1221b67fe9f59106bed9 \
    --reality-gate-lock /workspace/locked-test-checkout/locks/reality_gate_frozen.json \
    --reality-gate-lock-sha256 4e0d4d5cb7e42874bed4e1f93a3e016a5a248803d06d0acc9d3fb8e435e9a151 \
    --causal-receipt "$CAUSAL_RECEIPT_PATH" \
    --causal-receipt-sha256 "$CAUSAL_RECEIPT_SHA256" \
    --sensitivity-receipt "$SENSITIVITY_RECEIPT_PATH" \
    --sensitivity-receipt-sha256 "$SENSITIVITY_RECEIPT_SHA256" \
    --cost-receipt "$COST_RECEIPT_PATH" \
    --cost-receipt-sha256 "$COST_RECEIPT_SHA256" \
    --output-root /workspace/research-artifacts/evaluation
```

Unset variables, a digest mismatch, missing evidence, an out-of-order section,
or a second outcome-dependent analysis attempt are hard stops.

## 6. Post-run (same session)

1. Stop the instance (disk preserved); the instance is started again only by
   a new explicit instruction.
2. Copy off-instance: raw artifacts, completion receipt, score features,
   summaries, run logs → `artifacts/` on the local checkout; verify file
   counts and hashes against the `locked_test_collection_receipt`. Preserve the
   global-lock file and any staging directory in the incident bundle if the run
   stopped; do not treat either as a collection artifact or silently clean it.
3. Write the final report per this template into `log.md` + a report file;
   amendment-4 addendum (if run) as an explicitly exploratory section.
4. Commit + push. Move `calibration-locked-v1` and `locked-test-score-v1` to the
   final verified tooling commit only after the section-5.0 capability blockers
   are resolved prospectively. No Month-2 work without a new instruction.
