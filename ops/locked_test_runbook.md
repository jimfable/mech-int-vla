# Locked Test runbook (approved 2026-08-25; strictly before Locked Test opening)

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
cd mech-int-vla
git rev-parse HEAD                                  # == tag commit of calibration-locked-v1
git rev-list -n 1 tags/calibration-locked-v1        # must equal HEAD
git status --porcelain                              # must be empty
# per-artifact byte hashes: locks/calibration_frozen.json -> artifact_hashes
../tiny-vla-interp/.venv/bin/python ops/prepare_locked_test_artifacts.py \
    --repo-root . --environment-lock environment.lock
# prints 160-episode manifest sha256 1fd8c818… and writes
#   artifacts/manifests/locked-test-manifest-1fd8c818….json
#   artifacts/locked-test-authority.json            (locked_test_accessed: True)
# ANY digest mismatch -> STOP + report; do not start the instance.
```

## 1. Instance start + environment verification

1. `vastai start instance <ID>` (single RTX 5090, disk-preserved).
2. `vastai execute <ID> "nvidia-smi"` → expect 1× RTX 5090, 0% util.
3. Verify the instance layout used by the freeze (`/venv/main` python,
   `/workspace/hf-cache` with the policy snapshot, `/workspace/runstate`,
   LeRobot v0.6.0 + hf-libero==0.1.4, MuJoCo EGL). If the layout or any
   pinned package version differs from `environment-gpu.freeze` →
   STOP + report, no rollout.
4. Copy the current repo (the frozen HEAD from step 0) to
   `/workspace/locked-test-checkout`; `cd` there and verify
   `git rev-parse HEAD` == local HEAD and `calibration-locked-v1` == HEAD and
   clean tree. Copy the manifest + authority into `/workspace/runstate/`.

## 2. Dry run (2–3 episodes, fail-closed)

Run the cell script directly on the first three cell indices; each cell is one
episode and runs the full frozen chain (snapshot, instrumentalization,
rollout, raw artifact write, provenance). Expect per-cell wall time in the
Calibration envelope (≈6–8 min or less on this GPU); expect valid artifacts
with policy_revision `31d453f7…` and code_commit `18d64941…`.

```bash
/venv/main/bin/python /workspace/locked-test-checkout/ops/locked_test_cell.py \
    --index 0 --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c818….json \
    --cache-dir /workspace/hf-cache \
    --artifact-root /workspace/research-artifacts/raw \
    [--log-dir /workspace/run-logs/locked-test]
```

(Exact flag set as declared by `locked_test_cell.py --help`; the guard
`assert_locked_test_ready` runs before every episode.) Then run the supervisor
in plan-only mode and confirm `{"kind":"locked_test_plan_validated",
"episodes":160,…}`. Any deviation from the freeze (wrong manifest, authority,
tag, or artifact bytes) → STOP + report. Delete the three dry-run episodes'
raw artifacts (they are not part of the manifest collection) unless the
supervisor's resume logic accepts them — if it does not, the runbook keeps them
as pre-collection diagnostics and the supervisor is started fresh over the 160
manifest cells.

## 3. Full collection (160 rollouts)

```bash
/venv/main/bin/python /workspace/locked-test-checkout/ops/locked_test_supervisor.py \
    --repo-root /workspace/locked-test-checkout \
    --environment-lock /workspace/locked-test-checkout/environment.lock \
    --manifest /workspace/runstate/locked-test-manifest-1fd8c818….json \
    --authority /workspace/runstate/locked-test-authority.json \
    --cache-dir /workspace/hf-cache \
    --artifact-root /workspace/research-artifacts/raw \
    --cell-script /workspace/locked-test-checkout/ops/locked_test_cell.py \
    --log-dir /workspace/run-logs/locked-test \
    --completion-receipt /workspace/runstate/locked-test-complete.json
```

Supervision: log lines `starting/resume_validated` for 160 episode ids
(`libero_10-task5-locked-test-…`); stop on `unexplained staging`,
`authority mismatch`, or any `RuntimeError` — then STOP, report, resume only
via the same command (it resume-validates). `plan-only` first (step 2) also
validates the plan. Expected wall time ≈5–10 GPU h.

## 4. Scoring (frozen predictor/probe applied to Locked Test)

Frozen scoring tooling is `ops/locked_test_score.py` (Locked Test bound, tag
`locked-test-score-v1`, same predictor bundle and probe bytes as the freeze),
invoked with the Calibration scoring parameter pattern (see the Calibration
score log entries), now against the Locked Test manifest/authority and
`--raw-root …/raw/locked_test`. It writes the score allocation, features and
`score-feature-summary.json` with `locked_test_accessed: True`. Expected wall
time ≈11–18 GPU h. Budget gate: beyond 24 h → stop and report.

## 5. Evaluation — FIXED template (no improvisation; PREREG §11 order)

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
   check, matched-donor control (<5°). <30 valid pairs → inconclusive.
8. **Cost accounting:** GPU hours, instance charges, per-stage wall times,
   any budget-gate stops.
9. **Sensitivity (amendment items 1–3, separate section):**
   9a. rollout diagnostics: nearest-object identity accuracy + error/distance
       per step/cell;
   9b. patching dose × difficulty: alphas {0.5, 1.0} × 8 cells (sign +
       specificity, same definitions);
   9c. broken-successes ledger for the total 60-pair set.
10. **Decision-table mapping (start.md §12), pre-stated expectations:**
    toy evidence predicts M2 ≈ M1 (redundancy: internals substitute
    privileged state) with M2 ≫ M0 — reported exactly as measured, whatever
    the outcome; "weder Lift noch Spezifität" → Negativbefund-Publizierpfad.

## 6. Post-run (same session)

1. Stop the instance (disk preserved); the instance is started again only by
   a new explicit instruction.
2. Copy off-instance: raw artifacts, completion receipt, score features,
   summaries, run logs → `artifacts/` on the local checkout; verify file
   counts and hashes against receipts.
3. Write the final report per this template into `log.md` + a report file;
   amendment-4 addendum (if run) as an explicitly exploratory section.
4. Commit + push; tag scoring code `locked-test-score-v1` already exists at
   the tooling commit. No Month-2 work without a new instruction.