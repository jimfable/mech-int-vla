# Protocol amendments

`start.md` v2 is frozen. Every deviation from `PREREG.md` must be recorded here
before implementation, with the exception of filling hashes/versions explicitly
left to environment resolution.

## Required entry format

### YYYY-MM-DD — Short title

- **Prior protocol commit:** `<hash>`
- **Technical reason:**
- **Exact change:**
- **Affected hypotheses/metrics:**
- **Outcome visibility:** what Discovery, Calibration, or Locked Test outputs were
  visible before the decision
- **Bias risk and mitigation:**
- **Implementing commit:** `<hash>`

## Entries

### 2026-08-03 — Correct LIBERO-10 runtime task IDs

- **Prior protocol commit:** `247a22c50a649973440f9172c44202d13c27d8fc`
- **Technical reason:** The initial shortlist paired semantic task names with IDs
  from an incompatible task order. In the pinned `hf-libero==0.1.4` runtime,
  LeRobot constructs `LIBERO_10()` with `task_order_index=0`. The package's
  `libero_suite_task_map.py` and identity task order map the preregistered book,
  stove/moka-pot, and mug/microwave tasks to runtime IDs 5, 2, and 9,
  respectively—not 9, 3, and 2. Using the original IDs would silently run three
  different tasks.
- **Exact change:** In `configs/task_order.yaml`, replace task IDs `9, 3, 2` with
  `5, 2, 9`, preserving the task names, primary objects, symmetry orders, rank,
  selection rule, sample sizes, perturbations, hypotheses, and all metrics.
- **Affected hypotheses/metrics:** None. This is an identifier correction that
  makes execution match the already-preregistered semantic tasks.
- **Outcome visibility:** No simulator reset, policy inference, Discovery,
  Calibration, or Locked Test outcome had been run or viewed. The discrepancy was
  found by static inspection of the pinned package before the first rollout.
- **Bias risk and mitigation:** Negligible outcome-selection risk because semantic
  tasks and their order are unchanged and no outcomes were visible. The original
  erroneous IDs remain immutable in prior commit `247a22c`; this prospective
  amendment and the corrected executable config are committed before execution.
- **Implementing commit:** `24bf90f7ea9ce9a7c0580620623a490fd2dbf288`

### 2026-08-03 — Move VLM candidate to a causally accessible residual

- **Prior protocol commit:** `247a22c50a649973440f9172c44202d13c27d8fc`
- **Technical reason:** Static source inspection and a pre-rollout instrumentation
  audit showed that SmolVLA creates every layer's prefix K/V entry before executing
  the final VLM norm, then discards that final normalized prefix output during
  inference. A shift at the preregistered final-norm state token therefore cannot
  alter the cache or actions, even if the identical prefix forward is repeated.
- **Exact change:** Replace only the VLM-context candidate with the pre-norm residual
  state token entering
  `model.vlm_with_expert.get_vlm_model().text_model.layers[12].input_layernorm`
  during prefix-cache construction (the VLM residual after layer 11). Its probe and
  intervention use that same pre-norm residual coordinate system. A causal patch is
  applied in place to the state token before layer 12, so the residual branch and
  K/V entries for layers 12 onward are rebuilt from the intervention. The two expert
  locations, flow times, five-candidate count, selection preference, probe procedure,
  hypotheses, perturbations, sample sizes, and metrics are unchanged.
- **Affected hypotheses/metrics:** The VLM candidate's exact representation changes;
  the primary prediction comparison and causal criteria do not.
- **Outcome visibility:** No model inference, simulator reset, or Discovery,
  Calibration, or Locked Test outcome had been run or viewed. The issue was found
  using the pinned LeRobot source and synthetic unit tests.
- **Bias risk and mitigation:** Low outcome-selection risk because there were no
  outcomes and the replacement was selected solely for architectural causal
  accessibility. The depth is fixed prospectively at layer 12, matching the
  preregistered late-expert depth, and will not be tuned.
- **Implementing commit:** `PENDING_THIS_COMMIT`

### 2026-08-03 — Correct stale checkpoint state-shape metadata

- **Prior protocol commit:** `247a22c50a649973440f9172c44202d13c27d8fc`
- **Technical reason:** The pinned checkpoint's `config.json` declares a six-value
  `observation.state`, but both its pinned normalizer artifact and LeRobot v0.6.0's
  LIBERO processor use eight values: EEF position (3), EEF axis-angle (3), and two
  gripper positions. Truncating to the stale declaration would change the trained
  input and conflict with the checkpoint statistics.
- **Exact change:** Before policy/processor construction, replace only the loaded
  `observation.state` feature shape from `(6,)` to `(8,)`. Assert that raw state,
  processed state, and normalizer statistics are all length eight, and record both
  metadata values. No state value is added, removed, or synthesized.
- **Affected hypotheses/metrics:** None; this is a checkpoint compatibility metadata
  correction that preserves the trained eight-value input.
- **Outcome visibility:** No model inference, simulator reset, or Discovery,
  Calibration, or Locked Test output had been run or viewed.
- **Bias risk and mitigation:** Negligible. The value is fixed by two independent
  pinned runtime artifacts and guarded by fail-closed assertions.
- **Implementing commit:** `PENDING_THIS_COMMIT`

### 2026-08-03 — Fix condition timing and operational validity definitions

- **Prior protocol commit:** `247a22c50a649973440f9172c44202d13c27d8fc`
- **Technical reason:** The original protocol fixed ten settling steps but did not
  state whether they occur before or after a physical edit, and described
  penetration, instability, workspace membership, camera displacement, and task
  phase qualitatively. Those choices must be deterministic before Discovery.
- **Exact change:** Construct each episode reset with the wrapper's automatic wait
  disabled, set the fixed initial state, apply its condition, and then execute
  exactly ten registered dummy actions for every cell (including IID and render-only
  cells). A physical edit is valid iff all simulator/object values are finite; no
  contact involving the primary object has penetration deeper than 5 mm; its final
  free-joint linear speed is at most 0.05 m/s and angular speed at most 0.5 rad/s;
  its center lies within the tabletop XY rectangle and from 2 cm below to 50 cm
  above the table surface; and initial task success is false. Camera yaw composes
  about world Z at the existing optical center; lateral displacement follows the
  camera-local horizontal +X axis with the signed magnitude in config. Phase is
  `placed` when the primary placement predicate is true, otherwise `transport` when
  the backend grasp predicate is true and the primary object has moved at least
  2 cm from its post-settle start, otherwise `grasped` when the backend grasp
  predicate is true, else `pregrasp`. Existing contact/gripper/time constraints
  remain separate matching fields.
- **Affected hypotheses/metrics:** None; this makes validity, equal timing, camera
  transforms, and causal-pair strata executable.
- **Outcome visibility:** No model inference, simulator reset, or Discovery,
  Calibration, or Locked Test output had been run or viewed.
- **Bias risk and mitigation:** Low. Thresholds are fixed prospectively from common
  MuJoCo stability scales, every condition receives equal settling actions, and
  invalid episodes remain excluded exactly as preregistered.
- **Implementing commit:** `PENDING_THIS_COMMIT`
