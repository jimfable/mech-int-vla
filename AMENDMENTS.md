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
- **Implementing commit:** `aa863a6177a103ccb67d56e219367a0dc8c1ff03`

### 2026-08-03 — Correct primary-object runtime category aliases

- **Prior protocol commit:** `3831416c32d2902ecdf45776153c9330f103e705`
- **Technical reason:** An outcome-free audit of the pinned BDDL declarations found
  that task 5 exposes runtime object `black_book_1` with category `black_book`, not
  the generic configured alias `book`; task 9 exposes `white_yellow_mug_1` with
  category `white_yellow_mug`, not `yellow_and_white_mug`. The fail-closed resolver
  would therefore reject both tasks before any episode could reset. Task 2's
  `moka_pot` category is already exact.
- **Exact change:** Replace only the two `primary_object` lookup strings with their
  pinned LIBERO category names: `book` becomes `black_book`, and
  `yellow_and_white_mug` becomes `white_yellow_mug`. Also make each raw episode
  instance single-use because LeRobot advances its internal init-state index after
  reset; every condition must construct a fresh runtime to preserve paired init
  states. Task semantics, rank/order, perturbations, seeds, sample sizes, model
  features, hypotheses, and metrics are unchanged.
- **Affected hypotheses/metrics:** None; these are exact simulator identifiers and a
  pairing invariant.
- **Outcome visibility:** No policy forward, action, successful simulator
  construction/reset, observation, success value, or Discovery/Calibration/Test
  outcome had been run or viewed. The prior simulator attempt failed on a missing
  XML asset before initial-state loading.
- **Bias risk and mitigation:** Negligible. Values are forced by immutable BDDL
  declarations, tested against their real key pattern, and no semantic choice or
  outcome was available.
- **Implementing commit:** `a7e53048d82cab1284954efccd9b265d28dda7f8`

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
- **Implementing commit:** `aa863a6177a103ccb67d56e219367a0dc8c1ff03`

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
- **Implementing commit:** `aa863a6177a103ccb67d56e219367a0dc8c1ff03`

### 2026-08-03 — Freeze replay scoring and feature arithmetic

- **Prior protocol commit:** `f4565170f6ed9a7ba766d5fc1a55947f9b44e527`
- **Technical reason:** The preregistration names the M0/M1/M2 features and fixed
  transformations, but it does not define their array reductions, action scale,
  pose coordinates, coverage metric, circular statistics, intervention target,
  or cost-timer boundaries. The raw rollout artifact also cannot reconstruct
  camera/object counterfactual observations from pixels alone. Leaving those
  choices until Calibration labels are visible would create avoidable researcher
  degrees of freedom.
- **Exact change:** Score every valid pre-action state at a control step divisible
  by five in a separate, atomic, hash-linked sidecar produced by deterministic
  simulator replay of the saved reset and issued actions. Replay must match every
  saved uint8 camera frame exactly and every finite low-dimensional state value to
  absolute tolerance `1e-10`, reproduce terminal status/success, restore the exact
  MuJoCo state and camera arrays after every edit, and leave policy queues and RNG
  states unchanged. A mismatch publishes no sidecar. Both cameras receive
  brightness in uint8 space as `rint(clip(float32_pixel * multiplier, 0, 255))`;
  camera/object edits use the already frozen simulator transforms without advancing
  time. A transformed state failing the frozen finite/penetration/workspace checks
  is marked unavailable, stored as NaN with an explicit mask, and never invalidates
  the factual rollout.

  Noise seed `j` is `hash_seed("score-noise-v1", episode_id, control_step, j)`.
  The eight original chunks use `j=0..7`; the first four exact noise tensors are
  reused for every transformed and intervention call. Each primitive stores only
  the first ten postprocessed, unnormalized 7-D actions. The sidecar stores
  `(state,8,10,7)` original chunks, `(state,6,4,10,7)` transformed chunks, the
  selected representation for the corresponding calls, two `(state,4,10,7)`
  intervention arrays, seeds/masks, exact transform order, hashes, and costs. It is
  generated only after the Calibration probe is frozen, so no all-candidate
  counterfactual activations are collected.

  The per-action scale is the episode-equal weighted population standard deviation
  of all eight-by-ten original unnormalized Calibration actions at the five-step
  cadence, without labels. Nonfinite scales fail; a scale below `1e-8` is replaced
  by one and flagged. A standardized 7-D distance is
  `||delta / scale||_2 / sqrt(7)`. For each of the brightness, camera, and object
  families, M0 stores the mean and maximum over its two transforms, four draws, and
  ten actions. Brightness/camera render-equivariance each additionally store the
  mean over transform/draw of the flattened standardized 10x7 distance divided by
  `sqrt(70)`. Output uncertainty treats each of eight standardized 10x7 chunks as
  one 70-vector: covariance uses divisor eight and its ordinary trace; pairwise
  distance uses `||delta||_2/sqrt(70)`, with its mean and NumPy-linear 90th
  percentile over the 28 unordered pairs. Chunk norm is the mean raw
  `||chunk||_2/sqrt(70)` over eight draws; roughness is the mean raw
  `||a[t+1]-a[t]||_2/sqrt(7)` over eight draws and nine transitions.

  M1 appends: object-minus-EEF and goal-minus-object world-frame translations;
  canonical-sign relative quaternions `q_eef^-1 q_object` and
  `q_object^-1 q_goal` in WXYZ order; the two symmetry-aware yaw sine/cosine pairs;
  mean two-finger opening; primary contact and backend-grasp flags; normalized
  step; goal-present; and four phase indicators. Missing goal values are NaN plus
  `goal_present=0`. These columns, not the writer's reduced scalar array, define
  the raw M1 pose block. Coverage uses all valid Calibration states at the
  five-step cadence. Its continuous vector is the translations, symmetry pairs,
  opening, normalized step, contact/grasp, goal-present, and phase indicators,
  standardized by episode-equal Calibration mean/population SD with the same
  scale-one rule; distance is Euclidean RMS. Calibration queries exclude their
  entire episode and every episode with the same base init from both scaling and
  reference rows; Locked Test uses the full Calibration fit. Ties sort by distance,
  episode ID, then step. Feature one is the median distance to five successful
  same-phase states; feature two the nearest state across all phases; feature three
  the failure fraction among 25 nearest states across all phases. Insufficient
  neighbors yield NaN and the existing predictor missingness indicator.

  For M2, probe angles live in the scaled circle `phi=s*theta`. Object yaw `delta`
  expects `phi' = phi - s*delta`; its feature is mean symmetry-aware physical-angle
  error over both directions and four common draws. Brightness and camera each use
  mean `1-|mean(exp(i*phi))|` across the original plus two transformed values,
  averaged over four draws. Raw resultant norm is averaged over eight originals.
  Its distribution feature is the absolute robust z-distance from successful,
  same-phase Calibration state means, using median and `max(1.4826*MAD,1e-8)` with
  the same out-of-fold group exclusions; insufficient references yield NaN. Expert
  selections add eight-draw circular dispersion; VLM context omits that column
  because its prefix representation is architecturally upstream of flow noise.
  A zero probe vector yields NaN circular/intervention features.

  The `+/-10` degree intervention preserves the current raw probe-vector norm and
  targets its rotated vector. Its minimum-norm activation shift is
  `W^T(WW^T)^-1(z_target-z_current)` using the fitted probe coefficient `W`; it is
  applied to the VLM state token or broadcast across all expert action tokens only
  at the selected flow time. With four common draws, controllability gain is the
  absolute central finite difference of mean standardized yaw action divided by
  20 degrees in radians; specificity is the L2 norm of the mean standardized
  non-yaw central difference divided by the absolute yaw difference. A zero
  denominator yields NaN.

  Every component synchronizes CUDA immediately outside its measured interval,
  records CUDA-event and `perf_counter_ns` wall time, forward/intervention counts,
  absolute and incremental peak allocated bytes after a per-state peak reset, and
  logical plus compressed activation bytes. No unreported warmup calls are made.
  Shared original/transformed forwards are charged to M0, M1, and M2; intervention
  calls/activation bytes are additional M2 cost. Attempted and valid Calibration
  episode counts are both reported. For resource `R`, amortized per-deployment cost
  is `(R_calibration + 1000*R_inference)/1000`, alongside the unamortized terms.
- **Affected hypotheses/metrics:** Exact numerical values of all M0/M1/M2 features,
  secondary lead-time features, and resource-cost summaries. The primary estimand,
  data splits, transformations, predictor candidates, and success thresholds do
  not change.
- **Outcome visibility:** No valid simulator reset, policy action, success label,
  Discovery/Calibration/Test score, probe result, or intervention result had been
  observed. Choices came from a static sufficiency audit of the pinned source and
  synthetic artifact tests.
- **Bias risk and mitigation:** Low outcome-selection risk because the full
  arithmetic and replay failure behavior are fixed before any outcome. Remaining
  model/runtime compatibility risk is handled by fail-closed replay and publication
  of masks, hashes, raw primitives, and both logical and physical costs.
- **Implementing commit:** `PENDING_FUTURE_COMMIT`

### 2026-08-03 — Freeze causal matching and aggregation details

- **Prior protocol commit:** `f4565170f6ed9a7ba766d5fc1a55947f9b44e527`
- **Technical reason:** Section 10 fixes matching tolerances and causal success
  criteria but leaves deterministic pairing, action aggregation, random-control
  pooling, natural-manifold distances, Calibration alpha selection, and the two
  expert supporting-flow choices implicit.
- **Exact change:** Within each of the three registered pairing seeds, traverse
  recipients by SHA-256 seeded order; select the nearest eligible unused donor by
  the root-sum-square of tolerance-normalized gripper/EEF/object/time distances,
  tie-breaking on donor ID; and use each state at most once per seed, up to 20
  pairs. Pairs may recur across seeds. For each pair use the first ten unnormalized
  actions and the Calibration action scale defined above. Target effect is the mean
  standardized yaw change aligned by the sign of the mean natural donor-minus-
  recipient yaw change. Sign correctness separately uses the temporal mean of the
  per-step patched-change times natural-change product. Pair specificity is the L2
  norm of its mean standardized non-yaw shift divided by absolute mean standardized
  yaw shift; the confirmatory statistic is the median pair ratio.

  The probe projector must have numerical rank exactly two. For random control
  index `j=0..999`, generate a seeded Gaussian rank-two orthonormal row space and
  rescale its projected donor-recipient shift to the norm of the complete
  `alpha*P(h_d-h_r)` probe shift. Random statistic `j` is the median donor-aligned
  yaw effect over all valid pairs; these 1,000 medians form the comparison
  distribution. Five-nearest-neighbor distance is the arithmetic mean raw
  Euclidean distance to five selected-representation Calibration activations.
  Natural Calibration thresholds exclude the query episode/base init, use
  all remaining five-step-cadence states, and take the NumPy-linear 95th
  percentile; Locked Test queries use the full Calibration reference.

  Calibration chooses the smallest alpha in `{0.25,0.5,1.0}` whose median
  donor-aligned yaw effect is positive, sign-correctness is above 50%, and
  off-manifold rate is at most 5%; if none qualify, causal patching is frozen as
  inconclusive. Each architectural location uses its own Calibration-fitted probe.
  For early/late supporting analysis, select the lower-CV-MAE of `t=1.0` and
  `t=0.5`, tying to `t=1.0`, and freeze it before Locked Test. A location supports
  the direction iff its median donor-aligned effect is positive at the common
  frozen alpha. Matched-donor and Locked Test off-manifold results remain mandatory
  reported controls but add no unregistered confirmatory cutoff.
- **Affected hypotheses/metrics:** Causal target effect, specificity, random-control
  percentile, sign correctness, off-manifold rate, alpha selection, stability by
  seed/location, and the mechanistic-claim flag. Predictive estimands are unchanged.
- **Outcome visibility:** No valid simulator reset, policy action, success label,
  activation, probe, patch, or other Discovery/Calibration/Test result had been
  observed.
- **Bias risk and mitigation:** Low selection risk. Ambiguities are resolved before
  data, every control distribution and exclusion is retained, supporting candidates
  are selected only by the already registered Calibration probe metric, and failure
  to qualify an alpha is declared inconclusive rather than relaxed.
- **Implementing commit:** `23d0df6ac7420b1abd91ba52bb515924676988e3`
