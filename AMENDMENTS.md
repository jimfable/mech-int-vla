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
  time. A camera-only edit is unavailable only if it creates nonfinite simulator or
  camera values. An object-pose edit additionally must pass the frozen
  penetration/workspace checks. An unavailable transform is stored as NaN with an
  explicit mask and never invalidates the factual rollout.

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
  the raw M1 pose block. Each symmetry-aware yaw pair uses the yaw of its adjacent
  relative quaternion, not subtraction of separately extracted world Euler yaws.
  Coverage uses all valid Calibration states at the
  five-step cadence. Its continuous vector is the translations, symmetry pairs,
  opening, normalized step, contact/grasp, goal-present, and phase indicators,
  standardized by episode-equal Calibration mean/population SD with the same
  scale-one rule; distance is Euclidean RMS. Calibration queries exclude their
  entire episode and every episode with the same base init from both scaling and
  reference rows; Locked Test uses the full Calibration fit. Ties sort by distance,
  episode ID, then step. Feature one is the median distance to five successful
  same-phase states; feature two the nearest state across all phases; feature three
  the failure fraction among 25 nearest states across all phases. Insufficient
  neighbors yield NaN and the existing predictor missingness indicator. A reference
  row with any nonfinite coverage coordinate is ineligible; a nonfinite query vector
  receives NaN for all three coverage outputs rather than partial-coordinate
  distance or implicit imputation.

  For M2, probe angles live in the scaled circle `phi=s*theta`. Object yaw `delta`
  expects `phi' = phi - s*delta`; its feature is mean symmetry-aware physical-angle
  error over both directions and four common draws. Brightness and camera each use
  mean `1-|mean(exp(i*phi))|` across the original plus two transformed values,
  averaged over four draws. Raw resultant norm is averaged over eight originals.
  Its distribution feature is the absolute robust z-distance from successful,
  same-phase Calibration state means, using median and `max(1.4826*MAD,1e-8)` with
  the same out-of-fold group exclusions; fewer than five eligible references yield
  NaN. Expert selections add eight-draw circular dispersion; VLM context omits that
  column because its prefix representation is architecturally upstream of flow
  noise. A probe vector with raw L2 norm at most `1e-12` is numerically zero.
  Intervention availability is stored separately for each of the four common-noise
  draws. A numerically zero probe vector makes that draw unavailable; both
  aggregate intervention features are NaN unless all four draws are available,
  and a selective subset is never averaged. A numerically zero probe vector also
  yields NaN for any circular feature whose defined reduction contains that vector.

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
  logical plus compressed activation bytes. Compressed activation bytes are the
  byte length after zlib level-9 compression of deterministic NumPy `.npy` bytes
  (`allow_pickle=False`) for the contiguous float32 selected activation. No
  unreported warmup calls are made.
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
- **Implementing commit:** `882d753f83e930361e71e6e51ce63e633d667355`

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

### 2026-08-03 — Freeze deterministic failure-event annotation

- **Prior protocol commit:** `c9c0447ce4077f01027e75ba1b6e6d1f01899868`
- **Technical reason:** Section 5 fixes the semantic event order but does not define
  gripper-close polarity, the distance trend, landing/support detection, recovery
  from a workspace exit, event onset versus confirmation, or precedence. The raw
  trace has object-center height and gripper/object contact but not object-bottom
  height or a support-surface contact identity, so the literal drop clause requires
  a declared trace-level proxy before Discovery videos are inspected.
- **Exact change:** Annotate only valid failed episodes; successful episodes receive
  no failure event and invalid resets are excluded. Postprocessed LIBERO gripper
  action index 6 is `-1=open`, `+1=close`; zero is neither. A missed grasp candidate
  begins at the episode's first action with value greater than zero. It is confirmed
  only when frames `c+1..c+10` all lack primary/gripper contact, the ordinary
  least-squares slope of 3-D EEF/object distance over those ten equally spaced
  frames is strictly positive, and the final distance exceeds the first by at least
  5 mm. Its onset is `c+1` and confirmation is `c+10`; an incomplete ten-frame
  window cannot qualify.

  A drop candidate begins at frame `r` when primary/gripper contact changes true to
  false and the backend grasp predicate was true at `r-1`. Search forward for the
  first three-frame window `u..u+2` for which there is no gripper recontact anywhere
  from `r` through `u+2`, object center Z is at most the post-settle frame-0 center Z
  plus 5 mm on all three landing frames, and phase is not `placed` on all three.
  Recontact rejects that loss, but a later qualifying contact loss may be used. This
  stable initial-height proxy is called `initial-support-height proxy`, never
  literal support-surface contact. Drop onset is `r` and confirmation is `u+2`; no
  qualifying landing means no drop annotation.

  At the Reality Gate freeze, derive reachable object-center bounds separately for
  X/Y/Z as the float64 minima/maxima over frame 0 from every valid selected-task
  Discovery reproduction/assigned-yaw rollout plus frames 1 through terminal from
  every valid successful one, then expand each side by exactly 5 cm. Thus all valid
  initial states define the start envelope while failed excursions do not define
  reachability. An irrecoverable exit is the first frame of the final contiguous
  suffix in which the object center remains outside at least one expanded axis
  through the terminal frame, including onset 0 when an episode begins and remains
  outside; confirmation is terminal. Bounds, included and excluded episode hashes,
  and their success/validity reasons cannot be changed after `prereg-locked-v1`.

  For a failed episode choose the qualifying event with earliest onset; exact ties
  follow the registered textual order `missed_grasp`, `dropped_object`,
  `irrecoverable_workspace_exit`. Store both onset and confirmation. If none
  qualifies, require exactly 520 actions and assign `terminal_horizon` at step 520;
  a shorter failed episode without a qualifying event fails closed instead of
  inventing a horizon. The Reality Gate JSON must store and hash task identity,
  the exact primary-placement predicate key(s), exact Discovery episode IDs and both
  raw-artifact hashes, raw and expanded bounds, every threshold/proxy, implementation
  commit, and all Discovery annotations used in the preregistered video audit. The
  audit includes every expected selected-task Discovery episode, so video inclusion
  cannot be chosen from the observed event or outcome.
- **Affected hypotheses/metrics:** Failure-event subtype, alarm lead time, detection
  rate, and conditional lead-time summaries. Terminal success labels, primary
  paired log loss, task gates, and prediction features are unchanged.
- **Outcome visibility:** No valid simulator reset, action, success/failure label,
  video, trace, or Discovery/Calibration/Locked Test event had been observed. The
  action polarity came from pinned robosuite source; sufficiency and proxy choices
  came from static schema inspection only.
- **Bias risk and mitigation:** Low outcome-selection risk. The proxy is explicitly
  named and distinguishable from physical support contact; all thresholds,
  precedence, onset/confirmation rules, bounds cohort, and early-termination failure
  behavior are fixed before data and retained in a content-addressed freeze.
- **Implementing commit:** `f22714a579a82639b1b7ed650b548f40ccbdc69b`

### 2026-08-03 — Harden the post-Discovery lock evidence boundary

- **Prior protocol commit:** `596261b4c2b5b2d7aaaa6fca5f2e12612b9763df`
- **Technical reason:** After the Discovery outcomes and durable raw backup were
  visible, the repository still represented the Reality-Gate and failure-event
  receipts only as write-side objects. A lock consumer could therefore accept a
  self-consistent hand-written JSON payload without regenerating the deterministic
  Discovery manifest, gate arithmetic, orientation thresholds, or failure-event
  semantics. The orientation extraction contract also needed a raw-artifact
  constructor so caller-supplied state values could not be mistaken for rollout
  evidence.
- **Exact change:** Add a strict, duplicate-key-free lock JSON reader; reject
  symlinked/non-regular/oversized freeze files; require a validated
  `ProtocolConfig`; rehydrate and recompute the complete Reality-Gate receipt;
  add a raw-rollout quaternion extraction constructor with fixed equal-state,
  every-control-step-including-terminal weighting/cadence and extraction tests;
  rehydrate orientation eligibility and the canonical failure-event freeze with
  their owning validators; cross-bind all 40 artifact identities, validity and
  success outcomes, provenance, and hashes; and require the freeze's
  implementation commit to be an ancestor of the tagged lock commit. This is a
  two-commit implementation-then-lock workflow and does not alter task order,
  rollout assignments, thresholds, splits, estimands, or the Locked Test guard.
- **Affected hypotheses/metrics:** No estimand, threshold, sample size, or
  rollout behavior changes. The explicit orientation weighting/cadence is a
  post-Discovery operationalization of the already recorded eligibility check;
  its outcome is retained and disclosed rather than silently treated as
  preregistered before outcome visibility.
- **Outcome visibility:** The exact 40 Discovery rollouts, gate outcomes,
  orientation state extraction, and deterministic failure-event annotations were
  visible before this hardening decision. No Calibration output, Calibration
  manifest, Locked Test artifact, or protected-split label was accessed.
- **Bias risk and mitigation:** Moderate selection-risk disclosure because the
  orientation weighting/cadence was made explicit after Discovery. Mitigation is
  an immutable audit trail, raw-derived extraction, exact content hashes,
  fail-closed typed rehydration, a clean lock commit, and no protected-split
  access until the new guard passes. The original Discovery code commit remains
  separately recorded as `b491dc76641efe3a5c5d7eef6bb87af13d85f10b`.
- **Implementing commit:** `b41867e01ba50e6eec7fd869b4b18c0b8ea46a01`

### 2026-08-04 — Continue Calibration replay scoring with two disjoint workers

- **Prior protocol commit:** `08342a4ea05ca5a0b021ce4651db65eda5423073`
- **Technical reason:** The preregistered replay scorer was launched serially, but
  a later isolated two-worker benchmark on four already completed Calibration
  episodes increased aggregate throughput from `9.0420` to `15.5347` episodes per
  hour including model load (`1.7181x`). An independent audit over all 145 scored
  states in those episodes found that only the four explicitly recorded
  runtime/resource arrays (`original_cost`, `transformed_cost`,
  `intervention_minus_cost`, and `intervention_plus_cost`) differed. Every
  scientific array, action, selected activation, seed, transformation,
  availability mask, and non-cost metadata field was identical. Within the cost
  arrays, the deterministic counts and activation-byte fields were identical;
  differences were confined to CUDA-event time, wall time, and physical peak
  allocation fields. The audit receipt has SHA-256
  `68904e5285b029f7330cdeb43de85d35396f8e10b10f898298744ab086dc6d85`.
  Static dependency inspection and a regression test that perturbs all four cost
  arrays establish that none is an M0/M1/M2 predictor input. A read-only
  GPT-5.6-Sol-xhigh review therefore classified the outputs as scientifically
  equivalent, though not whole-sidecar byte-identical, and approved a guarded
  scheduling-only cutover.
- **Exact change:** Preserve and hash-freeze every already published authoritative
  serial sidecar without recomputation or replacement. Observe the serial runner's
  next JSON `score_completed` line, then send one interrupt to the exact
  PID/start-time/executable/cmdline identity, wait for both Python and its `flock`
  wrapper to exit, and require a fully loadable new sidecar, a stable rescan of all
  frozen hashes, no staging/publish-lock residue, and exclusive acquisition of the
  existing global score lock. Freeze the exact complement of valid manifest IDs
  at that boundary and assign it deterministically, by manifest order and without
  consulting labels, features, durations, state counts, or costs, to two disjoint
  workers. A coordinator holds the global lock throughout; each worker has its own
  lock and same-filesystem staging root, uses the unchanged locked checkout,
  weights, raw artifacts, probe, seeds, transforms, offline environment, and
  scoring functions, fully validates each staged sidecar, and atomically promotes
  it only if the authoritative destination does not exist. Any unexpected output,
  overwrite attempt, provenance mismatch, OOM, or worker failure stops both
  workers fail-closed. On completion, the unchanged serial finalizer runs with
  zero missing episodes to perform the original allocation audit, M0/M1/M2 feature
  construction, and predictor fitting. Locked Test remains closed.

  A hash-bound execution receipt assigns every authoritative sidecar to exactly
  one of three physical-cost modes: `serial`,
  `serial_with_equivalence_benchmark_contention`, or `two_worker`. Physical
  latency and memory costs are reported separately by mode; summed per-worker
  CUDA/wall time is not treated as parallel makespan, and per-process CUDA peaks
  are not treated as aggregate device peaks. Coordinator elapsed time, model-load
  and benchmark overhead, and device-level peak telemetry are separate. Logical
  counts and deterministic activation-byte totals remain aggregable. The receipt
  is provenance-only and is never exposed to the feature pipeline.
- **Affected hypotheses/metrics:** M0/M1/M2 scientific features, transformations,
  predictors, targets, splits, probe, estimands, and statistical rules do not
  change. Runtime and resource-cost summaries become explicitly mixed-execution
  descriptive measurements and must be stratified by execution mode. The observed
  speedup is post-hoc operational evidence, not a confirmatory scheduler claim.
- **Outcome visibility:** All 160 raw Calibration rollouts and their success/failure
  outcomes, the selected probe, 22 or more serially published scoring sidecars by
  the eventual cutover, the four-episode benchmark, and its equivalence and
  throughput results were visible before this decision. No Locked Test artifact,
  label, score, path contents, or protected-split output was accessed.
- **Bias risk and mitigation:** The four-episode audit supports high confidence for
  the observed 145 states but only medium confidence when generalized to the
  remaining episodes. Scheduling was chosen after throughput was observed, and
  the benchmark overlapped the serial process, so physical-cost and speed claims
  are susceptible to selection and contention bias. Mitigation is a prospective
  amendment before cutover, outcome-blind deterministic sharding, immutable serial
  hashes, per-sidecar provenance and execution-mode binding, continuous validation
  of scientific/deterministic fields, stratified cost reporting, unchanged final
  feature code, fail-closed publication, and continued Locked-Test exclusion.
- **Implementing commit:** `6cb3733a197f1374025fd08fee44d065e3350c04`

### 2026-08-04 — Recover the Calibration scoring cutover after inherited SIGINT ignore

- **Prior protocol commit:** `7fe9ebce54926bbb9b8ae3a47161f2fb478b5cab`
- **Technical reason:** The committed two-worker coordinator observed and durably
  recorded the exact 31/160 serial boundary, then sent its single planned SIGINT
  to the content- and process-identity-bound Python scorer. The scorer had been
  launched as a detached background job with SIGINT inherited as `SIG_IGN`.
  `/proc` therefore showed SIGINT ignored, unblocked, and not pending; both the
  Python process and its `flock` wrapper remained alive past the 120-second exit
  timeout. The coordinator failed closed before reacquiring the global lock,
  creating a continuation plan, or launching workers. It left only the original
  cutover intent, boundary, and one SIGINT intent/dispatch receipt. Episode 32
  subsequently completed normally and remains an immutable serial sidecar. A
  new read-only GPT-5.6-Sol-xhigh review approved a narrowly scoped recovery
  protocol; without every guard below, the serial scorer remains authoritative.
- **Exact change:** Append a recovery attempt without altering any original
  receipt. First close the original signal attempt truthfully as dispatched but
  ineffective because of the inherited ignore disposition; never attribute a
  later exit to SIGINT. From a stable, fully validated current serial inventory
  and log offset, wait for one fresh parsed `score_completed` record. Require it
  to be exactly the next manifest episode and bind its ID, digest, count, total,
  log identity, byte offset, and timestamp. Immediately revalidate the exact
  Python PID, start ticks, executable, complete argv, parent identity, runner
  hash, and signal masks; SIGTERM must be neither ignored, blocked, nor
  custom-caught. Fsync a hash-linked recovery intent, send SIGTERM exactly once
  to the Python PID only, then fsync its dispatch receipt. Never signal the
  wrapper first, never repeat or escalate the termination signal, and never use
  SIGKILL.

  After observing Python exit and natural wrapper exit, require no replacement
  or orphan scorer, exclusive reacquisition of the existing global flock, and a
  complete reload/hash validation of every authoritative sidecar. The post-exit
  inventory must be exactly the stable baseline plus the single freshly observed
  boundary episode, in manifest-prefix order, with all earlier hashes unchanged
  and the new combined digest equal to the logged digest. Any extra, missing,
  malformed, or changed sidecar, or any `.tmp-*`, `.publish.lock`, staging, or
  feature-finalization residue, aborts before planning. Record only that SIGTERM
  was dispatched and process exit was subsequently observed unless stronger
  causal evidence is available. Treat any already-started next episode as
  abandoned non-authoritative computation and report its cost separately when
  measurable, otherwise as unavailable. Freeze the actual manifest complement
  only after these checks, then use the already committed outcome-blind
  alternating two-worker assignment and all existing no-overwrite, provenance,
  execution-mode, finalizer, and Locked-Test guards. A restart after durable
  recovery dispatch may only wait and validate; it must never re-signal.
- **Affected hypotheses/metrics:** No scientific array, action, activation,
  seed, transformation, availability mask, probe, M0/M1/M2 feature, target,
  split, estimand, or statistical rule changes. The newly completed serial
  sidecars remain in the serial execution-cost stratum. The abandoned partial
  episode is excluded from the canonical episode denominator and is disclosed
  only as recovery overhead. Physical costs remain stratified by execution
  mode and are never predictor inputs.
- **Outcome visibility:** All 160 raw Calibration rollouts and their outcomes,
  the selected probe, the scheduling benchmark and equivalence audit, the first
  32 serial scoring sidecars, and the failed SIGINT attempt were visible before
  this recovery decision. No Locked Test path contents, artifact, label, score,
  or protected-split output was accessed.
- **Bias risk and mitigation:** The recovery is operational rather than
  outcome-adaptive: its future boundary is determined solely by the next
  manifest-order completion, and the later worker shards exclude labels,
  features, durations, state counts, and costs. Main risks are SIGTERM bypassing
  Python cleanup, observer delay, an unexpected extra publication, unavailable
  partial-episode cost, and premature lock release. Durable attempt receipts,
  exact identity/signal-mask checks, Python-only signalling, full post-exit
  validation, residue rejection, exclusive flock acquisition, immutable hashes,
  and fail-closed fallback to the serial scorer mitigate those risks.
- **Implementing commit:** `257309f64c60fc49a97babf6bc77603019ef1fb9`
