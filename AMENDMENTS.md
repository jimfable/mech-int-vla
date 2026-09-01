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

### 2026-08-05 — Complete Calibration feature finalization after a finalizer attribute defect

- **Prior protocol commit:** `56e191d4f3fe5c6f6996948acfcb4ecaf61e7a39`
- **Technical reason:** Raw Calibration scoring reached the full 160/160
  authoritative sidecars (40 immutable serial plus 120 two-worker promotions)
  and the execution receipt recorded `status="scoring_complete"`. The frozen
  continuation coordinator then launched its unchanged-serial finalizer, which
  validated all 160 sidecars, published the score-allocation receipt, the
  feature reference bundle and cohort, and fitted the M0/M1/M2 failure
  predictors. It aborted only while assembling the final summary, because
  `calibration_score_18d6494.py` read `FeatureCohort.metadata_sha256` — an
  attribute that does not exist. The digest property of `FeatureCohort` is
  named `provenance_sha256`, while the sibling class `FeatureReferenceBundle`
  exposes `metadata_sha256`. The defect is confined to receipt assembly and
  never touched a sidecar, a score, or a feature array. Because the
  coordinator promotes the authoritative feature root only after the staged
  summary exists, `attempt-0001` remained unpromoted and no authoritative
  feature artifact was published. The three scheduled heartbeats after the
  abort produced no supervising turn, so the failure stayed unobserved until
  this session.
- **Exact change:** Correct the attribute to `FeatureCohort.provenance_sha256`,
  the exact semantic counterpart of `FeatureReferenceBundle.metadata_sha256`
  with an identical implementation (`_metadata_sha256(self.to_metadata())`).
  The fix is applied only in `ops/calibration_score_18d6494.py`, which is not
  a member of `SCORING_SOURCE_FILES`; no file entering `scoring_source_sha256`
  is modified, so the provenance chain binding all 160 sidecars is unchanged.
  The corrected runner is deployed to the instance as a new file
  `calibration_score_18d6494_fix1.py`, leaving the original runner
  byte-identical so the frozen plan's `serial_runner_sha256` remains true.
  Because `_assert_arguments_match_plan` requires exactly that recorded runner
  hash, the continuation wrapper can no longer launch the corrected runner.
  The finalizer step is therefore executed directly, using the plan's exact
  `finalizer_command_prefix`, the identical locked environment, an exclusive
  `flock` on the plan's global lock, and a fresh
  `finalizer-staging/attempt-0002/features` root. The wrapper's post-run guards
  are reproduced explicitly: the staged summary must report
  `sidecar_count == 160` and `locked_test_accessed == false` before the staged
  root is promoted by a no-overwrite rename to the authoritative feature root,
  followed by the completion receipt.
- **Affected hypotheses/metrics:** None. No estimand, split, threshold, feature
  schema, probe, seed, transformation, label, or statistical rule changes. The
  160 score sidecars are untouched and remain immutable. The corrected line
  only records a digest inside a receipt and cannot enter any M0/M1/M2 feature
  or predictor input. Physical costs remain stratified by execution mode and
  stay outside predictor inputs.
- **Outcome visibility:** All 160 raw Calibration rollouts and outcomes, the
  complete score sidecar set, the execution receipt, and the unpromoted
  `attempt-0001` staging tree — including its fitted predictor metadata — were
  visible before this decision. The Calibration M2-versus-M1 comparison was
  read before this entry was written, so the entry is explicitly not blind to
  it; it is recorded here rather than presented as a prior decision. No Locked
  Test path, artifact, label, score, or protected-split output was accessed at
  any point.
- **Bias risk and mitigation:** The change is a defect correction in receipt
  assembly, not an outcome-adaptive choice: the corrected digest is a property
  of an already published cohort and cannot alter any prediction. Main risks
  are an unnoticed second difference between the original and corrected runner,
  silent divergence of the direct run from the frozen execution environment,
  and loss of the wrapper's promotion guards. These are mitigated by a
  byte-level diff proving exactly one changed line, an unmodified original
  runner, reuse of the plan's exact command prefix and locked environment
  variables, an exclusive global lock, a fresh staging attempt that never
  overwrites `attempt-0001`, explicit reproduction of the staged-summary
  guards, and a no-overwrite promotion. Determinism is verified independently
  by requiring the rerun's predictor metadata to reproduce the `attempt-0001`
  values exactly.
- **Implementing commit:** `b009029b84a48c4c1e1c44fbd48e0a33616557c7`

### 2026-08-05 — Fix the score source and failure step for alarm calibration

- **Prior protocol commit:** `f8bca592927260b8065049aa1af1bccdd1896795`
- **Technical reason:** `PREREG.md:43-45` fixes the alarm threshold on Calibration
  at an episode-level false-positive rate ≤ 10% among successful episodes, but it
  does not name which predicted probabilities that rate is measured on. Two
  sources exist: the base predictor refit on all Calibration rows, and
  group-out-of-fold predictions. The fitted bundle retains only scalar OOF
  metrics, not per-row OOF probabilities, so the source must be chosen and stated
  explicitly rather than inherited from an artifact. Separately, `PREREG.md:47-49`
  defines `t_failure` as "the first task-specific failure event when one can be
  identified", and the annotation artifact exposes both an `onset_step` and a
  later `confirmation_step`, so the mapping must also be stated.
- **Exact change:** (1) Calibrate the alarm threshold on **group-out-of-fold
  calibrated probabilities**, reconstructed deterministically from the frozen
  cohort and the already-selected family/hyperparameters, never on in-sample
  scores from the refit-on-all base model. This follows the protocol's existing
  Calibration currency: the Platt calibrator is fit from group-out-of-fold
  predictions (`PREREG.md:323-325`) and Kill Switch 1 is evaluated on
  group-out-of-fold Calibration predictions (`PREREG.md:328`). The refit-on-all
  model exists to be deployed on Locked Test, not to be evaluated in-sample.
  The reconstruction is verified against four frozen anchors before use: the
  recorded `fold_assignments`, the recorded per-model `oof_metrics`, the frozen
  Platt slope/intercept, and `calibration_data_sha256`
  (`215e52cf…488986`). `_make_group_folds` uses no RNG, so the fold partition is
  a deterministic function of episodes and groups. (2) Map `t_failure` to
  `onset_step`, not `confirmation_step`, as the first identifiable failure event.
  The threshold calibration itself uses only successful episodes and therefore
  needs no failure step; the mapping matters only for lead time.
- **Affected hypotheses/metrics:** No estimand, split, feature, probe, seed, or
  threshold value changes. This fixes how two already-preregistered quantities
  are computed. Both choices are the conservative direction: in-sample scores
  would push successful-episode probabilities toward zero and yield a threshold
  that is too low, silently exceeding the 10% cap on Locked Test and inflating
  detections for whichever model overfits more; `confirmation_step` would inflate
  lead times because an alarm must fire strictly before the failure step.
- **Outcome visibility:** All 160 Calibration rollouts and outcomes, the score
  sidecars, the published Calibration features, and the M0/M1/M2 Calibration OOF
  metrics were visible before this decision, including the 1.17% M2-over-M1
  log-loss difference. The Calibration failure annotations (107 successes, 53
  failures; 48 terminal-horizon, 5 irrecoverable workspace exits) were also
  visible. No Locked Test path, artifact, label, or output was accessed.
- **Bias risk and mitigation:** The risk is that a source is chosen after seeing
  which one flatters the M2 lead-time result. It is mitigated by fixing both
  choices before any alarm threshold or lead time is computed, by choosing the
  conservative option in both cases, by deriving both from protocol text that
  predates the results, and by binding the reconstruction to four frozen anchors
  that fail closed on any mismatch. The choice was reviewed read-only by an
  independent model (Fable 5) against the preregistration before implementation.
- **Implementing commit:** `cab0654c4f0698e27f3f488f5ab51a1fd5035d97`

### 2026-08-05 — Re-render counterfactual observations and re-score Calibration

- **Prior protocol commit:** `266b457bed063e6b093068a61bbdb761078e9b55`
- **Technical reason:** Scoring-time camera and object-pose counterfactuals were
  silent no-ops. `libero_runtime.py` read observations via robosuite's
  `_get_observations()` without `force_update=True`; under the pinned
  `robosuite==1.4.0` that serves cached observable values refreshed only inside
  `step()`/`reset()`. The transform helpers did edit MuJoCo state and call
  `sim.forward()`, but the cameras were never re-rendered, so the policy received
  a bit-identical observation and reproduced its output exactly. Measured across
  all 160 sidecars and 9,455 states: camera and object families show max
  |Δ activation| = 0 and max |Δ action| = 0, while photometric families change
  normally. Consequently five M0 columns
  (`m0_camera_action_drift_mean`/`_max`, `m0_object_action_drift_mean`/`_max`,
  `m0_camera_render_equivariance_error`) and two M2 columns
  (`m2_object_probe_equivariance_error_mean_rad`, constant at exactly the 15°
  transform magnitude, and `m2_camera_probe_circular_dispersion`) carried no
  information. The affected M2 column is the one aimed most directly at the
  preregistered primary variable, relative planar orientation.
- **Exact change:** Read non-advancing observations with `force_update=True` in
  `RawLiberoEpisode.current_raw_trace` and `counterfactual_observation`. A forced
  read calls `_update_observables(force=True)`, which advances every observable's
  sampling timer by one model timestep *without* advancing the simulation and
  refills `_obs_cache`; replay performs an arbitrary number of such reads per
  scored state, so an unrestored timer would shift the sampling phase of the next
  real `step` and desynchronise replay from the recorded rollout. The forced read
  is therefore wrapped in `_preserved_observable_sampling`, which snapshots and
  restores `_time_since_last_sample`, `_current_delay`, `_current_observed_value`
  and `_sampled` for every observable plus `_obs_cache`, making the read
  observationally pure. Rollout production is provably unaffected: the only
  `_raw_observation` call in `reset()` is unreachable because the protocol
  requires ten settle steps, and every rollout frame comes from `step()`.
  Raw rollouts, reset seeds and failure labels therefore remain valid and are
  **not** regenerated; only the score sidecars and everything derived from them
  are rebuilt.
- **Affected hypotheses/metrics:** No estimand, split, threshold, probe
  definition, feature schema or statistical rule changes. All 160 score sidecars,
  the feature cohort, the fitted predictors, the alarm thresholds and the
  lead-time result are invalidated and must be regenerated. The previously
  reported Calibration values — M2-over-M1 log-loss lift 1.17% and lead-time
  median paired difference 0.0 — are withdrawn as scientific statements: they
  were computed while M2's orientation features were constant, so they were never
  a fair test of internal geometry. `scoring_source_sha256` necessarily changes,
  so the bound probe and allocation receipts must be re-published against the new
  digest before Locked Test.
- **Outcome visibility:** All 160 raw rollouts and outcomes, the defective score
  sidecars, the published features, the M0/M1/M2 Calibration OOF metrics, the
  alarm thresholds, the lead-time null and the causal pair inventory were visible
  before this decision. No Locked Test path, artifact, label or output has ever
  been accessed.
- **Bias risk and mitigation:** The risk is that re-scoring is used to search for
  a more favourable result. It is mitigated by changing nothing except the
  observation refresh: the same manifest, seeds, transforms, probe procedure,
  feature schema, predictor family grid, thresholds and decision rules apply, and
  every preregistered criterion keeps its frozen value. The correction is
  outcome-blind — it was found by inspecting feature degeneracy, not model
  performance — and it removes a defect that suppressed M2, i.e. it works against
  the project's own thesis rather than for it. Regression tests now fail closed on
  both failure modes: one asserts that camera and object counterfactual
  observations differ from the factual one, the other asserts that forced reads
  leave the observable sampling state untouched. Both were verified to fail
  against the defective code and pass against the fix.

  Three further hardening changes land in the same commit, because every one of
  them touches `SCORING_SOURCE_FILES` and adding them later would force a second
  re-scoring run:

  1. **Inertness guard.** `scoring._reject_inert_transforms` rejects any sidecar
     in which a transform family produced actions *and* activations bit-identical
     to the factual rollout across every available state. Only "nothing changed
     at all" proves the policy saw an identical input; requiring both outputs to
     move would be wrong, since a small edit can leave the quantised activation
     untouched while still moving actions. The check is per episode, so a single
     insensitive state cannot trip it. Verified against the existing defective
     sidecars: all are rejected, i.e. this guard would have caught the defect at
     the first episode instead of after a full run.
  2. **Validate before publishing.** `score_replay_to_sidecar` now runs
     `_validate_arrays` on the freshly built arrays before writing. Previously
     validation happened only on load, so a defective sidecar could be published
     and only rejected hours later.
  3. **Content-keyed original activations.** `scoring_runtime` keyed its
     original-activation cache by `id(noise)`, tying correctness to caller-side
     tensor lifetime: a released tensor's id can be reused, and an intervention
     would then be measured against a different draw's original activation with
     no error raised. The cache is now keyed by a SHA-256 of the draw's bytes.
- **Implementing commit:** `773290d32482ca3d18b69a3a0bded4875d14f1fc`

### 2026-08-05 — Remove AMENDMENTS.md from the frozen configuration digest

- **Prior protocol commit:** `266b457bed063e6b093068a61bbdb761078e9b55`
- **Technical reason:** `AMENDMENTS.md` was a member of `FROZEN_CONFIG_FILES`, so
  every appended amendment changed `frozen_config_sha256`. Because
  `validate_bound_probe_artifact` and `audit_score_allocation` compare an
  artifact's recorded `config_sha256` against the digest recomputed at validation
  time, each legitimate amendment retroactively invalidated the bound probe, the
  score sidecars and the allocation receipts. Verified empirically on this tree:
  after the two 2026-08-05 amendments, `load_bound_probe_artifact` raises
  `BoundProbeError: bound probe config/source hashes differ from repository`.
  Since amendments are mandatory whenever the protocol is clarified, this
  guaranteed a fail-closed abort — most damagingly in the middle of Locked Test
  scoring, where it would have invited an ad hoc bypass.
- **Exact change:** Drop `AMENDMENTS.md` from `FROZEN_CONFIG_FILES`. The
  substantive protocol definitions — `start.md`, `PREREG.md`, `environment.lock`
  and the three `configs/*.yaml` files — remain frozen and still invalidate
  artifacts if they change, which is the intended behaviour. Amendment integrity
  is instead carried by git history, by the tracked-file and clean-worktree
  checks in `guard.py`, and by the implementing-commit hash recorded in each
  entry. Verified: appending an entry no longer moves `frozen_config_sha256`.
  Note that removing a member also changes the digest itself (the allowlist
  framing is hashed), so `frozen_config_sha256` moves once with this change and
  is then stable across future amendments.
- **Affected hypotheses/metrics:** None. No estimand, split, feature, probe,
  seed, threshold or statistical rule changes. This only stops a documentation
  file from silently invalidating scientific artifacts.
- **Outcome visibility:** Same as the entry above; no Locked Test access.
- **Bias risk and mitigation:** The risk is weakening provenance. It is bounded:
  the file removed is an append-only record, not a configuration input, and no
  quantity in the analysis reads it. Every file that actually parameterises the
  experiment stays in the digest, and the removal is recorded here and in the
  code comment at the definition site.
- **Implementing commit:** `773290d32482ca3d18b69a3a0bded4875d14f1fc`

### 2026-08-05 — Record the coverage-feature fold coupling as a known limitation

- **Prior protocol commit:** `266b457bed063e6b093068a61bbdb761078e9b55`
- **Technical reason:** `PREREG.md` §7 requires the coverage reference to exclude
  the query's own episode and base-init group, and the implementation does
  exactly that. It does not, however, exclude the *other* groups that share the
  query's cross-validation fold, and the predictor CV is grouped over the same 20
  base-init IDs. On average 20.9% of a row's 25 nearest neighbours fall in a
  given other fold, and for some rows all of them do, so a training row's
  `m1_all_phase_25nn_failure_fraction` — literally the failure rate of its
  neighbours — is computed partly from outcomes in its own validation fold. The
  preregistration did not anticipate this coupling; the implementation is
  faithful to it.
- **Exact change:** None to the feature definition. The coupling is recorded here
  with its direction, and a sensitivity analysis is added to the Calibration
  work: after re-scoring, refit M0/M1/M2 with the three coverage features and
  with `m2_probe_norm_success_same_phase_robust_z_abs` removed, and report the
  change alongside the headline numbers. Changing the feature definition now
  would require fold-dependent feature matrices, a deep restructuring of the
  pipeline at exactly the moment when a re-scoring run must be trustworthy; the
  risk of introducing a new defect outweighs the measured bias.
- **Affected hypotheses/metrics:** The three coverage features are shared
  *identically* by M1 and M2 and are absent from M0, so the preregistered primary
  contrast M2-versus-M1 differences them out; the optimism lands on M0-versus-M1
  and M0-versus-M2. One M2-only feature,
  `m2_probe_norm_success_same_phase_robust_z_abs`, uses the same
  success-conditioned reference and therefore does bias the primary contrast — in
  favour of M2. Locked Test is unaffected, since its reference is fitted on the
  whole Calibration split, which is disjoint from it.
- **Outcome visibility:** Same as the entries above; the biased quantities were
  visible before this decision, which is precisely why the direction of the bias
  is stated explicitly rather than left implicit.
- **Bias risk and mitigation:** The risk is presenting optimistic Calibration
  numbers as clean. It is mitigated by naming the affected comparisons, by noting
  that the residual primary-contrast bias favours M2 — so an M2 null result is
  conservative — and by the preregistered sensitivity analysis above. The primary
  estimand is decided on Locked Test, which carries none of this coupling.
- **Implementing commit:** `773290d32482ca3d18b69a3a0bded4875d14f1fc`

### 2026-08-07 — Move the calibration freeze tag to add Locked Test tooling

- **Prior protocol commit:** `a02b29de9bd255590aed895174c69c03e26bd23d`
- **Technical reason:** `assert_locked_test_ready` requires the
  `calibration-locked-v1` tag to sit exactly at HEAD. Locked Test cannot run
  without a collection runner for `SplitName.LOCKED_TEST`; the existing runners
  are hardcoded to Calibration, and no Locked Test runner was ever written. Any
  commit adding one moves HEAD away from the tag and the guard refuses. The
  freeze therefore has to be re-tagged, or Locked Test can never be executed at
  all.
- **Exact change:** Add `ops/locked_test_cell.py` and
  `ops/locked_test_supervisor.py`, then move the `calibration-locked-v1` tag to
  the commit containing them. Nothing else changes. The new runners are copies
  of the Calibration ones with three differences: the split is
  `SplitName.LOCKED_TEST`, the manifest digest is the Locked Test manifest
  `1fd8c818…`, and the checkout guard is replaced by a real call to
  `assert_locked_test_ready`, which is strictly stronger — it verifies the
  freeze file, the tag, a clean worktree, all nine frozen fields and the byte
  hashes of the four referenced artifacts before a single episode runs.
- **Affected hypotheses/metrics:** None. `locks/calibration_frozen.json` is
  byte-identical across the tag move, as are all four artifacts it references
  (predictor bundle, probe, reality gate lock, Calibration manifest). No
  estimand, split, threshold, probe, predictor, feature, seed or decision rule
  is touched. The frozen values — probe `early_expert_t1_0` at ridge alpha 1.0,
  histogram gradient boosting, alarm thresholds 0.5407/0.4927/0.5040, patch
  strength 0.25, and the Calibration metrics — are unchanged and remain
  verifiable through the guard.
- **Outcome visibility:** All Calibration results were visible before this
  decision, including the failing predictive-lift and lead-time comparisons and
  the causal preview. No Locked Test artifact, label, outcome or path has been
  accessed; the Locked Test raw set does not exist yet.
- **Bias risk and mitigation:** The risk is that a tag move becomes cover for
  changing an analysis decision after seeing Calibration results. Mitigation is
  that the change is fully auditable: `git diff` between the old and new tagged
  commits shows only two added files under `ops/`, the freeze file's digest is
  recorded here on both sides of the move, and the guard independently rehashes
  every referenced artifact at run time. A reviewer can verify in one command
  that no scientific quantity moved. The alternative — keeping the runners
  outside the repository to preserve the tag — would have made the Locked Test
  run unreproducible, which is strictly worse.
- **Implementing commit:** `e766c3abced6958314f49d2bad92a9d8992165f2`

### 2026-08-25 — Approved Locked Test evaluation amendments (toy validation phase)

- **Prior protocol commit:** `0808b23fabf0214b5230e277464dcdc9a576fe13`
- **Technical reason:** Five changes were approved by the study owner after a
  standalone toy stack (`../tiny-vla-interp`, nine toy experiments) was
  validated against the Calibration pattern. The toy's validated claims are
  documented in `../tiny-vla-interp/PARENT_STUDY_HANDOFF.md` (per-expectation
  status lines E2/E6/E8/E9/E10) and `../tiny-vla-interp/EXECUTIVE_SUMMARY_10.md`
  (P1–P3 results, CIs, decision-rule outcomes). Items 1–3 and 5 of this entry
  are approved in full; item 4 is approved strictly as a non-confirmatory
  addendum, executed only after the primary scoring, only on the Locked Test
  failure subset, with a budget of at most 3 GPU hours.
- **Exact change:**
  1. *Rollout diagnostics (secondary metrics).* Report, alongside the
     preregistered metrics, per-episode nearest-object identity accuracy and
     the error/remaining-distance ratio along the rollout, replacing any
     direction-cosine summaries. Applies only to the sensitivity section of
     the evaluation (PREREG §11 order: sensitivity analyses last). Rationale:
     toy Exp-8/8b showed direction cosines misreport healthy controls while
     identity and error/distance stay informative.
  2. *Dose × difficulty sweeps (secondary).* The confirmatory causal claim
     stays at the frozen `patch_strength = 0.25` (PREREG §10). Additionally
     report sign/specificity at the two unused calibration trial strengths
     (0.5, 1.0) crossed with the eight Locked Test condition cells, in the
     sensitivity section only. Rationale: toy Exp-7 showed steering/effect
     curves can change sign across difficulty.
  3. *Broken-successes ledger (secondary).* For the causal patching target
     pairs, report paired rescue/break counts (successes turned failures,
     failures turned successes) alongside target sign and specificity.
     Rationale: toy Exp-3/5/10b; "broken successes" is a first-class
     side-effect metric.
  4. *Waypoint extension (NON-CONFIRMATORY ADDENDUM, post-scoring).* A
     behavior-space internal steering trial on the Locked Test failure subset
     only, α ≈ 0.3 × action scale, holding a perception-time decoded target.
     Explicitly not part of any confirmatory claim; executed only after the
     primary scoring and only within 3 GPU hours; if the budget is exceeded
     the trial stops unfinished and is reported as incomplete. It is an
     exploratory postscript (AMENDMENTS header: exploratory only).
  5. *Data-seed policy (Month-2, no Month-1 effect).* Any future policy-
     training pilot must run ≥3 independent train-stream seeds plus a
     non-specific control band before claiming data-mix gains. Month-1 uses a
     fixed public checkpoint (start.md §4) and is unaffected.
  Implementation commits for the tooling: the Locked Test scoring entry
  (`ops/locked_test_score.py`, byte-faithful paramterization of the frozen
  calibration score path) and the CPU-only bookkeeping gate
  (`ops/prepare_locked_test_artifacts.py`), plus this runbook
  (`ops/locked_test_runbook.md`); the scoring-code tag becomes
  `locked-test-score-v1`.
- **Affected hypotheses/metrics:** None of the primary estimands change: the
  primary paired log-loss and lead-time estimands, the threshold bars, the
  decision table (start.md §12) and the confirmatory patch claim (frozen α)
  are untouched. Items 1–3 extend the mandatory sensitivity section; item 4
  is expressly outside the claim set; item 5 binds only Month-2.
- **Outcome visibility:** All Calibration outcomes were visible (they are the
  reason for this phase). No Locked Test rollout, label, score or pathway has
  been created or inspected; the Locked Test raw set does not exist.
- **Bias risk and mitigation:** The risk is that additional analyses become
  selective. Mitigations: (a) all five items were approved and written before
  any Locked Test rollout; (b) items 1–3 are confined to the last, sensitivity
  stage of the fixed analysis order and are reported in a separate section,
  never side-by-side with confirmatory claims; (c) item 4 cannot affect a
  confirmatory number and carries a hard budget; (d) the tooling is committed
  in the same commit as this entry, tagged, and the guard re-verifies every
  frozen artifact at runtime.
- **Implementing commit:** `886efe4e24964274cb5605aaf432bffd162ce837` (see runbook step 0; the
  commit adding this entry plus the tooling; the tag `calibration-locked-v1`
  is moved to that commit with the freeze payload unchanged, per the
  2026-08-07 precedent).

### 2026-09-01 — Make the approved Locked Test path fail-closed and executable

- **Prior protocol commit:** `50a955c2bf6a2156f04dfcbe9f7275defd9ccf2b`
- **Technical reason:** A prospective, code-only readiness audit after two
  aborted attempts found that the 2026-08-25 tooling could not safely execute
  the already-approved protocol. The collection scripts depended on an
  unstated `PYTHONPATH`, the documented preflight flags did not match the CLI,
  concurrent supervisors were not excluded, and resume/staging checks did not
  fully bind the executable and output paths. More importantly, the scoring
  entry point still called the Calibration feature/fitting path: it required a
  stale probe binding, could interpret the raw root incorrectly, and could
  refit predictors using Locked Test labels instead of applying the frozen
  predictors. There was no complete executable entry point for the fixed §11
  evaluation order. Finally, the Calibration freeze named the predictor pickle
  but omitted the source-bound probe, Calibration-only feature reference and
  predictor metadata needed to apply it independently. Its Brier values were
  averaged over state rows, contrary to §9's rule that each episode contributes
  total sample weight one.
- **Exact change:** Harden collection with self-contained imports, exact
  manifest/authority/repository path binding, an exclusive global lock,
  fail-closed writable/disk/runtime/GPU/offline-snapshot preflight, immutable
  resume validation and preservation of unexplained staging. Replace the
  Locked Test scoring flow with the split-specific feature builder and apply
  only the frozen predictor bundle and Platt calibrators; Locked Test labels may
  be read only for evaluation and are never passed to a fit or selection API.
  Add the deterministic evaluation entry point in the preregistered §11 order,
  excluding failed invalid-reset attempts while enforcing the per-cell 10%
  validity envelope. Extend `locks/calibration_frozen.json` and its guard with
  byte hashes for the bound probe, predictor metadata and both files of the
  full-Calibration reference bundle, and track those reference bytes. Recompute
  Calibration Brier scores with episode-total-one weights: M0
  `0.137469593953084`, M1 `0.06772738580612056`, M2
  `0.0676011578878377`. Restore the Calibration feature-builder source to its
  already-scored digest; Locked Test uses the separate locked builder.
- **Affected hypotheses/metrics:** No hypothesis, model, feature, label,
  threshold, alarm rule, probe, patch strength, decision bar, analysis order or
  Locked Test estimand changes. The only numerical correction is to the three
  descriptive Calibration Brier scores; Calibration log loss/AUROC and every
  frozen selection remain identical. The Locked Test Brier implementation uses
  the same episode-balanced estimand.
- **Outcome visibility:** All Calibration outcomes and the operational blocker
  were visible. No Locked Test rollout directory, success label, score feature,
  prediction or causal output existed or was inspected; only the deterministic
  160-cell manifest and authority bookkeeping existed. These repairs therefore
  precede all protected outcomes.
- **Bias risk and mitigation:** The main risk is that operational repair could
  conceal analytical discretion. Mitigation is structural: frozen inputs are
  expanded rather than replaced, predictor loading verifies both metadata and
  pickle against their committed digests, the scorer has no fitting path, the
  evaluation order/seed/bars are executable constants, invalid cases remain
  public, and direct fail-closed tests exercise tampering, concurrency, resume,
  split binding and label-independent prediction. Both freeze/scoring tags are
  moved only to the final clean, fully tested commit, following the 2026-08-07
  tag-move precedent.
- **Implementing commits:** `6a470be6a51dd573043e3152c0116c910cbe38d5`
  (repair, freeze expansion and evidence gate) and
  `278112a1c5621feb80b916bc658abec3e60d3da0` (clean-clone-discovered scorer
  import bootstrap plus its direct regression test). The following record-only
  commit fills these hashes without changing executable or scientific bytes.

### 2026-09-01 — Resolve unavailable secondary evidence without post-outcome substitution

- **Prior protocol commit:** `f535375a406d898501d5bf1252250d6bdad7742d`
- **Technical reason:** The final prospective capability audit found that three
  approved or preregistered reports cannot be computed from the frozen
  Calibration evidence as written. The selected circular probe decodes relative
  primary-object orientation, not the all-object positions needed by Amendment
  9a. The bound probe contains executable coefficients only for the selected
  architectural location, so the two-of-three-location criterion cannot be
  evaluated without a post-freeze refit. Finally, the ten-action causal
  intervention defines action effects but no closed-loop patched episode outcome,
  so the approved broken-success rescue/break counts have no registered outcome
  semantics. In contrast, the natural selected-representation activations needed
  by the preregistered off-manifold comparison already exist in the completed,
  pre-Locked-Test Calibration score sidecars and can be frozen without fitting or
  outcome-dependent selection.
- **Exact change:** Amendment 9a rollout diagnostics are mandatory-reported as
  `unavailable`, with reason `frozen_position_decoder_and_all_object_trace_absent`;
  no orientation, direction-cosine, primary-object-only or other proxy is allowed.
  Supporting-layer evidence is mandatory-reported as `unavailable`, and therefore
  `layer_support_passes=false` and the overall confirmatory mechanistic claim can
  never be positive in this run. The selected frozen layer is still evaluated in
  full at alpha 0.25, including all three pairing seeds, 1,000 norm-matched random
  subspaces, matched-donor and off-manifold controls; its descriptive result is not
  relabelled as multi-layer evidence. The broken-success ledger is
  mandatory-reported as `unavailable`, with reason
  `patched_closed_loop_outcome_not_defined`; no open-loop action sign or factual
  episode outcome may be substituted for a patched success/failure outcome.

  Before either freeze tag moves and before any Locked Test access, create one
  immutable, content-addressed Calibration activation-reference artifact from the
  already completed Calibration scoring sidecars. It contains the mean across the
  eight registered original noise draws for every valid five-step-cadence state at
  the selected candidate only, plus exact episode/state membership and source-file
  digests. Its builder verifies the bound-probe candidate, cohort membership,
  sidecar/raw links, dimensions and full expected membership; it performs no fit,
  label access, filtering by outcome or model selection. Locked Test off-manifold
  distances and their natural 95th-percentile threshold must use only this frozen
  artifact. All available secondary dose-by-difficulty results remain mandatory.
- **Affected hypotheses/metrics:** The primary paired log-loss estimand, Brier,
  AUROC, M2-vs-M0 comparison, alarm thresholds, lead-time estimands, condition
  rankings, fixed alpha, selected-layer causal estimands and analysis order do not
  change. The overall mechanistic claim is conservatively forced false because its
  preregistered multi-layer condition is unavailable. Three secondary reports gain
  explicit unavailable states rather than invented numerical proxies.
- **Outcome visibility:** Calibration artifacts and the code-only capability
  failures were visible. No Locked Test rollout, success label, score, prediction,
  activation, pairing, intervention or causal result existed or was inspected.
  No additional Calibration rollout or outcome was collected for this amendment.
- **Bias risk and mitigation:** The change follows from missing measurement
  definitions and can only weaken the mechanistic conclusion. Exact unavailable
  reasons are schema-validated, available analyses cannot be omitted, the
  activation reference is a deterministic materialization of already existing
  pre-test bytes, and the final freeze/score tags move only after clean-clone tests
  reproduce its content address and all entry points pass their preflights.
- **Implementing commit:** Pending; this amendment is committed prospectively
  before the affected implementation and artifact bytes.
