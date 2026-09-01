# Research log

This is the canonical, append-only research record. Entries use the same fields so
that experiments, negative results, decisions, and confidence can be audited.

## Entry format

### YYYY-MM-DD HH:MM TZ — ID: concise title

- **Stage:** setup | Discovery | Calibration | Locked Test | analysis | publication
- **Question:**
- **Pre-state / commit:**
- **Method:**
- **Inputs and controls:**
- **Results:**
- **Interpretation:**
- **Confidence:** high | medium | low, with justification
- **Decision:**
- **Next step:**
- **Artifacts:**
- **Compute / cost:**

---

### 2026-08-03 09:20 CEST — SETUP-001: pre-rollout feasibility and architecture audit

- **Stage:** setup
- **Question:** Are the public model, dataset, code, GitHub publishing credentials,
  and rented GPU sufficiently specified to write an executable preregistration?
- **Pre-state / commit:** repository did not yet exist; only `start.md` and
  `AGENTS.md` were present.
- **Method:** Read the frozen project document; inspected Hugging Face model and
  dataset metadata by immutable revision; inspected official LeRobot v0.6.0
  SmolVLA and LIBERO source; checked GitHub CLI authentication; attempted SSH with
  both available ED25519 keys.
- **Inputs and controls:** No policy inference or LIBERO rollout was run. Model Hub
  revision `31d453f7...`; dataset revision `a1aaacb7...`; LeRobot commit
  `30da8e687...`.
- **Results:** The checkpoint exposes a 16-layer VLM plus an action expert. Its
  `num_expert_layers=0` setting means the expert inherits 16 layers, not that the
  expert is absent. The implementation manually invokes layer submodules, so
  residual hooks must use layer-norm inputs. The dataset contains 40 tasks across
  LIBERO Spatial/Object/Goal/Long, including all shortlisted tasks. GitHub account
  `jimfable` is authenticated with repository scope. The GPU TCP/SSH endpoint is
  live and presents host key `SHA256:eD5dhvJUkNqEljtuS2NG1NvnKQD9P0FbE3pRRSiLqGc`,
  but currently rejects both local public keys; no shell was obtained.
- **Interpretation:** The research protocol and instrumentation points can be fixed
  without outcome leakage. GPU authentication is an infrastructure blocker for
  execution, not a scientific reason to change the protocol.
- **Confidence:** high for checkpoint/dataset revisions and architecture because
  they were read from immutable primary sources; medium for runtime compatibility
  until the exact environment is built; high that SSH authentication, rather than
  networking, is the immediate GPU issue.
- **Decision:** Pin LeRobot v0.6.0 and the immutable model revision; capture VLM
  state-token and expert residuals after layers 3 and 11 at fixed flow steps; keep
  simulator truth out of M2 inference features; proceed with preregistration and
  repository publication while retrying GPU access.
- **Next step:** Commit/push the preregistration, resolve the GPU key or recover the
  instance through available Vast metadata, then delegate bounded implementation
  work.
- **Artifacts:** `PREREG.md`, `configs/*.yaml`, `environment.lock`.
- **Compute / cost:** laptop metadata/source inspection only; no GPU seconds and no
  billable policy passes.

### 2026-08-03 10:05 CEST — SETUP-002: recover and verify the pinned GPU runtime

- **Stage:** setup
- **Question:** Can the exact policy/runtime stack be installed and exercised on
  the rented RTX 5090 without silently substituting mutable model inputs?
- **Pre-state / commit:** `247a22c50a649973440f9172c44202d13c27d8fc`;
  Vast SSH rejected the laptop's two initially attempted identities.
- **Method:** Attached the usable local ED25519 public key through the provider UI,
  read the instance operating guide, installed LeRobot from immutable v0.6.0 source
  commit `30da8e687...` into `/venv/main`, installed `hf-libero==0.1.4`, and replaced
  a mismatched CUDA 12.8 torchvision build with the official CUDA 13.0 wheel. Used
  parallel range downloads with post-download SHA-256 verification for the large
  checkpoint and wheels. Staged the policy and base-VLM metadata/tokenizer as exact
  offline Hugging Face snapshots.
- **Inputs and controls:** RTX 5090; Python 3.12.13; PyTorch 2.11.0+cu130;
  torchvision 0.26.0+cu130; Transformers 5.5.4; MuJoCo 3.8.1; robosuite 1.4.0;
  policy revision `31d453f7...`; base VLM revision `7b375e1b...`. No model forward,
  simulator reset, or episode outcome was run or viewed.
- **Results:** CUDA matrix multiplication and CUDA torchvision NMS both passed.
  The complete policy safetensor is 906,712,520 bytes and has SHA-256
  `9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8`.
  Its 500 state-dict keys include the full VLM, so the separate base VLM weights are
  unnecessary: the architecture/tokenizer can be built from the pinned local base
  snapshot and the policy state can then be loaded strictly. Both exact revisions
  resolve in network-free mode from `/workspace/hf-cache`.
- **Interpretation:** The GPU runtime and immutable model inputs are available.
  Strict offline policy construction remains the final pre-inference integration
  check.
- **Confidence:** high; exact revisions, file hashes, CUDA kernels, and local-only
  snapshot resolution were directly verified.
- **Decision:** Use the full policy checkpoint plus config/tokenizer-only pinned base
  snapshot, with `load_vlm_weights=False` before strict full-state policy loading.
- **Next step:** Freeze audited runtime code and run the strict offline model load.
- **Artifacts:** `environment.lock`, `environment-gpu.freeze`, remote
  `/workspace/hf-cache`, `/workspace/install.log`.
- **Compute / cost:** Installation/download time on the rented instance; only short
  CUDA smoke kernels, no billable policy forward passes.

### 2026-08-03 10:19 CEST — SETUP-003: pre-outcome runtime and instrumentation audit

- **Stage:** setup
- **Question:** Does the preregistered design map to causally active SmolVLA sites
  and an exact, terminal-state-preserving LIBERO execution path?
- **Pre-state / commit:** task-ID correction commit
  `24bf90f7ea9ce9a7c0580620623a490fd2dbf288`; no policy inference or simulator
  reset had occurred.
- **Method:** Independently implemented and audited typed frozen configs, manifest
  and split guards, model hooks, snapshot loading, and a raw single-episode LIBERO
  harness. Compared each path with immutable LeRobot/LIBERO source. Tested call
  classification, activation shapes, in-place causal shifts, simulator edits,
  state construction, terminal preservation, and fail-closed guards using synthetic
  unit backends.
- **Inputs and controls:** All 54 unit tests used synthetic tensors/backends. The
  selected semantic tasks, candidate count, flow times, perturbation cells, split
  sizes, hypotheses, and primary metrics were held fixed. No model output,
  simulator observation, success label, or failure rate was available.
- **Results:** Three prospective technical corrections were required and recorded
  in `AMENDMENTS.md`: (1) the originally named final VLM norm is action-inert after
  cache creation, so the fixed VLM candidate moves to the pre-norm state-token
  residual entering VLM layer 12; (2) checkpoint config metadata says six state
  values while its normalizer and LIBERO processor use eight; (3) equal post-edit
  settling and numerical validity/camera/phase rules needed operational definitions.
  Expert patches must mutate the residual tensor in place and name exactly one flow
  step. The raw harness bypasses the LeRobot wrapper's success autoreset and applies
  exactly ten dummy actions after every edit. The local suite passed: 54 tests,
  Ruff lint/format checks, and byte-compilation.
- **Interpretation:** The implementation now targets causal residual streams and
  should preserve both trained inputs and terminal simulator traces. Real runtime
  smoke checks can expose only compatibility bugs, not change the frozen scientific
  selection rules.
- **Confidence:** high for source-level causal accessibility and guard behavior;
  medium for LIBERO backend field compatibility until the first Discovery reset.
- **Decision:** Commit the implementation and amendment trail before strict model
  loading or simulator access. Fail closed on any architecture, shape, task-object,
  snapshot, or split mismatch.
- **Next step:** Record implementing commit hashes, push, sync the ephemeral GPU,
  and run offline strict policy construction followed by one Discovery IID reset.
- **Artifacts:** `src/mech_int_vla/`, `tests/`, `configs/`, `AMENDMENTS.md`.
- **Compute / cost:** laptop-only synthetic tests; no GPU policy passes or rollouts.

### 2026-08-03 10:28 CEST — SETUP-004: concrete-config loader compatibility failure

- **Stage:** setup
- **Question:** Does the committed offline loader construct the pinned checkpoint
  config under LeRobot v0.6.0?
- **Pre-state / commit:** `bab7e6e`; both immutable snapshots resolved offline and
  all 54 tests passed on the rollout host.
- **Method:** Invoked only the `load-policy` construction command with
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; inspected the resulting traceback
  and the immutable LeRobot `PreTrainedConfig.from_pretrained` implementation.
- **Inputs and controls:** Exact frozen GPU environment and snapshots. The failure
  occurred while parsing `config.json`, before model construction, weight loading,
  a forward pass, simulator initialization, or any outcome.
- **Results:** Calling `SmolVLAConfig.from_pretrained` directly causes draccus to
  reject the checkpoint's required `"type": "smolvla"` discriminator as an unknown
  concrete dataclass field. LeRobot's v0.6.0 dispatch contract is to call the
  registered `PreTrainedConfig.from_pretrained`, which uses that discriminator to
  choose `SmolVLAConfig` before parsing the remaining fields.
- **Interpretation:** This is a loader API mismatch in the integration code, not a
  model, data, or scientific-protocol result.
- **Confidence:** high; the traceback occurs deterministically at config decoding and
  the intended dispatch path is explicit in the pinned source.
- **Decision:** Dispatch via `PreTrainedConfig`, fail closed unless the returned type
  is exactly `SmolVLAConfig`, and add a unit-test assertion for that path. No protocol
  amendment is required because no representation, condition, metric, or selection
  rule changes.
- **Next step:** Commit/push the compatibility fix, resync it, and retry the same
  offline strict-load command.
- **Artifacts:** `src/mech_int_vla/snapshots.py`, `tests/test_snapshots.py`.
- **Compute / cost:** approximately one failed Python construction process; zero
  policy forward passes and zero simulator steps.

### 2026-08-03 10:40 CEST — SETUP-005: strict policy load passes; asset bundle blocked

- **Stage:** setup
- **Question:** After correcting registered config dispatch, does the entire pinned
  policy load strictly and can the first real LIBERO Discovery reset initialize?
- **Pre-state / commit:** `5d2911b99e3e937025eb039c425304bf76dfa5de`;
  remote working tree reconstructed exactly and clean from the committed archive.
- **Method:** Re-ran the offline `load-policy` command on CUDA, then initialized
  hf-libero's default noninteractive path config and attempted Discovery task rank 1,
  init state 0, IID condition 0. Inspected only setup exceptions. Queried the public
  LIBERO asset repository metadata and froze its exact current revision before any
  successful simulator construction.
- **Inputs and controls:** Policy `31d453f7...`, base VLM `7b375e1b...`, asset bundle
  `lerobot/libero-assets@0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`.
  No policy forward/action was requested. The simulator attempt failed while loading
  its XML arena, before a task initial state, observation, success value, or outcome
  was available.
- **Results:** Strict loading succeeded from local files. Runtime settings were
  `num_steps=10`, `chunk_size=50`, `n_action_steps=1`, and `empty_cameras=1`; all
  non-count normalization tensors for `observation.state` had shape `(8,)`, while
  the original checkpoint metadata was `(6,)` and corrected runtime metadata `(8,)`.
  The first LIBERO construction could not find
  `assets/scenes/libero_study_base_style.xml`. hf-libero attempted its unpinned
  default Hub download, but the provider's outbound HTTPS timed out, then raised
  `FileNotFoundError`. The public asset snapshot contains 586 repository entries and
  approximately 422 MB according to Hub metadata.
- **Interpretation:** Model/runtime integration is verified through strict weight and
  processor construction. LIBERO needs its separately distributed public assets
  staged offline, just as the model needed offline staging; this is not evidence
  about the task or policy.
- **Confidence:** high for strict load and asset cause because both emitted explicit
  checks/tracebacks; no confidence update about any research hypothesis is possible.
- **Decision:** Pin asset revision `0b3ea86...`, download it on the laptop where Hub
  access is reliable, hash the transfer, and place it at hf-libero's configured asset
  path. Keep the same first Discovery manifest cell.
- **Next step:** Stage the exact asset snapshot, retry task 1/init 0/IID reset, then
  either repair source-level API mismatches prospectively or start the Reality Gate.
- **Artifacts:** `environment.lock`; remote strict-load stdout and asset traceback
  retained in the task transcript pending structured capture.
- **Compute / cost:** one full model construction/weight load, zero forward passes;
  one failed simulator construction, zero simulator control steps.

### 2026-08-03 10:52 CEST — SETUP-006: verify assets, audit identifiers, pause idle GPU

- **Stage:** setup
- **Question:** Are all remaining LIBERO inputs locally verifiable, are selected task
  identifiers executable, and can idle rental cost be reduced without risking data?
- **Pre-state / commit:** `3831416c32d2902ecdf45776153c9330f103e705`;
  instance 46677323 running idle at the dashboard's `$0.344/hr` compute/storage rate.
- **Method:** Downloaded exact asset revision `0b3ea86...` on the laptop; verified
  each ordinary file against its Git blob SHA-1 and each LFS file against its
  SHA-256 OID; created and integrity-tested a zstd transfer archive. Audited the
  three pinned BDDL files/init-state tensors and LeRobot reset behavior without
  constructing a policy or simulator. Before pausing, checked process/GPU tables,
  synchronized disk writes, required a clean remote worktree, and rehashed the full
  checkpoint plus every transferred code/snapshot archive against off-instance
  copies. Used only Vast's Stop action, never Destroy.
- **Inputs and controls:** 586 asset files, 422,320,936 repository bytes; three
  selected BDDL tasks and their 50 pinned init states each. No policy forward,
  successful simulator construction/reset, observation, action, label, or outcome.
- **Results:** Every asset verified with no missing/unexpected files. Transfer archive
  size is 237,978,828 bytes with SHA-256
  `ad0626590c94ca126312ed52728d19d594a1f46e23c4185b6b6958cf349aa940`.
  Tasks 2/5/9 expose 50 finite init states with shapes `(50,45)`, `(50,47)`, and
  `(50,47)`. Source audit found two exact category mismatches: `black_book` and
  `white_yellow_mug` are required. It also found LeRobot increments its init-state
  index after reset, requiring one raw runtime per paired cell. With all relevant
  jobs gone and artifacts secured, Vast confirmed the instance as `Inactive` and
  displayed stopped storage cost `$0.037/hr` / `$0.89/day` versus `$0.344/hr`
  running, a 9.3-fold reduction. Vast warns restart depends on GPU availability.
- **Interpretation:** The assets and identifier corrections can be frozen without
  outcome leakage. Pausing preserves the 19 GB remote environment while avoiding
  idle compute charges; the exact saving is larger than the user's rough one-eighth
  estimate, though storage billing continues.
- **Confidence:** high; content-addressed file checks, immutable source/BDDL data,
  live process inspection, and Vast's own confirmation/dialog supplied the values.
- **Decision:** Commit the exact aliases/single-use invariant and tracked manifests.
  Keep the instance stopped while the transfer path and rollout runner are prepared;
  resume only when a concrete download or experiment is ready.
- **Next step:** Publish the corrected runtime, transfer the verified asset archive
  through a fast CDN path, resume the instance, and repeat the identical first IID
  Discovery reset.
- **Artifacts:** `artifacts/manifests/libero-assets-0b3ea86.*`, local
  `/tmp/libero-assets-0b3ea86.tar.zst`, `AMENDMENTS.md`, `environment.lock`.
- **Compute / cost:** GPU paused after approximately 2 h 26 m instance age; ongoing
  stopped storage rate `$0.037/hr`, no GPU/model/simulator work in this entry.

### 2026-08-03 10:56 CEST — SETUP-007: atomic deterministic rollout executor

- **Stage:** setup
- **Question:** Can one manifested episode be executed and recorded without wrapper
  autoresets, condition reuse, hidden open-loop actions, or partial artifacts?
- **Pre-state / commit:** `3831416c32d2902ecdf45776153c9330f103e705`;
  GPU inactive and no real model/simulator execution during implementation.
- **Method:** Composed the strict policy runtime, single-use raw LIBERO episode, and
  activation instrumentation in a narrow one-episode executor. Built synthetic
  success, horizon-truncation, invalid-reset, exception, protected-path, reuse, and
  contract tests. Reviewed the complete artifact schema and reran repository-wide
  tests/lint/format/compile checks.
- **Inputs and controls:** Synthetic 7-D actions, 8-D states, terminal frames, and
  five four-dimensional mock candidate activations. The executor requires the exact
  manifested policy/task/condition, fresh raw runtime, `n_action_steps=1`, ten
  denoising steps, one prefix plus ten internal calls per action, and unpatched
  activations.
- **Results:** The executor replans at every environment step, captures exactly the
  five frozen candidates, rejects an invalid post-settle reset before inference,
  preserves terminal state, and records action/reward/pose/state/contact/grasp/phase/
  predicate/scalar arrays plus revision/seeds/outcome metadata. Artifacts stage in a
  temporary directory, fsync data and metadata, publish by atomic rename, refuse
  overwrites, and cannot target config/lock paths. All 64 tests and all static checks
  pass.
- **Interpretation:** Discovery rollouts now have a fail-closed, crash-safe execution
  path. The remaining uncertainty is integration with a real processed observation
  and action, which will be checked on the first IID cell after asset staging.
- **Confidence:** high for tested control flow, capture cardinality, terminal and
  atomic-publication invariants; medium for real policy processor compatibility until
  the first action pass.
- **Decision:** Use a fresh runtime and artifact directory for every manifest cell;
  never retry a failed reset in the same wrapper; never overwrite a completed cell.
- **Next step:** Commit/push, stage the verified asset archive, resume the GPU, run the
  same first IID reset, then one complete Discovery episode if reset validity passes.
- **Artifacts:** `src/mech_int_vla/rollout.py`, `tests/test_rollout.py`.
- **Compute / cost:** laptop-only synthetic tests while the GPU remained inactive.

### 2026-08-03 11:14 CEST — SETUP-008: freeze downstream analysis machinery before outcomes

- **Stage:** setup
- **Question:** Can the preregistered probe, failure-predictor, and confirmatory
  evaluation rules be made executable before any real simulator outcome is seen?
- **Pre-state / commit:** `9e12e58cc6b7c606815f6d85f78c8e6f8e105a63`; no
  successful simulator reset or policy action had occurred.
- **Method:** Independently implemented and then centrally reviewed three
  dependency-light analysis modules using synthetic arrays only. The probe module
  uses sorted init-ID group folds, episode-equal weighted centered ridge fits, the
  frozen alpha grid, symmetry-aware circular MAE, and the fixed five-candidate
  one-standard-error preference. The predictor module uses outcome-independent
  init-ID group folds, per-episode total training weight one, weighted-median
  imputation with an indicator for every feature, the exact logistic/HGB candidate
  grids, M1-only raw OOF log-loss selection, model-specific OOF Platt maps, and
  full-Calibration refits. The evaluator implements the paired episode-level
  primary estimand, whole-init cluster percentile bootstrap, secondary metrics,
  alarm calibration, lead time, Wilson intervals, and explicit decision flags.
- **Inputs and controls:** Deterministic synthetic labels, features, activations,
  episode IDs, and base-init groups. No protected artifact path, model checkpoint,
  simulator, real success label, or Locked Test data was read.
- **Results:** The full repository suite passes 96 tests. Ruff lint and formatting,
  byte-compilation, and whitespace checks also pass. Probe artifacts expose
  canonical JSON/SHA-256 metadata; predictor artifacts expose data/pickle hashes,
  folds, preprocessing, coefficients where applicable, OOF metrics, and the
  black-box ceiling flag. Evaluation inputs fail closed on duplicate/unpaired
  episodes, malformed probabilities or cadence, missing failure events, and
  insufficient bootstrap clusters.
- **Interpretation:** Major analytic degrees of freedom are now encoded before
  outcomes: candidate/hyperparameter selection cannot be improvised after observing
  Discovery or Calibration behavior, and the headline Locked Test decision rule is
  an executable paired test rather than a retrospective analysis choice.
- **Confidence:** high for the encoded arithmetic and data-integrity rules because
  edge cases and deterministic replay are covered by unit tests; medium for the
  future feature-assembly boundary until a real rollout artifact exercises it.
- **Decision:** Commit and push these modules before resuming the GPU. Build the
  remaining feature-assembly and causal-pair utilities against these frozen APIs,
  without accessing Locked Test conditions.
- **Next step:** transfer the verified LIBERO assets and current commit when the
  stopped GPU can be scheduled, repeat the identical Discovery reset, and in
  parallel implement artifact-to-analysis assembly and causal matching contracts.
- **Artifacts:** `src/mech_int_vla/probes.py`,
  `src/mech_int_vla/predictors.py`, `src/mech_int_vla/evaluation.py`, their three
  test modules, and the optional `modeling` dependency in `pyproject.toml`.
- **Compute / cost:** laptop-only synthetic tests while the GPU remained
  inactive/unreachable; no model forward passes or simulator steps.

### 2026-08-03 11:40 CEST — SETUP-009: close capture, ingestion, causal, and lock-integrity gaps

- **Stage:** setup
- **Question:** Does the pre-outcome pipeline faithfully preserve the raw inputs,
  five-step scoring cadence, invalid-reset rule, causal controls, and Locked Test
  integrity needed for the registered claims?
- **Pre-state / commit:** `689caf0fd9cecdeb433b3cbda4dfac67cd011967`;
  no successful simulator reset, policy action, or research outcome was available.
- **Method:** Ran an independent line-by-line protocol/code audit, then split the
  fixes across isolated artifact-ingestion, causal-analysis, and lock-integrity
  workstreams. Added lossless uint8 storage for both 360x360 camera streams; limited
  activation hooks to pre-action steps divisible by five; implemented the single
  identical invalid-reset retry in a fresh runtime; and separated end-effector/object
  from object/goal symmetry scalars. Built a safe exact-set artifact loader and
  outcome-independent probe-cohort assembler. Implemented deterministic three-seed
  causal matching, rank-two probe/random projectors, first-ten-action effects, 5-NN
  checks, cluster intervals, and confirmatory decision flags. Hardened snapshot and
  lock guards with streamed byte hashes, tracked regular-file declarations, the
  exact HGB iteration count, the exact conservative alarm sentinel, and an exact
  20-init by 8-cell Locked Test evaluation entry point.
- **Inputs and controls:** Synthetic images, actions, poses, activations, manifests,
  labels, and temporary Git repositories only. Artifact rows are selected by reset
  validity and control-step stride before failure labels are attached. The real
  policy checkpoint bytes were not reloaded and no raw experiment artifact existed.
- **Results:** The executor metadata and trajectory schema now agree on 16 scalar
  features, raw-image provenance, scored control steps, retry provenance, and task
  semantics. The loader rejects unsafe archives, schema/cardinality/type drift,
  mixed task/split/revision/code cohorts, duplicates, and incomplete requested sets;
  it derives the registered relative yaw with the correct XYZW/WXYZ conventions.
  Causal controls enforce rank two, all three fixed alphas, norm matching, 1,000
  random controls, and at least 30 valid pairs. Confirmatory prediction evaluation
  refuses anything other than the exact Locked Test manifest and complete paired
  valid predictions. The full repository passes 153 tests plus Ruff lint/format,
  byte-compilation, and whitespace checks.
- **Interpretation:** The audit found material data-integrity defects that synthetic
  success-path tests had not exposed, but none involved empirical selection. Their
  correction before the first valid reset reduces both silent protocol drift and
  the chance that an incomplete Locked Test could be reported as confirmatory.
- **Confidence:** high for the encoded fail-closed contracts and synthetic replay;
  medium for real memory/runtime cost of lossless camera artifacts and repeated
  instrumentation installation until the first GPU episode exercises them.
- **Decision:** Freeze and publish this complete pre-outcome milestone. Do not start
  feature extraction until the remaining action-standardization and causal
  aggregation choices are written prospectively. Keep the same first manifested
  Discovery cell for the next GPU attempt.
- **Next step:** finish the prospective feature/causal operationalization, push and
  sync the commit, then stage the verified LIBERO asset archive and retry task rank
  1, init 0, IID on the original Vast disk as soon as it is schedulable.
- **Artifacts:** `src/mech_int_vla/{rollout,artifacts,causal,snapshots,guard,evaluation}.py`,
  `src/mech_int_vla/runtime_cli.py`, their regression tests, and this log.
- **Compute / cost:** laptop-only synthetic tests; Vast remained inactive at the
  previously verified `$0.037/hr` storage rate and the SSH endpoint never reached a
  server banner. Zero model forwards and zero simulator control steps.

### 2026-08-03 12:24 CEST — SETUP-010: freeze replay scores, exact features, and failure events

- **Stage:** setup
- **Question:** Can every input to the registered M0/M1/M2 comparison and failure
  lead-time analysis be computed deterministically from validated raw artifacts,
  without leaving post-outcome numerical or annotation choices open?
- **Pre-state / commit:** `c9c0447ce4077f01027e75ba1b6e6d1f01899868`;
  no successful simulator reset, policy action, success label, probe fit, or causal
  result had been observed.
- **Method:** Split the work into independent replay-scoring, feature-mathematics,
  and failure-event implementations, then audited their interfaces centrally.
  Added exact non-advancing camera/object observations with simulator restoration,
  a rank-two minimum-norm circular-probe intervention, and a deterministic replay
  orchestrator with explicit common-noise tensors, queue/RNG nonmutation checks,
  synchronized cost records, atomic content-linked sidecars, and a tamper-safe
  loader. Encoded the full action scale and M0/M1/M2 reductions, out-of-fold
  coverage geometry, circular statistics, probe-norm references, and nested feature
  ordering. Operationalized missed grasp, drop, workspace-exit, and horizon events,
  including exact artifact coverage and the noncircular Discovery bounds freeze.
- **Inputs and controls:** Synthetic uint8 images, simulator snapshots, action
  chunks, activations, poses, contacts, phases, terminal flags, and artifact hashes.
  The scorer reuses the exact first four noise objects across all transforms and
  interventions, restores every temporary edit byte-for-byte, refuses protected
  config/lock destinations and overwrites, and publishes nothing after a replay,
  RNG, queue, or schema mismatch. Coverage excludes the query episode and its full
  base-init group before both fitting and neighbor search. Failure bounds validate
  the complete expected artifact/hash set before reading validity or success.
- **Results:** The scorer stores all raw primitives and explicit availability masks
  in deterministic compressed NumPy archives. The feature layer implements 13 M0
  columns, 27 raw M1 state columns plus three coverage columns, and the frozen VLM
  or expert M2 increments. Relative yaw is derived from the declared relative
  quaternion; numerically zero probe vectors use the prospectively fixed `1e-12`
  floor; mixed intervention availability fails closed. Failure annotation records
  event onset and confirmation, rejects incomplete early failures, excludes failed
  wandering from reachable bounds, and requires every Discovery episode in the
  video audit. The repository passes 207 synthetic tests plus Ruff lint/format,
  byte-compilation, and whitespace checks.
- **Interpretation:** The remaining scoring uncertainty is now concentrated in one
  concrete pinned LeRobot/LIBERO adapter and the first real CUDA execution, rather
  than in the scientific arithmetic. The strict missingness behavior may reduce
  usable M2 features when a counterfactual is invalid, but it cannot silently favor
  the white-box method.
- **Confidence:** high for dependency-light reductions, provenance hashes,
  precedence, thresholds, and failure behavior; medium for private pinned
  `_get_action_chunk` and full-chunk postprocessor compatibility until exercised on
  the exact GPU runtime.
- **Decision:** Commit and push this pre-outcome contract before any GPU resume.
  Keep all four intervention draws mandatory, audit all Discovery videos, and never
  relax replay equality to recover a sidecar.
- **Next step:** finish the pinned runtime adapter against LeRobot commit
  `30da8e687a6dfc617fcd94afc367ac7071c376ce`, push it separately, then resume the
  preserved Vast instance only for asset staging and the first manifested IID
  Reality-Gate reset/action.
- **Artifacts:** `src/mech_int_vla/{scoring,features,failure_events,instrumentation,libero_runtime}.py`,
  their regression tests, `AMENDMENTS.md`, and this log entry.
- **Compute / cost:** laptop-only synthetic tests. Vast remained stopped at the
  verified `$0.037/hr` retained-disk rate versus `$0.344/hr` running (about 9.3x
  lower); zero new model forwards and zero simulator control steps.

### 2026-08-03 12:46 CEST — SETUP-011: bind scoring to the pinned runtime and artifacts

- **Stage:** setup
- **Question:** Can the frozen scorer be connected to the exact SmolVLA/LIBERO
  runtime, fitted-probe artifact, and Calibration/Locked Test feature pipeline
  without allowing compatible source drift or stale provenance links?
- **Pre-state / commit:** `5500fcd569f81ea8b6fa7a5825e0433c86d2fbb7`;
  the dependency-light scorer and feature arithmetic were frozen, but the concrete
  private-policy bridge, probe persistence, and artifact-to-feature integration
  were not yet committed. No valid reset, action, label, probe fit, or intervention
  result had been observed.
- **Method:** Implemented and independently audited the concrete replay adapter
  against LeRobot commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`.
  It creates explicit local-generator float32 noise of shape
  `(1,50,max_action_dim)`, invokes only `_get_action_chunk`, verifies the exact
  prefix-plus-ten-denoising call sequence, captures only the selected candidate,
  preserves the first factual activation for every reused noise object, applies
  rank-two minimum-norm circular shifts, postprocesses all 50 actions, and records
  synchronized per-call costs. Added canonical content-addressed `probe.json`
  publication/loading and strict raw/score pairing into immutable full-Calibration
  reference bundles plus deterministic M0/M1/M2 feature cohorts. The audit then
  added three fail-closed guards: the complete 487-file LeRobot Python tree hash,
  every frozen counterfactual-validity field, and raw/probe hash linkage before a
  sidecar can be published.
- **Inputs and controls:** Exact pinned source tree from the detached v0.6 commit;
  synthetic validated rollouts, sidecars, probes, simulator snapshots, action
  queues, RNG streams, activations, transforms, and provenance hashes. The source
  fingerprint uses path- and length-framed Python bytes and equals
  `79603648ff8d9889072449099da6e60b6a92fe0da84108e2bae1dc765b217ecd`.
  Probe loading rejects pickle, duplicate/unknown JSON keys, noncanonical bytes,
  symlinks, overwrites, and digest mismatches. Calibration coverage and robust probe
  norms exclude the query episode and base-init group; Locked Test uses only the
  supplied frozen Calibration references.
- **Results:** The private bridge reaches the atomic sidecar writer end-to-end in a
  dependency-light fake and refuses RTC/eval/shape/call-phase/queue/RNG/source/
  validity/probe/link drift. Feature integration validates all array names, dtypes,
  masks, exact score seeds, costs, task/cohort identities, outcomes, cadence, and
  one-to-one content hashes before reduction. The repository passes 248 synthetic
  tests plus Ruff lint/format, byte-compilation, and whitespace checks. Commit
  `882d753f83e930361e71e6e51ce63e633d667355` is pushed publicly.
- **Interpretation:** The remaining uncertainty is empirical/runtime reality rather
  than an unbound software contract. The real RTX 5090 must still establish pixel-
  exact LIBERO replay, actual hidden widths/hook phases, CUDA cost readings, and a
  successful first reset/action. No scientific result can yet be inferred.
- **Confidence:** high for local provenance, persistence, validation, and numerical
  integration; medium for the private pinned policy path until the first real
  CUDA call; zero update to the research hypotheses.
- **Decision:** Keep the same first manifested Discovery cell and all frozen
  thresholds. Do not weaken source, validity, replay, or hash checks to obtain a
  sidecar. Continue preparing persistence/orchestration locally while the retained
  GPU cannot be safely resumed.
- **Next step:** Resume instance 46677323 only through an authenticated Vast session,
  verify the preserved disk, transfer and rehash the already verified asset archive,
  then repeat task rank 1/init 0/IID reset and rollout. In parallel, add atomic
  feature-reference persistence and allocation-complete CLI orchestration without
  opening Locked Test.
- **Artifacts:** `src/mech_int_vla/{scoring_runtime,feature_pipeline,probes,scoring,snapshots}.py`,
  `environment.lock`, their regression tests, `AMENDMENTS.md`, and this log entry.
- **Compute / cost:** laptop-only synthetic tests. Both available Vast console paths
  were visibly signed out and no local Vast API configuration was present; the
  direct and proxy SSH endpoints still rejected access as expected for the stopped
  instance. No credentials were inspected or transmitted, no resume/stop/delete
  action occurred, and zero policy forwards or simulator steps ran. The last live-
  verified retained-disk price remains `$0.037/hr`; no current authenticated billing
  value was available in this run.

### 2026-08-03 13:14 CEST — SETUP-012: complete laptop reset preflight and harden persisted provenance

- **Stage:** setup / simulator preflight
- **Question:** Can the complete first-task Discovery reset allocation be
  constructed with the pinned LIBERO inputs off-GPU, and can the remaining
  score/feature/failure artifacts be persisted without caller-invented hashes or
  cross-split provenance contradictions?
- **Pre-state / commit:** `703e96b`; the asset-free simulator attempt had failed
  before initial-state loading, and no successful reset, policy action, outcome,
  probe fit, score, or intervention had been observed.
- **Method:** Built a disposable Python 3.12 laptop environment with
  `hf-libero==0.1.4` and the exact LeRobot commit. The first build exposed a CMake
  policy incompatibility in `egl-probe`; setting the documented minimum-policy
  compatibility allowed the free dependency build to finish. Reverified the local
  asset transfer archive SHA-256
  `ad0626590c94ca126312ed52728d19d594a1f46e23c4185b6b6958cf349aa940`,
  checked its paths/types, and installed its 586 files only into the disposable
  environment. Ran every task-rank-1 Discovery init/condition reset in a fresh
  runtime with macOS GLFW rendering. In parallel, independently implemented and
  reviewed canonical persistence for full-Calibration feature references/cohorts
  and Failure Event freezes/records. Added explicit path-and-length-framed hashes
  over the frozen protocol files and scoring/feature source allowlist, then made the
  real scoring adapter recompute and enforce them. Audits fixed canonical NaN
  hashing, compressed-member size bounds, mandatory per-event freeze membership,
  strict native freeze types, post-read file/layout checks, large-number handling,
  and the formerly impossible Calibration-vs-Test commit comparison.
- **Inputs and controls:** Pinned, free hf-libero/LeRobot packages; the verified
  asset archive; committed task/config/manifest generation; fresh single-use
  simulator instances; no checkpoint inference. The laptop used macOS/GLFW and is
  therefore explicitly not a substitute for the frozen Linux/EGL RTX 5090 runtime.
  The 40 reset cells used only their preregistered seeds and hash-balanced yaw
  assignments. Persistence reviews used synthetic validated artifacts and hostile
  schema/path/NaN/ZIP cases.
- **Results:** All 40/40 first-task Discovery reset cells returned exit code zero,
  settled for exactly ten no-op steps, produced an 8-value policy state, remained
  in phase `pregrasp`, and passed every frozen reset-validity check. All reported
  initial-success flags were false. No policy action was requested. The repository
  passes 282 synthetic/integration contract tests plus Ruff lint/format,
  byte-compilation, and whitespace checks. Feature and failure artifacts round-trip
  through content-addressed no-pickle directories; score publication now rejects
  stale repository configuration or computation-source links. Locked Test feature
  construction allows its necessarily later collection commit but still requires
  identical policy, base VLM, configuration hash, and scoring/feature source hash.
- **Interpretation:** Simulator construction, asset completeness, task/object
  resolution, condition application, settling, and reset validity now have positive
  cross-platform evidence across the full 40-cell allocation. This materially
  lowers—but does not eliminate—the GPU runtime risk. It says nothing about policy
  success, perturbation failure range, internal geometry, prediction, or causality.
- **Confidence:** high that the laptop reset-only allocation is internally valid
  because all cells used the executable manifest and fail-closed checks; medium that
  Linux/EGL will behave identically until the preserved GPU is resumed; zero update
  to any research hypothesis or behavioral gate.
- **Decision:** Preserve all registered thresholds and the selected first task.
  Treat the laptop run only as setup evidence. Continue building exact allocation
  receipts and Reality-Gate orchestration before any confirmatory rollout. Keep the
  Vast instance stopped while authentication is unavailable and no GPU job can run.
- **Next step:** Finish allocation-complete raw/score receipts and pure Reality-Gate
  evaluation, commit/push them, then resume the retained instance only through an
  authenticated session. On resume, reverify the preserved disk and asset manifest,
  run the same reset under Linux/EGL, and isolate the first action rollout as a smoke
  artifact until orchestration is frozen.
- **Artifacts:** `src/mech_int_vla/{failure_artifacts,feature_artifacts,provenance}.py`,
  hardened `failure_events.py`, `feature_pipeline.py`, and `scoring_runtime.py`,
  their regression tests, this log, and the updated status/ignore rules.
- **Compute / cost:** laptop-only CPU/simulator work. The GPU was never resumed,
  stopped, deleted, or otherwise mutated in this run; no new authenticated price was
  available. The last live-verified Vast prices remain `$0.037/hr` stopped storage
  and `$0.344/hr` running (about 9.3x lower while stopped).

### 2026-08-03 15:42 CEST — SETUP-013: establish Linux/EGL/CUDA and publish the first Discovery episode

- **Stage:** setup / Discovery Reality Gate
- **Question:** Does the exact committed SmolVLA/LIBERO stack execute the first
  manifested rank-1 IID cell on the retained RTX 5090 under Linux/EGL, including
  the frozen activation hooks and atomic raw-artifact writer?
- **Pre-state / commit:** `b491dc76641efe3a5c5d7eef6bb87af13d85f10b`;
  the GPU disk, checkpoint cache, Python environment, and exact source snapshots
  were retained, but no Linux/EGL reset, CUDA policy action, or registered rollout
  artifact had succeeded. The local worktree contained later uncommitted audit
  hardening and was deliberately not synchronized to the execution checkout.
- **Method:** Reauthenticated to Vast instance `46677323` through the provider proxy,
  verified the 200 GB disk and idle RTX 5090 read-only, transferred a task-specific
  LIBERO asset archive in individually hashed chunks, and reassembled it only after
  its SHA-256 matched
  `4fa1545f4022341fd76f8f88ce8c9380f4b3f69d9183ed315df8861cd1559195`.
  Preserved but renamed 83 AppleDouble metadata files out of Linux runtime globs.
  Two indirect arena-style dependencies omitted by the initial XML closure were
  recovered from the fully verified `lerobot/libero-assets` snapshot:
  `light-gray-floor-tile.png` (`a2aae4ba...acbc5`) and
  `light-gray-plaster.png` (`d0bdaf13...fe191`). Ran the exact reset, then a
  standalone one-action diagnostic that never invoked an artifact writer, and only
  after both passed launched the preregistered cell through Supervisor. Loaded the
  published directory with the committed fail-closed artifact loader and compared
  its full episode provenance to the regenerated manifest entry.
- **Inputs and controls:** policy revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`; base VLM revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`; exact 487-file LeRobot source hash
  `79603648ff8d9889072449099da6e60b6a92fe0da84108e2bae1dc765b217ecd`;
  rank-1 task 5, init 0, IID, reset seed 101000; Linux, `MUJOCO_GL=egl`, CUDA 13.0,
  NVIDIA GeForce RTX 5090 compute capability 12.0. Hub access was forced offline,
  the execution worktree was clean, and the diagnostic advanced exactly one action
  without writing a research artifact.
- **Results:** The exact reset settled for ten no-op steps, produced an 8-value
  policy state, and passed all validity checks with no initial success. The CUDA
  diagnostic emitted a finite 7-D action (SHA-256 `16b962c8...aea7caf`), made the
  required one prefix plus ten denoising calls, and captured all five finite frozen
  candidates at widths 720 or 960; peak allocated GPU memory was 1.271 GB. It then
  closed the simulator and removed instrumentation with no artifact. The registered
  episode `libero_10-task5-discovery-init00-cell0` succeeded after 164 control
  steps and terminated without truncation. Independent loading validated actions
  `(164, 7)`, both lossless camera arrays `(165, 360, 360, 3)`, the complete array
  schema, exact manifest metadata, no staging residue, metadata SHA-256
  `ff2f145576518a68f6efaad48d6b5a9e159859b74bf3e6ad2de26b31de6598d3`,
  and trajectory SHA-256
  `9e97ae27d8ec2902d07acc21a476bb28834cb58c66684ddccb5f53cd329a6a7a`
  over 42,463,750 bytes.
- **Interpretation:** This is the first positive end-to-end evidence for the exact
  GPU runtime, activation instrumentation, closed-loop policy, simulator, terminal
  handling, and atomic artifact schema. It is only one of ten IID reproduction
  cells and therefore cannot establish the six-of-ten reproduction gate, dynamic
  failure range, predictive advantage, internal geometry, or causality.
- **Confidence:** high for the recorded cell because the runtime and inputs were
  content-pinned, the smoke was separated from the writer, Supervisor reported
  expected exit status zero, and the destination was independently loaded and
  hashed; low for rank-1 gate passage until the remaining IID cells finish.
- **Decision:** Preserve the successful artifact immutably and run only the nine
  remaining rank-1 IID cells under a single external lock. Keep all 30 yaw cells
  closed until ten validated IID artifacts contain at least six successes.
- **Next step:** finish and independently validate the exact ten-IID set, evaluate
  the preregistered reproduction gate, and only on passage execute the 30 assigned
  yaw cells. Build an external per-file inventory and checksum-verified laptop
  backup before any instance stop.
- **Artifacts:** GPU
  `/workspace/research-artifacts/raw/discovery/libero_10-task5-discovery-init00-cell0/{metadata.json,trajectory.npz}`;
  GPU logs under `/workspace/run-logs`; retained exact asset archive
  `/workspace/libero-task5-assets.tar.zst.complete`; this log entry.
- **Compute / cost:** one exact reset, one excluded CUDA action, and one registered
  164-step rollout on the RTX 5090. Authenticated Vast pricing was reverified at
  `$0.344/hr` running versus `$0.037/hr` retained storage (about 9.3x lower stopped).
  The user added `$20` credit during execution. The instance was not stopped because
  the remaining IID batch began immediately, and it was never deleted, destroyed,
  recycled, or terminated.

### 2026-08-03 16:03 CEST — DISCOVERY-001: pass the rank-1 IID reproduction gate

- **Stage:** Discovery / Reality Gate reproduction phase
- **Question:** Does the frozen rank-1 policy succeed on at least six of the ten
  preregistered IID initial states before any yaw perturbation is executed?
- **Pre-state / commit:** execution remained fixed at
  `b491dc76641efe3a5c5d7eef6bb87af13d85f10b`; init 0 had already produced one
  independently validated success, while the other nine IID cells had never been
  executed on the registered GPU runtime.
- **Method:** Ran init IDs 1 through 9 sequentially as one-process-per-cell
  Supervisor children under an external file lock. Before resuming, the batch
  loaded and provenance-matched init 0. After every child exited, it used the
  committed artifact loader to validate the exact manifested episode metadata,
  full array schema, terminal state, and both file hashes before launching the next
  init. It then reloaded the exact ten-artifact set and computed the frozen
  six-of-ten gate; no perturbation cell was launched by this batch.
- **Inputs and controls:** rank-1 LIBERO task 5; condition index 0 (IID); base init
  IDs 0--9 in ascending order; policy revision
  `31d453f7edd78c839a8bbc39744a292686daf0de`; manifest SHA-256
  `24b5849a364b0798a66c6280cb3379de885a4247c519cf9625c05173a8af1dae`;
  exact Linux/EGL/CUDA environment from SETUP-013. Existing destinations were never
  overwritten, invalid resets would have retained the one identical retry receipt,
  and the batch halted on any child or loader failure.
- **Results:** All 10/10 IID resets were valid and all ten artifact directories
  loaded successfully with no staging residue. Eight succeeded: init 0 at 164
  steps, init 2 at 171, init 3 at 190, init 4 at 164, init 6 at 158, init 7 at 159,
  init 8 at 173, and init 9 at 163. Init 1 and init 5 remained unsuccessful through
  the exact 520-step horizon and were recorded as truncations. The corresponding
  trajectory SHA-256 prefixes in init order are `9e97ae27`, `96168b03`, `261b10d5`,
  `4dfa26fc`, `f406439c`, `2c65df3d`, `697c55be`, `fd882f99`, `35bef348`, and
  `a3047c37`. Supervisor exited with expected status zero. The complete IID raw set
  occupies approximately 578 MiB on the retained GPU disk.
- **Interpretation:** The 8/10 success rate passes the preregistered behavioral
  reproduction requirement and therefore authorizes the rank-1 yaw reality-gate
  phase. This result establishes baseline task competence but does not by itself
  establish a usable perturbation failure range or any mechanistic hypothesis.
- **Confidence:** high for the gate decision because the complete exact set was
  executed in manifest order and each immutable artifact was independently loaded;
  no missing, invalid, duplicate, mixed-commit, or extra IID cell entered the count.
- **Decision:** Keep rank 1 and open exactly its 30 hash-balanced yaw cells in
  manifested init-major order. Do not advance to ranks 2 or 3 unless rank 1 later
  fails the frozen perturbation validity/dynamic-range gate.
- **Next step:** execute and validate all 30 yaw cells, then require at least 27
  valid perturbations and an empirical failure rate between 0.20 and 0.80 inclusive.
  Generate a canonical per-file inventory and checksum-verified off-instance backup
  before any Vast stop.
- **Artifacts:** ten directories under GPU
  `/workspace/research-artifacts/raw/discovery/`; per-cell logs plus
  `/workspace/run-logs/rank1-iid-batch.log`; this log entry.
- **Compute / cost:** nine additional registered GPU rollouts totaling 2,218 control
  steps (six early successes and two 520-step truncations, plus the already logged
  init-0 success makes ten IID cells overall). The yaw batch began immediately after
  the validated gate, so the instance was neither idle nor stopped and was never
  deleted, destroyed, recycled, or terminated.

### 2026-08-03 17:22 CEST — DISCOVERY-002: complete the rank-1 yaw Reality Gate

- **Stage:** Discovery / Reality Gate perturbation phase
- **Question:** Does the frozen rank-1 policy produce a preregistered dynamic
  failure range under all 30 manifested yaw perturbations after the IID gate has
  passed?
- **Pre-state / commit:** execution remained pinned to
  `b491dc76641efe3a5c5d7eef6bb87af13d85f10b`; the ten IID cells had passed with
  eight successes. The authoritative Supervisor had already started the yaw batch
  in the frozen init-major order; no cell was restarted, duplicated, reordered, or
  overwritten during this continuation.
- **Method:** Reconnected through the verified Vast proxy and observed the existing
  Supervisor and its sole child process until it exited. The Supervisor revalidated
  the IID set, then ran the exact 30 yaw cells (condition indices 1--3) one
  process per cell under the existing reality-gate lock. After exit, an independent
  remote invocation of the committed fail-closed loader walked all raw discovery
  directories and validated metadata, provenance, array schema, terminal handling,
  and NumPy safety for every directory.
- **Inputs and controls:** rank-1 LIBERO task 5; yaw conditions from the frozen
  manifest; manifest SHA-256
  `24b5849a364b0798a66c6280cb3379de885a4247c519cf9625c05173a8af1dae`; exact
  Linux/EGL/CUDA environment and policy/base-VLM revisions from SETUP-013; no
  Locked Test access or later-stage calibration was opened.
- **Results:** The Supervisor exited with its authoritative summary
  `manifested_perturbations=30`, `valid_perturbations=30`,
  `invalid_perturbations=0`, `successes_among_valid=19`,
  `failures_among_valid=11`, `failure_rate=0.36666666666666664`,
  `validity_pass=true`, and `dynamic_range_pass=true`. An independent loader then
  validated all 40 exact Discovery directories (ten IID plus 30 yaw), found 40
  unique expected episode IDs, all 40 valid resets, and 27 total successes.
  No staging directory or active rollout process remained after Supervisor exit.
- **Interpretation:** Rank 1 passes the preregistered perturbation Reality Gate:
  the full 30-cell set is valid and its empirical failure rate lies inside the
  frozen inclusive interval [0.20, 0.80]. The result is a gate decision only; it
  does not establish the mechanistic hypothesis, predictive advantage, internal
  representation, or causal intervention effect.
- **Confidence:** high for completion and gate status because the authoritative
  receipt, clean Supervisor exit, exact ordered cell IDs, and independent loader
  agree. Confidence in any later scientific claim remains unchanged until the
  preregistered calibration and locked-test protocol is run.
- **Decision:** Preserve the complete raw set and receipt logs, create two
  independent read-only inventories, and perform a checksum-verified off-instance
  backup before considering a Vast stop. Do not launch any rank-2/3 or later-stage
  rollout in this continuation, and do not touch the retained instance until the
  backup audit is complete.
- **Next step:** return-copy the raw Discovery tree through the provider transfer
  endpoint into a durable local incomplete staging tree, independently compare all
  file paths/sizes/hashes, then commit/push this checkpoint and stop the instance
  only if no concrete GPU or transfer job remains and the backup is exact.
- **Artifacts:** GPU raw tree
  `/workspace/research-artifacts/raw/discovery/`; authoritative log
  `/workspace/run-logs/rank1-yaw-batch.log`; per-cell logs under
  `/workspace/run-logs/`; this log entry.
- **Compute / cost:** 30 registered yaw rollouts on the RTX 5090, including 19
  successes and 11 valid 520-step truncations or other failures as recorded by the
  immutable receipts. The instance remained running throughout execution; it was
  never deleted, destroyed, recycled, terminated, or otherwise mutated.

### 2026-08-03 17:44 CEST — BACKUP-001: return and verify the complete Discovery set

- **Stage:** artifact preservation / post-Reality-Gate audit
- **Question:** Can the complete canonical Discovery raw set and its irreplaceable
  receipts be preserved off-instance with an independently verifiable inventory
  before the GPU is stopped?
- **Pre-state:** the yaw Supervisor had exited cleanly and no rollout writer
  remained. The remote raw tree contained 40 immutable episode directories (2.8 GiB
  on disk; 2,921,163,613 raw bytes), with 179 GiB free on the retained 200 GiB
  disk. The laptop had 18 GiB free.
- **Method:** Generated two independent SHA-256 inventories after quiescence over
  the raw Discovery tree, run logs, run-state scripts, and exact asset closure.
  The canonical scope contained 126 files and 2,943,020,773 bytes; both inventories
  were byte-identical with SHA-256
  `163b09affb25bcb5a8d5a3a6a54dda1c497c696228f0d729d3dbf25a85dc8abd`. The
  provider return-copy endpoint could not be used from this laptop (transient
  console DNS and then 404/auth responses), so a verified SSH proxy fallback was
  used. Four persistent, host-key-verified SSH control connections streamed
  disjoint init partitions in parallel into a local `.incomplete` tree; no two
  streams targeted the same file and no remote source was changed.
- **Results:** All four streams exited zero. The local stage contains all 40
  artifact directories, 80 raw files, run logs, run-state, the 21 MiB asset
  closure, and both inventory copies. Hashing every canonical local file against
  the remote inventory found zero mismatches over all 126 files and
  2,943,020,773 bytes. A fresh local fail-closed loader independently loaded all
  40 copied artifacts: 40 unique IDs, 40 valid resets, 27 successes; first and
  last metadata/trajectory hashes match the remote receipts. The stage was then
  atomically renamed to `artifacts/raw-backup-ready/`; it is ignored by Git and
  remains available as the durable off-instance copy.
- **Interpretation:** The raw Discovery set and supporting receipts are now
  recoverable without the Vast disk. This satisfies the preservation precondition
  for an idle-instance stop; it does not authorize any Locked Test access or later
  protocol step by itself.
- **Confidence:** high for byte-level preservation and loader validity because the
  inventory was independently recomputed, the transfer used disjoint streams,
  every file was rehashed locally, and every artifact was loaded locally.
- **Decision:** Commit/push the completed yaw and backup receipts, perform one last
  read-only GPU/process check, then stop instance 46677323 only if it remains free
  of concrete jobs. Never delete, recycle, terminate, or overwrite the instance.
- **Artifacts:** durable local backup
  `artifacts/raw-backup-ready/`; canonical inventory copies within that directory;
  remote inventory files under `/workspace/runstate/`; this log entry.
- **Compute / cost:** no GPU work or rollout was launched during backup. The Vast
  instance stayed running while the transfer and verification were active and is
  eligible for the retained-storage rate only after the explicit stop check.

### 2026-08-03 17:47 CEST — COST-001: stop the idle Vast instance safely

- **Stage:** post-backup cost control
- **Pre-stop guards:** the authoritative yaw Supervisor was `EXITED`; no
  `runtime_cli` writer, rollout process, or GPU compute application remained; all
  four local transfer streams had exited zero; the 126-file inventory and local
  artifact-loader audit had passed; and `artifacts/raw-backup-ready/` was present.
  No Locked Test data was accessed and no remote artifact was deleted or changed.
- **Action:** issued only the reversible Vast API `stop_instance(46677323)` call.
  The subsequent read-only API state converged to `actual_status=exited`,
  `cur_state=stopped`, `intended_status=stopped`, and `next_state=stopped`.
  No destroy, delete, recycle, terminate, or reboot operation was issued.
- **Cost evidence:** the stopped instance reports `storage_total_cost`
  `$0.0370370370/hr`; the same instance's running `dph_total` is `$0.3437037037/hr`.
  Thus retained storage is about 9.28x cheaper than running, consistent with the
  previously verified rate and not an assumed factor.
- **Decision:** leave the preserved disk stopped while no concrete GPU job is
  authorized. Resume only after a future explicit calibration plan and the same
  read-only disk/asset checks; never recycle the instance.

### 2026-08-03 17:52 CEST — LOCK-AUDIT-001: defer the Calibration lock safely

- **Stage:** post-Reality-Gate protocol/guard audit
- **Question:** Is the repository ready to create the immutable
  `prereg-locked-v1` tag and open Calibration without changing the frozen
  protocol or silently weakening a guard?
- **Evidence:** A bounded read-only review checked the preregistration, guard
  implementation, current Git state, and the completed/verified Discovery backup.
  The result and backup commits are pushed, but the worktree still contains later
  uncommitted hardening; no tracked `locks/reality_gate_frozen.json` exists and no
  `prereg-locked-v1` tag exists. The Calibration guard requires a tracked freeze
  file, the tag exactly at `HEAD`, and a clean worktree. The existing guard payload
  shape also differs from the richer post-gate receipt (it expects a structured
  `selected_variable` and top-level policy revision), so a hand-written lock would
  not be a safe substitute for an integration-tested receipt.
- **Read-only probe:** A local attempt to derive orientation eligibility from the
  exact 40 backed-up `b491dc7` artifacts did not produce a decision; the
  uncommitted helper rejected its own constructed result because its factory-token
  wiring is incomplete. No eligibility value, variable switch, freeze file, tag,
  remote job, or artifact was created or changed by this probe.
- **Interpretation:** The Reality Gate behavioral result is complete, but the
  protocol lock is not yet auditable. The calibrated run must use a new clean lock
  commit (while retaining `b491dc7` as immutable Discovery provenance), and the
  exact weighting/cadence choice for orientation eligibility must be recorded
  before its value is used. This is an implementation/guard blocker, not a failed
  scientific gate.
- **Decision:** Do not tag, do not launch Calibration, and do not access Locked
  Test. Keep the Vast instance stopped. Finish and test the receipt-to-guard and
  failure-event freeze integration, commit the hardening separately with explicit
  outcome-visibility disclosure, then create `prereg-locked-v1` exactly at the
  clean lock commit only after the protocol guard passes.
- **Status:** the durable Discovery backup remains verified at
  `artifacts/raw-backup-ready/`; GitHub remains at the result/stop checkpoint;
  instance 46677323 is stopped at the retained-storage rate. No external blocker
  remains for the already completed Reality Gate, but Calibration is intentionally
  deferred pending this local guard work and authorization boundary.

### 2026-08-03 18:00 CEST — ORIENTATION-AUDIT-001: materialize Discovery-only lock evidence read-only

- **Stage:** post-Discovery receipt derivation; no protected split access
- **Inputs / integrity:** Re-loaded all 40 exact selected-task artifacts from
  `artifacts/raw-backup-ready/discovery/` with the fail-closed loader, regenerated
  the Discovery manifest at immutable execution commit `b491dc7`, and re-evaluated
  the point-estimate Reality Gate. No remote file, GPU process, or raw backup file
  was changed. The re-evaluated gate selected rank-1 `libero_10` task 5 and
  reproduced the previously recorded pass.
- **Orientation evidence:** Applied the explicit local helper contract of equal
  weight per recorded control state and every integer control step including the
  terminal frame. Quaternion extraction unit tests (identity, +/-90 degrees, and
  180 degrees) passed. The exact 40 artifacts contributed 11,822 states; 11,822
  were finite (fraction 1.0). Symmetry-aware resultant length was
  `0.6027090248077962`; physical circular SD was `0.5031504470045496` radians
  (`28.8283970735` degrees), above the fixed 15-degree criterion. State evidence
  hash: `1aa4dc5a6f6e33bd02f79468664948c3ffbdc796821fc4d50addea583d985a15`;
  candidate eligibility hash: `3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb`.
  The receipt-to-finalizer path now completes locally after correcting the missing
  trusted factory-token wiring; candidate lock-receipt hash is
  `17f033b935ea3f600373b5953cdc5bad5c0fd9dfd3dc1260d022acf3355f36f4`.
- **Failure-event evidence:** Applying the already amended deterministic rules to
  the same exact artifacts yielded 27 successful episodes with no event and 13
  annotated failures: 10 `terminal_horizon` and 3
  `irrecoverable_workspace_exit`; no invalid-reset or early-terminal cases. Raw
  object-center bounds were `[-0.4684745740206647, -0.19125620822373215,
  0.8829480023495043]` to `[-0.16332531682315798, 0.17258802482046695,
  1.1878611294197472]`; the frozen 5-cm expansion is
  `[-0.5184745740206648, -0.24125620822373217, 0.8329480023495043]` to
  `[-0.11332531682315798, 0.22258802482046697, 1.2378611294197472]`.
  A candidate freeze was not published because its implementation commit must be
  the eventual clean lock commit.
- **Interpretation / guard boundary:** The materialized values are Discovery
  evidence, not Calibration outcomes. The weighting/cadence choice is present in
  the uncommitted helper but was not explicitly recorded in `PREREG.md` or
  `AMENDMENTS.md` before outcome visibility, and the existing Calibration guard
  still does not validate the full Reality-Gate receipt, orientation evidence, and
  failure freeze as one round-trip. Therefore no freeze file, tag, Calibration
  manifest, GPU job, or Locked Test access was created. This entry is an explicit
  outcome-visibility disclosure and does not authorize a protocol lock.
- **Decision:** Retain the verified backup and stopped instance. Treat the local
  hardening and receipt-to-guard integration as unfinished; do not launch
  Calibration until a clean, tested lock commit and exact guard round-trip exist.

### 2026-08-03 18:35 CEST — LOCK-HARDENING-001: close the receipt-to-guard boundary

- **Stage:** post-Discovery implementation hardening; no protected split access
- **Reason:** The previous guard could validate hashes and a few semantic fields,
  but a caller could still fabricate a self-consistent Reality-Gate JSON payload.
  The orientation helper also accepted caller-supplied state values while marking
  them as rollout-derived.
- **Changes:** Added a raw-`RolloutArtifact` orientation constructor that extracts
  the stored EEF/object quaternions, enforces the complete frame cadence, runs
  identity/±90-degree/180-degree extraction checks, and retains the fixed
  equal-state/every-step-including-terminal contract. Caller-supplied state
  evaluation is now explicitly arithmetic-only. Added strict metadata
  rehydrators that regenerate every Discovery manifest under `ProtocolConfig`,
  reconstruct every 40-cell attempt, recompute Wilson metadata and point-estimate
  reproduction/dynamic gates, and validate canonical equivalence. The Calibration
  guard now rejects duplicate-key, symlinked, non-regular, or oversized lock
  files; rehydrates the typed Reality-Gate and failure-event freeze objects;
  cross-binds all 40 artifact hashes, validity/success outcomes, orientation
  sources, provenance, and content hashes; and requires the failure-freeze
  implementation commit to be an ancestor of the lock HEAD.
- **Verification:** The focused non-GPU suite passed (`222 passed` across the
  allocation, probe-artifact, artifact, feature, predictor, guard, manifest,
  provenance, feature-pipeline, Reality-Gate, and failure-artifact tests). The
  full collection remains unable to import the four GPU scoring tests because
  the local audit environment intentionally has no `torch`; no test changed or
  accessed the Vast instance. A fresh read-only derivation from all 40 backed-up
  artifacts still reproduces gate SHA
  `fd82aae6dd90462820a90448d3d75b649578f58ce898e94b31f4a23bfb6e2566`, orientation
  SHA `3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb`, and
  the strict guard payload validator accepts the complete candidate freeze.
- **Outcome visibility:** Discovery outcomes and the raw backup were already
  visible. No Calibration outcome, protected-split manifest, or Locked Test data
  was accessed. This entry records that ordering explicitly; it does not create
  a lock or authorize Calibration.
- **Decision:** Commit this implementation hardening separately. Only after the
  implementation commit is pushed will I materialize the freeze with that
  commit, make a lock-only commit, run the clean tagged guard, and decide whether
  any authorized Calibration runtime exists. Keep Vast instance 46677323 stopped.

### 2026-08-03 18:48 CEST — LOCK-MATERIALIZE-001: derive the immutable lock candidate

- **Stage:** Discovery-only lock materialization; no Calibration or Locked Test access
- **Inputs:** Re-loaded exactly the 40 backed-up rank-1 Discovery directories at
  immutable execution commit `b491dc76641efe3a5c5d7eef6bb87af13d85f10b` under
  the pinned `ProtocolConfig`. The generated freeze records implementation
  hardening commit `b41867e01ba50e6eec7fd869b4b18c0b8ea46a01`; no remote file or
  Vast instance state was changed.
- **Candidate artifact:** Wrote the canonical 129,099-byte
  `locks/reality_gate_frozen.json` candidate. The complete strict payload passed
  the typed guard validator at implementation HEAD before the lock-only commit.
  Payload SHA-256 is
  `64524c974e62c2ff500c385f049ce0589ca83c220caabc396358a9053051893c`;
  Reality-Gate receipt SHA-256 is
  `fd82aae6dd90462820a90448d3d75b649578f58ce898e94b31f4a23bfb6e2566`;
  orientation eligibility SHA-256 is
  `3599dab95b5bbc7ee4b3e6ea1872aa21d7aced6dd3ea61d7287cb6aee863a9fb`;
  nested lock receipt SHA-256 is
  `17f033b935ea3f600373b5953cdc5bad5c0fd9dfd3dc1260d022acf3355f36f4`;
  failure-event freeze SHA-256 is
  `dd42e46b055163ca7b8ca777e0bc1a04b9907eab265f87f7234e502a19839328`.
- **Evidence summary:** Orientation remains 11,822/11,822 finite states with
  SD `28.828397073481487` degrees. All 40 selected-task artifacts are retained;
  the failure freeze retains all 40 annotations and the fixed audit membership.
- **Decision:** Keep the candidate untagged until it is committed as the only
  lock evidence on top of the pushed implementation checkpoint. After that
  lock-only commit, require a clean worktree and `prereg-locked-v1` exactly at
  `HEAD`, run the public Calibration guard, and do not inspect or launch any
  protected split unless an authorized Calibration runtime is present. Keep
  Vast instance 46677323 stopped.

### 2026-08-03 19:02 CEST — CALIBRATION-BLOCK-001: provider did not allocate the GPU

- **Stage:** post-lock Calibration scheduling; no protected split access
- **Pre-state:** The immutable lock/tag `prereg-locked-v1` points to
  `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`; the public Calibration guard had
  passed at that exact commit. The deterministic rank-1 Calibration manifest
  was generated read-only (160 episodes, init states 10–29, SHA
  `6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f`).
- **Action:** Issued one reversible Vast `start_instance(46677323)` request for
  the concrete Calibration collection. Vast returned “Required resources are
  currently unavailable, state change queued”; eight read-only polls remained
  `actual_status=exited`, `cur_state=stopped`, `intended_status=stopped`, and
  `next_state=stopped`. SSH remained refused. No Supervisor, rollout writer,
  transfer, or GPU process was started, and no remote file or artifact changed.
- **Decision:** Do not retry or invent a second rollout path while the provider
  has not allocated the preserved instance. Leave it stopped at the verified
  retained-storage rate. The immutable lock remains valid at its tag; this
  administrative log checkpoint is deliberately after the lock commit, so any
  future Calibration attempt must run from the tagged lock worktree and first
  pass the guard there. No Calibration or Locked Test data was accessed.

### 2026-08-03 21:34 CEST — CALIBRATION-RESUME-001: restart, tag sync, and guarded runner preparation

- **Stage:** provider resume and protected Calibration preparation; no Calibration
  outcome or Locked Test access yet.
- **Provider state:** Reissued only the reversible Vast `start_instance(46677323)`
  action after the prior resource-queue response. Vast accepted it with
  `success=true`; read-only status is now `actual_status=running`,
  `cur_state=running`, `intended_status=running`, and `next_state=running`.
  Both the confirmed proxy route `ssh -p 37323 root@ssh9.vast.ai` and the direct
  route are reachable. The RTX 5090 is visible with 2 MiB used and 0% utilization
  before work. Running price remains `$0.3437037037/hr`; no stop is appropriate
  while the concrete Calibration job is active.
- **Remote audit:** Vast's required `/etc/vast-agents-guide.md` was read in
  full. The preserved `/workspace/run-bab7e6e` checkout is clean at Discovery
  commit `b491dc7`; the three old Discovery Supervisor programs are all
  `STOPPED` and were not restarted, so no remote rollout was duplicated,
  overwritten, or reordered. Discovery artifacts and receipts remain unchanged.
- **Immutable execution checkout:** Created a new, non-overlapping
  `/workspace/calibration-locked` checkout from the transferred Git bundle
  (bundle SHA-256
  `8f292320332cd0446c2dd22a3a7532aa4adedca1aa37412745c6715b850c8e54`). Its
  `HEAD` and `prereg-locked-v1` both equal
  `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`; the worktree is clean. The local
  exact-tag `assert_calibration_ready` guard passed and produced the canonical
  160-episode manifest SHA
  `6f5c7a5baa71eadfda1539e756d42ea6cec575316b6ab1245be7d3c5abfe3c3f`.
- **Portability finding:** Re-running the unchanged strict guard on Linux
  CPython 3.12.13 differs from the local macOS CPython 3.12.11 Wilson interval
  at two last-bit float values (~1e-17), so it rejects the otherwise identical
  lock as “dynamic-range receipt disagrees with recomputation.” This is a
  numerical guard portability defect, not an altered result or a protocol
  decision. The original tag was not moved. A hash-bound local guard-authority
  receipt, lock payload, full manifest, and runtime scripts are retained for
  audit; the remote runner additionally verifies tag/tree/lock/manifest hashes
  and uses the pure manifest reconstruction. No Locked Test data is inspected.
- **GPU/EGL smoke:** The tag checkout's real `discovery-reset` for the first
  manifested task returned a valid reset through Linux/EGL with the expected
  10 settle steps and no GPU/runtime error. The legacy policy-loader probe did
  not emit a JSON receipt, but left no persistent process and the GPU returned
  to 2 MiB/0%; the per-cell runner will load the pinned offline policy inside
  the Supervisor-managed child and fail closed on any error.
- **Decision:** Register one new `autostart=false`, `autorestart=false`
  Supervisor program (`mech_vla_calibration`) in the new checkout only. It
  serializes the exact 160 Calibration cells, resumes only byte/provenance-
  validated existing cells, refuses staging ambiguity and overwrite, and writes
  a completion receipt. Start it only after this checkpoint is committed and
  the final read-only no-active-job check passes; do not start any old Discovery
  service and do not instantiate Locked Test.

### 2026-08-03 21:39 CEST — CALIBRATION-RESUME-002: first-cell artifact preserved; runner validation patched

- **Observed execution:** The first corrected Supervisor launch reached the
  actual policy process for `libero_10-task5-calibration-init10-cell0`; model
  weights loaded offline, EGL initialized, and the RTX 5090 reached ~2.3 GiB
  and 11% utilization. The cell completed its atomic write, leaving exactly
  one Calibration artifact directory and no staging directory.
- **Failure:** The child then failed only while comparing the validated loader's
  nested `mappingproxy` metadata to the manifest (`TypeError: mappingproxy is not
  JSON serializable`). The parent exited before starting cell 1; no duplicate
  rollout, overwrite, or Locked Test access occurred. A read-only check confirmed
  Supervisor `EXITED`, no calibration/discovery process, GPU 2 MiB/0%, and the
  preserved cell-0 artifact.
- **Repair:** Updated both runner helpers to canonicalize `Mapping`/tuple values
  before hashing. The change is orchestration-only and does not alter the pinned
  policy, task, condition ordering, manifest, lock payload, or artifact writer.
  Local syntax/plan validation passed; the exact repaired script hashes now match
  the remote copies (`calibration_cell` `397bf4e7…`, `calibration_supervisor`
  `9710d1c2…`). The next Supervisor start must first resume-validate the existing
  cell-0 artifact, then continue at manifest index 1.
- **Decision:** Keep the instance running only for the resumed concrete
  Calibration job. Do not rerun cell 0 and do not touch any old Discovery
  service. The immutable `prereg-locked-v1` tag remains unchanged.

### 2026-08-03 21:41 CEST — CALIBRATION-RUNNING-001: resumable collection active

- **Resume evidence:** After the metadata-normalization repair, Supervisor
  accepted the existing `libero_10-task5-calibration-init10-cell0` artifact
  without rewriting it: `valid_reset=true`, `success=true`, 193 control steps,
  metadata SHA `946acfbb57e0ecce6920f153a6bada4b39d2595f3fb2f43219b4bd77a4bf1eff`,
  trajectory SHA `5169175ce56e96a2b38f1126a1e84f3e20763cc27432e6e383dad694d6180c75`.
- **Current job:** The same Supervisor process is now executing manifest index 1,
  `libero_10-task5-calibration-init10-cell1`, in its isolated child. Model
  weights and EGL initialization have completed; GPU telemetry is ~2.3 GiB and
  15% utilization. The raw Calibration directory count is exactly 1, with no
  Locked Test directory or completion receipt yet.
- **Integrity:** The prior import traceback remains only as historical log text
  from the fail-closed first attempt. The active parent/child both import the
  tag checkout, and the repaired script hashes match the locally committed
  copies. No Discovery service was restarted and no existing artifact is being
  overwritten.
- **Next step:** Leave `mech_vla_calibration` under Supervisor to continue in
  exact manifest order. Recheck counts/receipts after meaningful progress; stop
  the instance only after all concrete GPU work and any backup transfer are
  complete, never by deletion/termination/recycle.

### 2026-08-03 21:44 CEST — CALIBRATION-RUNNING-002: first two cells recorded

- **Progress:** Cell 0 remains the single resumed artifact. Cell 1 completed in
  exact manifest order with 520 control steps, `valid_reset=true`, and the
  expected terminal-horizon truncation (`success=false`). Its immutable hashes
  are metadata `846deb438f6ed87ad42d6fbff5c6308822deb17e5a8a98dda85a9271b43bcee5`
  and trajectory
  `f209a330b2c7531d593e065998c226d4a0af878731a1336fa37dec9914be2628`.
- **Current state:** The Supervisor has started cell 2 (`init10-cell2`) as the
  only rollout child; the remote Calibration tree contains exactly two episode
  directories and no completion receipt. GPU telemetry is healthy and the
  parent remains `RUNNING`.
- **Decision:** Continue the existing Supervisor job without intervention;
  this is ordinary protocol execution, not a new rollout path. Preserve the
  immutable tag and do not inspect or instantiate Locked Test.

### 2026-08-03 21:53 CEST — CALIBRATION-RUNNING-004: resumed job still advancing

- **Read-only recheck:** Vast reports instance 46677323 `running` at the known
  SSH endpoints. The remote Calibration checkout remains clean at
  `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`, exactly matching
  `prereg-locked-v1`; no old Discovery process or Locked Test process exists.
- **Progress:** The resumable Supervisor has now published and independently
  validated six Calibration cells in exact manifest order. Cells 4 and 5 are
  complete: cell 4 succeeded in 179 steps (metadata `7a7027b6…`, trajectory
  `58d788b9…`); cell 5 is a valid 520-step truncation (metadata `0c3f8c0a…`,
  trajectory `93bbf841…`). Cell 6 is the sole active child; no duplicate or
  overwrite path is present.
- **Decision:** Leave the concrete GPU job running under Supervisor and keep
  monitoring. No stop, recycle, terminate, or Locked Test access is permitted
  at this stage.

### 2026-08-04 00:37 CEST — CALIBRATION-RUNNING-005: overnight progress at 73/160

- **Read-only audit:** Vast instance 46677323 remains `running`; the protected
  checkout is clean at `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`, exactly the
  `prereg-locked-v1` tag. The old Discovery Supervisor programs remain stopped.
- **Progress:** The authoritative `mech_vla_calibration` Supervisor has
  atomically published and independently validated 73 Calibration artifact
  directories (manifest indices 0–72); no staging directory exists and no
  completion receipt exists yet. Index 73 (`init19-cell1`) is the sole active
  child. Recent cells remain valid and ordered, including index 72 with
  `success=true`, 197 control steps, metadata `34f5aab1…`, trajectory
  `5ae33c7e…`.
- **Decision:** No restart, duplicate, parallelization, overwrite, or protocol
  change. Leave the concrete GPU job running and continue read-only monitoring.

### 2026-08-04 01:37 CEST — CALIBRATION-RUNNING-006: overnight progress at 97/160

- **Read-only audit:** Vast 46677323 remains `running`; the serial Supervisor is
  healthy, GPU telemetry is active, the locked checkout is clean at
  `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`, and the old Discovery services
  remain stopped.
- **Progress:** 97 Calibration artifact directories (manifest indices 0–96)
  are complete and receipt-validated. There are zero staging directories and no
  completion receipt yet. Index 97 (`init22-cell1`) is the sole active child.
  Recent valid artifacts include index 96 (`success=true`, 155 steps; metadata
  `5eb580e4…`, trajectory `4295febe…`).
- **Decision:** Leave the authoritative process untouched; no duplication,
  reordering, overwrite, or Locked-Test access. Continue monitoring until the
  160-cell receipt appears.

### 2026-08-04 02:37 CEST — CALIBRATION-RUNNING-007: progress at 123/160

- **Read-only audit:** Vast 46677323 and `mech_vla_calibration` remain healthy;
  the locked checkout is clean and still exactly at `prereg-locked-v1`. Old
  Discovery services remain stopped, and no Locked-Test process exists.
- **Progress:** 123 Calibration artifact directories (indices 0–122) are
  complete and receipt-validated; staging count is zero and the completion
  receipt is not yet present. Index 123 (`init25-cell3`) is the only active
  child. Recent index 122 is valid/successful (171 steps; metadata
  `101e2c57…`, trajectory `8a81f367…`).
- **Decision:** Preserve the serial authoritative job unchanged and defer all
  analysis/backup actions until the full 160-cell receipt is present.

### 2026-08-04 03:40 CEST — CALIBRATION-RUNNING-008: progress at 149/160

- **Read-only audit:** Vast instance 46677323 is running and reachable through
  the verified SSH proxy. Supervisor `mech_vla_calibration` is healthy and
  `RUNNING`; the locked checkout is clean at the immutable
  `prereg-locked-v1` commit `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`.
- **Progress:** 149 Calibration artifact directories are present and
  receipt-validated (manifest indices 0–148). Staging is empty, no completion
  receipt exists yet, and index 149 is the sole active child. GPU telemetry is
  healthy (RTX 5090, 15% utilization at audit time).
- **Decision:** Leave the authoritative serial process untouched: no restart,
  duplicate, parallelization, overwrite, reorder, or Locked-Test access. Wait
  for the final cell and completion receipt before independent validation,
  backup, and post-calibration analysis.

### 2026-08-04 03:50 CEST — CALIBRATION-RUNNING-009: progress at 153/160

- **Read-only audit:** The existing Vast Supervisor remains healthy and
  `RUNNING` on the clean immutable `prereg-locked-v1` checkout; no competing
  calibration or Locked-Test process is present.
- **Progress:** 153 of 160 Calibration artifact directories are complete and
  staged count remains zero. Index 153 is the sole active child; the final
  completion receipt is not present yet. GPU telemetry remains responsive.
- **Decision:** Continue monitoring the same serial Supervisor only. No restart,
  duplicate, overwrite, reorder, or early analysis/Locked-Test access is
  permitted before the 160-cell receipt and independent backup validation.

### 2026-08-04 04:28 CEST — CALIBRATION-COMPLETE-001: receipt and remote validation passed

- **Completion:** The authoritative Supervisor exited cleanly after publishing
  all 160 Calibration cells. Its immutable completion receipt binds commit
  `18d64941bc8c899b06306fbec21d1c8d2c08f2ea`, tag `prereg-locked-v1`, and
  `locked_test_accessed=false`.
- **Independent validation:** A second read-only loader checked every cell's
  manifest provenance, metadata/trajectory hashes, task identity, reset flag,
  success flag, action count, and exact two-file layout. All 160 passed;
  107 succeeded, 53 reached the preregistered failure limit, 160 resets were
  valid, and total control steps were 47,067. The remote 320-file inventory
  digest is
  `2040ad899b95cc691b23afd58cfb9f03de63f5573c035ff8a77216c382840ca8`.
- **Backup state:** The new ignored local staging path
  `artifacts/calibration-backup-stage.incomplete/` is receiving the raw set via
  four disjoint, recoverable proxy-SSH Rsync streams. No file is written by
  more than one stream; finalization waits for a complete local inventory and
  digest match. The Vast instance remains running until that guard passes.
- **Decision:** Keep Locked Test closed and defer calibration analysis/freeze
  until the off-instance backup is complete and independently revalidated.

### 2026-08-04 04:41 CEST — CALIBRATION-BACKUP-001: transfer still active

- **Read-only status:** The Vast API reports instance 46677323 still
  `running`; direct SSH confirms `mech_vla_calibration` is `EXITED` and the RTX
  5090 is idle. The proxy status route temporarily timed out under the active
  transfer load, so no process-control action was attempted.
- **Backup progress:** Four non-overlapping proxy-Rsync streams remain active
  in `artifacts/calibration-backup-stage.incomplete/`, with 514 MB and ten
  trajectory files present at this checkpoint. The remote canonical set and
  completion receipt remain immutable; local finalization is blocked until all
  320 files and the independent inventory digest match.
- **Decision:** Preserve the streams and keep the instance running for the
  in-progress off-instance backup. Do not stop the instance, access Locked
  Test, or begin calibration freeze/analysis before checksum verification.

### 2026-08-04 05:40 CEST — CALIBRATION-BACKUP-002: transfer progressing

- **Read-only status:** Direct SSH and the Vast API still show instance
  46677323 `running`, Supervisor `EXITED`, GPU idle, 160 remote Calibration
  directories, and the completion receipt present. The locked checkout remains
  clean at the immutable commit.
- **Backup progress:** The four disjoint proxy-Rsync streams have reached about
  1.9 GiB and 28 local trajectory files in
  `artifacts/calibration-backup-stage.incomplete/`; no stream overlap or remote
  mutation is present. Local free space is about 17 GiB.
- **Decision:** Keep the streams and instance running until all 320 files,
  receipts, and the independent inventory digest are verified. Locked Test and
  Calibration freeze remain closed.

### 2026-08-04 06:41 CEST — CALIBRATION-BACKUP-003: transfer progressing

- **Read-only status:** Vast instance 46677323 remains `running` while the
  authoritative Supervisor is `EXITED`; direct SSH reports an idle RTX 5090,
  160 remote Calibration directories, a present completion receipt, and the
  clean immutable checkout.
- **Backup progress:** The four non-overlapping proxy-Rsync processes now hold
  approximately 3.4 GiB and 44 trajectory files in the ignored incomplete
  staging path. Local free space is approximately 16 GiB, sufficient for the
  remaining raw set with headroom.
- **Decision:** Continue the existing transfer streams unchanged. Do not stop
  the instance or open Locked Test until the complete local tree and independent
  inventory digest have been verified.

### 2026-08-04 07:04 CEST — CALIBRATION-ANALYSIS-001: read-only failure/probe run started

- **Guard:** The remote Calibration receipt and all 160 raw artifacts were
  already independently validated against the immutable `prereg-locked-v1`
  checkout and inventory digest
  `2040ad899b95cc691b23afd58cfb9f03de63f5573c035ff8a77216c382840ca8`. A
  read-only Sol-xhigh boundary review confirmed that the Discovery-derived
  failure bounds/rules may be applied unchanged to Calibration and that the
  exact stride-5 pre-action cohort assembly is protocol-correct when the full
  160-cell manifest is asserted first.
- **Execution:** A separate runner was launched on Vast PID `110887`, writing
  only to the new analysis path
  `/workspace/research-artifacts/analysis-staging/calibration-analysis-18d6494-probe-002`.
  It independently checks the guard authority, lock/freeze hashes, manifest
  topology, raw provenance, and then computes deterministic Failure-Event
  annotations, the five-candidate grouped circular probe CV, and the required
  Mean/Time/Proprioception/Random-label controls. The prior failed attempts
  were fail-closed schema/import corrections and created no raw or backup
  writes.
- **Isolation:** The four disjoint resumable backup streams continue unchanged
  in `artifacts/calibration-backup-stage.incomplete/`; no analysis process writes
  there or to the raw tree. Locked Test remains closed. M0/M1/M2 score/features
  will only start after this probe receipt, because no score sidecars exist yet.
- **Current result:** The runner is CPU-bound while validating/reading the
  immutable raw set (GPU idle); no result receipt is published yet. Vast stays
  running because the off-instance backup is incomplete.

### 2026-08-04 07:25 CEST — CALIBRATION-SCORING-001: serial M0/M1/M2 replay started

- **Inputs/guards:** The read-only probe analysis completed with 160 valid
  Calibration episodes, 9,455 stride-5 rows, selected candidate
  `early_expert_t1_0`, numerical probe digest
  `71a6bff1691aea4823556e256a4052f78e2126d9a6cc76437923ec25409d4afc`, and
  bound-probe digest
  `747c5fd8013a4ca54f17a3929df20228732cbe3c08b1b761090c5840fee94564`.
  The new scorer verifies the immutable tag/manifest and bound probe before
  any replay and is exclusively locked with `/workspace/runstate/calibration-score.lock`.
- **Execution:** Remote PID `111494` is running the serial replay from the
  Calibration raw set only. Each sidecar is atomically published under the new
  `calibration-score-18d6494-001` staging path and is resumable without
  overwriting existing sidecars. After all valid raws, the scorer will audit
  the score allocation, build the OOF feature cohort, and fit the frozen M0/M1/M2
  predictor family/Platt calibrators.
- **Isolation:** Locked Test is not instantiated or inspected. Raw artifacts,
  the four backup streams, and the probe analysis staging are read-only to this
  process. Vast GPU telemetry shows the pinned model loaded (~2.3 GiB VRAM);
  the first Calibration replay is still in progress and no sidecar has been
  published yet.

### 2026-08-03 21:47 CEST — CALIBRATION-RUNNING-003: four cells complete

- **Progress:** Calibration manifest indices 0–3 are now accounted for exactly
  once. Index 2 succeeded after 162 steps (metadata `2fad977d…`, trajectory
  `4b1b0c36…`); index 3 succeeded after 179 steps (metadata `9298a304…`,
  trajectory `b7043520…`). The earlier index-1 truncation remains valid and
  unchanged. All four directories were published atomically and independently
  revalidated by the parent.
- **Current state:** Index 4 (`init10-cell4`) is the only active child; the
  Supervisor remains `RUNNING`, GPU telemetry is healthy, and the remote raw
  tree count is 4/160. No Locked Test path has been instantiated.
- **Decision:** Continue under the same Supervisor process and preserve all
  receipts for the eventual off-instance backup.

### 2026-08-04 07:32 CEST — CALIBRATION-ANALYSIS-002: probe and failure analysis verified

- **Read-only result:** The remote analysis staging set is complete and a
  local copy was independently hash-checked. Failure annotations cover all
  160 valid-reset Calibration episodes: 107 successes, 53 failures, 53
  annotated terminal events, and zero early-terminal exclusions. The event
  classes are 48 terminal-horizon and 5 irrecoverable-workspace-exit events;
  no missed-grasp/drop event was inferred.
- **Probe result:** The exact stride-5 pre-action cohort contains 9,455 rows
  from 160 episodes. The grouped five-fold circular-probe CV selected
  `early_expert_t1_0` (mean MAE `0.2533213489664437` rad, SE
  `0.0246275630200348`) under the preregistered eligibility rule. The
  `vlm_context` candidate was ineligible at mean MAE
  `0.2740076992084197` rad. Mean, time-only, proprioception-only, and fixed
  reverse-row random-label controls were all computed outcome-independently.
- **Receipts:** Local payload hashes match the remote analysis receipts:
  failure annotations `d004e4db…999063a`, allocation
  `7cd831bd…7dbc1a`, probe analysis `a4cab1f…fcc5a5`, numerical probe
  `71a6bff1…d4afc`, and bound probe `747c5fd8…94564`. The analysis records
  `locked_test_accessed=false` and remains separate from raw/backup paths.
- **Decision:** Keep Locked Test closed. Continue the single resumable M0/M1/M2
  replay against the immutable raw set; no duplicate, overwrite, or protocol
  change is authorized.

### 2026-08-04 07:32 CEST — CALIBRATION-SCORING-002: M0/M1/M2 replay active

- **Read-only status:** Vast instance 46677323 is `running` and direct SSH is
  reachable. The calibration Supervisor is exited after its 160/160 receipt;
  the only active research process is the flock-protected scorer PID `111494`.
  It has published exactly one of the 160 valid Calibration score sidecars;
  the log reports `score_completed` for `libero_10-task5-calibration-init10-cell0`.
  GPU telemetry shows the pinned process using about 2.3 GiB VRAM.
- **Isolation:** The scorer writes only to
  `calibration-score-18d6494-001` and the later feature staging root. Raw
  artifacts, the four disjoint resumable backup streams, and the completed
  probe staging are not targets. Locked Test remains closed.
- **Decision:** Leave the scorer and all four backup streams running. Do not
  restart or parallelize scoring; stop the instance only after scoring/backup
  jobs are genuinely idle and all irreplaceable artifacts are secured.

### 2026-08-04 07:37 CEST — CALIBRATION-SCORING-003: active replay and backup checkpoint

- **Read-only status:** Vast API still reports `running` for instance 46677323
  (`dph_total=0.3437037037`; current inactive-storage field
  `storage_total_cost=0.0370370370`). Direct SSH confirms the same
  flock-protected scorer PID `111494` is alive at high CPU utilization with
  approximately 2.3 GiB GPU memory and no error receipt. The serial scorer has
  one validated sidecar and is processing the next 520-step Calibration cell.
- **Backup:** The four existing disjoint Rsync streams remain unchanged and
  have advanced to approximately 4.8 GiB / 134 local files in the ignored
  incomplete staging path. They still have active SSH children; no stream
  overlap or raw-tree write exists.
- **Decision:** Preserve both concrete jobs and the current immutable remote
  artifacts. No restart, parallel scoring, transfer-method switch, stop, or
  Locked-Test access is warranted.

### 2026-08-04 08:43 CEST — CALIBRATION-SCORING-004: serial replay advancing

- **Read-only status:** Vast instance 46677323 remains `running`; the
  collection Supervisor is cleanly exited after its immutable 160-episode
  receipt, with `locked_test_accessed=false`. The single flock-protected
  scorer PID `111494` is healthy and has advanced to six validated sidecars
  (Calibration cells `init10-cell0` through `init10-cell5`). No feature-stage
  publication or error output is present yet; GPU telemetry remains active.
- **Backup:** The four original disjoint Rsync streams remain active and
  unchanged at approximately 4.8 GiB / 134 local raw files. The incomplete
  staging tree is still not a final backup and is not used by scoring.
- **Decision:** Keep the serial scorer and existing backup streams running
  without restart, duplication, overwrite, reordering, or transfer-method
  changes. Locked Test stays closed and the Vast instance stays running while
  concrete jobs remain active.

### 2026-08-04 09:44 CEST — CALIBRATION-SCORING-005: replay remains healthy

- **Read-only status:** Vast 46677323 is still `running`; no
  `mech_vla_calibration` process remains because the authoritative collection
  already completed 160/160. The one flock-protected scorer is alive with
  active RTX 5090 utilization and has published 11 validated sidecars through
  `libero_10-task5-calibration-init11-cell2`. No feature-stage files or error
  output have appeared; this is expected until all 160 valid episodes are
  scored.
- **Backup:** The four disjoint Rsync streams remain active at approximately
  4.8 GiB / 134 files, with no target overlap or raw-artifact mutation.
- **Decision:** Leave both concrete jobs untouched. Do not restart,
  parallelize, reorder, stop, or access Locked Test; continue monitoring until
  the scorer publishes its downstream receipts and the backup independently
  verifies the canonical inventory.

### 2026-08-04 10:28 CEST — CALIBRATION-SCORING-BENCHMARK-001: isolated two-worker benchmark planned

- **Question and non-interference guard:** Test whether episode-level scheduling
  can accelerate the remaining Calibration M0/M1/M2 replay without changing any
  scientific computation. The authoritative flock-protected serial scorer PID
  `111494` is healthy at 15/160 validated sidecars and remains completely
  untouched. All benchmark writes are confined to the new
  `calibration-two-worker-benchmark-18d6494-001` staging tree with separate
  worker roots and locks; benchmark publication is exclusive/atomic and cannot
  overwrite the authoritative score root.
- **Frozen benchmark:** Re-score four already completed manifest episodes with
  two disjoint workers: worker 0 receives `init10-cell2` and `init10-cell4`;
  worker 1 receives `init10-cell3` and `init10-cell7`. Both dynamically use the
  exact frozen serial runner, authority/manifest, environment lock, offline
  model snapshot, raw artifacts, bound probe, seeds, transforms, replay and
  scoring functions. The benchmark script is content-bound at SHA-256
  `1318497b1d2ef39fc6558ef5481316c4fa8a2087b150c68d782fd722bf62620c`.
- **Measurements and decision rule:** Compare both `metadata.json` and
  `primitives.npz` byte-for-byte and record every array digest, total elapsed
  throughput versus the four uninterrupted serial publication intervals, plus
  one-second GPU utilization/VRAM, worker and serial CPU/RSS, and host RAM.
  Switching is forbidden unless every complete sidecar is byte-identical and
  throughput clearly improves. A switch would additionally require a narrow
  read-only Sol-xhigh protocol/provenance review before touching the serial
  process. Hash divergence or weak speedup means retaining the serial scorer.
- **Safety:** Worker parent-death guards, benchmark process-group cleanup,
  serial PID/start-time/executable/command identity checks, a 2,400-second
  watchdog, immutable reference rechecks, fresh-path requirements, and
  benchmark-script digest checks are fail-closed. Locked Test remains closed.

### 2026-08-04 10:49 CEST — CALIBRATION-SCORING-BENCHMARK-002: 1.72× throughput but byte-identity gate failed

- **Completed measurement:** The frozen two-worker benchmark completed all four
  previously scored episodes (145 total scored states) in `926.955328475` s,
  including two independent model loads. This is `15.534729191` episodes/hour
  versus `9.042013620` episodes/hour from the same four uninterrupted serial
  publication intervals (`1592.565617110` s), an observed throughput ratio of
  `1.718060804`. Worker episode wall times were 398.76/450.88 s and
  426.20/460.92 s, with approximately 31.2 s model load per worker.
- **Byte comparison:** The required whole-sidecar gate failed in all 4/4 cases:
  every benchmark `metadata.json` and `primitives.npz` SHA-256 differs from its
  authoritative serial counterpart. Array-level comparison is more specific:
  every scientific array (control steps, availability masks, seeds, original/
  transformed/intervention actions, and activations) is byte-identical in all
  four episodes, while exactly the four runtime-measurement arrays
  `original_cost`, `transformed_cost`, `intervention_minus_cost`, and
  `intervention_plus_cost` differ. The metadata consequently binds a different
  primitives digest/cost summary. This scientific equivalence does not satisfy
  the user's stricter complete-sidecar rule.
- **Resource profile:** Across 927 one-second samples, device-wide GPU
  utilization (including the untouched serial scorer) averaged 83.59% and
  peaked at 93%; total VRAM averaged 6,714.8 MiB and peaked at 6,993 MiB. Each
  benchmark worker peaked at 2,314 MiB VRAM and about 5.35/5.46 GB RSS; combined
  worker RSS averaged 9.80 GB and peaked at 10.81 GB. Combined benchmark-worker
  CPU averaged 589.5%, while the authoritative serial scorer averaged 318.4%.
  Host used RAM averaged 67.38 GB and peaked at 73.51 GB.
- **Integrity and backup:** The serial PID `111494`, start ticks, executable,
  parent, and full command remained identical throughout; it advanced normally
  to 18/160. The benchmark wrote nothing to the authoritative score root and
  records `locked_test_accessed=false`. The 15-file, 7.3-MiB remote benchmark
  set was copied to
  `artifacts/calibration-two-worker-benchmark-18d6494-001/`; every relative path,
  byte size, and SHA-256 matches remote. Key receipts are plan
  `c4225f26…130fa8`, resource samples `7b377190…38a61`, and final summary
  `0970accf…adf0f3`; the local 15-file backup receipt hashes to
  `99e8f0bf…abf85`.
- **Decision:** Retain the existing serial scorer. Do not stop it, shard the
  remaining manifest, create an amendment, or seek the conditional Sol-xhigh
  scheduling review: the prerequisite full-sidecar byte identity is false, so
  no switch is eligible regardless of the measured throughput increase. Locked
  Test remains closed.

### 2026-08-04 11:06 CEST — CALIBRATION-SCORING-EQUIVALENCE-001: cost-only divergence independently verified

- **Revised authorization and question:** The user explicitly authorized treating
  scheduling-dependent physical cost measurements as expected differences if all
  scientific primitives, masks, seeds, transformations, non-cost metadata, and
  M0/M1/M2 inputs are identical. No scheduling change has yet been made; the
  authoritative serial scorer remains healthy and reached 21/160 sidecars during
  this read-only audit.
- **Independent sidecar audit:** A new fail-closed reader reloaded both complete
  files for all four serial/benchmark episode pairs rather than trusting the prior
  summary. Exactly four arrays differ: `original_cost`, `transformed_cost`,
  `intervention_minus_cost`, and `intervention_plus_cost`. Every other array is
  byte-identical, including control steps, actions, selected activations, noise
  seeds, transform/intervention availability masks, and all transformation and
  intervention outputs. Every non-cost metadata field is identical; the sole JSON
  difference is `files.primitives_sha256`, which necessarily binds the differing
  cost-containing NPZ.
- **Cost-field localization:** Within those four arrays, `forward_count`,
  `intervention_count`, `logical_activation_bytes`, and
  `compressed_activation_bytes` are identical. Differences are confined to the
  physical runtime/resource fields `cuda_event_ms`, `wall_time_ns`, and, where the
  allocator state changed, `peak_allocated_bytes` and
  `incremental_peak_allocated_bytes`.
- **Feature exclusion:** Static dependency extraction from the frozen feature
  source found only control steps, masks, actions, and activations as score-array
  inputs to feature construction; no cost array enters an M0/M1/M2 primitive or
  feature schema. A synthetic end-to-end regression changed every dynamic cost
  column and independently rebuilt the Calibration feature cohort; all M0, M1,
  and M2 matrices remained bit-identical. The full feature-pipeline test file
  passes 19/19 tests.
- **Receipt and boundary:** The off-instance equivalence receipt is
  `artifacts/calibration-scoring-equivalence-audit-18d6494-001/equivalence-audit.json`
  at SHA-256 `68904e5285b029f7330cdeb43de85d35396f8e10b10f898298744ab086dc6d85`.
  It records `locked_test_accessed=false` and no writes to either sidecar set. The
  requested read-only Sol-xhigh review is in progress; no amendment, signal,
  shard, or worker launch is permitted before its verdict.

### 2026-08-04 13:35 CEST — CALIBRATION-SCORING-CUTOVER-002: cost-only scheduling switch executed

- **Protocol check before action:** The requested read-only GPT-5.6-Sol-xhigh
  scheduling review returned a conditional GO: a scheduling-only change is
  protocol-compatible when the scientific arrays, actions, activations, seeds,
  transformations, availability masks, and non-cost metadata remain identical,
  the four runtime-cost arrays are excluded from M0/M1/M2, completed serial
  sidecars are immutable, and future allocation is an outcome-blind manifest
  complement. Those conditions were recorded in `AMENDMENTS.md` and bound by
  the pre-action commits `6ad024c`, `6cb3733`, and `7fe9ebc`.
- **Exact equivalence:** The independent four-episode/145-state receipt
  `artifacts/calibration-scoring-equivalence-audit-18d6494-001/equivalence-audit.json`
  (SHA-256
  `68904e5285b029f7330cdeb43de85d35396f8e10b10f898298744ab086dc6d85`)
  reports exactly four differing arrays: `original_cost`, `transformed_cost`,
  `intervention_minus_cost`, and `intervention_plus_cost`. All scientific
  arrays and non-cost metadata are byte-identical. Within those arrays,
  deterministic counts and activation-byte fields are identical; only
  `cuda_event_ms`, `wall_time_ns`, and allocator peak fields differ. Static
  dependency extraction and a cost-perturbation regression both show that none
  of the four arrays is an M0/M1/M2 predictor input.
- **Failed first cutover, preserved:** The original coordinator observed
  serial boundary 31 (`libero_10-task5-calibration-init13-cell6`, digest
  `1eaf900ee6d67b232e6c50850631a8943283eb3714a5a09a92d9601c03f48716`) and
  dispatched its one planned SIGINT. The detached Python scorer inherited
  `SIGINT=SIG_IGN`, so the signal was durably recorded as ineffective and the
  coordinator failed closed without planning or launching workers. No receipt
  was rewritten and no sidecar was overwritten. A second read-only Sol-xhigh
  recovery review approved a narrowly bounded Python-only SIGTERM recovery;
  the recovery amendment and implementation were committed before signalling
  (`a19a50e`, `257309f`, bound checkout `fa64127`).
- **Clean recovery boundary:** At 13:35:03 CEST the recovery coordinator saw
  the fresh parsed completion for boundary 40,
  `libero_10-task5-calibration-init14-cell7`, digest
  `2c77eacdbf651c54f033bad12cbc405213e1319e327b21db7e8e9a49a3d7528f`.
  It revalidated the exact PID/start-tick/cmdline and signal masks, fsynced a
  hash-linked intent, sent exactly one `libc_pidfd_send_signal(SIGTERM)` to
  Python PID `111494`, never signalled the `flock` wrapper, never called
  `os.kill`, observed both exits, reacquired the global lock exclusively, and
  made no causal claim about the exit. The possibly started
  `init15-cell0` computation is explicitly non-authoritative with cost
  unavailable; it is not part of the frozen denominator.
- **Frozen allocation:** Independent validation rehashed all 40 frozen serial
  sidecars and reproduced inventory digest
  `3ec590dc746f74397c4ced256525290207588a3ca9eb620f659b2b21bbfc5528`.
  The plan digest is
  `7c6b9e92f3b570e576b8c4394af8c627641cdd940f1aa7f979e17fb6af5421a0` and
  its 120-ID manifest complement is exactly two disjoint alternating shards
  of 60 episodes, with labels, features, durations, state counts, and costs
  excluded from assignment. Every recovery, cutover, plan, and runtime receipt
  has `locked_test_accessed=false`.
- **Live continuation:** The recovery coordinator PID `133377` now holds the
  global score lock and exactly two workers are active: PID `134073` (worker 0)
  and PID `134074` (worker 1). Each has its own held lock and staging root,
  uses the unchanged locked checkout/probe/raw artifacts/model/seeds/transforms,
  and currently uses 2,314 MiB VRAM. At the latest check, two new promotions
  were atomically published (`42/160` authoritative sidecars); all 40 prior
  serial sidecars remain unchanged and the serial scorer is gone. The final
  execution receipt will report physical costs by execution mode (`serial`
  family, including the documented benchmark-contention subtype, versus
  `two_worker`) and will keep those costs outside predictor inputs. The full
  raw off-instance backup streams remain independent and incomplete; no worker
  writes to them. Locked Test remains closed.

### 2026-08-04 15:53 CEST — CALIBRATION-BACKUP-RESUME-001: resumable raw backup restored after app restart

- **Read-only diagnosis:** The app restart left the four earlier local proxy-
  Rsync processes absent; the incomplete local stage was intact at about 4.8
  GiB and 134 files. The remote Calibration raw tree remained immutable and
  the GPU scoring coordinator/workers were unaffected. No remote source or
  authoritative score path was modified.
- **Partition guard:** A fresh remote inventory check confirmed exactly four
  disjoint init ranges, each containing 80 canonical raw files:
  `init10–14`, `init15–19`, `init20–24`, and `init25–29`. No range overlaps
  another, and the local incomplete target is the sole destination.
- **Action:** Resumed the same SSH-proxy Rsync method with `--checksum` and
  `--partial`, without `--delete` or any remote write. Four separate local
  sessions now own the four ranges; the first resumed stream has already
  added two missing files. The transfer remains an incomplete staging copy and
  is not yet treated as a verified backup.
- **Concurrent science:** The guarded two-worker scorer remains authoritative
  and healthy at 58/160 published sidecars (18 promotions), with worker PIDs
  `134073`/`134074`, held per-worker locks, and 2,314 MiB VRAM each. No serial
  scorer, residue, or error receipt is present. The Vast instance stays running
  while GPU and transfer jobs are concrete; Locked Test remains closed.

### 2026-08-04 16:48 CEST — CALIBRATION-MONITOR-001: workers and resumed backup healthy

- **Live scoring:** The direct SSH route remains healthy after the proxy route
  timed out during banner exchange. The original Supervisor is `EXITED` after
  its completed 160-cell collection, while the guarded continuation coordinator
  and exactly two workers remain active. The authoritative score root has
  reached `66/160` sidecars (`26` two-worker promotions); worker PIDs
  `134073` and `134074` each hold their private lock and use 2,314 MiB VRAM.
  No serial scorer, publish residue, or Locked-Test access is present.
- **Backup:** The four checksum-aware, disjoint SSH-proxy Rsync sessions remain
  active in the local incomplete stage, which has grown to 150 raw files and
  approximately 5.3 GiB. It is still not treated as a verified backup; the
  instance remains running while both GPU and transfer jobs are concrete.
- **Decision:** Leave the authoritative coordinator, workers, and transfers
  unchanged. Continue read-only monitoring; do not stop the instance, inspect
  Locked Test, or launch any duplicate/reordered scorer.

### 2026-08-05 13:50 CEST — CALIBRATION-FEATURE-FINALIZATION-001: finalizer defect repaired, Calibration features published

- **Supervision gap:** The previous supervising agent's last observation was
  `124/160` sidecars at 2026-08-04 20:54 UTC. The two workers then finished
  normally (`60/60` each) and the execution receipt recorded
  `status="scoring_complete"` with the full 160 authoritative sidecars. The
  coordinator launched its unchanged-serial finalizer at 00:23 UTC, which
  aborted at 01:31 UTC. The three following scheduled heartbeats (02:56, 05:57,
  08:58 UTC) produced no supervising turn at all, so the abort went unobserved
  for roughly ten hours while the instance sat idle at 0% GPU utilisation.
- **Diagnosis (fail-closed, read-only first):** The finalizer log ended in
  `AttributeError: 'FeatureCohort' object has no attribute 'metadata_sha256'`
  raised at `calibration_score_18d6494.py:374` while assembling the final
  summary. Everything expensive had already succeeded: all 160 sidecars were
  revalidated, and the score-allocation receipt, feature reference bundle,
  feature cohort, and fitted M0/M1/M2 predictors were written. The cause is a
  name mismatch between sibling classes: `FeatureReferenceBundle` exposes
  `metadata_sha256`, whereas the digest property of `FeatureCohort` is named
  `provenance_sha256`, with an identical implementation
  (`_metadata_sha256(self.to_metadata())`). Because the coordinator promotes the
  authoritative feature root only after a staged summary exists, `attempt-0001`
  stayed unpromoted and no authoritative feature artifact had been published.
- **Provenance trap avoided:** The obvious repair — having
  `write_feature_cohort` return its digest — would have modified
  `src/mech_int_vla/feature_artifacts.py`, which is a member of
  `SCORING_SOURCE_FILES`. That would have changed `scoring_source_sha256` and
  broken the `code_sha256` guard against all 160 sidecars, which carry the old
  digest. The fix was therefore confined to `ops/calibration_score_18d6494.py`,
  which is not hashed into the provenance chain. No file entering
  `scoring_source_sha256` was touched.
- **Exact change and deployment:** One line, verified by byte-level diff:
  `feature_cohort.metadata_sha256` → `feature_cohort.provenance_sha256`. The
  corrected runner was deployed as a new file
  `/workspace/runstate/calibration_score_18d6494_fix1.py`; the original runner
  remains byte-identical at `6cf21bc2…98fd54`, so the frozen plan's
  `serial_runner_sha256` stays true.
- **Why the frozen wrapper could not be used:** `_assert_arguments_match_plan`
  requires the runner hash recorded in the immutable plan, so the continuation
  wrapper structurally refuses to launch a corrected runner. The finalizer step
  was therefore executed directly using the plan's exact
  `finalizer_command_prefix`, the identical locked environment (`PYTHONPATH`,
  `MUJOCO_GL=egl`, offline HF flags, etc.), an exclusive `flock` on the plan's
  global lock, and a fresh `finalizer-staging/attempt-0002/features` root. The
  run took 71 minutes (09:32–10:43 UTC).
- **Determinism independently confirmed:** `attempt-0002` reproduced
  `attempt-0001` exactly. The cohort `arrays.npz`/`metadata.json`, the reference
  bundle, the score-allocation receipt, and `predictors.json` are all
  byte-identical across the two attempts, and the cohort publish digest is the
  same `03c37788…5343ed`. The defect was thus purely in receipt assembly and
  changed no scientific quantity.
- **Guarded promotion:** The wrapper's post-run guards were reproduced
  explicitly — staged `sidecar_count == 160`, `locked_test_accessed == false`,
  destination absent, same filesystem — before promoting the staged root with
  `renameat2(RENAME_NOREPLACE)` to
  `/workspace/research-artifacts/analysis-staging/calibration-features-18d6494-001`.
  `completion.json` was then written exclusively (SHA-256 `67ff46d1…d579ef`),
  recording the direct-execution mode, both runner hashes, the single-line
  diff, the reproduced guards, and the determinism result. `attempt-0001` was
  preserved untouched as evidence. Locked Test remains closed and unaccessed
  (`locked_test_accessed=false` throughout).
- **Off-instance backup:** All 8 authoritative feature files (66 MB) were
  copied to `artifacts/calibration-features-18d6494-001/` and independently
  re-hashed; every file is byte-identical to the instance. Key receipts are
  cohort `03c37788…5343ed`, reference `c8cad407…56949b`, score allocation
  `e9a04dd4…17fb3` (self-verifying: hash equals directory name), predictors
  metadata `acecac7b…3adda7`, and feature summary `2c46f770…3433ac8`. Direct-SSH
  throughput measured ~1.9 MB/s, roughly twenty times the previously observed
  proxy rate.
- **First Calibration read-out (group-5-fold OOF, 160 episodes / 9,455 states /
  20 base-init groups; shared family logistic regression, C=1.0, selected on M1
  raw OOF log loss only):**

  | Model | calibrated AUROC | calibrated log loss |
  |---|---|---|
  | M0 (output-only) | 0.7889 | 0.47464 |
  | M1 (+ simulator state) | 0.9315 | 0.24832 |
  | M2 (+ internal geometry) | 0.9317 | 0.24541 |

  Relative log-loss improvement M1 over M0 is 47.68%; M2 over M0 is 48.30%;
  **M2 over M1 is 1.17%**. The M2−M1 AUROC delta is +0.0002. Kill-Switch 1 is
  **not** triggered (criterion: M0 or M1 calibrated group-OOF AUROC ≥ 0.95; M1
  reached 0.9315, close to but below the threshold), so the preregistered
  failure-mode order stands and no methodological switch is due.
- **Interpretation and its limits:** These are Calibration out-of-fold values
  used for model selection, **not** the primary estimand. The primary claim is
  the paired out-of-sample log-loss difference M2 versus M1 on the Locked Test
  Set, which remains closed. Read with that caveat, the pattern is nonetheless
  informative and points at the "lift over M0 but not over M1" row of the §12
  decision table: almost the entire predictive gain comes from privileged
  simulator state, while internal geometry adds 1.17% — well below the
  preregistered 3% threshold, and with an essentially unchanged AUROC. No
  threshold, feature, or split was altered in response to seeing this.
- **Decision:** Calibration scoring and feature finalization are complete and
  backed up. Do not open Locked Test. Next preregistered steps are the
  remaining Calibration-side calibration work — alarm thresholds at
  episode-level FPR 10% with k=3 consecutive exceedances for the Lead-Time
  secondary question, and intervention strength for the §8 causal protocol —
  followed by a clean tracked freeze and the `calibration-locked-v1` tag before
  any Locked Test access. The instance is stopped once the remaining
  irreplaceable artifacts are secured; its disk is preserved.

### 2026-08-05 17:20 CEST — CALIBRATION-ALARM-001: alarm thresholds fixed, Lead-Time claim fails against M1

- **Scope:** Preregistered alarm-threshold calibration and the Lead-Time
  secondary question, on Calibration only. Locked Test was never opened
  (`locked_test_accessed=false`). Implemented as
  `ops/calibration_alarm_18d6494.py`, executed in the frozen instance
  environment (python 3.12.13, numpy 2.2.6, scikit-learn 1.9.0).
- **Two protocol decisions fixed before running** (see the AMENDMENTS entry
  "Fix the score source and failure step for alarm calibration"): the threshold
  is calibrated on **group-out-of-fold calibrated probabilities**, never on
  in-sample scores from the refit-on-all base model; and `t_failure` maps to
  the annotation `onset_step`, never `confirmation_step`. Both are the
  conservative direction. The frozen bundle retains only scalar OOF metrics, so
  the per-row OOF vectors were reconstructed deterministically
  (`_make_group_folds` uses no RNG) and bound to independently pinned anchors:
  cohort digest `03c37788…5343ed`, predictor metadata `acecac7b…3adda7`,
  calibration data `215e52cf…488986`, the recorded `fold_assignments`, the
  recorded per-model `oof_metrics`, and the frozen Platt map. All anchors
  reproduced exactly; the script fails closed otherwise.
- **Independent review:** The score-source question and the finished script were
  reviewed read-only by an independent model (Fable 5) before execution. That
  review caught two blocking defects in the first draft: the cohort row list is
  `provenance["rows"]`, not `provenance["records"]`; and base-init identifiers
  must stay **`int`**, because `_canonical_identifier` hashes `10` and `"10"`
  differently — with strings the data hash becomes `5c1c7c49…3a7ea1` instead of
  the frozen `215e52cf…488986`. The script was rewritten to use the frozen
  validated loader `load_feature_cohort`, which reproduces the exact fit-time
  inputs and would have prevented both defects.
- **Alarm thresholds** (episode-level FPR ≤ 10% over the 107 successful
  episodes, alarm after k=3 consecutive exceedances, cadence 5). All three
  models land on the same realised rate of 9.35% (10 of 107 successful
  episodes):

  | Model | threshold | realised episode FPR |
  |---|---|---|
  | M0 | 0.5716 | 9.35% |
  | M1 | 0.4259 | 9.35% |
  | M2 | 0.3874 | 9.35% |

- **Lead time** (53 failed episodes, 18 init clusters, cluster bootstrap):

  | Comparison | median lead (baseline → M2) | median paired difference | 90% interval | detection rate |
  |---|---|---|---|---|
  | **M2 vs M1** (preregistered) | 440.0 → 440.0 | **0.0** | [0.0, 0.0] | 100% → 100% |
  | M2 vs M0 (secondary) | 395.0 → 440.0 | 85.0 | [60.0, 95.0] | 83.0% → 100% |

  The preregistered internal lead-time claim requires a median paired
  `(lead_M2 − lead_M1) ≥ 5` control steps (`PREREG.md:50`). Observed is **0.0
  with a degenerate [0.0, 0.0] interval**: on Calibration, internal signals
  raise the alarm on exactly the same control step as M1 in every failed
  episode. The claim **fails**, and it fails cleanly rather than marginally.
  Against M0 the picture is the familiar one: M2 alarms 85 steps earlier and
  detects all 53 failures where M0 detects 83%.
- **Important caveat on what "lead" means here:** 48 of the 53 failures are
  `terminal_horizon` events, where onset equals confirmation equals the 520-step
  horizon — the policy simply never finishes rather than committing an
  identifiable failure. Only 5 are `irrecoverable_workspace_exit` with a genuine
  earlier onset. A median lead of 440 steps therefore mostly means "it is
  visible early that this episode will not succeed", not "an imminent failure
  event was anticipated". The M2-versus-M1 null is unaffected by this caveat,
  since both models are measured against identical failure steps.
- **Convergent read-out:** Two independent preregistered Calibration criteria now
  point the same way. Predictive lift M2 over M1 is 1.17% against a 3% threshold,
  and lead-time gain M2 over M1 is 0.0 steps against a 5-step threshold. Both
  large-looking gains (48.3% log loss, 85 steps) are M2-over-M0, i.e. internal
  signals substituting for privileged simulator state rather than exceeding it.
  This is the §12 row "lift only M2>M0, not M2>M1". These remain Calibration
  values; the primary estimand is decided on the closed Locked Test.
- **Backup:** The receipt `alarm-lead-time-summary.json`
  (SHA-256 `ea93f872…30e7a2`) was copied off-instance and re-hashed locally to
  the identical digest.
- **Decision:** Do not adjust any threshold, feature, or model in response to
  this result. Proceed to the remaining Calibration work — intervention-strength
  calibration for the §8 causal protocol, which requires GPU — then the tracked
  freeze and `calibration-locked-v1` tag before any Locked Test access.

### 2026-08-05 17:45 CEST — CALIBRATION-BACKUP-COMPLETE-001: raw set fully secured, instance stopped

- **Raw backup completed and verified:** The Calibration raw set is now fully
  off-instance. All 400 remote files (14.6 GB) were transferred and then
  independently re-hashed on both sides: every file is byte-identical, and the
  remote-only set is empty. This closes the long-standing partial backup, which
  had been stuck at 166 of 400 files since 2026-08-04.
- **Failed first attempt, recorded:** The initial resume silently transferred
  nothing. macOS ships rsync 2.6.9, which does not support `--info=stats2`; the
  process printed a usage message and exited 0, so the wrapper reported success.
  The corrected run used `--stats` instead. Lesson: a zero exit code from a
  wrapped transfer is not evidence of transfer; the file count was.
- **Legacy staging residue identified, not deleted:** The earlier partial backup
  was written one directory level deeper (`./raw/calibration/...` rather than
  `./calibration/...`), so it survives alongside the verified mirror as 166
  extra files (~6 GB). Hashing shows 162 are exact duplicates of the verified
  mirror and 4 are truncated fragments — one per each of the four parallel
  streams Codex had been running, each smaller than its verified counterpart
  (e.g. 9.5 MB against 42.2 MB for `init22-cell3`). The residue carries no
  unique data. It is left in place pending an explicit decision rather than
  removed autonomously.
- **Instance stopped:** No scoring, alarm, or supervisor process was running.
  Instance 46677323 is `stopped`/`exited`, still exists, and retains its 200 GB
  disk. Every irreplaceable artifact — raw rollouts, score sidecars, features,
  predictors, and all receipts — is now verified off-instance.
- **Decision:** Next is the §8 causal protocol. Order deliberately puts the cheap
  gate first: build Calibration candidate states and count how many valid
  donor/recipient pairs exist under the frozen tolerances (phase, contact,
  gripper opening 0.01, eef and object position 2 cm, normalized time 0.10, all
  non-primary predicates, orientation difference 30-90 degrees) before spending
  GPU time. `PREREG.md:356` declares patching inconclusive rather than negative
  below 30 valid pairs, so the pair inventory can end the phase cheaply. Only
  then calibrate the intervention strength alpha over {0.25, 0.5, 1.0}, choosing
  the smallest value with the expected target-action sign and an off-manifold
  rate ≤ 5%. The natural activation distribution needed for that off-manifold
  reference is already present in the score sidecars and needs no new compute.

### 2026-08-05 20:18 CEST — CALIBRATION-CAUSAL-PAIRS-001: pair inventory passes the gate with room to spare

- **Purpose:** Phase 1 of the §8 causal protocol, deliberately run before any GPU
  work. `PREREG.md:356` declares patching *inconclusive rather than negative*
  below 30 valid pairs, so counting pairs first can end the causal phase cheaply.
  Implemented as `ops/calibration_pair_inventory_18d6494.py`. It chooses no
  alpha, patches nothing, and runs no model.
- **Result — the gate is wide open.** From the 9,455 scored Calibration states,
  **19,829 valid confirmatory donor/recipient pairs** exist under the frozen
  tolerances, against a required minimum of 30. 19,335 pairs (97.5%) join states
  from different episodes and 14,848 (74.9%) from different base-init clusters,
  so the supply is not an artifact of pairing neighbouring steps within one
  trajectory. 3,062 distinct states participate and all 20 clusters are
  represented. Orientation differences span the full admissible window: minimum
  30.0°, median 53.1°, maximum 90.0°.
- **Where the constraint actually binds:** states fall into four exact-match
  buckets — pregrasp/no-contact 5,105, transport/contact 2,951,
  pregrasp/contact 1,199, grasped/contact 200. Marginal pass rates on random
  pairs are 0.72% for the 2 cm end-effector tolerance, 7.8% for the 2 cm object
  tolerance, 23.5% for normalized time and 53.7% for the orientation window. The
  end-effector constraint is doing nearly all the work; the tolerances are not
  loose. The realised 19,829 is about 16x what independence would predict, which
  is expected because states along stereotyped trajectories are positively
  correlated in phase, time and pose.
- **Note on `symmetry_order = 2`:** the symmetry-aware orientation difference is
  folded into [0°, 90°], so the preregistered 30-90° confirmatory window covers
  the upper two thirds of the attainable range rather than a narrow slice.
- **Implementation note:** exhaustive pairing is O(n^2) (~44M comparisons) and is
  not tractable in pure Python, so eligibility is evaluated in two stages —
  bucket on the criteria that must match exactly, apply a vectorised numeric
  prefilter with a 1e-6 slack margin, then confirm every survivor with the frozen
  `pair_eligibility` itself. The prefilter can only admit a superset, so the
  reported counts are those of the frozen implementation. 18,118,786 same-bucket
  comparisons were made; the whole run takes about 4 seconds because `np.load` on
  an npz decompresses only accessed members, and the two 75 MB image arrays per
  episode are never touched.
- **Independent verification:** an independent model (Fable 5) re-derived the
  inventory from scratch with **no prefilter and no bucketing** — a full exact
  O(n^2) recount using an independent rotation-matrix quaternion path — and
  reproduced every figure to the last digit, including the orientation quantiles.
  It further confirmed that the empty exclusion dictionary is genuine rather than
  a bypass (the prefilter-versus-exact gap is exactly 0, and the closest
  same-bucket approach to a tolerance boundary is 1.5e-8), that the quaternion
  convention matches `construct_m1_raw_pose` to 4e-14 rad with eef and object not
  swapped, that positions are in metres so 0.02 is the 2 cm of `PREREG.md:349`,
  and that 275 brute-forced sample pairs show no false positives and, critically,
  no false negatives. Its one substantive finding — that the cohort digest was
  used only as a directory name and never verified against artifact bytes — was
  fixed by switching to the validated `load_feature_cohort`; the hardened script
  reproduces a byte-identical receipt.
- **Environment:** the Vast instance refused to start (`resources_unavailable`;
  the RTX 5090 is currently allocated elsewhere), so this CPU-only phase ran
  locally in a matched environment built with `uv` — python 3.12.11 and
  numpy 2.2.6 against the instance's 3.12.13 and 2.2.6. No GPU is required for
  a pair inventory, and every downstream quantity will be recomputed on the
  instance during alpha calibration.
- **Receipt:** `artifacts/calibration-pair-inventory-18d6494-001/pair-inventory.json`,
  SHA-256 `6e1659b4…d832133`, `locked_test_accessed=false`.
- **Decision:** Pair supply is not the limiting factor, so the causal phase
  proceeds. Next is alpha calibration over {0.25, 0.5, 1.0} on the GPU, choosing
  the smallest value with the expected target-action sign and an off-manifold
  rate ≤ 5%. Two implementation consequences follow from this inventory. First,
  `select_pairs` is itself O(n^2) over candidates and cannot be handed all 9,455
  states directly; the pairing stage will need the same bucketed treatment, with
  its selection semantics preserved exactly. Second, with 19,829 pairs available
  the 20-pairs-per-seed limit is not supply-constrained, so the three
  preregistered seeds will genuinely diversify rather than exhaust the pool.

### 2026-08-05 20:55 CEST — CALIBRATION-DEFECT-001: camera and object-pose counterfactuals are silent no-ops

- **STOP-THE-LINE FINDING.** While inspecting the existing intervention data as a
  free preview of the causal question, the M2 column
  `m2_object_probe_equivariance_error_mean_rad` turned out to be *constant* at
  0.2617993877991494 rad across all 6,211 finite rows — exactly 15.0 degrees,
  which is precisely the magnitude of the object-yaw transform itself. That is
  the algebraic signature of a probe prediction that does not move at all when
  the object is rotated.
- **Direct measurement over all 160 sidecars / 9,455 states:**

  | transform family | states | max abs delta activation | max abs delta action |
  |---|---|---|---|
  | brightness_0_85 | 9455 | 0.2539 | 2.107 |
  | brightness_1_15 | 9455 | 0.1953 | 2.098 |
  | camera_yaw_neg_3 | 9455 | **0** | **0** |
  | camera_yaw_pos_3 | 9455 | **0** | **0** |
  | object_yaw_neg_15 | 6836 | **0** | **0** |
  | object_yaw_pos_15 | 7042 | **0** | **0** |

  The camera and object-pose counterfactuals are bit-identical to the factual
  rollout. Photometric transforms work, and probe-shift interventions genuinely
  perturb activations (max 0.0625), so the sidecars are not globally frozen —
  only the geometric transforms are inert.
- **Root cause, located in code:** `libero_runtime.py:819` calls
  `_backend(self.wrapper)._get_observations()` **without `force_update=True`**.
  Under the pinned `robosuite==1.4.0` (`environment-gpu.freeze:146`),
  `_get_observations(force_update=False)` returns *cached* observable values that
  are refreshed only inside `step()`/`reset()`. The transform helpers
  (`apply_object_yaw`, `apply_camera_transform`) do write MuJoCo state and call
  `sim.forward()`, but nothing re-renders the cameras. The policy therefore
  receives a bit-identical observation, and a deterministic policy on the same
  noise draw reproduces the original actions and activations exactly. Robosuite's
  own docstring names this exact use case as requiring `force_update=True`.
  Brightness is unaffected because it edits the stored pixel arrays directly.
  Rollout-time initial conditions are unaffected because `reset()` runs settle
  `step()` calls that refresh observables. The unit tests missed it because
  `tests/test_libero_runtime.py:116` uses a fake backend whose
  `_get_observations()` regenerates observations statelessly.
- **Draw alignment ruled out as an explanation:** transformed draw *k* shares
  original draw *k*'s noise object by construction (`scoring.py:813-820`,
  `894-899`). Fingerprinting one state shows transformed family-4 draw 0 is
  bit-equal to original draw 0 while differing from draws 1-3 by 0.78-0.88, so
  the zeros are matched-draw bit-equality against mutually distinct draws, not a
  misalignment artifact.
- **Blast radius, verified column by column on the published cohort:**
  - **M0: 5 of 13 columns degenerate** — `m0_camera_action_drift_mean`/`_max`,
    `m0_object_action_drift_mean`/`_max` (constant 0), and
    `m0_camera_render_equivariance_error` (constant 0). The remaining 8 columns
    are healthy.
  - **M2: 2 of 8 increment columns degenerate** —
    `m2_object_probe_equivariance_error_mean_rad` (constant 15°) and
    `m2_camera_probe_circular_dispersion` (≤1.7e-16). Brightness dispersion,
    resultant norm, robust-z, flow-noise dispersion, controllability and
    specificity are healthy.
  - **M1 is unaffected**, as are the raw rollouts and the failure labels.
- **What this does to the headline:** the entire counterfactual-action-drift arm
  of M0 carried zero information, so M0's 0.7889 AUROC rests only on brightness
  and output-dispersion features. More damaging, the single M2 feature aimed most
  directly at the project's primary variable — relative planar **orientation** —
  was a constant. The Calibration read-out "M2 over M1 is 1.17%" and the
  lead-time null were therefore computed with M2's on-thesis features unable to
  contribute. **That comparison is not a fair test of internal geometry and must
  not be reported as one.** The earlier CALIBRATION-ALARM-001 and
  CALIBRATION-FEATURE-FINALIZATION-001 conclusions stand as arithmetic but are
  suspended as science pending re-scoring.
- **Not previously documented:** no entry in `log.md`, `AMENDMENTS.md` or
  `PREREG.md` acknowledges inert transforms, and `PREREG.md:208-216` explicitly
  presupposes that camera yaw produces output-equivariance error and object yaw
  produces counterfactual action drift. There is no legitimate protocol reason
  for exact zeros.
- **Why this is recoverable:** the Locked Test set has never been opened, the raw
  rollouts are intact and fully backed up, and the defect is confined to
  scoring-time counterfactual observation refresh. A corrected re-scoring before
  any Locked Test access stays inside the preregistered framework.
- **Decision:** Halt the causal phase. Do not calibrate alpha, do not freeze
  Calibration, and do not approach Locked Test on the current sidecars. The fix
  is a one-call change (`force_update=True`) in a file inside
  `SCORING_SOURCE_FILES`, so it necessarily invalidates `scoring_source_sha256`
  and every score sidecar bound to it — this requires an `AMENDMENTS.md` entry
  and a full re-scoring of all 160 episodes, followed by rebuilt features,
  refitted predictors, and repeated alarm calibration. That is a multi-hour GPU
  job and a deliberate protocol decision, so it is put to the user rather than
  taken autonomously.

### 2026-08-06 01:10 CEST — CALIBRATION-RESCORE-001: fix verified on hardware, re-scoring under way

- **The fix works.** The first re-scored episode
  (`libero_10-task5-calibration-init10-cell0`) shows every transform family
  moving the policy output, where the defective run showed exact zeros:

  | family | defective run | re-scored |
  |---|---|---|
  | brightness_0_85 | 0.2539 | 0.125 |
  | brightness_1_15 | 0.1953 | 0.125 |
  | camera_yaw_neg_3 | **0** | **0.5645** |
  | camera_yaw_pos_3 | **0** | **0.5898** |
  | object_yaw_neg_15 | **0** | **0.4883** |
  | object_yaw_pos_15 | **0** | **0.7188** |

  (max absolute activation delta; action deltas are ~2.05 for every family.) The
  geometric transforms now move activations *more* than the photometric ones,
  which is physically sensible: a 15° object rotation changes the rendered scene
  far more than a 15% brightness scale. Object-transform availability is 13/39
  and 12/39 states, consistent with the previously observed ~33% rejection rate,
  so the validity checks still bite. The inertness guard did not fire and the
  sidecar published normally.
- **Pre-run gates, all passed:**
  - Full test suite green *on the instance*, including the four torch-dependent
    modules that cannot run locally.
  - Two further defects surfaced at this gate and were fixed. First, a
    pre-existing platform bug: `wilson_interval(0, 10).lower` returned
    2.8e-17 instead of 0 on x86 while passing on ARM, so the endpoint is now
    pinned exactly. Second, and more seriously, the end-to-end scoring test
    mocked `apply_condition` as a no-op and its fake policy never read the
    images — it was structurally blind to the very defect being fixed. The new
    inertness guard caught it, which is the first evidence that the guard earns
    its keep.
  - Probe re-bound rather than refitted. The numerical probe reproduced
    **bit-identically** (`71a6bff1…`), as did the rollout allocation receipt
    (`7cd831bd…`), the four selection controls, the 1-SE candidate choice, and
    all 160 failure annotations (107 success / 53 failure). Exactly two recorded
    values changed — `configuration_sha256` and `scoring_source_sha256` — which
    are precisely the two that had to. This confirms the probe never depended on
    the defective counterfactuals. New bound probe: `e94269a1…`.
- **Commit topology:** `COLLECTION_COMMIT`/`COLLECTION_TAG` stay pinned at
  `18d6494`/`prereg-locked-v1` and continue to validate the authority receipt,
  the collection receipt and the manifest reconstruction. The new
  `calibration-rescore-v1` tag identifies the commit the scoring code runs at;
  the runner requires `HEAD` to equal it. Conflating the two would have broken
  manifest reconstruction and rejected every raw artifact.
- **Execution choice:** the serial runner, not the two-worker continuation
  machinery. Serial is slower but resumable and free of the cutover/signal
  recovery paths that previously cost a day of debugging. Robustness is worth
  more than the wall-clock saving here.
- **Open:** the observed first-episode time is longer than the pre-fix baseline
  suggested, so the wall-clock estimate is being measured rather than assumed
  before it is reported. Forced re-renders add roughly eleven extra two-camera
  renders per scored state, which is the expected source of any slowdown.
- **Decision:** let the run proceed. Every scored episode is validated before it
  is published, so a residual inert transform aborts within minutes instead of
  surviving the full run. Locked Test remains closed.

### 2026-08-06 19:40 CEST — CALIBRATION-RESCORE-002: re-scoring complete, first fair M0/M1/M2 read-out

- **All 160 episodes re-scored** with working counterfactuals. Two processes
  walked the manifest from both ends (83 forward, 77 reverse) and met at
  `init20-cell2`, where the reverse worker terminated on the exclusive-write
  guard exactly as intended — no overwrite, no data loss. The forward worker
  then validated the 77 foreign sidecars through the normal loader path and ran
  finalization. Zero errors across the run; the inertness guard never fired,
  i.e. all six transform families reached the policy input in every episode.
- **Every previously degenerate column now carries information:**

  | column | before | now (distinct / variance) |
  |---|---|---|
  | `m0_camera_action_drift_mean` | constant 0 | 9455 / 0.036 |
  | `m0_camera_action_drift_max` | constant 0 | 9455 / 0.090 |
  | `m0_object_action_drift_mean` | constant 0 | 6211 / 0.013 |
  | `m0_object_action_drift_max` | constant 0 | 6211 / 0.074 |
  | `m0_camera_render_equivariance_error` | constant 0 | 9455 / 0.037 |
  | `m2_object_probe_equivariance_error_mean_rad` | constant 15.0° | 6211 / 0.021 |
  | `m2_camera_probe_circular_dispersion` | ~1e-16 | 9455 / 0.0092 |

- **Corrected Calibration read-out** (group-5-fold OOF, 160 episodes / 9,455
  states; selected family **histogram gradient boosting**, learning rate 0.03,
  200 iterations, 7 leaves, min 20 samples per leaf):

  | model | AUROC | log loss |
  |---|---|---|
  | M0 | 0.8277 | 0.43347 |
  | M1 | 0.9366 | 0.24636 |
  | M2 | 0.9378 | 0.24537 |

  M1 over M0 is 43.16%, M2 over M0 is 43.39%, and **M2 over M1 is 0.40%**
  against the preregistered 3% threshold. Kill-Switch 1 is not triggered
  (M1 AUROC 0.9366 < 0.95).
- **The repair helped M0, not M2.** Against the defective run, M0 AUROC rose
  from 0.7889 to 0.8277 — the counterfactual action-drift arm was dead and now
  carries real signal — while the M2-over-M1 lift *fell* from 1.17% to 0.40%.
  The earlier conclusion is therefore not merely reproduced but strengthened,
  and this time it rests on a fair comparison in which M2's orientation features
  were able to contribute.
- **New substantive finding:** `m2_object_probe_equivariance_error_mean_rad` was
  previously pinned at exactly 15.0°, the full transform magnitude, meaning the
  probe prediction did not move at all under object rotation. Its median is now
  12.8° under the same 15° rotation, so the representation tracks roughly 15% of
  the actual rotation. The geometry is present but weak — which is a coherent
  mechanistic explanation for why M2 adds so little.
- **Model family changed** from logistic regression to histogram gradient
  boosting. This is protocol-conformant: selection is on M1 raw OOF log loss
  only, and with corrected features a different family wins.
- **Backup:** the full re-scored set is off-instance and independently verified —
  320 score files (589 MB) byte-identical, plus the feature root (68 MB) and the
  re-bind analysis artifacts. New digests: cohort `ef51efbb…`, predictors
  `47daa982…`, bound probe `e94269a1…`, score allocation `89dfae2f…`.
  `locked_test_accessed=false` throughout.
- **Instance stopped** at genuine idle; it still exists with its 200 GB disk.
- **Decision:** the defective Calibration numbers are formally superseded. Next,
  on CPU: alarm-threshold calibration and Lead Time, the causal pair inventory,
  and the preregistered coverage-feature sensitivity refit — all against the new
  cohort. Only the §8 alpha calibration needs the GPU again. Locked Test remains
  closed.

### 2026-08-06 20:15 CEST — CALIBRATION-RESCORE-003: alarm, pairs and leakage sensitivity on corrected data

- **Alarm thresholds and Lead Time, recomputed.** All frozen anchors reproduced
  against the new cohort before any threshold was computed. Thresholds land at a
  realised 9.35% episode FPR (10 of 107 successes) for all three models: M0
  0.5407, M1 0.4927, M2 0.5040.

  | comparison | median lead (baseline → M2) | median paired difference | 90% interval | detection rate |
  |---|---|---|---|---|
  | **M2 vs M1** (preregistered) | 435 → 435 | **0.0** | [0.0, 0.0] | 100% → 100% |
  | M2 vs M0 (secondary) | 405 → 435 | 60.0 | [45.0, 70.0] | 92.5% → 100% |

  The internal lead-time claim requires a median paired gain ≥ 5 control steps.
  Observed is again **exactly 0.0 with a degenerate interval**: with corrected
  counterfactuals, internal signals still raise the alarm on precisely the same
  control step as M1 in every failed episode. M0 improved (detection 83% → 92.5%,
  lead 395 → 405), which is the expected consequence of its drift features
  becoming informative.
- **Causal pair inventory reproduces exactly.** 19,829 eligible confirmatory
  pairs, 19,335 cross-episode, 14,848 cross-cluster, 3,062 participating states,
  orientation quantiles 30.0/53.1/90.0 — identical to the pre-fix inventory to
  every digit. This is the expected invariant: pairing depends only on factual
  simulator state, which the scoring defect never touched. It doubles as a
  cross-check that the re-scored cohort carries the same episodes and rows.
- **Leakage sensitivity refit — and a correction to an earlier claim.** Dropping
  the four label-informed columns (three shared by M1/M2, one M2-only):

  | model | with the columns | without |
  |---|---|---|
  | M0 | 0.8277 / 0.43347 | 0.8277 / 0.43347 (unchanged, as it has none) |
  | M1 | 0.9366 / 0.24636 | 0.9335 / 0.25320 |
  | M2 | 0.9378 / 0.24537 | 0.9338 / 0.25017 |

  M2 over M1 moves from **0.40% to 1.20%**; M1 over M0 from 43.17% to 41.59%.
  The earlier amendment reasoned that the residual bias favoured M2, making an
  M2 null conservative. **That reasoning was wrong and is corrected here:**
  removing the leak-prone features *increases* M2's margin. The three shared
  coverage features are strongly predictive, so while present they absorb
  variance that M2's increment might otherwise explain. The direction of the
  bias is therefore against M2, not for it. Both figures nonetheless fall well
  short of the preregistered 3% threshold, so the conclusion is unchanged while
  the stated reasoning behind it is not.
- **Convergent read-out on corrected data:** predictive lift M2 over M1 is 0.40%
  (1.20% without leak-prone features) against a 3% threshold, and lead-time gain
  is 0.0 steps against a 5-step threshold. Both preregistered internal-signal
  criteria fail on Calibration, now measured with M2's orientation features
  fully functional. The large gains — 43.4% log loss, 60 steps of lead — are
  M2 over M0, i.e. internal signals substituting for privileged simulator state
  rather than exceeding it.
- **Receipts:** alarm `9732b235…`, pair inventory unchanged at `6e1659b4…`
  content, leakage sensitivity written to
  `artifacts/calibration-leakage-sensitivity-001/`. All carry
  `locked_test_accessed=false`.
- **Decision:** Calibration-side prediction work is complete. Remaining before
  the freeze: the §8 alpha calibration, which is the only step still needing the
  GPU. Locked Test stays closed.

### 2026-08-06 20:40 CEST — CALIBRATION-ALPHA-001: off-manifold half of the alpha calibration passes

- **Design:** `PREREG.md:345-347` picks alpha from {0.25, 0.5, 1.0} as the
  smallest value with the expected target-action sign *and* an off-manifold rate
  ≤ 5%. The two conditions need different resources, so the off-manifold half —
  pure geometry on the patched activation vector — was evaluated first on CPU.
  An alpha failing here is disqualified regardless of its action effect, so this
  ordering can retire the whole GPU stage for free.
- **Pairs:** `select_pairs_for_three_seeds` returned the full 60 attempted pairs
  (20 per seed over three deterministic seeds), consistent with the 19,829
  eligible edges in the inventory — the seeds diversify rather than exhaust.
- **Result — every alpha passes, none is off-manifold:**

  | alpha | off-manifold rate | median patched 5-NN distance |
  |---|---|---|
  | 0.25 | 0.0% | 2.697 |
  | 0.5 | 0.0% | 2.719 |
  | 1.0 | 0.0% | 2.786 |

  The natural 95th percentile is 3.891, so every patched activation sits well
  inside the natural distribution. The frozen ≤5% constraint is satisfied by all
  three strengths, and the preregistered preference for the smallest makes
  **alpha = 0.25** the candidate, pending the sign condition.
- **Substantive observation:** the patched 5-NN distance barely responds to
  alpha — 2.697 to 2.786 across a fourfold increase in strength. The intervention
  therefore moves the activation very little in the metric that matters for
  naturalness. Read together with the weak probe decoding (~0.25 rad) and the
  15%-of-rotation equivariance tracking, this is the same picture from a third
  angle: the probe subspace carries a real but small share of the representation.
  It also means the off-manifold criterion is not the binding constraint here,
  contrary to the pre-run expectation that flow-matching models would resist
  linear patching.
- **Decision:** proceed to the GPU sign stage. It must apply
  `h_r' = h_r + alpha·P(h_d − h_r)` at the recipient state, run a patched forward
  pass, and check that the yaw action moves in the donor-aligned direction. Only
  then is alpha frozen. Locked Test remains closed.

### 2026-08-07 05:05 CEST — CALIBRATION-ALPHA-002: sign stage blocked by the adapter's intervention contract

- **Status: not completed.** Three overnight attempts at the GPU sign stage each
  failed for a different reason. The off-manifold half remains done and passing;
  only the sign condition is outstanding, so alpha is not yet frozen.
- **Defect 1 — instrumentation lifetime.** `patch()` refuses to run while the
  hooks are uninstalled. An outer `with instrumentation` block does not survive,
  because `SmolVLAScoringAdapter.predict_action_chunk` itself enters
  `self.instrumentation` as a context manager (`scoring_runtime.py:1044-1048`)
  and its `__exit__` calls `remove()`. Every adapter inference therefore
  uninstalls the hooks behind it. Fixed by re-installing immediately before each
  patch context.
- **Defect 2 — candidate name is not a hook name.** The probe candidate
  `early_expert_t1_0` encodes hook *and* flow step together, while
  `instrumentation.patch` expects the hook alone. Fixed by resolving through the
  frozen `candidate_target` map (`early_expert_t1_0 -> expert_layer_4`, step 0),
  the same mapping the scoring adapter uses, so the patch lands where the probe
  was fitted.
- **Defect 3 — structural, still open.** With both fixed, the patch is applied
  and the forward runs, but the adapter aborts with
  `selected activation patch marker disagrees` (`scoring_runtime.py:951`). The
  adapter validates that its own `patched` flag matches the observed patch
  marker, and that flag is driven solely by `intervention_degrees`, i.e. the
  frozen ±10° probe-shift intervention. An externally supplied donor patch is
  therefore rejected by design. This is a guard behaving correctly, not a bug:
  the scoring adapter was built for the M2 controllability feature, not for
  Section 10 donor patching.
- **Consequence:** the sign stage needs its own inference path that applies the
  patch and reads the resulting action chunk without the adapter's
  intervention-marker contract, while still reusing the frozen preprocessing,
  noise draws and postprocessing so the actions stay comparable to the recorded
  ones. That is a deliberate piece of work, not another quick retry.
- **Process failures worth recording.** (1) A monitor reported a crashed run as
  "alive" for 2.5 hours because its `pgrep` pattern matched its own wrapper
  process; the run had died in the first minute and the GPU idled. The same
  mistake had already occurred once earlier in the project. Monitors now capture
  the PID at launch and check that PID. (2) The O(n^2) pair selection (~35 min
  per attempt) was flagged in this log as needing bucketing and was then not
  implemented, so every failed attempt paid it in full. (3) Testing the patch
  mechanics in isolation — which finally exposed defect 3 in two minutes —
  should have come before any full run.
- **Instance stopped.** All artifacts remain backed up off-instance and verified.
  Locked Test remains closed and unaccessed.

### 2026-08-07 09:00 CEST — CALIBRATION-ALPHA-003: alpha frozen at 0.25; patching shows no specific effect

- **Calibration is now complete.** The sign stage ran over all 60 preregistered
  pairs (20 per seed, three deterministic seeds) at the frozen probe location
  `early_expert_t1_0` → `expert_layer_4`, flow step 0.
- **Structural blocker resolved.** `SmolVLAScoringAdapter.predict_action_chunk`
  validates that its internal `patched` flag matches the observed patch marker,
  and that flag is driven solely by its frozen ±10° probe-shift intervention, so
  a Section 10 donor patch is rejected by design. The runner now reproduces the
  adapter's inference core exactly — same `inference_mode`, instrumentation
  scope, defensive batch clone and postprocessor — and reads only the action
  chunk, which is all the sign condition needs. Verified in isolation before the
  full run: the baseline forward is bit-deterministic across repeats and all
  three alphas patch cleanly. Two further defects surfaced in that isolated test
  within two minutes each: the candidate name encodes hook *and* flow step and
  must be resolved through `candidate_target`, and the policy emits 50 actions
  while sidecars and `summarize_action_effect` use the first `CHUNK_ACTIONS`.
- **Alpha selection.** All three alphas satisfy the off-manifold constraint
  (0.0% each) and all three exceed a 50% sign rate as point estimates, so the
  preregistered rule — smallest qualifying value — freezes **alpha = 0.25**.
- **But the effect is indistinguishable from chance:**

  | alpha | sign correct | two-sided binomial p | 90% interval |
  |---|---|---|---|
  | 0.25 | 34/60 = 56.7% | 0.366 | [45.2%, 67.6%] |
  | 0.5 | 36/60 = 60.0% | 0.155 | [48.6%, 70.7%] |
  | 1.0 | 31/60 = 51.7% | 0.897 | [40.3%, 62.9%] |

  Every interval contains 50%. `PREREG.md:369-374` requires, for the
  confirmatory claim, "sign correctness exceeds 50% with a 90% cluster interval
  above 50%" — on Calibration no alpha comes close to that bar.
- **Specificity is failed by more than an order of magnitude.** Median
  off-target ratio is 4.94 / 3.63 / 5.51 for the three alphas, against a
  preregistered bound of 0.25. The patch moves non-yaw action dimensions three
  to five times *more* than the targeted yaw axis. Median donor-aligned target
  effects are 1.6e-05 to 2.1e-04, i.e. numerically negligible.
- **Interpretation, stated carefully.** These are Calibration values whose
  preregistered purpose is to *select* the patch strength, not to decide the
  causal claim; the confirmatory test belongs to Locked Test. Alpha is duly
  frozen at 0.25 and the protocol is satisfied. But the preview is unambiguous:
  a patch along the frozen probe subspace neither moves the target action
  reliably in the donor-aligned direction nor does so specifically. Combined
  with the weak decoding (~0.25 rad), the ~15% equivariance tracking, the
  near-zero movement on the activation manifold, and the absent predictive and
  lead-time lift over M1, five independent measurements now point the same way:
  the probe subspace carries a real but small share of the representation, and
  the policy's behaviour does not hinge on it in a way this intervention can
  demonstrate.
- **Receipts:** `alpha-sign.json` and `alpha-off-manifold.json`, both copied
  off-instance. The pair selection is cached at
  `alpha-pairs-cache.json` on the instance so it never costs 35 minutes again.
  `locked_test_accessed=false` throughout. Instance stopped.
- **Decision:** Calibration is closed. What remains is the tracked freeze plus
  the `calibration-locked-v1` tag, and then the Locked Test — which stays shut
  until the user explicitly opens it.

### 2026-08-07 09:40 CEST — CALIBRATION-FREEZE-001: Calibration frozen, Locked Test authorized but not opened

- **`locks/calibration_frozen.json` is committed and `calibration-locked-v1` is
  tagged at that commit.** `assert_locked_test_ready` accepts it: repository at
  `a02b29d`, tag exactly at HEAD, worktree clean including untracked files, all
  nine required fields present, every referenced artifact git-tracked with
  matching bytes. Locked Test is now *authorized* — and remains *unopened*.
- **Nothing was chosen at freeze time.** Every value is read from artifacts that
  already existed. The only quantity computed here is the Brier score, which the
  guard requires and no earlier step produced. It is derived from out-of-fold
  probabilities reconstructed deterministically and bound to the frozen
  calibration data hash `8713c4cb…` and the recorded per-model OOF log losses;
  a mismatch aborts rather than reporting numbers other than those the
  predictors were selected on. Both anchors reproduced exactly.
- **The Calibration manifest is now tracked.** The guard requires each referenced
  artifact to be a git-tracked file. The manifest previously existed only on the
  instance; it is deterministically reconstructible and reproduces its frozen
  digest `6f5c7a5b…` byte for byte, so no GPU was needed to recover it.
- **Frozen contents:**

  | field | value |
  |---|---|
  | representation probe | `early_expert_t1_0`, ridge alpha 1.0 |
  | predictor | histogram gradient boosting (lr 0.03, 200 iter, 7 leaves, min 20) |
  | alarm thresholds | M0 0.5407, M1 0.4927, M2 0.5040 |
  | patch strength | 0.25 |
  | Calibration metrics | M0 0.4335 / 0.1946 / 0.8277, M1 0.2464 / 0.0994 / 0.9366, M2 0.2454 / 0.0992 / 0.9378 (log loss / Brier / AUROC) |

- **Note on `predictor.coefficient_hash`:** the selected family is histogram
  gradient boosting, which has no coefficient vector, so the field carries the
  frozen predictor-metadata digest `47daa982…`, which identifies the fitted
  bundle exactly. The guard requires a SHA-256; it does not require it to be a
  literal coefficient hash.
- **Locked Test cost, for the record.** The Locked Test rollouts do not exist:
  the raw set contains only 160 Calibration and 40 Discovery episodes, which is
  the intended state. Opening Locked Test therefore means collecting 160 fresh
  rollouts (~5-10 GPU hours) and scoring them (~11-18 GPU hours), i.e. one to
  two days of GPU, before any primary number exists.
- **Decision:** stop here. The freeze is the last reversible step; opening the
  Locked Test is not. It stays closed until the user explicitly authorizes it.

## 2026-08-25 — BLOCKER: Locked Test dry-run cannot start (instance destroyed, negative balance)

- **State:** Amendments 1-3+5 and addendum-4 terms approved (AMENDMENTS.md
  2026-08-25); tooling + runbook committed; Locked Test manifest reconstructed
  and its digest reproduces the frozen 1fd8c818... byte for byte; authority
  written (locked_test_accessed: True); tags calibration-locked-v1 and
  locked-test-score-v1 at HEAD 9daa4df; freeze payload and four artifacts
  re-verified byte-identical. Locked Test raw set does not exist.
- **Blocker:** 1) Vast instance 46677323 (the only instance; storage "never
  delete") reports "not found or no longer exists" — the disk, /venv/main,
  /workspace/hf-cache, runstate and research-artifacts live copy are gone.
  2) Account balance is -$4.28 with the billing threshold enabled; recent card
  charges show requires_action/failed, so no instance can be rented or started.
- **Freeze protection applied:** deviation from the runbook's Step 1 (expected
  existing instance, disk preserved) -> STOP, report, do not continue. No
  Locked Test rollout, no scoring, no improvised re-provisioning.
- **Record intact:** all frozen artifacts are mirrored and byte-verified in
  this repo; the Calibration raw set has the off-instance backup marker
  (artifacts/raw-backup-ready/.complete). The pinned environment recipe
  (environment-gpu.freeze, LeRobot v0.6.0, hf-libero 0.1.4, policy snapshot
  31d453f7...) is versioned.
- **Resume path (requires human action, in order):** (1) restore billing
  (add credits / fix card); (2) provision a new 1x RTX 5090 instance; (3)
  rebuild the environment per environment-gpu.freeze; (4) restore/verify the
  raw calibration artifacts from the off-instance backup (byte hashes against
  the freeze manifest); (5) rerun runbook Step 0-2 (including dry-run on 2-3
  episodes) before any full collection. Locked Test stays closed until then.

### 2026-09-01 10:27 CEST — LOCKED-READINESS-002: execution repaired; prospective evidence gate remains NO-GO

- **Stage:** Locked Test, strictly pre-access
- **Question:** Will the complete Locked Test collection, frozen scoring and
  fixed-order evaluation execute from the committed freeze without path drift,
  refitting on protected labels, silent resume corruption or a late discovery
  that mandatory evidence cannot be produced?
- **Pre-state / commit:** `50a955c2bf6a2156f04dfcbe9f7275defd9ccf2b`;
  no Locked Test raw directory, score sidecar, prediction or causal result
  existed. `mein_verständnis.md` was unrelated user material and was excluded
  locally from Git status without reading, editing, staging or deleting it.
- **Method:** Performed a static and executable audit of collection, scoring,
  freeze, evaluation and runbook paths. Added direct fail-closed operational
  tests; reconstructed the Calibration freeze from its original cohort; loaded
  the real bound probe, full-Calibration reference and predictor bundle; queried
  Vast read-only with raw JSON; ran the post-score capability check on the real
  frozen artifacts; then ran the complete isolated Python 3.12 contract suite.
- **Inputs and controls:** Frozen manifest `1fd8c818…`; collection commit
  `18d64941…`; policy `31d453f7…`; bound probe `e94269a1…`; feature reference
  `4441c760…` (9,455 Calibration states, 160 source episodes); predictor pickle
  `2b41854a…` plus canonical metadata `47daa982…`; Calibration data
  `8713c4cb…`. No GPU model, simulator, Locked Test outcome or protected score
  was loaded. Predictor inference tests expose matrices and names but no label
  argument, and AST/contracts reject all fit/selection/calibration calls.
- **Results:**
  - Collection is now self-contained and resumable with a single global lock,
    exact tracked executable/environment bytes, machine-readable RTX
    5090/CUDA/offline-snapshot/disk preflight, immutable cell validation,
    staging preservation and content-bound completion receipt.
  - Scoring now uses `build_locked_test_features`, accepts the Calibration probe
    only through explicit cross-split compatibility, applies the committed
    all-Calibration M0/M1/M2 predictor and Platt calibrators without a label
    argument, excludes 0–2 invalid resets per 20-episode condition cell, and
    content-addresses allocation, cohort, predictions and summary. Three invalid
    resets in one cell, stale sidecars, extra raw/score paths and orphan staging
    all abort.
  - The freeze now names all executable dependencies: bound probe, both
    reference-bundle files, predictor metadata/pickle, probe, Reality Gate and
    Calibration manifest. It reproduces byte-for-byte at SHA-256 `52412cfb…`.
    The previously row-weighted descriptive Brier values were corrected to the
    §9 episode-total-one estimand: M0 `0.137469593953084`, M1
    `0.06772738580612056`, M2 `0.0676011578878377`; log loss, AUROC and every
    frozen selection reproduced unchanged.
  - The fixed-order evaluator verifies 160 raw artifacts, allowed invalid-reset
    exclusions, prediction/raw/freeze hashes, M2-vs-M1 and M2-vs-M0 paired
    comparisons, Brier/AUROC intervals, lead time, condition rankings, costs and
    immutable causal/sensitivity receipts before publishing one report.
  - The real pre-access capability gate correctly returned nonzero for three
    missing preregistered evidence inputs: (1) amendment 9a requires an object-
    position decoder and all-object pose trace, but the freeze has only the
    circular `theta_rel` probe and primary-object state; (2) §10's two-of-three
    supporting-layer claim lacks executable frozen non-selected-layer probe
    coefficients; (3) the feature reference lacks the natural Calibration
    activation matrix required for the activation-space 5-NN off-manifold
    threshold. No angular/coverage proxy or post-access refit was substituted.
  - Verification: 446 tests collected; 443 passed and 3 optional-runtime tests
    skipped. The new/affected focused suite also passed after import formatting;
    byte-diff checks and Python compilation passed. A no-hardlink clean clone at
    `7a0c6411…` reproduced all three frozen file hashes, ran all five operational
    CLIs by absolute path with caller `PYTHONPATH` removed, passed the same full
    suite, and remained Git-clean. Both authorization tags deliberately remained
    at `50a955c2…`. The authenticated Vast state is still `instances=[]`, balance
    `-4.275898916619454` with the negative-balance threshold enabled.
- **Interpretation:** The two earlier operational failure modes are removed from
  Collection and frozen Scoring. The study as a whole is nevertheless not safe
  to open: the prospective gate proves that the already-required causal and
  sensitivity report cannot be computed honestly from the old freeze. This is
  a scientific-input blocker, not a software exception to bypass.
- **Confidence:** high. Every positive readiness claim has direct synthetic or
  real-artifact execution evidence; the NO-GO is reproduced on the exact frozen
  bytes before protected access. Remote GPU behavior remains untested because no
  instance exists, which is why the remote preflight remains mandatory.
- **Decision:** Keep Locked Test closed. Do not move `calibration-locked-v1` or
  `locked-test-score-v1` from `50a955c…`, do not provision/collect even after
  billing is restored, and do not invent proxy sensitivity results. Commit and
  push the repair/audit record without authorization tags.
- **Next step:** Study owner must prospectively choose whether to (a) approve a
  pre-access amendment that collects/freezes the missing Calibration position,
  supporting-layer and natural-activation evidence, or (b) amend the report to
  mark the impossible secondary diagnostics unavailable and remove the unsupported
  multi-layer claim. Separately, restore Vast billing and provision/rebuild one
  RTX 5090 only after that scientific choice is resolved; rerun runbook §0 and
  §5.0 before section 1.
- **Artifacts:** `AMENDMENTS.md` 2026-09-01 entry; `ops/locked_test_runbook.md`;
  collection/scoring/evaluation/postscore scripts and their direct tests;
  expanded `locks/calibration_frozen.json`; tracked feature-reference arrays and
  metadata (to be included in the implementing commit).
- **Compute / cost:** Local CPU-only verification and package-cache use; no GPU
  seconds, no simulator episodes and no new Vast charge.

### 2026-09-01 12:15 CEST — LOCKED-READINESS-003: complete path repaired and locally verified

- **Stage:** Locked Test, strictly pre-access
- **Question:** After choosing conservative unavailable states for measurements
  that were never frozen, is every remaining mandatory collection, scoring,
  causal, sensitivity and evaluation step executable and fail-closed before the
  protected run begins?
- **Pre-state / commit:** Prospective amendment
  `35674dd0c37d8b83a9cfb57ac87f6290e6aa36bb`; no Locked Test raw directory,
  score sidecar, prediction, activation, pairing, intervention, causal receipt or
  outcome existed or was inspected. The unrelated local
  `mein_verständnis.md` remained excluded without reading or modification.
- **Method:** Materialized the selected-layer natural-activation reference solely
  from the existing 160-episode Calibration rescore cohort; required exact raw,
  sidecar, BoundProbe, feature-reference, row-order, candidate-width and source
  hash links; averaged all eight preregistered factual draws in float64; froze its
  natural leave-self-out five-neighbor distance distribution. Implemented a
  separate content-addressed causal producer and sensitivity producer, then made
  the final evaluator reload every pair-evidence file and recompute summaries.
  Exercised tampering, interrupted/resumed random-control chunks, insufficient
  donor availability, zero-yaw ratios, empty difficulty cells and the exact
  conservative unavailable markers. Rebuilt the freeze with the new artifact
  bindings and ran the complete isolated Python 3.13 suite plus Ruff and byte-diff
  checks.
- **Inputs and controls:** Frozen Locked manifest `1fd8c818…`; selected probe
  `early_expert_t1_0`; BoundProbe `e94269a1…`; Calibration score allocation
  `89dfae2f…`; feature reference `4441c760…`; activation reference
  `cb210e825…` (9,455 states × 720 dimensions, 160 episodes); metadata
  `b7662e62…`; arrays `574bad7f…`; natural-distance p95
  `3.890758912438606`; selected alpha 0.25; three pairing seeds; 20 registered
  slots per seed; 1,000 norm-matched random two-dimensional subspaces per valid
  pair. No labels, fitting, model selection or new Calibration collection entered
  the activation-reference build.
- **Results:**
  - The activation-reference artifact is standalone, content-addressed and bound
    by both file hashes in the canonical Calibration freeze. The updated builder
    reproduces the freeze byte-for-byte at SHA-256
    `eb39e6952ad8864c8f9ae88a07f382b0efcbe18fd36a1221b67fe9f59106bed9`;
    the frozen scoring-source digest remains `9452c066…`.
  - Causal execution now streams random-subspace shifts in O(d) working memory,
    checkpoints every 25 controls, validates deterministic replay/noise/queue
    state, preserves resumable content-addressed evidence and reports fewer than
    30 eligible confirmatory pairs as inconclusive instead of crashing.
  - Sensitivity always emits the complete alpha {0.5, 1.0} × eight-cell grid.
    Cells with no valid pair remain explicit unavailable rows. Position-trace,
    supporting-layer and patched closed-loop outcome evidence use only the exact
    prospectively approved unavailable states; no proxy can make the overall
    mechanistic claim positive.
  - Producer-to-evaluator integration passed with all 60 evidence slots and
    1,000 controls per valid pair, including invalid donor slots and empty dose
    cells. The full repository suite passed: 473 tests, 3 optional-runtime skips
    and 3 passing subtests. Ruff, `py_compile` and `git diff --check` also passed.
  - The executable/scientific implementation is commit
    `1150570b4250278c1c4f492cd2944be0c561a439`. The authenticated external state
    remains `instances=[]` and balance `-4.275898916619454`; no GPU run was
    attempted.
- **Interpretation:** The known code, schema, memory, resume and missing-evidence
  failure modes are closed locally. This establishes software start-readiness,
  not a preferred scientific result and not proof of the still-unrun remote CUDA
  environment. The mandatory remote preflight remains the final fail-closed test.
- **Confidence:** high for local executable contracts and artifact integrity;
  moderate for remote runtime readiness until the exact replacement RTX 5090
  environment exists and passes the machine-readable preflight.
- **Decision:** The Locked Test remains unopened. After the record commit, move
  both authorization tags only to the final clean verified commit and validate a
  no-hardlink clone. Operational execution may begin only after the user restores
  Vast billing/provisions the replacement instance and the remote preflight is
  green.
- **Next step:** Commit this record, run the clean-clone verification from the
  final tagged commit, then wait for the external Vast prerequisite. Do not
  provision, collect or evaluate protected data while that prerequisite is
  absent.
- **Compute / cost:** Local CPU-only validation and package-cache use; no GPU
  seconds, simulator episodes, Locked Test reads or new Vast charges.
