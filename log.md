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
