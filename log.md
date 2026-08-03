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
