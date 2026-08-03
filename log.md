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
