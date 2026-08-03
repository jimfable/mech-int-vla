# Does Internal Geometry Break Before Robot Behavior Does?

This repository tests whether pre-registered white-box geometry and controllability
signals improve held-out prediction of closed-loop failures for a public SmolVLA
policy on LIBERO, beyond a strong simulator-privileged non-internal baseline.

The primary estimand is the paired locked-test log-loss difference between:

- **M1:** output-only counterfactual/uncertainty features plus privileged
  simulator-state coverage; and
- **M2:** M1 plus internal geometry-consistency and controllability features.

The frozen project rationale is in [`start.md`](start.md). The executable protocol
is in [`PREREG.md`](PREREG.md). Changes to the protocol are recorded only in
[`AMENDMENTS.md`](AMENDMENTS.md), and all work/results are recorded in
[`log.md`](log.md).

## Current status

The exact CUDA runtime and checkpoint pass strict offline loading, the five causal
capture sites pass synthetic tests, and the deterministic single-episode executor
is implemented. It stores both raw camera streams losslessly, captures activations
at the frozen five-step cadence, and performs the one required fresh-runtime retry
after an invalid reset without running the policy. Safe artifact ingestion,
circular-probe selection, shared M0/M1/M2 predictor fitting, causal matching and
patch evaluation, and confirmatory statistical evaluation are implemented and
tested against synthetic data. Deterministic factual replay, the six frozen
counterfactual score transforms, per-draw probe interventions, exact numerical
M0/M1/M2 reductions, leakage-safe coverage features, and task-specific failure
event annotation are also executable and content-addressed. The lock guards verify
the checkpoint and frozen analysis artifacts by content and require exact Locked
Test coverage. The first
simulator construction exposed a separately distributed LIBERO asset dependency
before an initial state was loaded; its exact snapshot is pinned and
content-verified. No successful simulator reset, policy action, success label,
probe result, or intervention result has yet been observed. The Vast instance is
deliberately stopped between concrete jobs to avoid idle GPU charges.

## Reproducibility contract

- Model, dataset, LeRobot, and LIBERO revisions are immutable inputs recorded in
  `environment.lock`.
- Discovery, Calibration, and Locked Test groups never share base initialization
  IDs.
- Calibration chooses one representation/probe, one failure-predictor family,
  alarm thresholds, and intervention strength.
- Locked Test access is guarded in code and occurs only after a calibration-freeze
  commit.
- Frames are never treated as independent experimental units. Episode-level
  summaries are primary and uncertainty is clustered by base initialization.

## Verification and rollout entry points

Install the lightweight local package and run the synthetic contract suite:

```bash
python -m pip install -e '.[test]'
pytest
```

On the pinned GPU runtime, exact snapshots default to network-free resolution:

```bash
python -m mech_int_vla.runtime_cli snapshots \
  --environment-lock environment.lock --cache-dir /workspace/hf-cache
python -m mech_int_vla.runtime_cli load-policy \
  --environment-lock environment.lock --cache-dir /workspace/hf-cache
```

Discovery execution is intentionally split into a reset-only compatibility check
and an atomic full rollout. Both select a cell from the committed manifest; the
full rollout additionally refuses a dirty worktree or an existing artifact:

```bash
python -m mech_int_vla.runtime_cli discovery-reset \
  --repo-root "$PWD" --task-rank 1 --init-id 0 --condition-index 0
python -m mech_int_vla.runtime_cli discovery-rollout \
  --repo-root "$PWD" --environment-lock environment.lock \
  --cache-dir /workspace/hf-cache --task-rank 1 --init-id 0 --condition-index 0
```

Large raw episode arrays are excluded from Git. Content-addressed input manifests
and verification summaries live under `artifacts/manifests/`; completed run
metadata records the exact code, policy, task, condition, seeds, validity, and
terminal outcome alongside `trajectory.npz`.
