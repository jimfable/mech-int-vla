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

Pre-rollout setup and preregistration. No policy rollout, success label, probe
result, or intervention result had been observed when the initial preregistration
was written.

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
