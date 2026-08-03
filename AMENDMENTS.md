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
- **Implementing commit:** `PENDING_THIS_COMMIT`
