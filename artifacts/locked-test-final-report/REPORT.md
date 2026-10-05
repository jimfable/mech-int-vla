# Locked Test — final report (single preregistered evaluation, 2026-09-17)

Report JSON: `report.json` (SHA-256 `f7d51bf8da4fae2107e7be73113c87825da66c9adbc65726e7731a9b0fc8c814`),
produced once by `ops/locked_test_evaluate.py` at commit `7edc069b…` on Vast instance
51243106 (1× RTX 5090, `/venv/main`, Python 3.12.13) from: manifest `1fd8c818…`,
prediction receipt `86d85ed1…`, Calibration freeze `eb39e695…`, reality-gate lock
`4e0d4d5c…`, causal receipt `5fc67e65…`, sensitivity receipt `e6dee93e…`, cost receipt
`b2547d51…`. Bootstrap: 10,000 replicates, seed 260803, init-ID clusters, 90 % intervals.
No model was trained or selected on Locked Test data.

## 1. Data integrity
160/160 artifacts; provenance code commit `18d64941…`, policy `31d453f7…`, task 5,
split `locked_test`. Two invalid resets (`init47-cell5`, `init48-cell5`, both in cell 5
"planar_diagonal_4_5cm", rate 0.10 — at the ≤ 10 % limit, cell retained). All other
envelope rates 0. **158 valid episodes** enter every estimand.

## 2. Primary estimand — paired log loss, M2 vs M1
| | M1 (privileged state) | M2 (internals) |
|---|---|---|
| log loss | 0.5197 | 0.5173 |
| Brier | 0.1616 | 0.1615 |
| AUROC | 0.8317 | 0.8358 |

Δ log loss (M2 − M1) = **−0.0024**, relative lift **+0.47 %**, 90 % CI on the lift **[−1.2 %, +2.1 %]** (Δ log loss CI [−0.0111, +0.0063]; corrected 2026-09-17 21:50 CEST — an earlier version misread the absolute interval as percent)
(20 clusters, 158 episodes). Preregistered bar ≥ 3 % lift: **not met** (`primary_claim_succeeds: false`).

## 3. Brier / AUROC per model (90 % CI)
| model | AUROC | Brier | log loss |
|---|---|---|---|
| M0 (vision baseline) | 0.667 [0.626, 0.710] | 0.219 [0.185, 0.256] | 0.637 [0.553, 0.727] |
| M1 | 0.831 [0.786, 0.876] | 0.162 [0.123, 0.205] | 0.520 [0.404, 0.649] |
| M2 | 0.835 [0.791, 0.879] | 0.162 [0.123, 0.204] | 0.517 [0.404, 0.643] |

## 4. M2 vs M0 (substitution ceiling)
Δ log loss = **−0.120**, relative lift **+18.8 %**, 90 % CI [−0.187, −0.052] → **succeeds**.
Note (added 2026-09-18): M2 ⊃ M1, so this measures privileged state plus internals over outputs; whether internals alone could replace the state ("outputs + internals") was not among the preregistered models and is untested.

## 5. Lead time at 10 % episode-FPR (Calibration-frozen thresholds M0 0.5407, M1 0.4927, M2 0.5040)
60 failed episodes; detection rate 98.3 % for both M1 and M2; median lead M1 435 steps,
M2 422.5 steps; **median paired difference 0.0, 90 % CI [0, 0]** → bar ≥ 5 steps **not met**.

## 6. Condition rankings (AUROC / log loss; M0 → M1 → M2)
| cell | condition | M0 | M1 | M2 |
|---|---|---|---|---|
| 0 | iid | 0.504 / 0.716 | 0.694 / 0.667 | 0.712 / 0.663 |
| 1 | yaw −37.5° | 0.713 / 0.537 | 0.780 / 0.416 | 0.763 / 0.438 |
| 2 | yaw −22.5° | 0.566 / 0.492 | 0.696 / 0.439 | 0.706 / 0.437 |
| 3 | yaw +22.5° | 0.725 / 0.635 | 0.933 / 0.487 | 0.951 / 0.456 |
| 4 | yaw +37.5° | 0.801 / 0.590 | 0.976 / 0.210 | 0.980 / 0.204 |
| 5 | planar diagonal 4.5 cm | 0.665 / 0.821 | 0.858 / 0.700 | 0.860 / 0.708 |
| 6 | camera yaw −5° | 0.602 / 0.404 | 0.665 / 0.323 | 0.658 / 0.316 |
| 7 | camera yaw +5° | 0.514 / 0.921 | 0.732 / 0.934 | 0.729 / 0.935 |

## 7. Causal patching (selected layer `early_expert_t1_0`, frozen α = 0.25)
60 pairs attempted, **52 valid** (≥ 30, so not inconclusive). Sign-correct 25/52 =
**48.1 %**, 90 % CI [35.6 %, 62.2 %] → `sign_passes: false`. Median donor-aligned target
effect −1.5e−4 versus random-control 95th percentile 1.0e−4; **median off-target ratio
4.03** (bar ≤ 0.25) → `specificity_passes: false`; `random_control_passes: false`;
matched-donor (<5°) control sign rate 0.538; off-manifold rate 0.596; positive seeds 1/3
(`seed_stability_passes: false`). Supporting layers: `unavailable`
(`frozen_supporting_layer_coefficients_absent`) → confirmatory multi-layer claim
`unsupported`, `succeeds: false` (as prospectively fixed).

## 8. Cost accounting (operator-supplied stage receipt)
| stage | wall | GPU h | charges |
|---|---|---|---|
| collection | 15,339 s | 4.26 | $2.91 |
| scoring | 52,277 s | 14.52 | $9.90 |
| evaluation (§5.0 preflight) | 152 s | 0 | $0.03 |
| causal patching | 9,934 s | 2.76 | $1.88 |
| sensitivity | 464 s | 0.13 | $0.09 |
| **total** | **78,166 s** | **21.67** | **$14.81** |

No budget-gate stop. (Setup, failed hosts and the evaluator run itself are outside the receipt.)

## 9. Sensitivity
9a. `unavailable_preaccess_missing_position_trace` (`frozen_position_decoder_and_all_object_trace_absent`).
9b. Dose × difficulty, α ∈ {0.5, 1.0} × 8 cells: every cell `specificity_passes: false`
(median off-target ratios 1.6–8.0); sign-correct rates range 0.0–0.75 with 4–15 valid pairs
per cell, no consistent direction.
9c. `unavailable` (`patched_closed_loop_outcome_not_defined`).

## 10. Decision-table mapping (start.md §12)
Pre-stated expectation: M2 ≈ M1, M2 ≫ M0. Measured: M2 vs M1 lift +0.47 % (fails ≥ 3 %),
M2 vs M0 lift +18.8 % (succeeds), lead-time gain 0 steps (fails), causal sign and
specificity fail, multi-layer support unavailable. **Neither lift nor specificity →
negative-result publication path.**
