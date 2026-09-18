"""EXPLORATORY, POST-HOC (not preregistered): internals without simulator state, evaluated on the
Locked Test.

Everything is fitted on the Calibration split (all 160 episodes; labels permitted there) and
applied frozen to the 158 valid Locked Test episodes.  This is a labelled post-hoc analysis: the
Locked Test outcomes were already read by the single preregistered evaluation; nothing here is
confirmatory.  Activations are the frozen selected candidate (early_expert_t1_0), mean over the
8 original draws in float64 -- the same transformation as the Calibration activation reference.

Models: M0, M1, M2 (re-fits with frozen HGB hyper-parameters, raw probabilities, no Platt),
act_logreg (logistic regression on 720 standardized activations, C by grouped CV on Calibration),
M0+act (HGB), M0+M2inc (HGB), M2inc (HGB); plus the official frozen predictions of M0/M1/M2 from
the Locked Test prediction receipt as reference rows.
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ACT = ROOT / "artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c"
CAL_COH = next((ROOT / "artifacts/calibration-features-rescore-001/cohort").glob("*/"))
LT = ROOT / "artifacts/locked-test-run-51243106/research-artifacts"
LT_COH = next((LT / "locked-test-features/cohort").glob("*/"))
PRED = ROOT / "artifacts/locked-test-final-report/predictions.json"
PRIMARY = {0, 50, 100, 150, 200}
SEED = 260803
HGB = dict(learning_rate=0.03, max_iter=200, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=0.0)


def load_cohort(path):
    z = np.load(path / "arrays.npz"); m = json.load(open(path / "metadata.json"))
    rec = m["records"]; cols = m["provenance"]["columns"]
    inc = [i for i, c in enumerate(cols["M2"]) if c not in set(cols["M1"])]
    assert len(inc) == 8
    d = dict(X0=z["m0_matrix"], X1=z["m1_matrix"], X2=z["m2_matrix"], X2inc=z["m2_matrix"][:, inc],
             y=np.array([bool(r["terminal_failure_label"]) for r in rec]),
             ep=np.array([r["episode_id"] for r in rec]), grp=np.array([int(r["base_init_state_id"]) for r in rec]),
             step=np.array([int(r["control_step"]) for r in rec]))
    d["ep_ids"], d["ep_inv"] = np.unique(d["ep"], return_inverse=True)
    d["w"] = 1.0 / np.bincount(d["ep_inv"])[d["ep_inv"]]
    n = len(d["ep_ids"])
    d["ep_y"] = np.array([d["y"][d["ep_inv"] == k][0] for k in range(n)])
    d["ep_grp"] = np.array([d["grp"][d["ep_inv"] == k][0] for k in range(n)])
    d["prim"] = np.isin(d["step"], list(PRIMARY))
    return d

cal = load_cohort(CAL_COH); lt = load_cohort(LT_COH)

# Calibration activations from the frozen reference, aligned by (episode_id, control_step)
act = np.load(ACT / "arrays.npz"); am = json.load(open(ACT / "metadata.json"))
ep_of = {e["episode_index"]: e["episode_id"] for e in am["episodes"]}
key = {(ep_of[int(i)], int(s)): r for r, (i, s) in enumerate(zip(act["episode_index"], act["control_step"]))}
cal["H"] = act["activation_vectors"][[key[(e, int(s))] for e, s in zip(cal["ep"], cal["step"])]].astype(np.float64)

# Locked Test activations from the score sidecars: mean over the 8 original draws (float64)
H_lt = np.full((len(lt["y"]), 720), np.nan)
row_of = {(e, int(s)): r for r, (e, s) in enumerate(zip(lt["ep"], lt["step"]))}
n_sidecars = 0
for d in sorted(glob.glob(str(LT / "scores/locked_test/*/"))):
    eid = Path(d).name; p = np.load(Path(d) / "primitives.npz")
    A = p["original_activation"].astype(np.float64).mean(axis=1)  # (states, 720)
    for s, vec in zip(p["control_step"], A):
        r = row_of.get((eid, int(s)))
        if r is not None: H_lt[r] = vec
    n_sidecars += 1
assert n_sidecars == 158 and not np.isnan(H_lt).any(), (n_sidecars, int(np.isnan(H_lt).any(axis=1).sum()))
lt["H"] = H_lt
print(f"calibration rows {len(cal['y'])} eps {len(cal['ep_ids'])} | locked test rows {len(lt['y'])} eps {len(lt['ep_ids'])} failed {int(lt['ep_y'].sum())} clusters {len(np.unique(lt['grp']))}", flush=True)

# ------------------------------------------------------------- fit on Calibration, apply to Locked Test
def fit_hgb(Xc, Xl):
    return HistGradientBoostingClassifier(random_state=SEED, **HGB).fit(Xc, cal["y"], sample_weight=cal["w"]).predict_proba(Xl)[:, 1]

def fit_logreg(Xc, Xl, Cs=(0.001, 0.01, 0.1, 1.0)):
    best, best_ll = None, np.inf
    for C in Cs:
        ll = []
        for tr, te in GroupKFold(n_splits=5).split(Xc, cal["y"], cal["grp"]):
            sc = StandardScaler().fit(Xc[tr])
            m = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(Xc[tr]), cal["y"][tr], sample_weight=cal["w"][tr])
            q = np.clip(m.predict_proba(sc.transform(Xc[te]))[:, 1], 1e-6, 1 - 1e-6)
            ll.append(np.average(-(cal["y"][te] * np.log(q) + (~cal["y"][te]) * np.log(1 - q)), weights=cal["w"][te]))
        if np.mean(ll) < best_ll: best, best_ll = C, float(np.mean(ll))
    sc = StandardScaler().fit(Xc)
    m = LogisticRegression(C=best, max_iter=3000).fit(sc.transform(Xc), cal["y"], sample_weight=cal["w"])
    return m.predict_proba(sc.transform(Xl))[:, 1], best

# official frozen predictions (with Platt) as reference rows
pred = json.load(open(PRED)); rows = pred["records"]
assert pred["kind"] == "locked_test_frozen_predictions" and len(rows) == 9988
frozen = {m: np.full(len(lt["y"]), np.nan) for m in ("M0", "M1", "M2")}
for r in rows:
    i = row_of.get((r["episode_id"], int(r["control_step"])))
    if i is None: continue
    for m in frozen: frozen[m][i] = r["probabilities"][m]
assert all(not np.isnan(v).any() for v in frozen.values())

# ------------------------------------------------------------- scoring on Locked Test
n_ep = len(lt["ep_ids"])
def episode_scores(p):
    q = np.clip(p, 1e-6, 1 - 1e-6); ll = np.zeros(n_ep); br = np.zeros(n_ep); pm = np.zeros(n_ep)
    for k in range(n_ep):
        m = (lt["ep_inv"] == k) & lt["prim"]; yy = lt["ep_y"][k]
        ll[k] = np.mean(-(yy * np.log(q[m]) + (not yy) * np.log(1 - q[m]))); br[k] = np.mean((q[m] - yy) ** 2); pm[k] = np.mean(q[m])
    return ll, br, pm

rng = np.random.default_rng(SEED); clusters = np.unique(lt["ep_grp"])
boot = [np.concatenate([np.where(lt["ep_grp"] == c)[0] for c in rng.choice(clusters, len(clusters), replace=True)]) for _ in range(10000)]
def ci(f): v = np.array([f(i) for i in boot]); return [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
def auroc(pm, idx): return roc_auc_score(lt["ep_y"][idx], pm[idx])

results, LL = {}, {}
def evaluate(name, p, extra=None):
    ll, br, pm = episode_scores(p); LL[name] = ll; idx = np.arange(n_ep)
    r = dict(log_loss=float(ll.mean()), brier=float(br.mean()), auroc=float(auroc(pm, idx)),
             log_loss_ci=ci(lambda i: ll[i].mean()), auroc_ci=ci(lambda i: auroc(pm, i)))
    if extra: r.update(extra)
    results[name] = r
    print(f"{name:16s} logloss {r['log_loss']:.4f} [{r['log_loss_ci'][0]:.3f},{r['log_loss_ci'][1]:.3f}]  brier {r['brier']:.4f}  auroc {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]", flush=True)

t0 = time.time()
for m in ("M0", "M1", "M2"): evaluate(f"frozen_{m}", frozen[m], {"source": "locked-test prediction receipt 86d85ed1 (Platt-calibrated)"})
evaluate("M0", fit_hgb(cal["X0"], lt["X0"]))
evaluate("M1", fit_hgb(cal["X1"], lt["X1"]))
evaluate("M2", fit_hgb(cal["X2"], lt["X2"]))
p, C = fit_logreg(cal["H"], lt["H"]); evaluate("act_logreg", p, {"chosen_C": C})
evaluate("M0+act", fit_hgb(np.hstack([cal["X0"], cal["H"]]), np.hstack([lt["X0"], lt["H"]])))
evaluate("M0+M2inc", fit_hgb(np.hstack([cal["X0"], cal["X2inc"]]), np.hstack([lt["X0"], lt["X2inc"]])))
evaluate("M2inc", fit_hgb(cal["X2inc"], lt["X2inc"]))
print(f"fits + bootstrap: {time.time()-t0:.0f}s", flush=True)

for a, b in [("act_logreg", "M0"), ("M0+act", "M0"), ("M0+M2inc", "M0"), ("act_logreg", "M1"), ("M0+act", "M1"), ("M0+M2inc", "M1"), ("M2", "M1"), ("frozen_M2", "frozen_M1")]:
    d = LL[a] - LL[b]; lo, hi = ci(lambda i: d[i].mean()); ref = LL[b].mean()
    results[f"{a}_vs_{b}"] = dict(delta_log_loss=float(d.mean()), delta_ci=[lo, hi], relative_lift=float(-d.mean() / ref), relative_lift_ci=[-hi / ref, -lo / ref])
    print(f"{a:12s} vs {b:10s}: delta {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  lift {-d.mean()/ref*100:+.1f}% [{-hi/ref*100:+.1f}%, {-lo/ref*100:+.1f}%]", flush=True)

results["_meta"] = dict(kind="exploratory_internals_only_locked_test", preregistered=False, post_hoc=True,
                        fit_split="calibration (all 160 episodes)", eval_split="locked_test (158 valid episodes)",
                        clusters=int(len(clusters)), bootstrap=dict(replicates=10000, seed=SEED, level=0.9), hgb=HGB,
                        note="Re-fit rows use raw probabilities (no Platt); frozen_* rows are the official Platt-calibrated receipt predictions. AUROC comparable across all rows; log loss comparable within re-fit rows.")
json.dump(results, open(OUT / "results.json", "w"), indent=1, sort_keys=True); print("written", OUT / "results.json")
