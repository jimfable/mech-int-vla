"""EXPLORATORY (post-hoc, not preregistered): can internals stand in for privileged state?

Calibration split only; the Locked Test is not touched.  Uses the frozen Calibration
activation reference (9,455 states x 720, selected candidate early_expert_t1_0, mean over
8 draws) and the frozen Calibration feature cohort (M0/M1/M2 matrices + labels), aligned by
(episode_id, control_step).  Grouped 5-fold CV by initial state; per-episode losses over the
primary steps {0,50,100,150,200}; every episode has total weight one; 90% cluster-bootstrap
intervals over the 20 initial states (10,000 replicates, seed 260803).

Models:
  M0, M1, M2            -- frozen feature sets, frozen HGB hyper-parameters (reference re-fits)
  act_logreg            -- logistic regression on the 720 standardized activations (nested C)
  M0+act (HGB)          -- outputs + raw internals, no simulator state
  M0+M2inc (HGB)        -- outputs + the 8 internal M2 columns, no simulator state
  M2inc (HGB)           -- the 8 internal columns alone
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
ACT = ROOT / "artifacts/calibration-activation-reference-001/cb210e82571cda4ebf3b3a66499357eeb26bfee1ac5c5ea6d5560da5f5bc684c"
COH = next((ROOT / "artifacts/calibration-features-rescore-001/cohort").glob("*/"))
OUT = Path(__file__).resolve().parent
PRIMARY = {0, 50, 100, 150, 200}
SEED = 260803
HGB = dict(learning_rate=0.03, max_iter=200, max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=0.0)

# ---------------------------------------------------------------- load + align
act = np.load(ACT / "arrays.npz")
am = json.load(open(ACT / "metadata.json"))
ep_id_of_index = {e["episode_index"]: e["episode_id"] for e in am["episodes"]}
act_key = {(ep_id_of_index[int(i)], int(s)): r for r, (i, s) in enumerate(zip(act["episode_index"], act["control_step"]))}

coh = np.load(COH / "arrays.npz")
cm = json.load(open(COH / "metadata.json"))
records = cm["records"]
cols = cm["provenance"]["columns"]
order = np.array([act_key[(r["episode_id"], int(r["control_step"]))] for r in records])
assert len(order) == len(records) == 9455 and len(set(order.tolist())) == 9455
H = act["activation_vectors"][order].astype(np.float64)
X0, X1, X2 = coh["m0_matrix"], coh["m1_matrix"], coh["m2_matrix"]
m2inc_idx = [i for i, c in enumerate(cols["M2"]) if c not in set(cols["M1"])]
assert len(m2inc_idx) == 8
X2inc = X2[:, m2inc_idx]
y = np.array([bool(r["terminal_failure_label"]) for r in records])
ep = np.array([r["episode_id"] for r in records])
grp = np.array([int(r["base_init_state_id"]) for r in records])
step = np.array([int(r["control_step"]) for r in records])
ep_ids, ep_inv = np.unique(ep, return_inverse=True)
n_ep = len(ep_ids)
w = 1.0 / np.bincount(ep_inv)[ep_inv]  # each episode total weight one
ep_y = np.array([y[ep_inv == k][0] for k in range(n_ep)])
ep_grp = np.array([grp[ep_inv == k][0] for k in range(n_ep)])
print(f"rows {len(y)}  episodes {n_ep}  failed {int(ep_y.sum())}  clusters {len(np.unique(grp))}", flush=True)

# ---------------------------------------------------------------- models
def oof_hgb(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        clf = HistGradientBoostingClassifier(random_state=SEED, **HGB).fit(X[tr], y[tr], sample_weight=w[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p

def oof_logreg(X, Cs=(0.001, 0.01, 0.1, 1.0)):
    p = np.zeros(len(y)); chosen = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, grp):
        best, best_ll = None, np.inf
        for C in Cs:  # inner grouped CV on the training fold only
            ll = []
            for itr, ite in GroupKFold(n_splits=4).split(X[tr], y[tr], grp[tr]):
                sc = StandardScaler().fit(X[tr][itr])
                m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr][itr]), y[tr][itr], sample_weight=w[tr][itr])
                q = np.clip(m.predict_proba(sc.transform(X[tr][ite]))[:, 1], 1e-6, 1 - 1e-6)
                ll.append(np.average(-(y[tr][ite] * np.log(q) + (~y[tr][ite]) * np.log(1 - q)), weights=w[tr][ite]))
            if np.mean(ll) < best_ll: best, best_ll = C, np.mean(ll)
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=best, max_iter=2000).fit(sc.transform(X[tr]), y[tr], sample_weight=w[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]; chosen.append(best)
    return p, chosen

# ---------------------------------------------------------------- episode-level scoring
prim = np.isin(step, list(PRIMARY))
def episode_scores(p):
    q = np.clip(p, 1e-6, 1 - 1e-6)
    ll = np.full(n_ep, np.nan); br = np.full(n_ep, np.nan); pm = np.full(n_ep, np.nan)
    for k in range(n_ep):
        m = (ep_inv == k) & prim
        yy = ep_y[k]
        ll[k] = np.mean(-(yy * np.log(q[m]) + (not yy) * np.log(1 - q[m])))
        br[k] = np.mean((q[m] - yy) ** 2)
        pm[k] = np.mean(q[m])
    return ll, br, pm

def auroc(pm, idx):
    return roc_auc_score(ep_y[idx], pm[idx])

rng = np.random.default_rng(SEED)
clusters = np.unique(ep_grp)
boot_idx = [np.concatenate([np.where(ep_grp == c)[0] for c in rng.choice(clusters, len(clusters), replace=True)]) for _ in range(10000)]
def ci(stat_fn):
    vals = np.array([stat_fn(idx) for idx in boot_idx])
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))

results = {}
def evaluate(name, p, extra=None):
    ll, br, pm = episode_scores(p)
    all_idx = np.arange(n_ep)
    r = dict(log_loss=float(ll.mean()), brier=float(br.mean()), auroc=float(auroc(pm, all_idx)),
             log_loss_ci=ci(lambda i: ll[i].mean()), auroc_ci=ci(lambda i: auroc(pm, i)))
    if extra: r.update(extra)
    results[name] = r; results[name]["_ll"] = ll.tolist()
    print(f"{name:14s} logloss {r['log_loss']:.4f} [{r['log_loss_ci'][0]:.3f},{r['log_loss_ci'][1]:.3f}]  brier {r['brier']:.4f}  auroc {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]", flush=True)

t0 = time.time()
evaluate("M0", oof_hgb(X0))
evaluate("M1", oof_hgb(X1))
evaluate("M2", oof_hgb(X2))
p_act, Cs = oof_logreg(H); evaluate("act_logreg", p_act, {"chosen_C_per_fold": Cs})
evaluate("M0+act", oof_hgb(np.hstack([X0, H])))
evaluate("M0+M2inc", oof_hgb(np.hstack([X0, X2inc])))
evaluate("M2inc", oof_hgb(X2inc))
print(f"fits + bootstrap: {time.time()-t0:.0f}s", flush=True)

# paired comparisons (relative lift = -delta / mean loss of the reference)
pairs = [("act_logreg", "M0"), ("M0+act", "M0"), ("M0+M2inc", "M0"), ("act_logreg", "M1"), ("M0+act", "M1"), ("M0+M2inc", "M1"), ("M2", "M1")]
for a, b in pairs:
    la, lb = np.array(results[a]["_ll"]), np.array(results[b]["_ll"])
    d = la - lb
    lo, hi = ci(lambda i: d[i].mean())
    lift = -d.mean() / lb.mean()
    lift_ci = (-hi / lb.mean(), -lo / lb.mean())
    results[f"{a}_vs_{b}"] = dict(delta_log_loss=float(d.mean()), delta_ci=[lo, hi], relative_lift=float(lift), relative_lift_ci=list(lift_ci))
    print(f"{a:10s} vs {b:3s}: delta {d.mean():+.4f} [{lo:+.4f},{hi:+.4f}]  lift {lift*100:+.1f}% [{lift_ci[0]*100:+.1f}%, {lift_ci[1]*100:+.1f}%]", flush=True)

for k in list(results):
    if isinstance(results[k], dict): results[k].pop("_ll", None)
results["_meta"] = dict(kind="exploratory_internals_only_calibration", preregistered=False, split="calibration",
                        rows=int(len(y)), episodes=int(n_ep), failed_episodes=int(ep_y.sum()), clusters=int(len(clusters)),
                        primary_steps=sorted(PRIMARY), bootstrap=dict(replicates=10000, seed=SEED, level=0.9),
                        hgb=HGB, activation_reference=str(ACT.name), cohort=str(COH.name),
                        note="Raw out-of-fold probabilities (no Platt step), so log loss is not directly comparable with the frozen Calibration numbers; AUROC is.")
json.dump(results, open(OUT / "results.json", "w"), indent=1, sort_keys=True)
print("written", OUT / "results.json")
