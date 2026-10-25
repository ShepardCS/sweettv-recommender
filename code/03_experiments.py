# Step 3. Methods + temporal hold-out comparison (MAP@5).
#
# Fit: weeks 11-26. Validate: weeks 27-30 (strictly later, like the real test).
# Methods:
#   M1  Implicit-feedback matrix factorization (ALS, Hu-Koren 2008), own numpy
#       implementation. Confidence c_ui = 1 + alpha * log1p(seconds/300).
#   M2  Personalized PageRank on the programme co-view graph.
# References:
#   B0  popularity Top-5 (required baseline)
#   R0  personal rewatch: user's own most-watched shows, popularity backfill
#   H1  hybrid: rewatch head + MF tail (candidate switching)
# All methods: candidate pool = labelled shows airing in the target window;
# repeats allowed; cold-start fallback = popularity Top-5. Seed fixed.

import numpy as np
import pandas as pd
import scipy.sparse as sp
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, split_val, pool_for_weeks, build_truth,
                    popularity_top, mapk, SEED, VAL_FIT_WEEKS, VAL_TARGET_WEEKS)

rng = np.random.default_rng(SEED)
t0 = time.time()

inter = load_interactions()
fit, tgt = split_val(inter)
pool = pool_for_weeks(*VAL_TARGET_WEEKS)
truth = build_truth(tgt, pool)
print(f"fit rows {len(fit)}, target users with truth {len(truth)}, pool {len(pool)}")

# ---- aggregate fit interactions per (user, show) ----
agg = (fit.groupby(["user_id", "tv_show_id"])
       .agg(sec=("overlap_sec", "sum"), weeks=("week", "nunique"))
       .reset_index())

users = np.sort(agg.user_id.unique())
items = np.sort(agg.tv_show_id.unique())
uidx = {u: i for i, u in enumerate(users)}
iidx = {s: j for j, s in enumerate(items)}
n_u, n_i = len(users), len(items)
print(f"matrix {n_u} x {n_i}")

rows = agg.user_id.map(uidx).to_numpy()
cols = agg.tv_show_id.map(iidx).to_numpy()
sec = agg.sec.to_numpy(dtype=np.float64)

pool_mask = np.fromiter((s in pool for s in items), bool, n_i)
pop5 = popularity_top(fit, pool)


def topk_from_scores(score_vec, k=5):
    s = np.where(pool_mask, score_vec, -np.inf)
    idx = np.argpartition(-s, k)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(items[j]) for j in idx]


def evaluate(name, preds):
    m, n = mapk(truth, preds)
    print(f"{name:28s} MAP@5 = {m:.5f}   ({n} users, {time.time()-t0:.0f}s)", flush=True)
    return m

results = {}

# ---- B0 popularity ----
preds = {u: pop5 for u in truth}
results["B0 popularity"] = evaluate("B0 popularity", preds)

# ---- R0 personal rewatch (own shows by total seconds, backfill popularity) ----
own_rank = (agg.sort_values("sec", ascending=False)
            .groupby("user_id")["tv_show_id"].agg(list).to_dict())
def rewatch_list(u):
    lst = [s for s in own_rank.get(u, []) if s in pool][:5]
    for p in pop5:
        if len(lst) >= 5: break
        if p not in lst: lst.append(p)
    return lst
preds = {u: rewatch_list(u) for u in truth}
results["R0 rewatch"] = evaluate("R0 rewatch+popback", preds)

# ---- M1 ALS implicit MF ----
def als(factors=64, reg=0.1, alpha=40.0, iters=10, sec_scale=300.0):
    conf = 1.0 + alpha * np.log1p(sec / sec_scale)
    Cui = sp.csr_matrix((conf, (rows, cols)), shape=(n_u, n_i))
    X = rng.standard_normal((n_u, factors)) * 0.01
    Y = rng.standard_normal((n_i, factors)) * 0.01
    I = np.eye(factors)
    for it in range(iters):
        for mat, other, XY in ((Cui, Y, X), (Cui.T.tocsr(), X, Y)):
            OtO = other.T @ other + reg * I
            for r in range(mat.shape[0]):
                lo, hi = mat.indptr[r], mat.indptr[r + 1]
                if lo == hi:
                    XY[r] = 0.0
                    continue
                idx = mat.indices[lo:hi]
                c = mat.data[lo:hi]
                Oi = other[idx]
                A = OtO + (Oi * (c - 1.0)[:, None]).T @ Oi
                b = (Oi * c[:, None]).sum(axis=0)
                XY[r] = np.linalg.solve(A, b)
    return X, Y

def mf_preds(X, Y):
    out = {}
    for u in truth:
        if u in uidx:
            out[u] = topk_from_scores(X[uidx[u]] @ Y.T)
        else:
            out[u] = pop5
    return out

# ---- M2 Personalized PageRank (dense float32 power iteration, BLAS) ----
def ppr_matrix():
    B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
    A = (B.T @ B).astype(np.float32).toarray()
    np.fill_diagonal(A, 0.0)
    colsum = A.sum(axis=0)
    colsum[colsum == 0] = 1.0
    return A / colsum                        # column-stochastic transition

def ppr(P, alpha=0.5, iters=20):
    tu = [u for u in truth if u in uidx]
    B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
    V = np.zeros((n_i, len(tu)), dtype=np.float32)
    for k, u in enumerate(tu):
        v = B[uidx[u]].toarray().ravel().astype(np.float32)
        s = v.sum()
        V[:, k] = v / s if s > 0 else 0.0
    R = V.copy()
    for _ in range(iters):
        R = (1 - alpha) * V + alpha * (P @ R)
    out = {}
    for k, u in enumerate(tu):
        out[u] = topk_from_scores(R[:, k])
    for u in truth:
        if u not in out:
            out[u] = pop5
    return out

P_trans = ppr_matrix()
for a in (0.3, 0.5, 0.7):
    results[f"M2 PPR a={a}"] = evaluate(f"M2 PPR alpha={a}", ppr(P_trans, alpha=a))

X, Y = als()
results["M1 MF-ALS f64"] = evaluate("M1 MF-ALS f=64 a=40", mf_preds(X, Y))

# ---- H1 hybrid: rewatch head + MF backfill ----
mfp = mf_preds(X, Y)
def hybrid(u, head=3):
    lst = [s for s in own_rank.get(u, []) if s in pool][:head]
    for p in mfp.get(u, pop5):
        if len(lst) >= 5: break
        if p not in lst: lst.append(p)
    return lst
for head in (2, 3):
    preds = {u: hybrid(u, head) for u in truth}
    results[f"H1 rewatch{head}+MF"] = evaluate(f"H1 rewatch head={head} + MF", preds)

print("\n=== summary ===")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{k:28s} {v:.5f}")
print("total %.0f s" % (time.time() - t0))
