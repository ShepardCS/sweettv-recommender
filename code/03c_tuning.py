# Step 3c. Final-strategy tuning on the hold-out (fit 11-26, validate 27-30).
# Candidates: PPR alpha grid + weighted personalization; recency-weighted
# popularity; rewatch ranked by distinct weeks; reciprocal-rank fusion (RRF)
# of PPR / rewatch / popularity. Winner becomes the designated submission.

import numpy as np
import scipy.sparse as sp
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, split_val, pool_for_weeks, build_truth,
                    popularity_top, mapk, VAL_FIT_WEEKS, VAL_TARGET_WEEKS)

t0 = time.time()
inter = load_interactions()
fit, tgt = split_val(inter)
pool = pool_for_weeks(*VAL_TARGET_WEEKS)
truth = build_truth(tgt, pool)

agg = (fit.groupby(["user_id", "tv_show_id"])
       .agg(sec=("overlap_sec", "sum"), weeks=("week", "nunique"),
            last_week=("week", "max"))
       .reset_index())
users = np.sort(agg.user_id.unique())
items = np.sort(agg.tv_show_id.unique())
uidx = {u: i for i, u in enumerate(users)}
n_u, n_i = len(users), len(items)
rows = agg.user_id.map(uidx).to_numpy()
cols = agg.tv_show_id.map({s: j for j, s in enumerate(items)}).to_numpy()
sec = agg.sec.to_numpy(dtype=np.float64)
pool_mask = np.fromiter((s in pool for s in items), bool, n_i)
pop5 = popularity_top(fit, pool)
results = {}

def topk(score_vec, k=5):
    s = np.where(pool_mask, score_vec, -np.inf)
    idx = np.argpartition(-s, k)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(items[j]) for j in idx]

def ev(name, preds):
    m, n = mapk(truth, preds)
    results[name] = m
    print(f"{name:36s} MAP@5 = {m:.5f} ({time.time()-t0:.0f}s)", flush=True)

# ---- popularity: recency-weighted (half-life 4 weeks) ----
w_end = VAL_FIT_WEEKS[1]
fit2 = fit.copy()
fit2["w"] = 0.5 ** ((w_end - fit2.week) / 4.0)
popw = (fit2.groupby("tv_show_id")
        .agg(score=("w", "sum")).reset_index())
popw = popw[popw.tv_show_id.isin(pool)].sort_values("score", ascending=False)
pop5_recency = popw.tv_show_id.head(5).astype(int).tolist()
ev("B0r popularity recency hl=4w", {u: pop5_recency for u in truth})

# ---- rewatch by distinct weeks (regularity), backfill popularity ----
own_by_weeks = (agg.sort_values(["weeks", "sec"], ascending=False)
                .groupby("user_id")["tv_show_id"].agg(list).to_dict())
def rew(u, src):
    lst = [int(s) for s in src.get(u, []) if s in pool][:5]
    for p in pop5:
        if len(lst) >= 5: break
        if p not in lst: lst.append(p)
    return lst
ev("R0w rewatch by weeks", {u: rew(u, own_by_weeks) for u in truth})

# ---- PPR variants ----
B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
A = (B.T @ B).astype(np.float32).toarray()
np.fill_diagonal(A, 0.0)
colsum = A.sum(axis=0); colsum[colsum == 0] = 1.0
P = A / colsum
tu = [u for u in truth if u in uidx]
Bw = sp.csr_matrix((np.log1p(sec / 300.0), (rows, cols)), shape=(n_u, n_i))

def make_V(weighted):
    M = Bw if weighted else B
    V = np.zeros((n_i, len(tu)), dtype=np.float32)
    for k, u in enumerate(tu):
        v = M[uidx[u]].toarray().ravel().astype(np.float32)
        s = v.sum()
        V[:, k] = v / s if s > 0 else 0.0
    return V

ppr_ranks = {}
def run_ppr(alpha, weighted, iters=20):
    V = make_V(weighted)
    R = V.copy()
    for _ in range(iters):
        R = (1 - alpha) * V + alpha * (P @ R)
    preds = {}
    for k, u in enumerate(tu):
        preds[u] = topk(R[:, k])
    for u in truth:
        preds.setdefault(u, pop5)
    tag = f"PPR a={alpha} {'wsec' if weighted else 'bin'}"
    ev(tag, preds)
    return R

R03w = run_ppr(0.3, True)
run_ppr(0.1, True)
run_ppr(0.5, True)
R03 = run_ppr(0.3, False)
run_ppr(0.2, False)

# ---- RRF fusion of full rankings: PPR(0.3,wsec) + rewatch-by-weeks + popularity ----
# build PPR rank lists once (top-50 per user) from R03w
def ppr_toplist(R, n=50):
    out = {}
    for k, u in enumerate(tu):
        s = np.where(pool_mask, R[:, k], -np.inf)
        idx = np.argpartition(-s, n)[:n]
        idx = idx[np.argsort(-s[idx])]
        out[u] = [int(items[j]) for j in idx]
    return out
ppr50 = ppr_toplist(R03w)

def rrf_preds(w_ppr=1.0, w_rew=1.0, w_pop=1.0, K=60.0):
    preds = {}
    for u in truth:
        sc = {}
        for r, s in enumerate(ppr50.get(u, [])):
            sc[s] = sc.get(s, 0.0) + w_ppr / (K + r + 1)
        own = [s for s in own_by_weeks.get(u, []) if s in pool][:50]
        for r, s in enumerate(own):
            sc[s] = sc.get(s, 0.0) + w_rew / (K + r + 1)
        for r, s in enumerate(popw.tv_show_id.head(50).astype(int)):
            sc[s] = sc.get(s, 0.0) + w_pop / (K + r + 1)
        if not sc:
            preds[u] = pop5
        else:
            preds[u] = [int(s) for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:5]]
            for p in pop5:
                if len(preds[u]) >= 5: break
                if p not in preds[u]: preds[u].append(p)
    return preds

ev("RRF ppr+rew+pop 1/1/1", rrf_preds())
ev("RRF ppr+rew+pop 2/1/1", rrf_preds(w_ppr=2.0))
ev("RRF ppr+pop 1/0/1", rrf_preds(w_rew=0.0))
ev("RRF ppr+rew 1/1/0", rrf_preds(w_pop=0.0))

print("\n=== summary ===")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{k:36s} {v:.5f}")
