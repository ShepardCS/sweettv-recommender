# Step 3d. Second tuning round: rewatch head + PPR backfill, PPR a=0.7 wsec,
# RRF with heavier rewatch weight. Same hold-out.

import numpy as np
import scipy.sparse as sp
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, split_val, pool_for_weeks, build_truth,
                    popularity_top, mapk, VAL_TARGET_WEEKS)

t0 = time.time()
inter = load_interactions()
fit, tgt = split_val(inter)
pool = pool_for_weeks(*VAL_TARGET_WEEKS)
truth = build_truth(tgt, pool)
agg = (fit.groupby(["user_id", "tv_show_id"])
       .agg(sec=("overlap_sec", "sum"), weeks=("week", "nunique"))
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

def topk_idx(score_vec, k):
    s = np.where(pool_mask, score_vec, -np.inf)
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]

def ev(name, preds):
    m, n = mapk(truth, preds)
    results[name] = m
    print(f"{name:40s} MAP@5 = {m:.5f} ({time.time()-t0:.0f}s)", flush=True)

own_by_weeks = (agg.sort_values(["weeks", "sec"], ascending=False)
                .groupby("user_id")["tv_show_id"].agg(list).to_dict())

# PPR weighted personalization
B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
A = (B.T @ B).astype(np.float32).toarray()
np.fill_diagonal(A, 0.0)
colsum = A.sum(axis=0); colsum[colsum == 0] = 1.0
P = A / colsum
tu = [u for u in truth if u in uidx]
Bw = sp.csr_matrix((np.log1p(sec / 300.0), (rows, cols)), shape=(n_u, n_i))
V = np.zeros((n_i, len(tu)), dtype=np.float32)
for k, u in enumerate(tu):
    v = Bw[uidx[u]].toarray().ravel().astype(np.float32)
    s = v.sum()
    V[:, k] = v / s if s > 0 else 0.0

def run_ppr(alpha, iters=20):
    R = V.copy()
    for _ in range(iters):
        R = (1 - alpha) * V + alpha * (P @ R)
    return R

for alpha in (0.6, 0.7, 0.85):
    R = run_ppr(alpha)
    preds = {u: [int(items[j]) for j in topk_idx(R[:, tu.index(u)], 5)] if u in uidx else pop5
             for u in truth}
    ev(f"PPR a={alpha} wsec", preds)

# best PPR for backfill
R = run_ppr(0.5)
ppr20 = {}
for k, u in enumerate(tu):
    ppr20[u] = [int(items[j]) for j in topk_idx(R[:, k], 20)]

def rewatch_ppr(u):
    lst = [int(s) for s in own_by_weeks.get(u, []) if s in pool][:5]
    for p in ppr20.get(u, pop5):
        if len(lst) >= 5: break
        if p not in lst: lst.append(p)
    for p in pop5:
        if len(lst) >= 5: break
        if p not in lst: lst.append(p)
    return lst
ev("rewatch-weeks head + PPR0.5 backfill", {u: rewatch_ppr(u) for u in truth})

# RRF with heavier rewatch
popw = (fit.groupby("tv_show_id")["user_id"].nunique().reset_index(name="n"))
popw = popw[popw.tv_show_id.isin(pool)].sort_values("n", ascending=False)
pop50 = popw.tv_show_id.head(50).astype(int).tolist()
def rrf_preds(w_ppr, w_rew, w_pop, K=60.0):
    preds = {}
    for u in truth:
        sc = {}
        for r, s in enumerate(ppr20.get(u, [])):
            sc[s] = sc.get(s, 0.0) + w_ppr / (K + r + 1)
        own = [s for s in own_by_weeks.get(u, []) if s in pool][:50]
        for r, s in enumerate(own):
            sc[s] = sc.get(s, 0.0) + w_rew / (K + r + 1)
        for r, s in enumerate(pop50):
            sc[s] = sc.get(s, 0.0) + w_pop / (K + r + 1)
        lst = [int(s) for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:5]]
        for p in pop5:
            if len(lst) >= 5: break
            if p not in lst: lst.append(p)
        preds[u] = lst
    return preds
ev("RRF ppr/rew/pop 1/2/1", rrf_preds(1, 2, 1))
ev("RRF ppr/rew/pop 1/3/1", rrf_preds(1, 3, 1))

print("\n=== summary ===")
for k, v in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{k:40s} {v:.5f}")
