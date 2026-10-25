# Step 4. Fit on ALL training weeks (11-30), predict for the 1,260 test users
# (weeks 31-42), write one submission file per final configuration, and dump
# the two example users' Top-5 per method for the report.
#
# Final configurations (hyperparameters chosen on the 27-30 hold-out, step 3):
#   B0  popularity Top-5 (already submitted: private 0.27809)
#   M1  MF ALS (implicit 0.7.3), conf = 1+10*log1p(sec/300), f=128
#   M2  PPR on co-view graph, alpha=0.85, personalization log1p(sec/300)
#   R0  rewatch ranked by distinct weeks, popularity backfill
#   H1  rewatch-weeks head + PPR(0.5) backfill + popularity tail  <- designated
#   H2  RRF rank fusion of PPR / rewatch / popularity, weights 1/3/1
# Candidate pool = labelled shows airing in weeks 31-42; repeats allowed;
# cold-start fallback = popularity Top-5; seed = 42.

import numpy as np
import pandas as pd
import scipy.sparse as sp
import json, time, sys, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, load_shows, test_pool, test_users,
                    popularity_top, SEED)
import implicit

t0 = time.time()
BASE = r"C:\exam_sweettv"
SUBS = os.path.join(BASE, "submissions")
os.makedirs(SUBS, exist_ok=True)
os.makedirs(os.path.join(BASE, "report"), exist_ok=True)

inter = load_interactions()
pool = test_pool()
tusers = test_users()
shows = load_shows().set_index("tv_show_id")

agg = (inter.groupby(["user_id", "tv_show_id"])
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
pop5 = popularity_top(inter, pool)
print(f"full fit: {n_u} users x {n_i} shows, pool {len(pool)}", flush=True)

def topk_list(score_vec, k=5):
    s = np.where(pool_mask, score_vec, -np.inf)
    idx = np.argpartition(-s, k)[:k]
    idx = idx[np.argsort(-s[idx])]
    return [int(items[j]) for j in idx]

def backfill(lst, source):
    for p in source:
        if len(lst) >= 5: break
        if p not in lst: lst.append(int(p))
    return lst

# ---- B0 popularity ----
preds_B0 = {u: list(pop5) for u in tusers}

# ---- R0 rewatch by regularity (distinct weeks, ties by seconds) ----
own_by_weeks = (agg.sort_values(["weeks", "sec"], ascending=False)
                .groupby("user_id")["tv_show_id"].agg(list).to_dict())
def rewatch_list(u):
    lst = [int(s) for s in own_by_weeks.get(u, []) if s in pool][:5]
    return backfill(lst, pop5)
preds_R0 = {u: rewatch_list(u) for u in tusers}

# ---- M1 MF ALS (implicit library, best hold-out config) ----
conf = 1.0 + 10.0 * np.log1p(sec / 300.0)
Cui = sp.csr_matrix((conf, (rows, cols)), shape=(n_u, n_i)).astype(np.float32)
mf = implicit.als.AlternatingLeastSquares(
    factors=128, regularization=0.05, iterations=20,
    random_state=SEED, use_gpu=False)
mf.fit(Cui, show_progress=False)
X, Y = mf.user_factors, mf.item_factors
preds_M1 = {u: (topk_list(X[uidx[u]] @ Y.T) if u in uidx else list(pop5)) for u in tusers}
print("MF done %.0fs" % (time.time() - t0), flush=True)

# ---- M2 PPR (weighted personalization) ----
B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
A = (B.T @ B).astype(np.float32).toarray()
np.fill_diagonal(A, 0.0)
colsum = A.sum(axis=0); colsum[colsum == 0] = 1.0
P = A / colsum
Bw = sp.csr_matrix((np.log1p(sec / 300.0), (rows, cols)), shape=(n_u, n_i))
known = [u for u in tusers if u in uidx]
V = np.zeros((n_i, len(known)), dtype=np.float32)
for k, u in enumerate(known):
    v = Bw[uidx[u]].toarray().ravel().astype(np.float32)
    s = v.sum()
    V[:, k] = v / s if s > 0 else 0.0

def run_ppr(alpha, iters=20):
    R = V.copy()
    for _ in range(iters):
        R = (1 - alpha) * V + alpha * (P @ R)
    return R

R85 = run_ppr(0.85)
preds_M2 = {u: topk_list(R85[:, k]) for k, u in enumerate(known)}
for u in tusers:
    preds_M2.setdefault(u, list(pop5))
print("PPR done %.0fs" % (time.time() - t0), flush=True)

# ---- H1 designated final: rewatch-weeks head + PPR(0.5) backfill + pop tail ----
R50 = run_ppr(0.5)
ppr20 = {u: topk_list(R50[:, k], 20) for k, u in enumerate(known)}
def hybrid_H1(u):
    lst = [int(s) for s in own_by_weeks.get(u, []) if s in pool][:5]
    lst = backfill(lst, ppr20.get(u, []))
    return backfill(lst, pop5)
preds_H1 = {u: hybrid_H1(u) for u in tusers}

# ---- H2 RRF fusion 1/3/1 ----
popw = (inter.groupby("tv_show_id")["user_id"].nunique().reset_index(name="n"))
popw = popw[popw.tv_show_id.isin(pool)].sort_values("n", ascending=False)
pop50 = popw.tv_show_id.head(50).astype(int).tolist()
def hybrid_H2(u, w_ppr=1.0, w_rew=3.0, w_pop=1.0, K=60.0):
    sc = {}
    for r, s in enumerate(ppr20.get(u, [])):
        sc[s] = sc.get(s, 0.0) + w_ppr / (K + r + 1)
    own = [int(s) for s in own_by_weeks.get(u, []) if s in pool][:50]
    for r, s in enumerate(own):
        sc[s] = sc.get(s, 0.0) + w_rew / (K + r + 1)
    for r, s in enumerate(pop50):
        sc[s] = sc.get(s, 0.0) + w_pop / (K + r + 1)
    lst = [int(s) for s, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:5]]
    return backfill(lst, pop5)
preds_H2 = {u: hybrid_H2(u) for u in tusers}

# ---- write submissions ----
template = pd.read_csv(os.path.join(BASE, "data", "submission.csv"),
                       dtype={"user_id": np.uint64})
def write_sub(preds, name):
    out = template.copy()
    out["tv_show_id"] = [" ".join(map(str, preds[u])) for u in out.user_id]
    path = os.path.join(SUBS, f"submission_{name}.csv")
    out.to_csv(path, index=False)
    assert all(len(set(preds[u])) == 5 for u in out.user_id), name
    print("wrote", path)

write_sub(preds_R0, "rewatch_weeks")
write_sub(preds_M1, "mf_als_f128")
write_sub(preds_M2, "ppr_a085_wsec")
write_sub(preds_H1, "hybrid_rewatch_ppr")   # designated final candidate
write_sub(preds_H2, "rrf_1_3_1")

# ---- two example users: heaviest viewer + lightest viewer among test users ----
hist_count = agg.groupby("user_id")["tv_show_id"].nunique()
in_hist = [u for u in tusers if u in hist_count.index]
heavy = int(max(in_hist, key=lambda u: hist_count[u]))
no_hist = [u for u in tusers if u not in uidx]
cold = int(no_hist[0]) if no_hist else int(min(in_hist, key=lambda u: hist_count[u]))
print("heavy:", heavy, int(hist_count[heavy]), "shows; cold/light:", cold,
      int(hist_count.get(cold, 0)), "shows")

def describe(ids):
    out = []
    for s in ids:
        if s in shows.index:
            r = shows.loc[s]
            out.append({"id": int(s), "title": str(r.title), "category": str(r.category)})
        else:
            out.append({"id": int(s), "title": "?", "category": "?"})
    return out

examples = {}
for name, preds in (("B0_popularity", preds_B0), ("M1_MF_ALS", preds_M1),
                    ("M2_PPR", preds_M2), ("R0_rewatch", preds_R0),
                    ("H1_hybrid", preds_H1), ("H2_RRF", preds_H2)):
    examples[name] = {str(heavy): describe(preds[heavy]), str(cold): describe(preds[cold])}

watched = {}
for u in (heavy, cold):
    h = agg[agg.user_id == u].sort_values(["weeks", "sec"], ascending=False).head(10)
    watched[str(u)] = [{"id": int(r.tv_show_id),
                        "title": str(shows.loc[r.tv_show_id].title) if r.tv_show_id in shows.index else "?",
                        "sec": int(r.sec), "weeks": int(r.weeks)} for r in h.itertuples()]

with open(os.path.join(BASE, "report", "examples.json"), "w", encoding="utf-8") as f:
    json.dump({"heavy_user": heavy, "cold_user": cold, "examples": examples,
               "watched_train": watched, "pop5": describe(pop5)},
              f, ensure_ascii=False, indent=1)
print("examples.json written, total %.0f s" % (time.time() - t0))
