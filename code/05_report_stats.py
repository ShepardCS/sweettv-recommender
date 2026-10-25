# Step 5. Statistics for the report, computed on the hold-out
# (fit weeks 11-26, validate weeks 27-30):
#   - MAP@5, reach (share of users with >=1 relevant in Top-5), coverage
#     (distinct recommended items), popularity-head bias per method
#   - a worked AP@5 example for one validation user (AI-disclosure section)

import numpy as np
import scipy.sparse as sp
import json, sys, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, load_shows, split_val, pool_for_weeks,
                    build_truth, popularity_top, mapk, apk, SEED, VAL_TARGET_WEEKS)
import implicit

inter = load_interactions()
fit, tgt = split_val(inter)
pool = pool_for_weeks(*VAL_TARGET_WEEKS)
truth = build_truth(tgt, pool)
shows = load_shows().set_index("tv_show_id")

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

# methods on hold-out (same configs as step 4)
preds_all = {}
preds_all["B0_popularity"] = {u: list(pop5) for u in truth}

own_by_weeks = (agg.sort_values(["weeks", "sec"], ascending=False)
                .groupby("user_id")["tv_show_id"].agg(list).to_dict())
preds_all["R0_rewatch"] = {u: backfill([int(s) for s in own_by_weeks.get(u, []) if s in pool][:5], pop5)
                           for u in truth}

conf = 1.0 + 10.0 * np.log1p(sec / 300.0)
Cui = sp.csr_matrix((conf, (rows, cols)), shape=(n_u, n_i)).astype(np.float32)
mf = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.05,
                                          iterations=20, random_state=SEED, use_gpu=False)
mf.fit(Cui, show_progress=False)
X, Y = mf.user_factors, mf.item_factors
preds_all["M1_MF_ALS"] = {u: (topk_list(X[uidx[u]] @ Y.T) if u in uidx else list(pop5))
                          for u in truth}

B = sp.csr_matrix((np.ones_like(sec), (rows, cols)), shape=(n_u, n_i))
A = (B.T @ B).astype(np.float32).toarray()
np.fill_diagonal(A, 0.0)
colsum = A.sum(axis=0); colsum[colsum == 0] = 1.0
P = A / colsum
Bw = sp.csr_matrix((np.log1p(sec / 300.0), (rows, cols)), shape=(n_u, n_i))
tu = [u for u in truth if u in uidx]
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
R85 = run_ppr(0.85)
preds_all["M2_PPR"] = {u: topk_list(R85[:, k]) for k, u in enumerate(tu)}
for u in truth:
    preds_all["M2_PPR"].setdefault(u, list(pop5))

R50 = run_ppr(0.5)
ppr20 = {u: topk_list(R50[:, k], 20) for k, u in enumerate(tu)}
preds_all["H1_hybrid"] = {u: backfill(backfill([int(s) for s in own_by_weeks.get(u, []) if s in pool][:5],
                                               ppr20.get(u, [])), pop5)
                          for u in truth}

# popularity head: top 10% items by distinct fit viewers
popcnt = agg.groupby("tv_show_id")["user_id"].nunique().sort_values(ascending=False)
head = set(popcnt.head(max(1, len(popcnt) // 10)).index.astype(int))

stats = {}
for name, preds in preds_all.items():
    m, n = mapk(truth, preds)
    hits = sum(1 for u, a in truth.items() if set(preds[u][:5]) & a)
    allrec = [s for u in truth for s in preds[u][:5]]
    stats[name] = {
        "map5": round(m, 5),
        "users": n,
        "reach_ge1": hits,
        "reach_share": round(hits / len(truth), 4),
        "coverage_items": len(set(allrec)),
        "head_share": round(sum(1 for s in allrec if s in head) / len(allrec), 4),
    }
    print(name, stats[name], flush=True)

# worked AP@5 example: a mid-size validation user, H1 predictions
cand = sorted(truth, key=lambda u: len(truth[u]))
worked_user = None
for u in cand:
    if 2 <= len(truth[u]) <= 4 and u in own_by_weeks:
        pr = preds_all["H1_hybrid"][u]
        k_hits = [p in truth[u] for p in pr]
        if 0 < sum(k_hits) < 5:
            worked_user = u
            break
pr = preds_all["H1_hybrid"][worked_user]
rel = truth[worked_user]
def title(s):
    return str(shows.loc[s].title) if s in shows.index else "?"
worked = {
    "user_id": int(worked_user),
    "relevant_weeks27_30": [{"id": int(s), "title": title(s)} for s in sorted(rel)],
    "predicted_top5": [{"rank": i + 1, "id": int(p), "title": title(p),
                        "relevant": bool(p in rel)} for i, p in enumerate(pr)],
    "ap5": apk(rel, pr),
    "denominator": min(5, len(rel)),
}
print(json.dumps(worked, ensure_ascii=False, indent=1))
with open(r"C:\exam_sweettv\report\stats.json", "w", encoding="utf-8") as f:
    json.dump({"holdout_stats": stats, "worked_example": worked,
               "truth_users": len(truth), "pool_size": len(pool)}, f,
              ensure_ascii=False, indent=1)
print("stats.json written")
