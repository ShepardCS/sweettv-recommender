# Step 3b. Matrix factorization via the `implicit` library (ALS, Hu-Koren).
# The hand-rolled numpy ALS in 03_experiments.py scored far below popularity,
# so it is replaced by the battle-tested implementation (disclosed in the
# report). Same hold-out: fit weeks 11-26, validate weeks 27-30.

import numpy as np
import scipy.sparse as sp
import time, sys, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interactions, split_val, pool_for_weeks, build_truth,
                    popularity_top, mapk, SEED, VAL_TARGET_WEEKS)
import implicit

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
sec = agg.sec.to_numpy(dtype=np.float32)
weeks = agg.weeks.to_numpy(dtype=np.float32)
pool_idx = np.array([j for j, s in enumerate(items) if s in pool])
pop5 = popularity_top(fit, pool)

def run(name, conf, factors=64, reg=0.05, iters=20):
    Cui = sp.csr_matrix((conf, (rows, cols)), shape=(n_u, n_i))
    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=reg, iterations=iters,
        random_state=SEED, use_gpu=False)
    model.fit(Cui, show_progress=False)
    preds = {}
    for u in truth:
        if u in uidx:
            ids, _ = model.recommend(uidx[u], Cui[uidx[u]], N=5,
                                     filter_already_liked_items=False,
                                     items=pool_idx)
            preds[u] = [int(items[j]) for j in ids]
        else:
            preds[u] = pop5
    m, n = mapk(truth, preds)
    print(f"{name:34s} MAP@5 = {m:.5f} ({n} users, {time.time()-t0:.0f}s)", flush=True)

run("MF conf=1+40*log1p(sec/300) f64", 1.0 + 40.0 * np.log1p(sec / 300.0))
run("MF conf=40 binary f64", np.full_like(sec, 40.0))
run("MF conf=1+10*log1p(sec/300) f128", 1.0 + 10.0 * np.log1p(sec / 300.0), factors=128)
run("MF conf=weeks*20 f64", weeks * 20.0)
