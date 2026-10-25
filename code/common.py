# Shared utilities: data loading, temporal split, MAP@5, candidate pools, fallback.
#
# Validation split (temporal, documented in the report):
#   fit on weeks 11-26, validate on weeks 27-30. The Kaggle test window
#   (weeks 31-42) is later than all training weeks, so validation also places
#   the target window strictly after the fitting window.
#
# Candidate pool rule (identical for every method): only labelled programmes
# that AIR in the target window can be recommended (you cannot watch a
# programme that is not broadcast). Repeats are allowed: recommending a
# programme the user already watched is legitimate on TV (news, running
# series) and the competition does not forbid it.
#
# Cold-start fallback (identical for every method): the popularity Top-5.

import duckdb
import numpy as np
import pandas as pd
import os

BASE = r"C:\exam_sweettv"
WORK = os.path.join(BASE, "data_work")
SEED = 42

VAL_FIT_WEEKS = (11, 26)
VAL_TARGET_WEEKS = (27, 30)


def load_interactions():
    return duckdb.sql(f"SELECT * FROM '{WORK}/interactions.parquet'").df()


def load_shows():
    return duckdb.sql(f"SELECT * FROM '{WORK}/shows.parquet'").df()


def pool_for_weeks(w_lo, w_hi):
    """Labelled programme ids airing in schedule weeks [w_lo, w_hi] (train files)."""
    con = duckdb.connect()
    df = con.execute(f"""
      SELECT DISTINCT tv_show_id FROM (
        SELECT tv_show_id,
               week(strptime(start_time, '%d.%m.%Y %H:%M:%S')) AS week
        FROM read_csv_auto('{BASE}/data/export_arh_11-20-final.csv', header=true, types={{'start_time':'VARCHAR'}})
        UNION ALL
        SELECT tv_show_id,
               week(strptime(start_time, '%d.%m.%Y %H:%M:%S')) AS week
        FROM read_csv_auto('{BASE}/data/export_arh_21-30-final.csv', header=true, types={{'start_time':'VARCHAR'}})
      )
      WHERE tv_show_id != 0 AND week BETWEEN {w_lo} AND {w_hi}
    """).df()
    return set(df.tv_show_id.tolist())


def test_pool():
    df = duckdb.sql(f"SELECT tv_show_id FROM '{WORK}/test_pool.parquet'").df()
    return set(df.tv_show_id.tolist())


def test_users():
    return pd.read_csv(os.path.join(BASE, "data", "submission.csv"),
                       dtype={"user_id": np.uint64})["user_id"].tolist()


def apk(actual_set, predicted5, k=5):
    """AP@5 with the course convention: denominator = min(k, |relevant|)."""
    if not actual_set:
        return None
    hits, score = 0, 0.0
    for i, p in enumerate(predicted5[:k]):
        if p in actual_set:
            hits += 1
            score += hits / (i + 1)
    return score / min(k, len(actual_set))


def mapk(truth, preds, k=5):
    """truth: dict user -> set of relevant ids; preds: dict user -> list of 5 ids.
    Averaged over users in `truth` (users the system fails to serve get AP=0)."""
    scores = []
    for u, actual in truth.items():
        p = preds.get(u, [])
        a = apk(actual, p, k)
        if a is not None:
            scores.append(a if p else 0.0)
    return float(np.mean(scores)), len(scores)


def popularity_top(inter_fit, pool, n=5):
    """Shows ranked by distinct fit-window viewers, restricted to pool."""
    df = (inter_fit.groupby("tv_show_id")["user_id"].nunique()
          .reset_index(name="n_users"))
    df = df[df.tv_show_id.isin(pool)]
    df = df.sort_values(["n_users", "tv_show_id"], ascending=[False, True])
    return df.tv_show_id.head(n).tolist()


def split_val(inter):
    fit = inter[(inter.week >= VAL_FIT_WEEKS[0]) & (inter.week <= VAL_FIT_WEEKS[1])]
    tgt = inter[(inter.week >= VAL_TARGET_WEEKS[0]) & (inter.week <= VAL_TARGET_WEEKS[1])]
    return fit, tgt


def build_truth(inter_target, pool):
    """Ground truth per user in the target window, limited to the candidate pool
    (a show that does not air in the window can never be predicted correctly;
    pool-restricted truth keeps the comparison about ranking, not availability).
    """
    t = inter_target[inter_target.tv_show_id.isin(pool)]
    return t.groupby("user_id")["tv_show_id"].agg(set).to_dict()
