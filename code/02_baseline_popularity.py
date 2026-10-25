# Step 2. Popularity baseline (required reference, not one of the two methods).
#
# Popularity = number of DISTINCT training users who watched the programme
# (weeks 11-30), restricted to the candidate pool = labelled programmes that
# actually air in the test window (weeks 31-42). Every test user gets the same
# global Top-5. Also reports test-user coverage of the training log.

import duckdb, os, pandas as pd

BASE = r"C:\exam_sweettv"
WORK = os.path.join(BASE, "data_work")
SUBS = os.path.join(BASE, "submissions")
os.makedirs(SUBS, exist_ok=True)

con = duckdb.connect()
inter = f"{WORK}/interactions.parquet"
pool  = f"{WORK}/test_pool.parquet"

top5 = con.execute(f"""
SELECT i.tv_show_id, count(DISTINCT i.user_id) AS n_users
FROM '{inter}' i
JOIN '{pool}' p USING (tv_show_id)
GROUP BY i.tv_show_id
ORDER BY n_users DESC, i.tv_show_id
LIMIT 5
""").df()
print(top5)

shows = con.execute(f"SELECT * FROM '{WORK}/shows.parquet' WHERE tv_show_id IN ({','.join(map(str, top5.tv_show_id))})").df()
print(shows)

template = pd.read_csv(os.path.join(BASE, "data", "submission.csv"))
ids = " ".join(map(str, top5.tv_show_id.tolist()))
out = template.copy()
out["tv_show_id"] = ids
out.to_csv(os.path.join(SUBS, "submission_baseline.csv"), index=False)
print("rows:", len(out), "->", os.path.join(SUBS, "submission_baseline.csv"))

# coverage of test users in the training interactions
cov = con.execute(f"""
SELECT count(*) AS test_users,
       sum(CASE WHEN i.user_id IS NOT NULL THEN 1 ELSE 0 END) AS with_history
FROM read_csv_auto('{BASE}/data/submission.csv', types={{'user_id':'UBIGINT'}}) t
LEFT JOIN (SELECT DISTINCT user_id FROM '{inter}') i USING (user_id)
""").df()
print(cov)
