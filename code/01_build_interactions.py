# Step 1. Join the view log to the broadcast schedule -> user-programme interactions.
#
# Join rule: same channel (view.vsetv_id = schedule.channel_id) AND the viewing
# interval overlaps the programme airing interval. A (user, programme) pair is
# kept as "watched" if the total overlap across all airings is >= 180 seconds
# OR >= 50% of one airing's duration (covers short programmes). Thresholds are
# justified in the report against the duration distribution (median view 396 s).
#
# tv_show_id = 0 rows are unlabelled airtime, NOT items: they are dropped from
# the item side (they can never be recommended because the submission needs ids).
#
# Output: data_work/interactions.parquet  (user_id, tv_show_id, week, overlap_sec, n_airings)
#         data_work/shows.parquet         (tv_show_id, title, category, genres)
#         data_work/test_pool.parquet     (labelled tv_show_id airing in weeks 31-42)
#
# Environment: Python 3.10.11, duckdb 1.5.6, pandas 2.3.3. Runtime printed below.

import duckdb
import os
import time

BASE = r"C:\exam_sweettv"
DATA = os.path.join(BASE, "data")
WORK = os.path.join(BASE, "data_work")
os.makedirs(WORK, exist_ok=True)

t0 = time.time()
con = duckdb.connect()

# ---- schedule: train (weeks 11-30) and test (weeks 31-42) ----
con.execute(f"""
CREATE VIEW sched_raw AS
SELECT *, 'train' AS part FROM read_csv_auto('{DATA}/export_arh_11-20-final.csv', header=true, types={{'start_time':'VARCHAR'}})
UNION ALL
SELECT *, 'train' FROM read_csv_auto('{DATA}/export_arh_21-30-final.csv', header=true, types={{'start_time':'VARCHAR'}})
UNION ALL
SELECT *, 'test'  FROM read_csv_auto('{DATA}/export_arh_31-42-final.csv', header=true, types={{'start_time':'VARCHAR'}})
""")

con.execute("""
CREATE TABLE sched AS
SELECT
  channel_id,
  strptime(start_time, '%d.%m.%Y %H:%M:%S') AS s_start,
  strptime(start_time, '%d.%m.%Y %H:%M:%S') + to_seconds(duration) AS s_end,
  duration AS prog_duration,
  tv_show_id,
  tv_show_title,
  tv_show_category,
  tv_show_genre_1, tv_show_genre_2, tv_show_genre_3,
  week(strptime(start_time, '%d.%m.%Y %H:%M:%S')) AS week,
  part
FROM sched_raw
WHERE tv_show_id IS NOT NULL AND tv_show_id != 0
""")

print("labelled schedule rows:", con.execute("SELECT part, count(*) FROM sched GROUP BY part").fetchall())

# ---- view log (weeks 11-30 only) ----
con.execute(f"""
CREATE TABLE views AS
SELECT
  user_id,
  vsetv_id AS channel_id,
  start_time AS v_start,   -- ISO format, safely auto-parsed by read_csv_auto
  stop_time  AS v_stop
FROM read_csv_auto('{DATA}/dataset11-30.csv', header=true, types={{'user_id':'UBIGINT'}})
""")
print("view rows:", con.execute("SELECT count(*) FROM views").fetchone()[0])

# ---- interval-overlap join on train schedule ----
con.execute("""
CREATE TABLE hits AS
SELECT
  v.user_id,
  s.tv_show_id,
  s.week,
  epoch(least(v.v_stop, s.s_end) - greatest(v.v_start, s.s_start)) AS overlap_sec,
  s.prog_duration
FROM views v
JOIN sched s
  ON v.channel_id = s.channel_id
 AND v.v_start < s.s_end
 AND v.v_stop  > s.s_start
WHERE s.part = 'train'
""")
print("raw overlap hits:", con.execute("SELECT count(*) FROM hits").fetchone()[0])

# ---- watched = overlap >= 180 s OR >= 50% of programme duration (per airing hit) ----
con.execute("""
CREATE TABLE interactions AS
SELECT user_id, tv_show_id, week,
       SUM(overlap_sec) AS overlap_sec,
       COUNT(*) AS n_airings
FROM hits
WHERE overlap_sec >= 180 OR overlap_sec >= 0.5 * prog_duration
GROUP BY user_id, tv_show_id, week
""")
print("interactions (user,show,week):", con.execute("SELECT count(*) FROM interactions").fetchone()[0])
print("distinct users:", con.execute("SELECT count(DISTINCT user_id) FROM interactions").fetchone()[0])
print("distinct shows:", con.execute("SELECT count(DISTINCT tv_show_id) FROM interactions").fetchone()[0])

con.execute(f"COPY interactions TO '{WORK}/interactions.parquet' (FORMAT PARQUET)")

# ---- show metadata (most frequent title/category per id, over all schedule) ----
con.execute(f"""
COPY (
  SELECT tv_show_id,
         arg_max(tv_show_title, cnt) AS title,
         arg_max(tv_show_category, cnt) AS category,
         arg_max(tv_show_genre_1, cnt) AS genre_1
  FROM (
    SELECT tv_show_id, tv_show_title, tv_show_category, tv_show_genre_1, count(*) AS cnt
    FROM sched GROUP BY ALL
  ) GROUP BY tv_show_id
) TO '{WORK}/shows.parquet' (FORMAT PARQUET)
""")

# ---- candidate pool: labelled shows airing in the test window (weeks 31-42) ----
con.execute(f"""
COPY (
  SELECT DISTINCT tv_show_id FROM sched WHERE part = 'test'
) TO '{WORK}/test_pool.parquet' (FORMAT PARQUET)
""")
print("test-window labelled shows:", con.execute("SELECT count(DISTINCT tv_show_id) FROM sched WHERE part='test'").fetchone()[0])

# ---- week sanity ----
print("train weeks:", con.execute("SELECT min(week), max(week) FROM interactions").fetchall())
print("done in %.1f s" % (time.time() - t0))
