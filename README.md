# sweet.tv – TV Program Recommender (Final ML Exam)

Solution for the Kaggle competition
[sweettv-tv-program-recommender](https://www.kaggle.com/competitions/sweettv-tv-program-recommender):
recommend 5 TV programmes to each of 1,260 test users for weeks 31-42 of 2020, metric MAP@5.

Author: Vinokurov Alexander (Kaggle: `shepardcs`).

## Results (Kaggle, 25.10.2026)

| Submission | Private MAP@5 | Public MAP@5 |
|---|---:|---:|
| **Final: hybrid rewatch-head + PPR backfill** (`submission_hybrid_rewatch_ppr.csv`) | **0.31531** | 0.33720 |
| RRF fusion PPR/rewatch/popularity 1/3/1 | 0.30812 | 0.32788 |
| Personalized PageRank α=0.85 | 0.29414 | 0.31195 |
| Popularity baseline | 0.27809 | 0.28234 |

## Pipeline

1. `code/01_build_interactions.py` – interval-overlap join of the view log to the broadcast
   schedule (same channel, overlap ≥180 s or ≥50% of the airing) → user-programme interactions.
2. `code/02_baseline_popularity.py` – required popularity baseline.
3. `code/common.py` – temporal split (fit weeks 11-26 / validate 27-30), MAP@5, candidate pools.
4. `code/03_experiments.py`, `03b_mf_implicit.py`, `03c_tuning.py`, `03d_tuning2.py` –
   hold-out comparison and tuning: matrix factorization (implicit ALS), Personalized PageRank
   on the co-view graph, rewatch rule, hybrids.
5. `code/04_final_submission.py` – fit on all training weeks, write the submission files.
6. `code/05_report_stats.py` – numbers for the report (reach, coverage, worked AP@5 example).
7. `code/kaggle_submit.py` – Kaggle REST submission helper (KGAT bearer token).

`submission_hybrid_rewatch_ppr.csv` in the repository root is the designated final submission
that produced the reported private score.

## Reproducing

```
pip install -r requirements.txt
# put the competition data into data\ (accept the rules on Kaggle first)
python code/01_build_interactions.py
python code/04_final_submission.py
```

Environment: Windows 11, Python 3.10.11; numpy 2.2.6, scipy 1.15.3, pandas 2.3.3,
duckdb 1.5.6, implicit 0.7.3. Seed 42. Full pipeline runs in under a minute.
