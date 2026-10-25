# Kaggle REST submission helper (the official CLI does not accept KGAT tokens).
# The legacy /competitions/submissions/url/... endpoints return 404 now; the
# working flow (discovered 25.10.2026) is the service-style API:
#   1) POST /api/v1/competitions.CompetitionApiService/StartSubmissionUpload
#      JSON {competitionName, fileName, contentLength, lastModifiedEpochSeconds}
#      -> {token, createUrl}
#   2) PUT the CSV bytes to createUrl (Google Cloud Storage upload URL)
#   3) POST /api/v1/competitions.CompetitionApiService/CreateSubmission
#      JSON {competitionName, blobFileTokens: token, submissionDescription}
# Usage: python kaggle_submit.py <csv_path> "<description>"
#        python kaggle_submit.py --list          (poll recent submissions/scores)

import json, os, sys, time
import requests

SLUG = "sweettv-tv-program-recommender"
API = "https://www.kaggle.com/api/v1"
TOKEN_PATH = os.path.expanduser(r"~\.kaggle\access_token")

with open(TOKEN_PATH) as f:
    HEADERS = {"Authorization": "Bearer " + f.read().strip()}

def list_submissions():
    r = requests.get(f"{API}/competitions/submissions/list/{SLUG}", headers=HEADERS)
    r.raise_for_status()
    subs = r.json()
    if not subs:
        print("(no submissions yet)")
    for s in subs[:8]:
        print(json.dumps({k: s.get(k) for k in
            ("ref", "fileName", "date", "description", "status",
             "publicScore", "privateScore")}, ensure_ascii=False))
    return subs

def submit(path, description):
    r = requests.post(f"{API}/competitions.CompetitionApiService/StartSubmissionUpload",
                      headers=HEADERS,
                      json={"competitionName": SLUG,
                            "fileName": os.path.basename(path),
                            "contentLength": os.path.getsize(path),
                            "lastModifiedEpochSeconds": int(os.path.getmtime(path))})
    r.raise_for_status()
    info = r.json()
    token, create_url = info["token"], info["createUrl"]

    with open(path, "rb") as f:
        up = requests.put(create_url, data=f)
    print("upload status:", up.status_code)
    up.raise_for_status()

    r = requests.post(f"{API}/competitions.CompetitionApiService/CreateSubmission",
                      headers=HEADERS,
                      json={"competitionName": SLUG,
                            "blobFileTokens": token,
                            "submissionDescription": description})
    print("submit step:", r.status_code, r.text[:500])
    r.raise_for_status()

if __name__ == "__main__":
    if sys.argv[1] == "--list":
        list_submissions()
    else:
        submit(sys.argv[1], sys.argv[2])
        time.sleep(25)
        list_submissions()
