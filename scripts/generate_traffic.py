'''
Sends realistic prediction requests to the app, to feed the Grafana dashboard.

    python scripts/generate_traffic.py                       # 200 normal requests
    python scripts/generate_traffic.py --drift 25            # reading / writing scores 25 points lower: data drift
    python scripts/generate_traffic.py --invalid-rate 0.1    # 10 % invalid forms

Students are sampled from the training data (notebooks/data/stud.csv, "dvc pull" if missing).
'''
import argparse
import random
import time

import pandas as pd
import requests

FORM_FIELDS = {
    "gender": "gender",
    "race_ethnicity": "ethnicity",
    "parental_level_of_education": "parental_level_of_education",
    "lunch": "lunch",
    "test_preparation_course": "test_preparation_course",
    "reading_score": "reading_score",
    "writing_score": "writing_score",
}

INVALID_CHANGES = [{"gender": ""}, {"reading_score": "abc"}, {"writing_score": "150"}]


def build_form(student, drift):
    form = {FORM_FIELDS[column]: student[column] for column in FORM_FIELDS}
    for score in ["reading_score", "writing_score"]:
        form[score] = int(min(100, max(0, student[score] - drift)))
    return form


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--delay", type=float, default=0.2, help="seconds between requests")
    parser.add_argument("--drift", type=float, default=0, help="points removed from reading / writing scores")
    parser.add_argument("--invalid-rate", type=float, default=0.05, help="share of invalid forms")
    args = parser.parse_args()

    students = pd.read_csv("notebooks/data/stud.csv")
    statuses = {}

    for i in range(args.requests):
        form = build_form(students.sample(1).iloc[0], args.drift)
        if random.random() < args.invalid_rate:
            form.update(random.choice(INVALID_CHANGES))

        status = requests.post(f"{args.url}/predictdata", data=form, timeout=60).status_code
        statuses[status] = statuses.get(status, 0) + 1
        print(f"\r{i + 1}/{args.requests} requests, status codes: {statuses}", end="", flush=True)
        time.sleep(args.delay)

    print()


if __name__ == "__main__":
    main()
