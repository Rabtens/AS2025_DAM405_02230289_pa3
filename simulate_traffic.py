"""
simulate_traffic.py - realistic traffic against the running service
------------------------------------------------------------------
Start the service first (python serve.py), then run this script. Phases:

  1. normal      400 single-record requests drawn from the held-out test set
  2. bad input   malformed requests (missing field, text where a number is
                 expected, empty body, too many records) -> HTTP 400, WARNING
  3. suspicious  valid but unusual input (unknown category, age 120)
                 -> served, WARNING
  4. drift check on normal traffic                -> expected: no drift
  5. drifted     500 requests from a shifted population: an older, longer-
                 hours, more professional workforce (e.g. the service is
                 opened to a new client segment)   -> drift WARNINGs
  6. final metrics

Every response is checked; a short report is printed and saved to
artifacts/traffic_report.json for the write-up.
(print() is fine here - this is a client script, not the service.)
"""

import json
import time

import numpy as np
import requests

from pa3.config import ARTIFACT_DIR, SEED
from pa3.data import load_dataset, split

URL = "http://127.0.0.1:8000"
rng = np.random.default_rng(SEED)
s = requests.Session()
report = {}


def post(body, raw=False):
    if raw:
        return s.post(f"{URL}/predict", data=body, headers={"Content-Type": "application/json"},
                      timeout=10)
    return s.post(f"{URL}/predict", json=body, timeout=10)


def phase(title):
    print(f"\n=== {title} ===")


def send_records(df, label):
    codes, probs, t0 = [], [], time.perf_counter()
    for rec in df.to_dict("records"):
        r = post({"records": [rec]})
        codes.append(r.status_code)
        if r.ok:
            probs.append(r.json()["predictions"][0]["probability"])
    secs = time.perf_counter() - t0
    pos = float(np.mean(np.array(probs) >= 0.5))
    print(f"{label}: {len(codes)} requests in {secs:.1f}s, "
          f"HTTP 200 = {codes.count(200)}, predicted >50K = {pos:.1%}")
    return {"requests": len(codes), "ok": codes.count(200), "positive_rate": round(pos, 4)}


def main():
    _, X_test, _, y_test, _ = split(load_dataset())
    print("health:", s.get(f"{URL}/health", timeout=5).json())

    phase("1. normal traffic")
    normal = X_test.sample(400, random_state=SEED)
    report["normal"] = send_records(normal, "normal")

    phase("2. invalid requests -> HTTP 400")
    rec = X_test.iloc[0].to_dict()
    bad = {"missing field": {"records": [{k: v for k, v in rec.items() if k != "age"}]},
           "text in numeric": {"records": [{**rec, "hours_per_week": "forty"}]},
           "empty records": {"records": []},
           "too many records": {"records": [rec] * 1001}}
    for name, body in bad.items():
        r = post(body)
        print(f"{name:18s} -> {r.status_code} {r.json()['error']}")
    r = post("{not json", raw=True)
    print(f"{'not JSON':18s} -> {r.status_code} {r.json()['error']}")

    phase("3. unusual but valid input -> served with WARNING")
    for name, body in {"unknown category": {"records": [{**rec, "occupation": "Astronaut"}]},
                       "age out of range": {"records": [{**rec, "age": 120}]}}.items():
        r = post(body)
        print(f"{name:18s} -> {r.status_code} {r.json()['predictions'][0]}")

    phase("4. drift check on normal traffic")
    d = s.get(f"{URL}/drift", timeout=30).json()
    print(f"status={d['status']} samples={d['samples']} drifted={d['drifted_features']}")
    report["drift_before"] = {f: v["psi"] for f, v in d["features"].items()}

    phase("5. drifted traffic (new client segment)")
    shifted = X_test.sample(500, random_state=SEED + 1).copy()
    shifted["age"] = np.clip(shifted["age"] + rng.normal(12, 3, len(shifted)), 17, 90).round()
    shifted["hours_per_week"] = np.clip(shifted["hours_per_week"] * 1.3, 1, 99).round()
    mask = rng.random(len(shifted)) < 0.4
    shifted.loc[mask, "occupation"] = rng.choice(["Exec-managerial", "Prof-specialty"], mask.sum())
    report["drifted"] = send_records(shifted, "drifted")

    d = s.get(f"{URL}/drift", timeout=30).json()
    print(f"status={d['status']} samples={d['samples']} drifted={d['drifted_features']}")
    for f, v in d["features"].items():
        extra = (f"KS p={v['ks_p_value']:.1e}  mean {v['ref_mean']:.1f} -> {v['live_mean']:.1f}"
                 if v["type"] == "numeric" else f"top: {v['live_top_category']}")
        print(f"  {f:15s} PSI={v['psi']:.3f}  {v['status']:8s} {extra}")
    report["drift_after"] = {f: v["psi"] for f, v in d["features"].items()}

    phase("6. final monitoring metrics")
    m = s.get(f"{URL}/metrics", timeout=5).json()
    print(json.dumps(m, indent=2))
    report["metrics"] = m
    (ARTIFACT_DIR / "traffic_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
