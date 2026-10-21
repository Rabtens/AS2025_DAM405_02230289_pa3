"""
serve.py - the registered model behind a small REST API, with logging and monitoring
-----------------------------------------------------------------------------------
Loads models:/adult-income-classifier@champion from the MLflow Model Registry and
serves it with Flask.

Endpoints
  GET  /health    model name, version and uptime
  POST /predict   {"records": [ {age, workclass, ...}, ... ]}  (or a single record)
                  -> probability and label for each record
  GET  /metrics   monitoring signals (also writes a SUMMARY log line)
  GET  /drift     data-drift check of the recent inputs against the training data

Logging levels used here
  DEBUG     raw request payload (only when LOG_LEVEL=DEBUG)
  INFO      every request: request_id, records, latency, status; SUMMARY lines
  WARNING   request served but suspicious (unknown category, value outside the
            training range); invalid client input rejected with HTTP 400;
            data drift detected
  ERROR     unexpected failure inside the service (HTTP 500), with traceback
  CRITICAL  the model cannot be loaded at start-up - the process exits

Run:  python serve.py                 (port 8000)
"""

import os
import sys
import time
import uuid

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, g, jsonify, request
from mlflow import MlflowClient

from pa3.config import (CATEGORICAL_FEATURES, FEATURES, MODEL_ALIAS, MODEL_NAME,
                        NUMERIC_FEATURES, REFERENCE_FILE, TRACKING_URI)
from pa3.drift import DriftDetector
from pa3.logging_config import get_logger, setup_logging
from pa3.monitoring import ServiceMonitor

setup_logging("service")
log = get_logger("service")
# our own "request handled" line replaces Werkzeug's plain access log
import logging  # noqa: E402
logging.getLogger("werkzeug").setLevel(logging.WARNING)

SUMMARY_EVERY = 100        # requests between automatic SUMMARY lines
DRIFT_EVERY = 250          # records between automatic drift checks
MAX_RECORDS = 1000         # largest batch accepted in one request


class InvalidInput(ValueError):
    """Client sent something we cannot score (-> HTTP 400)."""


# ------------------------------------------------------------- start-up --
def load_champion():
    """Load the champion model from the registry; CRITICAL + exit if impossible."""
    mlflow.set_tracking_uri(TRACKING_URI)
    try:
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
        reference = pd.read_csv(REFERENCE_FILE)
    except Exception as e:                                   # noqa: BLE001
        log.critical("service cannot start - model or reference data unavailable",
                     extra={"fields": {"model": f"{MODEL_NAME}@{MODEL_ALIAS}",
                                       "error": f"{type(e).__name__}: {e}"}})
        sys.exit(1)
    return model, mv, reference


model, model_version, reference = load_champion()
cat_encoder = model.named_steps["prep"].named_transformers_["cat"]
KNOWN_CATEGORIES = {f: set(c) for f, c in zip(CATEGORICAL_FEATURES, cat_encoder.categories_)}
RANGES = {f: (reference[f].min(), reference[f].max()) for f in NUMERIC_FEATURES}

monitor = ServiceMonitor()
drift = DriftDetector(reference, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
records_since_drift_check = 0
app = Flask(__name__)
log.info("service ready", extra={"fields": {
    "model": MODEL_NAME, "version": model_version.version, "alias": MODEL_ALIAS,
    "run_id": model_version.run_id[:8], "features": len(FEATURES)}})


# ----------------------------------------------------------- validation --
def parse_records(payload) -> tuple[pd.DataFrame, list[str]]:
    """Turn the JSON body into a DataFrame; raise InvalidInput or return warnings."""
    if payload is None:
        raise InvalidInput("request body must be JSON")
    records = payload.get("records", [payload]) if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise InvalidInput("expected {'records': [ {...}, ... ]} or a single record object")
    if len(records) > MAX_RECORDS:
        raise InvalidInput(f"at most {MAX_RECORDS} records per request")
    df = pd.DataFrame(records)
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise InvalidInput(f"missing feature(s): {missing}")
    df = df[FEATURES].copy()
    for f in NUMERIC_FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        if df[f].isna().any():
            raise InvalidInput(f"feature '{f}' must be numeric")
    warnings = []
    for f in CATEGORICAL_FEATURES:
        df[f] = df[f].astype(str)
        unknown = sorted(set(df[f]) - KNOWN_CATEGORIES[f])
        if unknown:
            warnings.append(f"unknown {f} {unknown}")
    for f, (lo, hi) in RANGES.items():
        out = df[(df[f] < lo) | (df[f] > hi)]
        if len(out):
            warnings.append(f"{f} outside training range [{lo:g}, {hi:g}] in {len(out)} record(s)")
    return df, warnings


# ------------------------------------------------------------ endpoints --
@app.before_request
def start_timer():
    g.start = time.perf_counter()
    g.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME, "version": model_version.version,
                    "alias": MODEL_ALIAS})


@app.post("/predict")
def predict():
    global records_since_drift_check
    rid = g.request_id
    payload = request.get_json(silent=True)
    log.debug("request payload", extra={"fields": {"request_id": rid, "payload": payload}})
    n, n_pos, warnings = 0, 0, []
    try:
        df, warnings = parse_records(payload)
        n = len(df)
        for w in warnings:
            log.warning("input served with warning",
                        extra={"fields": {"request_id": rid, "issue": w}})
        proba = model.predict_proba(df)[:, 1]
        labels = (proba >= 0.5).astype(int)
        n_pos = int(labels.sum())
        drift.add(df.to_dict("records"))
        records_since_drift_check += n
        body = {"request_id": rid, "model_version": model_version.version,
                "predictions": [{"probability": round(float(p), 4), "label": int(l),
                                 "income": ">50K" if l else "<=50K"}
                                for p, l in zip(proba, labels)]}
        status = 200
    except InvalidInput as e:
        log.warning("invalid request rejected",
                    extra={"fields": {"request_id": rid, "reason": str(e)}})
        body, status = {"request_id": rid, "error": str(e)}, 400
    except Exception as e:                                   # noqa: BLE001
        log.error("prediction failed", exc_info=True,
                  extra={"fields": {"request_id": rid, "error": f"{type(e).__name__}: {e}"}})
        body, status = {"request_id": rid, "error": "internal error"}, 500

    latency = (time.perf_counter() - g.start) * 1000
    monitor.record(latency, status, n_records=n, n_positive=n_pos, n_warnings=len(warnings))
    log.info("request handled", extra={"fields": {
        "request_id": rid, "endpoint": "/predict", "status": status, "records": n,
        "positive": n_pos, "latency_ms": round(latency, 2)}})

    if monitor.requests % SUMMARY_EVERY == 0:
        monitor.summary()
    if records_since_drift_check >= DRIFT_EVERY:
        records_since_drift_check = 0
        drift.check()
    return jsonify(body), status


@app.get("/metrics")
def metrics():
    return jsonify({"model_version": model_version.version, **monitor.summary()})


@app.get("/drift")
def drift_report():
    return jsonify(drift.check())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    log.info("starting HTTP server", extra={"fields": {"port": port}})
    app.run(host="127.0.0.1", port=port, debug=False)
