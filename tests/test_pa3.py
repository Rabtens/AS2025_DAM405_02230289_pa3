"""
tests/test_pa3.py - automated checks of the assignment requirements
-------------------------------------------------------------------
Run after `python train.py`:   python -m pytest -v tests/
"""

import json
import logging

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient

from pa3.config import (CATEGORICAL_FEATURES, EXPERIMENT, LOG_DIR, MODEL_ALIAS, MODEL_NAME,
                        NUMERIC_FEATURES, REFERENCE_FILE, TRACKING_URI)
from pa3.drift import DriftDetector, psi
from pa3.logging_config import get_logger, setup_logging

mlflow.set_tracking_uri(TRACKING_URI)


# --------------------------------------------------------------- tracking --
def test_experiment_has_baselines_and_automl_runs():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT])
    names = set(runs["tags.mlflow.runName"])
    assert {"baseline_dummy", "baseline_logreg", "baseline_lgbm_default", "flaml_automl"} <= names
    tracked = runs[runs["tags.stage"].isin(["baseline", "automl"])]
    assert tracked["metrics.cv_roc_auc"].notna().all()
    assert tracked["metrics.test_roc_auc"].notna().all()


def test_automl_beats_logistic_baseline():
    runs = mlflow.search_runs(experiment_names=[EXPERIMENT]).set_index("tags.mlflow.runName")
    assert runs.loc["flaml_automl", "metrics.cv_roc_auc"] > runs.loc["baseline_logreg",
                                                                     "metrics.cv_roc_auc"]


def test_champion_registered_and_predicts():
    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    assert int(mv.version) >= 1
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    ref = pd.read_csv(REFERENCE_FILE).head(20)
    p = model.predict_proba(ref)[:, 1]
    assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all()


# ---------------------------------------------------------------- logging --
def test_logging_writes_json_with_timestamp_and_level():
    setup_logging("test")
    get_logger("test").warning("hello", extra={"fields": {"request_id": "abc"}})
    for h in logging.getLogger("pa3").handlers:
        h.flush()
    last = json.loads((LOG_DIR / "test.log").read_text().strip().splitlines()[-1])
    assert last["level"] == "WARNING" and last["request_id"] == "abc"
    assert "T" in last["timestamp"]                 # ISO-8601


# ------------------------------------------------------------------ drift --
@pytest.fixture(scope="module")
def reference():
    return pd.read_csv(REFERENCE_FILE)


def test_psi_zero_for_identical_distributions():
    p = np.array([0.2, 0.3, 0.5])
    assert psi(p, p) == pytest.approx(0.0)


def test_no_drift_on_sample_from_same_distribution(reference):
    det = DriftDetector(reference, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    det.add(reference.sample(500, random_state=1).to_dict("records"))
    assert det.check()["drifted_features"] == []


def test_drift_flagged_when_age_shifts(reference):
    det = DriftDetector(reference, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    live = reference.sample(500, random_state=2).copy()
    live["age"] = live["age"] + 15
    det.add(live.to_dict("records"))
    assert "age" in det.check()["drifted_features"]


# ---------------------------------------------------------------- service --
@pytest.fixture(scope="module")
def client():
    import serve
    return serve.app.test_client()


def test_predict_valid_record(client, reference):
    r = client.post("/predict", json={"records": reference.head(3).to_dict("records")})
    assert r.status_code == 200 and len(r.get_json()["predictions"]) == 3


def test_predict_invalid_record_returns_400(client):
    r = client.post("/predict", json={"records": [{"age": 30}]})
    assert r.status_code == 400 and "missing" in r.get_json()["error"]


def test_server_error_returns_500_and_logs_error(client, reference, monkeypatch, caplog):
    import serve

    def broken(_):
        raise RuntimeError("simulated model failure")
    monkeypatch.setattr(serve.model, "predict_proba", broken)
    logging.getLogger("pa3").addHandler(caplog.handler)
    r = client.post("/predict", json={"records": reference.head(1).to_dict("records")})
    logging.getLogger("pa3").removeHandler(caplog.handler)
    assert r.status_code == 500
    assert any(rec.levelname == "ERROR" and rec.exc_info for rec in caplog.records)


def test_metrics_count_requests_and_errors(client):
    m = client.get("/metrics").get_json()
    assert m["requests"] >= 3 and m["client_errors"] >= 1 and m["server_errors"] >= 1
    assert 0 < m["error_rate"] < 1 and m["latency_ms_p95"] > 0
