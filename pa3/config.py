"""
pa3/config.py - every name, path and setting shared across the project
----------------------------------------------------------------------
Training, explanation and serving all import from here, so the feature list,
the MLflow experiment and the registered-model name can never drift apart.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------------- data --
DATA_FILE = ROOT / "data" / "adult.csv"
TARGET = "income_gt_50k"                       # 1 if income > $50K, else 0
NUMERIC_FEATURES = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL_FEATURES = ["workclass", "marital_status", "occupation", "relationship",
                        "native_country"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Protected attributes: kept in the CSV for a fairness check, but deliberately
# NOT used as model inputs (see report section on design decisions).
PROTECTED = ["sex", "race"]
TEST_SIZE = 0.2
SEED = 42

# ----------------------------------------------------------------- mlflow --
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT = "adult-income-automl"
MODEL_NAME = "adult-income-classifier"
MODEL_ALIAS = "champion"

# ----------------------------------------------------------------- automl --
AUTOML_TIME_BUDGET = int(os.getenv("AUTOML_TIME_BUDGET", "300"))   # seconds
# lrl1 (FLAML's L1 logistic regression) is left out: it expects scaled one-hot
# input and is covered by the logistic-regression baseline instead.
AUTOML_ESTIMATORS = ["lgbm", "xgboost", "rf", "extra_tree"]

# ---------------------------------------------------------------- outputs --
ARTIFACT_DIR = ROOT / "artifacts"          # plots and reports written by scripts
LOG_DIR = Path(os.getenv("PA3_LOG_DIR", str(ROOT / "logs")))   # tests use their own folder
REFERENCE_FILE = ARTIFACT_DIR / "reference_sample.csv"   # training sample for drift checks
