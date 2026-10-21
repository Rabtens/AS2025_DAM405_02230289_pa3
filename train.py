"""
train.py - baselines vs FLAML AutoML, tracked in MLflow, best model registered (LO6)
----------------------------------------------------------------------------------
Workflow
  1. load data/adult.csv, stratified 80/20 split (test set is only touched for the
     final scores, never for choosing a model)
  2. train three baselines, each in its own MLflow run:
        baseline_dummy          always predicts the majority class (floor)
        baseline_logreg         one-hot + scaled logistic regression (classic baseline)
        baseline_lgbm_default   LightGBM with default settings (isolates the value of tuning)
  3. run FLAML AutoML (5 learners, 5-fold CV, ROC AUC, time budget) in a parent run;
     the best configuration found for each learner is logged as a nested child run
  4. every run logs: parameters, 5-fold CV ROC AUC (selection metric), test metrics,
     training time, the full sklearn Pipeline (preprocessing + model), ROC/PR and
     confusion-matrix plots
  5. the run with the best CV ROC AUC is registered as 'adult-income-classifier'
     and given the alias 'champion'; a fairness check on the champion is logged

Run:  python train.py          (AUTOML_TIME_BUDGET=60 python train.py for a quicker run)
"""

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from flaml import AutoML
from lightgbm import LGBMClassifier
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from pa3.config import (ARTIFACT_DIR, AUTOML_ESTIMATORS, AUTOML_TIME_BUDGET, EXPERIMENT,
                        MODEL_ALIAS, MODEL_NAME, REFERENCE_FILE, SEED, TRACKING_URI)
from pa3.data import linear_preprocessor, load_dataset, split, tree_preprocessor
from pa3.logging_config import get_logger, setup_logging

setup_logging("training")
log = get_logger("train")
ARTIFACT_DIR.mkdir(exist_ok=True)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Types inside a LightGBM/XGBoost/sklearn pipeline that skops (MLflow 3's safe
# serialiser) cannot verify on its own. We built these models ourselves, so we
# trust exactly these types and nothing else.
# LightGBM's own defaults expressed in FLAML's search space. Passed as a starting
# point so the search begins from the default model and can only move on if CV
# finds something better (FLAML "warm start").
LGBM_DEFAULTS = {"n_estimators": 100, "num_leaves": 31, "min_child_samples": 20,
                 "learning_rate": 0.1, "log_max_bin": 8, "colsample_bytree": 1.0,
                 "reg_alpha": 1 / 1024, "reg_lambda": 1 / 1024}

TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "lightgbm.basic.Booster",
                 "lightgbm.sklearn.LGBMClassifier", "xgboost.sklearn.XGBClassifier",
                 "xgboost.core.Booster", "collections.OrderedDict",
                 "lightgbm.basic._ConfigAliases", "numpy.dtype"]


# ------------------------------------------------------------------ helpers --
def test_metrics(model, X, y) -> dict:
    """Threshold-free (ROC AUC, PR AUC, log loss) and thresholded (0.5) metrics."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"test_roc_auc": roc_auc_score(y, proba),
            "test_pr_auc": average_precision_score(y, proba),
            "test_log_loss": log_loss(y, proba),
            "test_f1": f1_score(y, pred, zero_division=0),
            "test_precision": precision_score(y, pred, zero_division=0),
            "test_recall": recall_score(y, pred, zero_division=0),
            "test_accuracy": accuracy_score(y, pred)}


def log_evaluation_plots(model, X, y, name: str) -> None:
    """ROC, precision-recall and confusion matrix as one PNG artifact."""
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax[0], name=name)
    PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax[1], name=name)
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=ax[2], colorbar=False,
                                          display_labels=["<=50K", ">50K"])
    for a, t in zip(ax, ["ROC curve", "Precision-recall", "Confusion matrix (t=0.5)"]):
        a.set_title(t)
    fig.tight_layout()
    mlflow.log_figure(fig, "evaluation.png")
    plt.close(fig)


def log_model(pipe, X_train) -> None:
    sig = infer_signature(X_train.head(100), pipe.predict_proba(X_train.head(100))[:, 1])
    mlflow.sklearn.log_model(pipe, name="model", signature=sig,
                             input_example=X_train.head(3),
                             skops_trusted_types=TRUSTED_TYPES)


def baseline_run(name, pipe, X_train, y_train, X_test, y_test, params) -> dict:
    """Cross-validate, fit, evaluate and log one baseline in its own MLflow run."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"stage": "baseline", "model_family": type(pipe[-1]).__name__})
        mlflow.log_params(params)
        t0 = time.perf_counter()
        cv = cross_val_score(pipe, X_train, y_train, cv=CV, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        train_s = time.perf_counter() - t0
        m = {"cv_roc_auc": cv.mean(), "cv_roc_auc_std": cv.std(), "train_time_s": train_s,
             **test_metrics(pipe, X_test, y_test)}
        mlflow.log_metrics(m)
        log_evaluation_plots(pipe, X_test, y_test, name)
        log_model(pipe, X_train)
        log.info("run finished", extra={"fields": {
            "run": name, "cv_roc_auc": round(m["cv_roc_auc"], 4),
            "test_roc_auc": round(m["test_roc_auc"], 4), "train_time_s": round(train_s, 1)}})
        return {"run_id": run.info.run_id, "run_name": name, **m}


# --------------------------------------------------------------------- main --
def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    df = load_dataset()
    X_train, X_test, y_train, y_test, prot_test = split(df)
    log.info("data split", extra={"fields": {
        "train_rows": len(X_train), "test_rows": len(X_test),
        "positive_rate_train": round(y_train.mean(), 4)}})

    # reference sample of training inputs - the baseline for drift detection
    X_train.sample(5000, random_state=SEED).to_csv(REFERENCE_FILE, index=False)

    results = []
    # ---------------------------------------------------------- baselines --
    results.append(baseline_run(
        "baseline_dummy", Pipeline([("prep", tree_preprocessor()),
                                    ("model", DummyClassifier(strategy="prior"))]),
        X_train, y_train, X_test, y_test, {"strategy": "prior"}))
    results.append(baseline_run(
        "baseline_logreg", Pipeline([("prep", linear_preprocessor()),
                                     ("model", LogisticRegression(max_iter=2000, C=1.0))]),
        X_train, y_train, X_test, y_test, {"C": 1.0, "penalty": "l2", "encoding": "one-hot"}))
    results.append(baseline_run(
        "baseline_lgbm_default", Pipeline([("prep", tree_preprocessor()),
                                           ("model", LGBMClassifier(random_state=SEED,
                                                                    verbose=-1))]),
        X_train, y_train, X_test, y_test, {"config": "lightgbm defaults"}))

    # ------------------------------------------------------------- AutoML --
    prep = tree_preprocessor().fit(X_train)
    Xtr_enc = pd.DataFrame(prep.transform(X_train), columns=prep.get_feature_names_out())
    automl = AutoML()
    flaml_log = ARTIFACT_DIR / "flaml_search.log"
    settings = {"task": "classification", "metric": "roc_auc", "time_budget": AUTOML_TIME_BUDGET,
                "estimator_list": AUTOML_ESTIMATORS, "eval_method": "cv", "n_splits": 5,
                "split_type": "stratified", "seed": SEED, "log_file_name": str(flaml_log),
                "log_type": "all",                       # record every trial, not just improvements
                "starting_points": {"lgbm": LGBM_DEFAULTS}, "n_jobs": -1,
                "verbose": 0, "mlflow_logging": False}   # we log trials ourselves (below)
    log.info("AutoML search started", extra={"fields": {
        "tool": "FLAML", "time_budget_s": AUTOML_TIME_BUDGET, "estimators": AUTOML_ESTIMATORS}})

    with mlflow.start_run(run_name="flaml_automl") as parent:
        mlflow.set_tags({"stage": "automl", "tool": "FLAML"})
        mlflow.log_params({k: (",".join(v) if isinstance(v, list) else v)
                           for k, v in settings.items()
                           if k not in ("log_file_name", "starting_points")})
        mlflow.log_param("warm_start", "lgbm defaults")
        t0 = time.perf_counter()
        automl.fit(Xtr_enc, y_train, **settings)
        search_s = time.perf_counter() - t0
        best_cv_auc = 1 - automl.best_loss
        log.info("AutoML search finished", extra={"fields": {
            "best_estimator": automl.best_estimator, "best_cv_roc_auc": round(best_cv_auc, 4),
            "search_time_s": round(search_s, 1)}})

        # one nested child run per learner: its best config and CV score
        for est, cfg in automl.best_config_per_estimator.items():
            loss = automl.best_loss_per_estimator.get(est)
            if cfg is None or loss is None:
                log.warning("learner produced no valid trial", extra={"fields": {"learner": est}})
                continue
            with mlflow.start_run(run_name=f"flaml_best_{est}", nested=True):
                mlflow.set_tags({"stage": "automl_trial", "learner": est})
                mlflow.log_params({f"hp_{k}": v for k, v in cfg.items()})
                mlflow.log_metric("cv_roc_auc", 1 - loss)
            log.debug("learner best", extra={"fields": {"learner": est, "cv_roc_auc": 1 - loss}})

        # final model = our preprocessing + FLAML's best estimator (refit on all train data)
        pipe = Pipeline([("prep", prep), ("model", automl.model.estimator)])
        # sanity check: wrapping must not change FLAML's predictions
        Xte_enc = pd.DataFrame(prep.transform(X_test), columns=prep.get_feature_names_out())
        assert np.allclose(pipe.predict_proba(X_test)[:, 1], automl.predict_proba(Xte_enc)[:, 1])

        # re-score the winning configuration on the SAME 5 folds as the baselines,
        # so the comparison is like for like (FLAML's internal folds differ)
        cv = cross_val_score(clone(pipe), X_train, y_train, cv=CV, scoring="roc_auc")
        m = {"cv_roc_auc": cv.mean(), "cv_roc_auc_std": cv.std(),
             "flaml_internal_cv_roc_auc": best_cv_auc, "train_time_s": search_s,
             **test_metrics(pipe, X_test, y_test)}
        mlflow.log_params({"best_estimator": automl.best_estimator,
                           **{f"best_{k}": v for k, v in automl.best_config.items()}})
        mlflow.log_metrics(m)
        mlflow.log_artifact(str(flaml_log), "automl")
        mlflow.log_dict(automl.best_config, "automl/best_config.json")
        log_evaluation_plots(pipe, X_test, y_test, "flaml_automl")

        # search progress: every trial's CV AUC (by learner) + best-so-far line
        trials = pd.read_json(flaml_log, lines=True).dropna(subset=["validation_loss"])
        trials["cv_auc"] = 1 - trials["validation_loss"]
        trials.to_csv(ARTIFACT_DIR / "automl_trials.csv", index=False)
        mlflow.log_metric("n_trials", len(trials))
        fig, ax = plt.subplots(figsize=(7, 3.4))
        for learner, g in trials.groupby("learner"):
            ax.scatter(g["wall_clock_time"], g["cv_auc"], s=12, alpha=.6, label=learner)
        ax.step(trials["wall_clock_time"], trials["cv_auc"].cummax(), where="post",
                color="black", lw=1.2, label="best so far")
        ax.set_ylim(max(0.80, trials["cv_auc"].min() - 0.01), trials["cv_auc"].max() + 0.005)
        ax.set_xlabel("search time (s)"); ax.set_ylabel("5-fold CV ROC AUC")
        ax.set_title(f"FLAML search: {len(trials)} trials", loc="left")
        ax.grid(alpha=.3); ax.legend(fontsize=7, frameon=False, ncol=3, loc="lower right")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        fig.tight_layout(); fig.savefig(ARTIFACT_DIR / "automl_search.png", dpi=200)
        mlflow.log_figure(fig, "automl/search_progress.png"); plt.close(fig)

        log_model(pipe, X_train)
        results.append({"run_id": parent.info.run_id, "run_name": "flaml_automl", **m})

    # ------------------------------------------------------- compare + pick --
    table = pd.DataFrame(results).set_index("run_name")
    table.drop(columns="run_id").round(4).to_csv(ARTIFACT_DIR / "model_comparison.csv")
    best_name = table.drop(index="baseline_dummy")["cv_roc_auc"].idxmax()   # chosen on CV only
    best = table.loc[best_name]
    log.info("model selected", extra={"fields": {
        "run": best_name, "cv_roc_auc": round(best["cv_roc_auc"], 4),
        "test_roc_auc": round(best["test_roc_auc"], 4)}})

    # -------------------------------------------------- register + fairness --
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", MODEL_NAME)
    client.update_registered_model(MODEL_NAME, description=(
        "Predicts whether annual income exceeds $50K (UCI Adult). Trained by "
        "DAM405 PA3 train.py; preprocessing is part of the model pipeline."))
    client.update_model_version(MODEL_NAME, mv.version, description=(
        f"Selected run {best_name}: best 5-fold CV ROC AUC {best['cv_roc_auc']:.4f} "
        f"(test ROC AUC {best['test_roc_auc']:.4f})."))
    for k in ("cv_roc_auc", "test_roc_auc", "test_f1"):
        client.set_model_version_tag(MODEL_NAME, mv.version, k, f"{best[k]:.4f}")
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, mv.version)
    log.info("model registered", extra={"fields": {
        "name": MODEL_NAME, "version": mv.version, "alias": MODEL_ALIAS, "source_run": best_name}})

    champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    proba = champion.predict_proba(X_test)[:, 1]
    pred = proba >= 0.5
    rows = []
    for attr in prot_test.columns:
        for grp, idx in prot_test.groupby(attr).groups.items():
            mask = prot_test.index.isin(idx)
            if mask.sum() < 50:
                continue
            pos = y_test[mask] == 1
            rows.append({"attribute": attr, "group": grp, "n": int(mask.sum()),
                         "actual_positive_rate": y_test[mask].mean(),
                         "predicted_positive_rate": pred[mask].mean(),
                         "true_positive_rate": pred[mask][pos].mean() if pos.any() else np.nan})
    fair = pd.DataFrame(rows).round(3)
    fair.to_csv(ARTIFACT_DIR / "fairness_check.csv", index=False)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_table(fair, "fairness_check.json")
    print("\n" + table[["cv_roc_auc", "test_roc_auc", "test_pr_auc", "test_f1",
                        "test_accuracy", "train_time_s"]].round(4).to_string())
    print(f"\nRegistered {MODEL_NAME} v{mv.version} (@{MODEL_ALIAS}) from run {best_name}")


if __name__ == "__main__":
    main()
