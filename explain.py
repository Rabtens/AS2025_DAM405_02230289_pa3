"""
explain.py - SHAP explanations for the registered champion model
---------------------------------------------------------------
Loads models:/adult-income-classifier@champion (the same model the service uses)
and explains it on 2,000 held-out test rows with SHAP:

  global  shap_importance.png   mean |SHAP| per feature (what matters overall)
          shap_beeswarm.png     direction and spread of each feature's effect
          shap_dependence.png   how the effect of age / capital gain / education /
                                hours changes with the value
          shap_categories.png   average effect of each category of relationship
                                and occupation (categoricals need their own view)
  local   shap_waterfall_high.png / _low.png   why one person got a high / low score
  check   importance_comparison.csv  SHAP ranking vs. the model's own gain importance

SHAP values are in log-odds units (the model's raw output); positive pushes the
prediction towards ">50K". Everything is logged to an MLflow run 'explain_champion'.
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from mlflow import MlflowClient

from pa3.config import (ARTIFACT_DIR, CATEGORICAL_FEATURES, EXPERIMENT, MODEL_ALIAS,
                        MODEL_NAME, NUMERIC_FEATURES, SEED, TRACKING_URI)
from pa3.data import load_dataset, split
from pa3.logging_config import get_logger, setup_logging

setup_logging("training")
log = get_logger("explain")
OUT = ARTIFACT_DIR / "shap"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    mlflow.log_figure(fig, f"shap/{name}")
    plt.close(fig)


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    pipe = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    log.info("explaining model", extra={"fields": {
        "model": MODEL_NAME, "version": mv.version, "estimator": type(model).__name__}})

    _, X_test, _, y_test, _ = split(load_dataset())
    X_raw = X_test.sample(2000, random_state=SEED)
    names = list(prep.get_feature_names_out())
    X_enc = pd.DataFrame(prep.transform(X_raw), columns=names, index=X_raw.index)

    # TreeExplainer: exact, fast SHAP values for tree ensembles
    explainer = shap.TreeExplainer(model)
    sv = explainer(X_enc)
    if sv.values.ndim == 3:                       # some models return one set per class
        sv = sv[:, :, 1]
    # show readable category names (not integer codes) in the plots
    sv.display_data = X_raw[names].to_numpy()
    base = float(np.atleast_1d(sv.base_values)[0])

    with mlflow.start_run(run_name="explain_champion"):
        mlflow.set_tags({"stage": "explainability", "model_version": mv.version,
                         "explainer": "shap.TreeExplainer"})
        mlflow.log_params({"n_rows_explained": len(X_raw), "model_version": mv.version})

        # ------------------------------------------------------------ global
        mean_abs = pd.Series(np.abs(sv.values).mean(0), index=names).sort_values(ascending=False)
        fig = plt.figure(figsize=(7, 4)); shap.plots.bar(sv, max_display=10, show=False)
        plt.title("Mean |SHAP| – global feature importance", loc="left"); save(fig, "shap_importance.png")

        fig = plt.figure(figsize=(7, 4.5)); shap.plots.beeswarm(sv, max_display=10, show=False)
        plt.title("SHAP beeswarm – direction of each feature's effect", loc="left")
        save(fig, "shap_beeswarm.png")

        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
        for ax, f in zip(axes, ["age", "capital_gain", "education_num", "hours_per_week"]):
            i = names.index(f)
            ax.scatter(X_raw[f], sv.values[:, i], s=5, alpha=.35, color="#2563eb")
            ax.axhline(0, color="grey", lw=.8)
            ax.set_xlabel(f); ax.set_ylabel("SHAP value (log-odds)")
            if f == "capital_gain":                      # mostly zeros, long right tail
                ax.set_xscale("symlog", linthresh=1000)
                ax.set_xlim(left=0)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.suptitle("SHAP dependence: effect vs. feature value", x=0.01, ha="left")
        fig.tight_layout(); save(fig, "shap_dependence.png")

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
        cat_effects = {}
        for ax, f in zip(axes, ["relationship", "occupation"]):
            i = names.index(f)
            eff = pd.Series(sv.values[:, i], index=X_raw.index).groupby(X_raw[f]).agg(["mean", "size"])
            eff = eff[eff["size"] >= 20].sort_values("mean")
            cat_effects[f] = eff["mean"].round(3).to_dict()
            ax.barh(eff.index, eff["mean"], color=np.where(eff["mean"] > 0, "#dc2626", "#2563eb"))
            ax.axvline(0, color="grey", lw=.8); ax.set_xlabel("mean SHAP value (log-odds)")
            ax.set_title(f, loc="left")
            for s in ("top", "right"): ax.spines[s].set_visible(False)
        fig.tight_layout(); save(fig, "shap_categories.png")

        # ------------------------------------------------------------- local
        proba = pipe.predict_proba(X_raw)[:, 1]
        cases = {"high": int(np.argmax(proba)), "low": int(np.argmin(proba)),
                 "borderline": int(np.argmin(np.abs(proba - 0.5)))}
        local = {}
        for label, idx in cases.items():
            fig = plt.figure(figsize=(7, 4)); shap.plots.waterfall(sv[idx], max_display=8, show=False)
            plt.title(f"{label} case: P(>50K) = {proba[idx]:.2f}", loc="left")
            save(fig, f"shap_waterfall_{label}.png")
            local[label] = {"probability": round(float(proba[idx]), 4),
                            "actual": int(y_test[X_test.index.get_indexer([X_raw.index[idx]])[0]]),
                            "record": X_raw.iloc[idx].to_dict()}

        # --------------------------------------- SHAP vs model's own importance
        gain = getattr(model, "feature_importances_", None)
        comp = pd.DataFrame({"mean_abs_shap": mean_abs})
        if gain is not None:
            comp["model_importance"] = pd.Series(gain, index=names)
            comp["model_importance_rank"] = comp["model_importance"].rank(ascending=False).astype(int)
        comp["shap_rank"] = range(1, len(comp) + 1)
        comp.round(4).to_csv(OUT / "importance_comparison.csv")
        mlflow.log_table(comp.reset_index(names="feature").round(4), "shap/importance_comparison.json")

        findings = {"model_version": mv.version, "base_value_log_odds": round(base, 4),
                    "base_probability": round(1 / (1 + np.exp(-base)), 4),
                    "mean_abs_shap": mean_abs.round(4).to_dict(),
                    "category_effects": cat_effects, "local_cases": local}
        (OUT / "findings.json").write_text(json.dumps(findings, indent=2, default=str))
        mlflow.log_dict(findings, "shap/findings.json")
        mlflow.log_metrics({f"mean_abs_shap_{k}": v for k, v in mean_abs.items()})

    log.info("explanations written", extra={"fields": {
        "top_features": list(mean_abs.index[:5]), "folder": "artifacts/shap"}})
    print(mean_abs.round(4).to_string())


if __name__ == "__main__":
    main()
