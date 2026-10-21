"""
pa3/drift.py - data-drift detection: live inputs vs. the training reference
--------------------------------------------------------------------------
For every feature the detector compares a sliding window of recent live inputs
with a 5,000-row sample of the training data:

  numeric      Kolmogorov-Smirnov two-sample test  -> statistic + p-value
               Population Stability Index (PSI) on 10 reference-quantile bins
  categorical  PSI on category frequencies (unseen categories get their own bin)

Decision rule (per feature):
  PSI >= 0.20  -> DRIFT     (logged as WARNING)   - the usual "significant shift" level
  PSI >= 0.10  -> moderate  (logged as INFO)      - worth watching
  otherwise    -> stable
The KS p-value is reported as supporting evidence but not used alone: with
hundreds of samples, KS flags even tiny, harmless differences.

Why not just compare means (as in Practical 5)? A mean misses changes in spread
and shape and does not exist for categorical features; PSI and KS do not.
"""

from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from pa3.logging_config import get_logger

log = get_logger("drift")
EPS = 1e-4                     # smoothing so empty bins do not give log(0)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftDetector:
    def __init__(self, reference: pd.DataFrame, numeric: list[str], categorical: list[str],
                 window: int = 500, min_samples: int = 200,
                 psi_drift: float = 0.20, psi_watch: float = 0.10):
        self.ref = reference.reset_index(drop=True)
        self.numeric, self.categorical = numeric, categorical
        self.window: deque = deque(maxlen=window)
        self.min_samples = min_samples
        self.psi_drift, self.psi_watch = psi_drift, psi_watch
        # numeric bins: deciles of the reference (duplicates removed for skewed
        # features such as capital_gain, where most values are 0)
        self.edges = {f: np.unique(np.quantile(self.ref[f], np.linspace(0, 1, 11)))
                      for f in numeric}

    def add(self, records: list[dict]) -> None:
        self.window.extend(records)

    def _numeric(self, f: str, live: pd.Series) -> dict:
        edges = self.edges[f].copy()
        edges[0], edges[-1] = -np.inf, np.inf
        ref_p = np.histogram(self.ref[f], edges)[0] / len(self.ref)
        live_p = np.histogram(live, edges)[0] / len(live)
        ks = ks_2samp(self.ref[f], live)
        return {"type": "numeric", "psi": psi(ref_p, live_p), "ks_stat": float(ks.statistic),
                "ks_p_value": float(ks.pvalue), "ref_mean": float(self.ref[f].mean()),
                "live_mean": float(live.mean())}

    def _categorical(self, f: str, live: pd.Series) -> dict:
        cats = sorted(set(self.ref[f]) | set(live))
        ref_p = self.ref[f].value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        live_p = live.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        top = live.value_counts(normalize=True).head(1)
        return {"type": "categorical", "psi": psi(ref_p, live_p),
                "unseen_categories": sorted(set(live) - set(self.ref[f])),
                "live_top_category": f"{top.index[0]} ({top.iloc[0]:.0%})"}

    def check(self) -> dict:
        """Compare the current window with the reference and log the outcome."""
        n = len(self.window)
        if n < self.min_samples:
            log.info("drift check skipped - not enough live data",
                     extra={"fields": {"samples": n, "needed": self.min_samples}})
            return {"status": "insufficient_data", "samples": n}
        live = pd.DataFrame(list(self.window))
        features = {}
        for f in self.numeric:
            features[f] = self._numeric(f, pd.to_numeric(live[f], errors="coerce").dropna())
        for f in self.categorical:
            features[f] = self._categorical(f, live[f].astype(str))
        for f, r in features.items():
            r["psi"] = round(r["psi"], 4)
            r["status"] = ("drift" if r["psi"] >= self.psi_drift else
                           "moderate" if r["psi"] >= self.psi_watch else "stable")
            fields = {"feature": f, **{k: (round(v, 4) if isinstance(v, float) else v)
                                       for k, v in r.items()}}
            if r["status"] == "drift":
                log.warning("data drift detected", extra={"fields": fields})
            elif r["status"] == "moderate":
                log.info("moderate shift - watch this feature", extra={"fields": fields})
        drifted = [f for f, r in features.items() if r["status"] == "drift"]
        summary = {"status": "drift" if drifted else "ok", "samples": n,
                   "drifted_features": drifted, "features": features}
        (log.warning if drifted else log.info)(
            "drift check complete", extra={"fields": {"samples": n, "drifted_features": drifted
                                                      or "none"}})
        return summary
