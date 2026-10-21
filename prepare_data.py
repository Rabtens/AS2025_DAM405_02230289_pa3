"""
prepare_data.py - download the Adult Census Income dataset once and save it
---------------------------------------------------------------------------
Source: UCI Machine Learning Repository, "Adult" (Becker & Kohavi, 1996),
licensed CC BY 4.0, fetched through shap.datasets.adult (32,561 rows).

The CSV is committed to the repository (data/adult.csv), so this script only
needs to be re-run if the file is missing.
"""

import shap

from pa3.config import DATA_FILE, TARGET
from pa3.logging_config import get_logger, setup_logging

setup_logging("training")
log = get_logger("data")

RENAME = {"Age": "age", "Workclass": "workclass", "Education-Num": "education_num",
          "Marital Status": "marital_status", "Occupation": "occupation",
          "Relationship": "relationship", "Race": "race", "Sex": "sex",
          "Capital Gain": "capital_gain", "Capital Loss": "capital_loss",
          "Hours per week": "hours_per_week", "Country": "native_country"}

if __name__ == "__main__":
    X, y = shap.datasets.adult(display=True)          # human-readable categories
    df = X.rename(columns=RENAME)
    for col in df.select_dtypes("category").columns:
        # the raw data marks missing values with "?" - make that an explicit category
        df[col] = df[col].astype(str).str.strip().replace("?", "Unknown")
    df[TARGET] = y.astype(int)
    DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    log.info("dataset saved", extra={"fields": {
        "rows": len(df), "columns": df.shape[1], "positive_rate": round(df[TARGET].mean(), 4),
        "file": str(DATA_FILE.name)}})
