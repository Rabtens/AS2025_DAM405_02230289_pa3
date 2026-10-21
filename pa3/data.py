"""
pa3/data.py - loading, splitting and preprocessing the Adult dataset
--------------------------------------------------------------------
The preprocessing lives inside the saved model (a scikit-learn Pipeline),
so the service sends raw records and gets exactly the transformation used
in training - no train/serve skew.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pa3.config import (CATEGORICAL_FEATURES, DATA_FILE, FEATURES, NUMERIC_FEATURES,
                        PROTECTED, SEED, TARGET, TEST_SIZE)


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} not found - run `python prepare_data.py` first")
    df = pd.read_csv(DATA_FILE)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    return df


def split(df: pd.DataFrame):
    """Stratified 80/20 split. Returns X_train, X_test, y_train, y_test, protected_test."""
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=SEED,
                                   stratify=df[TARGET])
    return (train[FEATURES], test[FEATURES], train[TARGET].to_numpy(),
            test[TARGET].to_numpy(), test[PROTECTED])


def tree_preprocessor() -> ColumnTransformer:
    """For tree models: numbers pass through, categories -> integer codes.
    Unseen categories at serving time map to -1 instead of crashing."""
    return ColumnTransformer(
        [("num", "passthrough", NUMERIC_FEATURES),
         ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
          CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False)


def linear_preprocessor() -> ColumnTransformer:
    """For linear models: scaled numbers, one-hot categories."""
    return ColumnTransformer(
        [("num", StandardScaler(), NUMERIC_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False)
