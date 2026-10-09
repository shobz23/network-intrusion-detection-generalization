from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_label_column(df):
    for c in df.columns:
        if c.lower() == "label":
            return c
    raise ValueError("Could not find Label column.")


def load_csv(path, sample=None, seed=42):
    print(f"Reading: {path.name}")
    df = normalize_columns(pd.read_csv(path, low_memory=False))

    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=seed)

    return df


def prepare_xy(df):
    label_col = find_label_column(df)

    y_raw = df[label_col].astype(str).str.strip().str.upper()
    y = pd.Series(np.where(y_raw.eq("BENIGN"), 0, 1), name="target")

    X = df.drop(columns=[label_col]).replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.isna().mean() < 0.95]
    X = X.loc[:, X.nunique(dropna=True) > 1]

    return X, y


def align_features(*dfs):
    common = set(dfs[0].columns)

    for df in dfs[1:]:
        common = common.intersection(df.columns)

    common = sorted(common)

    return [df[common] for df in dfs]


def fpr(cm):
    tn, fp, fn, tp = cm.ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def evaluate(name, model, Xtr, Xte, ytr, yte):
    print(f"\nTraining {name}...")

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    cm = confusion_matrix(yte, pred, labels=[0, 1])

    result = {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "balanced_accuracy": balanced_accuracy_score(yte, pred),
        "precision_attack": precision_score(
            yte, pred, pos_label=1, zero_division=0
        ),
        "recall_attack": recall_score(
            yte, pred, pos_label=1, zero_division=0
        ),
        "f1_attack": f1_score(
            yte, pred, pos_label=1, zero_division=0
        ),
        "false_positive_rate": fpr(cm),
    }

    return result


def main():

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    seed = 42

    # TRAIN FILE 1
    friday = load_csv(
        data_dir / "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        sample=30000,
        seed=seed
    )

    # TRAIN FILE 2
    thursday = load_csv(
        data_dir / "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        sample=30000,
        seed=seed
    )

    # TEST FILE
    wednesday = load_csv(
        data_dir / "Wednesday-workingHours.pcap_ISCX.csv",
        sample=30000,
        seed=seed
    )

    X1, y1 = prepare_xy(friday)
    X2, y2 = prepare_xy(thursday)
    Xtest, ytest = prepare_xy(wednesday)

    X1, X2, Xtest = align_features(X1, X2, Xtest)

    # combine training files
    Xtrain = pd.concat([X1, X2], ignore_index=True)
    ytrain = pd.concat([y1, y2], ignore_index=True)

    print("\nTrain rows:", len(Xtrain))
    print("Test rows:", len(Xtest))
    print("Features:", Xtrain.shape[1])

    print("\nTrain distribution:")
    print(ytrain.value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

    print("\nTest distribution:")
    print(ytest.value_counts().rename(index={0: "BENIGN", 1: "ATTACK"}))

    features = list(Xtrain.columns)

    logprep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            features
        )
    ])

    rfprep = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            features
        )
    ])

    log = Pipeline([
        ("preprocess", logprep),
        ("model", LogisticRegression(
            max_iter=600,
            class_weight="balanced"
        ))
    ])

    rf = Pipeline([
        ("preprocess", rfprep),
        ("model", RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1
        ))
    ])

    results = [
        evaluate(
            "Logistic Regression",
            log,
            Xtrain,
            Xtest,
            ytrain,
            ytest
        ),

        evaluate(
            "Random Forest",
            rf,
            Xtrain,
            Xtest,
            ytrain,
            ytest
        )
    ]

    df_results = pd.DataFrame(results)

    print("\n=== MULTI-FILE TRAINING RESULTS ===")

    print(
        df_results[
            [
                "model",
                "accuracy",
                "balanced_accuracy",
                "precision_attack",
                "recall_attack",
                "f1_attack",
                "false_positive_rate",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
