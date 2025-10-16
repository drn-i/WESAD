import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except Exception:
    HAS_SGK = False

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def choose_cv(n_splits, y, groups, seed=42):
    if HAS_SGK:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/m14_merged.csv", help="Merged CSV from readme_parser")
    ap.add_argument("--target", default="stress", choices=["stress"], help="Target to model")
    ap.add_argument("--output-dir", default="artifacts", help="Where to write CSV outputs")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, index_col=0)
    if "subject" not in df.columns or "label" not in df.columns:
        raise ValueError("Expected columns 'subject' and 'label' in merged CSV")

    # Binary stress target: 1 = stress (label==2), 0 = not-stress (0 or 1)
    y = (df["label"] == 2).astype(int).values
    groups = df["subject"].astype(str).values

    # Features: drop label & subject (everything else is numeric already)
    X = df.drop(columns=["label", "subject"])
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    X = X.values

    # Save correlations (numeric only)
    corr = pd.DataFrame(np.corrcoef(X.T), index=feature_names, columns=feature_names)
    corr.to_csv(out / "correlation_matrix.csv")

    # Correlation of each feature with target
    target_corr = []
    for i, col in enumerate(feature_names):
        xi = X[:, i]
        ok = np.isfinite(xi)
        if ok.sum() >= 3 and y[ok].std() > 0 and xi[ok].std() > 0:
            r = np.corrcoef(xi[ok], y[ok])[0, 1]
            target_corr.append({"feature": col, "corr_with_stress": r})
    pd.DataFrame(target_corr).sort_values("corr_with_stress", ascending=False)\
        .to_csv(out / "target_corr_stress.csv", index=False)

    # CV
    cv = choose_cv(args.n_splits, y, groups, args.seed)

    # RandomForest
    rf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1
    )

    rf_metrics = []
    rf_imps = []

    for fold, (tr, va) in enumerate(cv.split(X, y, groups), start=1):
        rf.fit(X[tr], y[tr])
        pred = rf.predict(X[va])
        proba = rf.predict_proba(X[va])[:, 1]

        rf_metrics.append({
            "fold": fold,
            "accuracy": accuracy_score(y[va], pred),
            "f1": f1_score(y[va], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[va], proba) if len(np.unique(y[va])) == 2 else np.nan
        })
        rf_imps.append(pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
            "fold": fold
        }))

    rf_metrics_df = pd.DataFrame(rf_metrics)
    rf_metrics_df.loc[len(rf_metrics_df)] = ["mean",
        rf_metrics_df["accuracy"].mean(),
        rf_metrics_df["f1"].mean(),
        rf_metrics_df["roc_auc"].mean()
    ]
    rf_metrics_df.to_csv(out / "metrics_rf_stress.csv", index=False)

    pd.concat(rf_imps).groupby("feature", as_index=False)["importance"].mean()\
        .sort_values("importance", ascending=False)\
        .to_csv(out / "feature_importance_rf_stress.csv", index=False)

    # XGBoost (optional if installed)
    if HAS_XGB:
        xgb_metrics = []
        xgb_imps = []
        for fold, (tr, va) in enumerate(cv.split(X, y, groups), start=1):
            # balance pos weight neg/pos
            pos = (y[tr] == 1).sum()
            neg = (y[tr] == 0).sum()
            spw = float(neg) / float(pos) if pos > 0 else 1.0

            xgb = XGBClassifier(
                n_estimators=args.n_estimators,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=args.seed,
                scale_pos_weight=spw
            )
            xgb.fit(X[tr], y[tr])
            pred = xgb.predict(X[va])
            proba = xgb.predict_proba(X[va])[:, 1]

            xgb_metrics.append({
                "fold": fold,
                "accuracy": accuracy_score(y[va], pred),
                "f1": f1_score(y[va], pred, zero_division=0),
                "roc_auc": roc_auc_score(y[va], proba) if len(np.unique(y[va])) == 2 else np.nan
            })
            xgb_imps.append(pd.DataFrame({
                "feature": feature_names,
                "importance": xgb.feature_importances_,
                "fold": fold
            }))

        xgb_metrics_df = pd.DataFrame(xgb_metrics)
        xgb_metrics_df.loc[len(xgb_metrics_df)] = ["mean",
            xgb_metrics_df["accuracy"].mean(),
            xgb_metrics_df["f1"].mean(),
            xgb_metrics_df["roc_auc"].mean()
        ]
        xgb_metrics_df.to_csv(out / "metrics_xgb_stress.csv", index=False)

        pd.concat(xgb_imps).groupby("feature", as_index=False)["importance"].mean()\
            .sort_values("importance", ascending=False)\
            .to_csv(out / "feature_importance_xgb_stress.csv", index=False)

    print(f"Saved outputs to {out.resolve()}")

if __name__ == "__main__":
    main()
