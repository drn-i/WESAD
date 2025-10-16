"""
Feature selection for WESAD (or any tabular dataset) using:
- RandomForest (Gini-based feature_importances_)
- XGBoost (gain-based feature_importances_)
- Permutation importance on a holdout set
- SHAP (TreeExplainer) mean(|SHAP|) on a subsample

Outputs (in --output-dir):
- metrics_{model}.json
- feature_importances_{model}.csv              (native model importances)
- permutation_importance.csv
- shap_mean_abs.csv
- feature_ranking_combined.csv                 (rank-aggregated)
- top_k_features.json
- plots/*.png                                  (top-K barplots)

Supports binary or multiclass targets:
- Binary: pass --positive-values and optional --negative-values
- Multiclass: omit both and it will treat target as categorical with >=2 classes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd

from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", type=Path, required=True, help="CSV with features and target column.")
    p.add_argument("--target-col", type=str, required=True, help="Target label column.")
    p.add_argument("--positive-values", nargs="*", default=None,
                   help="Values to map to positive class (binary). If omitted, multiclass mode.")
    p.add_argument("--negative-values", nargs="*", default=None,
                   help="Values to map to negative class (binary). Optional.")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=20, help="How many top features to export.")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/feature_selection/wesad"))
    p.add_argument("--shap-subsample", type=int, default=300,
                   help="Number of training rows to use for SHAP (0 disables SHAP).")
    p.add_argument("--models", nargs="*", default=["random_forest", "xgboost"],
                   choices=["random_forest", "xgboost"], help="Which models to run.")
    return p.parse_args()


def split_xy(
    df: pd.DataFrame,
    target_col: str,
    positive_values: Optional[List[str | int | float]] = None,
    negative_values: Optional[List[str | int | float]] = None,
) -> Tuple[pd.DataFrame, pd.Series, bool]:
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Decide binary vs multiclass
    if positive_values is not None:
        # Binary mapping to 0/1
        pos_set = set(str(v).lower() for v in positive_values)
        if negative_values is not None:
            neg_set = set(str(v).lower() for v in negative_values)
        else:
            neg_set = set(str(v).lower() for v in y_raw.unique() if str(v).lower() not in pos_set)

        def map_bin(v):
            s = str(v).lower()
            return 1 if s in pos_set else 0 if s in neg_set else np.nan

        y = y_raw.map(map_bin).astype(float)
        if y.isna().any():
            missing = df.loc[y.isna(), target_col].value_counts().to_dict()
            raise ValueError(f"Unmapped target values in binary mapping: {missing}")
        y = y.astype(int)
        return X, y, True
    else:
        # Multiclass (string labels)
        y = y_raw.astype(str)
        return X, y, False


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category", "bool")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )
    return pre, num_cols, cat_cols


def feature_names_after_preprocessor(pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str]) -> List[str]:
    names = []
    if "num" in pre.named_transformers_:
        names += list(num_cols)
    if "cat" in pre.named_transformers_:
        ohe = pre.named_transformers_["cat"].named_steps["onehot"]
        cat_out = ohe.get_feature_names_out(cat_cols)
        names += list(cat_out)
    return names


def fit_random_forest(
    X_train, y_train, pre: ColumnTransformer, binary: bool, seed: int
) -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = Pipeline([
        ("pre", pre),
        ("model", RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
            n_jobs=-1, random_state=seed, class_weight="balanced" if binary else None
        )),
    ])
    # Simple CV to tune a couple params
    grid = {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_split": [2, 5],
    }
    scoring = "roc_auc" if binary else "accuracy"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_, {"best_params": search.best_params_, "best_score_cv": search.best_score_}


def fit_xgboost(
    X_train, y_train, pre: ColumnTransformer, binary: bool, seed: int
) -> Tuple[Pipeline, Dict[str, Any]]:
    if binary:
        clf = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",
            random_state=seed, n_estimators=400, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            max_depth=6, n_jobs=-1
        )
        scoring = "roc_auc"
        grid = {
            "model__max_depth": [4, 6],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__learning_rate": [0.03, 0.07],
            "model__n_estimators": [300, 600],
        }
    else:
        # Multiclass
        clf = xgb.XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
            random_state=seed, n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            max_depth=6, n_jobs=-1
        )
        scoring = "accuracy"
        grid = {
            "model__max_depth": [4, 6],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__learning_rate": [0.03, 0.07],
            "model__n_estimators": [300, 600],
        }

    pipe = Pipeline([("pre", pre), ("model", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    search = GridSearchCV(pipe, grid, scoring=scoring, cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    return search.best_estimator_, {"best_params": search.best_params_, "best_score_cv": search.best_score_}


def evaluate_model(
    pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, binary: bool
) -> Dict[str, Any]:
    y_pred = pipe.predict(X_test)
    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
    if binary:
        y_proba = pipe.predict_proba(X_test)[:, 1]
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, (y_proba >= 0.5).astype(int), average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_test, (y_proba >= 0.5).astype(int)).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        auc = roc_auc_score(y_test, y_proba)
        metrics.update({
            "roc_auc": float(auc),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        })
    else:
        pr, rc, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
        metrics.update({
            "macro_f1": float(np.mean(f1)),
            "labels": sorted(list(set(y_test))),
        })
    return metrics


def extract_model_importances(
    pipe: Pipeline, pre: ColumnTransformer, num_cols: List[str], cat_cols: List[str], model_name: str
) -> pd.DataFrame:
    # Get final model
    model = pipe.named_steps["model"]
    # Transformed feature names
    feat_names = feature_names_after_preprocessor(pre, num_cols, cat_cols)
    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        imp = np.asarray(model.feature_importances_, dtype=float)
        if imp.shape[0] == len(feat_names):
            df = pd.DataFrame({"feature": feat_names, f"{model_name}_importance": imp})
            df = df.sort_values(by=f"{model_name}_importance", ascending=False)
            return df
    # Fallback
    return pd.DataFrame({"feature": feat_names, f"{model_name}_importance": np.nan})


def compute_permutation_importance(
    pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, binary: bool
) -> pd.DataFrame:
    scoring = "roc_auc" if binary else "accuracy"
    perm = permutation_importance(pipe, X_test, y_test, n_repeats=20, random_state=0, scoring=scoring, n_jobs=-1)
    # ColumnTransformer drops names; need transformed names:
    pre: ColumnTransformer = pipe.named_steps["pre"]
    num_cols = pre.transformers_[0][2] if pre.transformers_ else []
    cat_cols = pre.transformers_[1][2] if pre.transformers_ and len(pre.transformers_) > 1 else []
    feat_names = feature_names_after_preprocessor(pre, list(num_cols), list(cat_cols))
    res = pd.DataFrame({
        "feature": feat_names,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    }).sort_values(by="perm_importance_mean", ascending=False)
    return res


def compute_shap(
    pipe: Pipeline, X_train: pd.DataFrame, shap_subsample: int
) -> pd.DataFrame:
    if shap_subsample <= 0:
        return pd.DataFrame(columns=["feature", "shap_mean_abs"])
    # Get preprocessed matrix and feature names
    pre: ColumnTransformer = pipe.named_steps["pre"]
    Xt = pre.fit_transform(X_train)  # ensure fitted
    feat_names = feature_names_after_preprocessor(pre,
                                                  pre.transformers_[0][2] if pre.transformers_ else [],
                                                  pre.transformers_[1][2] if len(pre.transformers_) > 1 else [])
    model = pipe.named_steps["model"]
    # pick subset rows
    n = Xt.shape[0]
    idx = np.random.RandomState(0).choice(n, size=min(shap_subsample, n), replace=False)
    Xt_sub = Xt[idx]
    # SHAP TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer(Xt_sub)
        # For multiclass, shap_vals is list; use mean of abs across classes
        if isinstance(shap_vals.values, list) or getattr(shap_vals, "values", None) is None:
            # new SHAP API may return Explanation, handle generically
            vals = getattr(shap_vals, "values", None)
            if vals is None:
                return pd.DataFrame(columns=["feature", "shap_mean_abs"])
        values = shap_vals.values
        if values.ndim == 3:
            # [n_samples, n_features, n_classes]
            arr = np.mean(np.abs(values), axis=(0, 2))
        else:
            # [n_samples, n_features]
            arr = np.mean(np.abs(values), axis=0)
        df = pd.DataFrame({"feature": feat_names, "shap_mean_abs": arr})
        df = df.sort_values(by="shap_mean_abs", ascending=False)
        return df
    except Exception:
        # SHAP can fail depending on model params / SHAP version; return empty
        return pd.DataFrame(columns=["feature", "shap_mean_abs"])


def rank_aggregate(
    dfs: List[pd.DataFrame],
    cols: List[str],
    top_k: int
) -> pd.DataFrame:
    # Left-join on 'feature', compute rank per column, then average rank
    agg = None
    for df, col in zip(dfs, cols):
        if df is None or df.empty or col not in df.columns:
            continue
        temp = df[["feature", col]].copy()
        temp = temp.dropna()
        temp[f"rank_{col}"] = temp[col].rank(ascending=False, method="average")
        temp = temp[["feature", f"rank_{col}"]]
        agg = temp if agg is None else pd.merge(agg, temp, on="feature", how="outer")
    if agg is None:
        return pd.DataFrame(columns=["feature", "avg_rank"])
    rank_cols = [c for c in agg.columns if c.startswith("rank_")]
    agg["avg_rank"] = agg[rank_cols].mean(axis=1)
    agg = agg.sort_values(by="avg_rank", ascending=True)
    return agg.head(top_k)


def barplot_top(df: pd.DataFrame, col: str, k: int, title: str, out_path: Path) -> None:
    top = df.dropna().head(k)
    if top.empty:
        return
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    sns.barplot(data=top, y="feature", x=col, orient="h", color="#4e79a7")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in CSV.")

    X, y, binary = split_xy(df, args.target_col, args.positive_values, args.negative_values)
    pre, num_cols, cat_cols = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed, shuffle=True
    )

    results: Dict[str, Any] = {"binary": binary, "n_features": int(X.shape[1])}
    importance_frames: List[pd.DataFrame] = []
    importance_cols: List[str] = []

    if "random_forest" in args.models:
        rf_pipe, rf_cv = fit_random_forest(X_train, y_train, pre, binary, args.seed)
        rf_metrics = evaluate_model(rf_pipe, X_test, y_test, binary)
        (out_dir / "metrics_random_forest.json").write_text(json.dumps({"cv": rf_cv, "test": rf_metrics}, indent=2))
        # importances
        rf_imp = extract_model_importances(rf_pipe, pre, num_cols, cat_cols, "rf")
        rf_imp.to_csv(out_dir / "feature_importances_random_forest.csv", index=False)
        importance_frames.append(rf_imp.rename(columns={"rf_importance": "importance"}))
        importance_cols.append("importance")
        # permutation
        rf_perm = compute_permutation_importance(rf_pipe, X_test, y_test, binary)
        rf_perm.to_csv(out_dir / "permutation_importance.csv", index=False)
        # plot
        barplot_top(rf_imp, "rf_importance", args.top_k, "RF Feature Importances", plots_dir / "rf_top.png")
        barplot_top(rf_perm, "perm_importance_mean", args.top_k, "Permutation Importances", plots_dir / "perm_top.png")

    if "xgboost" in args.models:
        xgb_pipe, xgb_cv = fit_xgboost(X_train, y_train, pre, binary, args.seed)
        xgb_metrics = evaluate_model(xgb_pipe, X_test, y_test, binary)
        (out_dir / "metrics_xgboost.json").write_text(json.dumps({"cv": xgb_cv, "test": xgb_metrics}, indent=2))
        # importances (gain by default; ensure the underlying booster importance_type is gain)
        model = xgb_pipe.named_steps["model"]
        try:
            model.set_params(importance_type="gain")
        except Exception:
            pass
        # To get importances after preprocessing, we still rely on model.feature_importances_
        xgb_imp = extract_model_importances(xgb_pipe, pre, num_cols, cat_cols, "xgb")
        xgb_imp.to_csv(out_dir / "feature_importances_xgboost.csv", index=False)
        importance_frames.append(xgb_imp.rename(columns={"xgb_importance": "importance"}))
        importance_cols.append("importance")
        # SHAP on subsample
        shap_df = compute_shap(xgb_pipe, X_train, args.shap_subsample)
        shap_df.to_csv(out_dir / "shap_mean_abs.csv", index=False)
        barplot_top(xgb_imp, "xgb_importance", args.top_k, "XGBoost Gain Importances", plots_dir / "xgb_top.png")
        if not shap_df.empty:
            barplot_top(shap_df, "shap_mean_abs", args.top_k, "SHAP |mean value|", plots_dir / "shap_top.png")

    # Combine rankings
    combined = rank_aggregate(
        dfs=[df for df in [importance_frames[0] if importance_frames else None]],
        cols=["importance"],
        top_k=args.top_k,
    )

    # If we also have permutation or shap, incorporate their ranks as well
    try:
        perm_df = pd.read_csv(out_dir / "permutation_importance.csv")
        perm_df = perm_df.rename(columns={"perm_importance_mean": "perm"})
        shap_df = pd.read_csv(out_dir / "shap_mean_abs.csv") if (out_dir / "shap_mean_abs.csv").exists() else pd.DataFrame()
        # Merge rank tables
        pieces = []
        if importance_frames:
            imp_union = pd.concat(importance_frames, axis=0)
            imp_union = imp_union.groupby("feature", as_index=False)["importance"].mean().sort_values(by="importance", ascending=False)
            imp_union["rank_importance"] = imp_union["importance"].rank(ascending=False)
            pieces.append(imp_union[["feature", "rank_importance"]])
        if not perm_df.empty:
            perm_df["rank_perm"] = perm_df["perm"].rank(ascending=False)
            pieces.append(perm_df[["feature", "rank_perm"]])
        if not shap_df.empty and "shap_mean_abs" in shap_df.columns:
            shap_df["rank_shap"] = shap_df["shap_mean_abs"].rank(ascending=False)
            pieces.append(shap_df[["feature", "rank_shap"]])

        if pieces:
            agg = pieces[0]
            for p in pieces[1:]:
                agg = pd.merge(agg, p, on="feature", how="outer")
            rank_cols = [c for c in agg.columns if c.startswith("rank_")]
            agg["avg_rank"] = agg[rank_cols].mean(axis=1)
            combined = agg.sort_values(by="avg_rank").head(args.top_k)
    except Exception:
        pass

    combined = combined.reset_index(drop=True)
    combined.to_csv(out_dir / "feature_ranking_combined.csv", index=False)

    top_k = combined["feature"].tolist()[: args.top_k]
    (out_dir / "top_k_features.json").write_text(json.dumps({"top_k": top_k}, indent=2))
    print(json.dumps({
        "binary": bool(df[args.target_col].nunique() <= 2 or args.positive_values is not None),
        "n_samples": int(df.shape[0]),
        "n_features_input": int(X.shape[1]),
        "top_k_features_path": str(out_dir / "top_k_features.json"),
        "ranking_path": str(out_dir / "feature_ranking_combined.csv")
    }, indent=2))


if __name__ == "__main__":
    main()
