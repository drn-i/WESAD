import os
import re
import sys
import json
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

WESAD_LABEL_MAP = {1: 'baseline', 2: 'stress'}

# --------- Actionable filters (built-in) ---------
#SUBJECT_META = {"age","height","weight","gender_ female","gender_ male", "smoker_NO","smoker_YES","coffee_today_YES","sport_today_YES","feel_ill_today_YES"}
SUBJECT_META = {"smoker_NO","smoker_YES","coffee_today_YES","sport_today_YES","feel_ill_today_YES"}

LABEL_DERIVED_PAT = re.compile(r"^[012]_(mean|std|min|max)$")
ACC_AXIS_MEANMINMAX_PAT = re.compile(r"^ACC_[xyz]_(mean|min|max)$")
FLAT_MEANMINMAX_PAT = re.compile(r"^(TEMP|BVP|Resp)_(mean|min|max)$")

PREFERRED_KEEP = [
    # EDA
    "EDA_phasic_mean","EDA_phasic_std","EDA_phasic_max","EDA_phasic_min",
    "EDA_smna_std","EDA_tonic_std","EDA_tonic_min",
    # Acc magnitude
    "net_acc_mean","net_acc_std","net_acc_max","net_acc_min",
    # Resp/BVP
    "Resp_std","BVP_std","BVP_peak_freq",
    # Temp dynamics
    "TEMP_slope",
]

ZEROISH = {
    "ACC_x_mean","ACC_y_mean","ACC_z_mean",
    "ACC_x_min","ACC_y_min","ACC_z_min",
    "ACC_x_max","ACC_y_max","ACC_z_max",
    "BVP_mean","BVP_min","BVP_max",
    "TEMP_mean","TEMP_min","TEMP_max",
    "Resp_mean","Resp_min","Resp_max",
    "0_mean","0_std","0_min","0_max",
}

def create_wesad_prompt(row, feature_names):
    lines = ["<s>[INST] Analyze the following physiological data from a 30-second window to determine the user's state."]
    for feature in feature_names:
        if feature in row:
            val = row[feature]
            try:
                lines.append(f"- {feature.replace('_',' ').title()}: {float(val):.4f}")
            except Exception:
                lines.append(f"- {feature.replace('_',' ').title()}: {val}")
    lines.append("What is the user's stress level? [/INST]")
    return "\n".join(lines)

def save_jsonl(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} samples to {path}")

def actionable_filtered_feature_list(df):
    cols = set(df.columns)
    drop = set()
    drop |= (SUBJECT_META & cols)
    drop |= {c for c in cols if LABEL_DERIVED_PAT.match(c)}
    drop |= {c for c in cols if ACC_AXIS_MEANMINMAX_PAT.match(c)}
    drop |= {c for c in cols if FLAT_MEANMINMAX_PAT.match(c)}
    drop |= (ZEROISH & cols)

    numeric_cols = df.select_dtypes(include="number").columns
    keep = [c for c in numeric_cols if c not in drop and c not in {"label","subject"}]

    # Prioritize preferred present
    preferred_present = [c for c in PREFERRED_KEEP if c in df.columns]
    seen = set()
    final_keep = []
    for c in preferred_present + keep:
        if c not in seen and c not in {"label","subject"}:
            final_keep.append(c)
            seen.add(c)
    if not final_keep:
        raise RuntimeError("No features left after filtering; relax rules.")
    return final_keep, sorted(drop)

def top_n_features_from_importance(importance_file, n_top, df):
    if not os.path.exists(importance_file):
        raise FileNotFoundError(f"Missing importance file: {importance_file}")
    fi = pd.read_csv(importance_file)
    feats = [f for f in fi.nlargest(n_top, 'importance')['feature'].tolist() if f in df.columns]
    if not feats:
        raise RuntimeError("Top-N selection produced empty list; check file/columns.")
    return feats

def split_subjects(df, train_n, test_n):
    subjects = sorted(df["subject"].unique().tolist())
    if len(subjects) < train_n + test_n:
        raise RuntimeError(f"Not enough subjects ({len(subjects)}) for split {train_n}/{test_n}.")
    train_subjects = subjects[:train_n]
    test_subjects = subjects[train_n:train_n+test_n]
    dtr = df[df["subject"].isin(train_subjects)].copy()
    dte = df[df["subject"].isin(test_subjects)].copy()
    return dtr, dte, train_subjects, test_subjects

def build_records(df_part, feats):
    records = []
    for _, r in df_part.iterrows():
        lbl = WESAD_LABEL_MAP.get(int(r["label"]), None)
        if lbl is None:
            continue
        records.append({
            "prompt": create_wesad_prompt(r, feats),
            "completion": f"{lbl}</s>"
        })
    return records

def main():
    ap = argparse.ArgumentParser(description="Prepare WESAD JSONL for LLM fine-tuning with multiple split/feature modes.")
    ap.add_argument("--data-file", type=str, default="data/m14_merged.csv", help="Merged WESAD CSV (subject,label + features).")
    ap.add_argument("--importance-file", type=str, default="artifacts/feature_importance_xgb_stress.csv", help="XGBoost importance CSV.")
    ap.add_argument("--feature-mode", type=str, choices=["top","filtered","all"], required=True, help="Feature selection mode.")
    ap.add_argument("--n-top", type=int, default=15, help="Top-N features when feature-mode=top.")
    ap.add_argument("--split-method", type=str, choices=["subject_12_3","subject_10_5","random_80_20"], required=True, help="Split strategy.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--output-prefix", type=str, required=True, help="Prefix for output files.")
    args = ap.parse_args()

    if not os.path.exists(args.data_file):
        print(f"Error: {args.data_file} not found.")
        sys.exit(1)
    df = pd.read_csv(args.data_file)
    if "subject" not in df.columns or "label" not in df.columns:
        print("Error: expected 'subject' and 'label' columns in data.")
        sys.exit(1)

    # Feature selection
    if args.feature_mode == "all":
        feats = [c for c in df.select_dtypes(include="number").columns if c not in {"label","subject"}]
        dropped = []
    elif args.feature_mode == "filtered":
        feats, dropped = actionable_filtered_feature_list(df)
    else:  # top
        feats = top_n_features_from_importance(args.importance_file, args.n_top, df)
        dropped = []

    # Splits
    if args.split_method == "subject_12_3":
        dtr, dte, tr_subj, te_subj = split_subjects(df, 12, 3)
    elif args.split_method == "subject_10_5":
        dtr, dte, tr_subj, te_subj = split_subjects(df, 10, 5)
    else:
        dtr, dte = train_test_split(df, test_size=0.2, random_state=args.random_state, stratify=df["label"])
        tr_subj = sorted(dtr["subject"].unique().tolist())
        te_subj = sorted(dte["subject"].unique().tolist())

    print(f"Split: {args.split_method}")
    print(f"  Train subjects ({len(tr_subj)}): {tr_subj}")
    print(f"  Test  subjects ({len(te_subj)}): {te_subj}")
    print(f"  Train samples: {len(dtr)} | Test samples: {len(dte)}")
    print("  Train label counts:\n", dtr["label"].value_counts().sort_index())
    print("  Test  label counts:\n", dte["label"].value_counts().sort_index())
    print(f"Feature mode: {args.feature_mode} | #features used: {len(feats)}")
    if dropped:
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/llm_filtered_dropped_features.txt", "w") as f:
            for c in dropped:
                f.write(f"{c}\n")
        print("  Wrote artifacts/llm_filtered_dropped_features.txt")

    # Records
    train_records = build_records(dtr, feats)
    test_records  = build_records(dte, feats)

    # Write
    train_out = f"{args.output_prefix}_train.jsonl"
    test_out  = f"{args.output_prefix}_test.jsonl"
    save_jsonl(train_records, train_out)
    save_jsonl(test_records, test_out)

if __name__ == "__main__":
    main()