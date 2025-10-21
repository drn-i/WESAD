import pandas as pd
import json
import sys
import argparse
from sklearn.model_selection import train_test_split

# --- Configuration ---
WESAD_DATA_FILE = 'm14_merged.csv'
FEATURE_IMPORTANCE_FILE = 'feature_importance_xgb_stress.csv'
HYPERTENSION_DATA_FILE = 'hypertension_dataset.csv'

N_TOP_FEATURES = 15
WESAD_LABEL_MAP = {1: 'baseline', 2: 'stress'}
HYPERTENSION_LABEL_MAP = {'No': 'baseline', 'Yes': 'stress'}
WESAD_EXCLUDE_COLS = ['label', 'subject', 'Unnamed: 0']
HYPERTENSION_FEATURES = [
    'Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI', 
    'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status'
]

# --- Prompt Engineering Functions ---

def create_wesad_prompt(row, feature_names):
    """Generates a text prompt for WESAD data."""
    prompt_lines = ["<s>[INST] Analyze the following physiological data from a 30-second window to determine the user's state."]
    for feature in feature_names:
        if feature in row:
            clean_name = feature.replace('_', ' ').title()
            value = row[feature]
            prompt_lines.append(f"- {clean_name}: {value:.4f}")
    prompt_lines.append("What is the user's stress level? [/INST]")
    return "\n".join(prompt_lines)

def create_hypertension_prompt(row):
    """Generates a text prompt for Hypertension data."""
    prompt_lines = ["<s>[INST] Analyze the following health risk factors to determine the user's state."]
    for feature in HYPERTENSION_FEATURES:
        if feature in row:
            clean_name = feature.replace('_', ' ').title()
            value = row[feature]
            prompt_lines.append(f"- {clean_name}: {value}")
    prompt_lines.append("What is the user's stress level? [/INST]")
    return "\n".join(prompt_lines)

def save_to_jsonl(dataset, filename):
    """Saves a list of dictionaries to a .jsonl file."""
    with open(filename, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + '\n')
    print(f"Successfully saved {len(dataset)} samples to {filename}")

# --- Processing Modes ---

def process_wesad(mode):
    """Prepares WESAD data, splitting 80/20."""
    print(f"--- Processing WESAD (mode: {mode}) ---")
    try:
        df = pd.read_csv(WESAD_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {WESAD_DATA_FILE}.")
        sys.exit(1)

    if mode == 'top':
        try:
            df_feats = pd.read_csv(FEATURE_IMPORTANCE_FILE)
            feature_names = df_feats.nlargest(N_TOP_FEATURES, 'importance')['feature'].tolist()
            print(f"Using Top {N_TOP_FEATURES} features.")
        except FileNotFoundError:
            print(f"Error: Could not find {FEATURE_IMPORTANCE_FILE}.")
            sys.exit(1)
    elif mode == 'full':
        feature_names = [col for col in df.columns if col not in WESAD_EXCLUDE_COLS]
        print(f"Using {len(feature_names)} features.")
    else:
        print(f"Error: Unknown mode '{mode}'. Use 'top' or 'full'.")
        sys.exit(1)

    df_filtered = df[df['label'].isin(WESAD_LABEL_MAP.keys())].copy()
    
    print("Performing 80/20 train-test split (stratified)...")
    df_train, df_test = train_test_split(
        df_filtered, test_size=0.2, random_state=42, stratify=df_filtered['label']
    )
    
    # Process and save training data
    train_dataset = []
    df_train['label_text'] = df_train['label'].map(WESAD_LABEL_MAP)
    for _, row in df_train.iterrows():
        train_dataset.append({
            "prompt": create_wesad_prompt(row, feature_names),
            "completion": f"{row['label_text']}</s>"
        })
    save_to_jsonl(train_dataset, f'train_dataset_{mode}.jsonl')

    # Process and save test data
    test_dataset = []
    df_test['label_text'] = df_test['label'].map(WESAD_LABEL_MAP)
    for _, row in df_test.iterrows():
        test_dataset.append({
            "prompt": create_wesad_prompt(row, feature_names),
            "completion": f"{row['label_text']}</s>"
        })
    save_to_jsonl(test_dataset, f'wesad_test_dataset_{mode}.jsonl')
    print("WESAD preparation complete.")

def process_hypertension():
    """Prepares the full Hypertension dataset for testing."""
    print("--- Processing Hypertension (mode: test) ---")
    try:
        df = pd.read_csv(HYPERTENSION_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {HYPERTENSION_DATA_FILE}.")
        sys.exit(1)

    df_filtered = df[df['Has_Hypertension'].isin(HYPERTENSION_LABEL_MAP.keys())].copy()
    df_filtered['label_text'] = df_filtered['Has_Hypertension'].map(HYPERTENSION_LABEL_MAP)

    dataset = []
    for _, row in df_filtered.iterrows():
        dataset.append({
            "prompt": create_hypertension_prompt(row),
            "completion": f"{row['label_text']}</s>"
        })
    save_to_jsonl(dataset, 'hypertension_test_dataset.jsonl')
    print("Hypertension preparation complete.")

# --- Main Executor ---
def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for LLM fine-tuning.")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['wesad-top', 'wesad-full', 'hypertension'],
        help="The dataset and mode to prepare."
    )
    args = parser.parse_args()

    if args.mode == 'wesad-top':
        process_wesad('top')
    elif args.mode == 'wesad-full':
        process_wesad('full')
    elif args.mode == 'hypertension':
        process_hypertension()

if __name__ == "__main__":
    main()
