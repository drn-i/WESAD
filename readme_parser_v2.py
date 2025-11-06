import os
import re
import pandas as pd
import argparse


class rparser:
    VALUE_EXTRACT_KEYS = {
        "age": {'search_key': 'Age', 'delimiter': ':'},
        "height": {'search_key': 'Height', 'delimiter': ':'},
        "weight": {'search_key': 'Weight', 'delimiter': ':'},
        "gender": {'search_key': 'Gender', 'delimiter': ':'},
        "dominant_hand": {'search_key': 'Dominant', 'delimiter': ':'},
        "coffee_today": {'search_key': 'Did you drink coffee today', 'delimiter': '? '},
        "coffee_last_hour": {'search_key': 'Did you drink coffee within the last hour', 'delimiter': '? '},
        "sport_today": {'search_key': 'Did you do any sports today', 'delimiter': '? '},
        "smoker": {'search_key': 'Are you a smoker', 'delimiter': '? '},
        "smoke_last_hour": {'search_key': 'Did you smoke within the last hour', 'delimiter': '? '},
        "feel_ill_today": {'search_key': 'Do you feel ill today', 'delimiter': '? '}
    }

    parse_file_suffix = '_readme.txt'

    def __init__(self, data_path, output_path="/kaggle/working/"):
        """
        Initialize the parser with:
        - data_path: path to the WESAD dataset (read-only OK)
        - output_path: where to save parsed files (must be writable)
        Example:
            parser = rparser(
                "/kaggle/input/wesad-wearable-stress-affect-detection-dataset/WESAD",
                "/kaggle/working/"
            )
        """
        self.DATA_PATH = data_path if data_path.endswith('/') else data_path + '/'
        self.OUTPUT_PATH = output_path if output_path.endswith('/') else output_path + '/'

        if not os.path.exists(self.OUTPUT_PATH):
            os.makedirs(self.OUTPUT_PATH, exist_ok=True)

        # Collect subject directories (e.g., S2, S3, S4, ...)
        self.readme_locations = {
            subject_directory: os.path.join(self.DATA_PATH, subject_directory, '')
            for subject_directory in os.listdir(self.DATA_PATH)
            if re.match(r'^S[0-9]{1,2}$', subject_directory)
        }

        readme_csv_path = os.path.join(self.OUTPUT_PATH, 'readmes.csv')

        if not os.path.isfile(readme_csv_path):
            print('Parsing Readme files...')
            self.parse_all_readmes()
        else:
            print(f'Files already parsed. Found at {readme_csv_path}')

        self.merge_with_feature_data()

    def parse_readme(self, subject_id):
        with open(self.readme_locations[subject_id] + subject_id + self.parse_file_suffix, 'r') as f:
            x = f.read().split('\n')

        readme_dict = {}

        for item in x:
            for key, val in self.VALUE_EXTRACT_KEYS.items():
                if item.startswith(val['search_key']):
                    _, v = item.split(val['delimiter'])
                    readme_dict[key] = v.strip()
                    break
        return readme_dict

    def parse_all_readmes(self):
        dframes = []

        for subject_id, path in self.readme_locations.items():
            readme_dict = self.parse_readme(subject_id)
            df = pd.DataFrame(readme_dict, index=[subject_id])
            dframes.append(df)

        df = pd.concat(dframes)

        output_path = os.path.join(self.OUTPUT_PATH, 'readmes.csv')
        df.to_csv(output_path)
        print(f"✅ Parsed readmes saved to: {output_path}")

    def merge_with_feature_data(self):
        feat_path = 'data/may14_feats4.csv'
        if not os.path.isfile(feat_path):
            print('⚠️ No feature data available. Skipping merge...')
            return

        readme_df_path = os.path.join(self.OUTPUT_PATH, 'readmes.csv')
        if not os.path.isfile(readme_df_path):
            print(f'⚠️ Readme CSV not found at {readme_df_path}')
            return

        feat_df = pd.read_csv(feat_path, index_col=0)
        df = pd.read_csv(readme_df_path, index_col=0)

        dummy_df = pd.get_dummies(df)
        dummy_df['subject'] = dummy_df.index.str[1:].astype(int)

        keep_cols = [c for c in dummy_df.columns if 'subject' in c or c in [
            'age', 'height', 'weight',
            'gender_ female', 'gender_ male',
            'coffee_today_YES', 'sport_today_YES',
            'smoker_NO', 'smoker_YES', 'feel_ill_today_YES'
        ]]
        dummy_df = dummy_df[keep_cols]

        merged_df = pd.merge(feat_df, dummy_df, on='subject')
        merged_path = os.path.join(self.OUTPUT_PATH, 'm14_merged.csv')
        merged_df.to_csv(merged_path)
        print(f"✅ Merged data saved to: {merged_path}")


# --- Command line usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse WESAD readme files.")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to WESAD dataset directory.")
    parser.add_argument("--output_path", type=str, default="/kaggle/working/",
                        help="Writable output directory (default: /kaggle/working/)")
    args = parser.parse_args()

    rparser(args.data_path, args.output_path)
