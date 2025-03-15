import os
import pandas as pd
import argparse

def copy_first_200_rows(origin_folder, target_folder):
    os.makedirs(target_folder, exist_ok=True)

    for filename in os.listdir(origin_folder):
        if filename.endswith(".csv"):
            origin_path = os.path.join(origin_folder, filename)
            target_path = os.path.join(target_folder, filename)
            df = pd.read_csv(origin_path)
            df.head(200).to_csv(target_path, index=False)

            print(f"Processed: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Copy first 200 rows of CSV files.")
    parser.add_argument("--origin", required=True, help="Path to the origin folder.")
    parser.add_argument("--target", required=True, help="Path to the target folder.")

    args = parser.parse_args()

    copy_first_200_rows(args.origin, args.target)

if __name__ == "__main__":
    main()