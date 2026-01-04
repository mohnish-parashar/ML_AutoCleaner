import pandas as pd
import numpy as np
import random
import string
import os

# -----------------------------
# Corruption Functions
# -----------------------------

def introduce_nulls(df, null_prob=0.1):
    df = df.copy()
    for col in df.columns:
        mask = np.random.rand(len(df)) < null_prob
        df.loc[mask, col] = np.nan
    return df


def introduce_wrong_dtypes(df, prob=0.1):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and random.random() < prob:
            df.loc[df.sample(frac=0.1).index, col] = "invalid"
        elif pd.api.types.is_object_dtype(df[col]) and random.random() < prob:
            df.loc[df.sample(frac=0.1).index, col] = np.random.randint(100, 999)
    return df


def introduce_typos(df, prob=0.1):
    df = df.copy()

    def typo(word):
        if not isinstance(word, str) or len(word) < 2:
            return word
        i = random.randint(0, len(word) - 1)
        return word[:i] + random.choice(string.ascii_lowercase) + word[i + 1:]

    for col in df.select_dtypes(include="object").columns:
        for idx in df.sample(frac=prob).index:
            df.at[idx, col] = typo(df.at[idx, col])

    return df


def introduce_outliers(df, prob=0.05):
    df = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        idx = df.sample(frac=prob).index
        mean = df[col].mean()
        std = df[col].std()
        df.loc[idx, col] = mean + (10 * std)
    return df


def introduce_inconsistent_formats(df, prob=0.1):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        for idx in df.sample(frac=prob).index:
            val = df.at[idx, col]
            if isinstance(val, str):
                df.at[idx, col] = random.choice([
                    val.upper(),
                    val.lower(),
                    f" {val} ",
                    val.replace("-", "/")
                ])
    return df


# -----------------------------
# Dataset Corruption Pipeline
# -----------------------------

def corrupt_dataframe(df):
    df = introduce_nulls(df)
    df = introduce_wrong_dtypes(df)
    df = introduce_typos(df)
    df = introduce_outliers(df)
    df = introduce_inconsistent_formats(df)
    return df


# -----------------------------
# Folder-Level Processing
# -----------------------------

def corrupt_datasets_in_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            input_path = os.path.join(input_folder, file)
            output_name = file.replace(".csv", "_corrupted.csv")
            output_path = os.path.join(output_folder, output_name)

            print(f"Corrupting: {file}")

            df = pd.read_csv(input_path)
            corrupted_df = corrupt_dataframe(df)
            corrupted_df.to_csv(output_path, index=False)

    print("\nAll datasets corrupted successfully.")




# -----------------------------
# Example Usage
# -----------------------------

if __name__ == "__main__":
    corrupt_datasets_in_folder(
        input_folder="clean_datasets",
        output_folder="corrupted_datasets"
    )
