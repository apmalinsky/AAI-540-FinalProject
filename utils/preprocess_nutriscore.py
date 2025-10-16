# Preprocessing Script Used in CI/CD Pipeline

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define our feature engineering function
def process_data_chunk(df_chunk):
    """
    Performs feature engineering steps on a single chunk of the raw data.
    """
    # Extract 'text' value from product_name field if it's a list of dicts.
    def extract_product_name(name_field):
        if isinstance(name_field, list) and len(name_field) > 0:
            return name_field[0].get("text", None)
        return name_field
    
    # Apply cleaning
    df_chunk["product_name"] = df_chunk["product_name"].apply(extract_product_name)
    df_chunk['product_name'] = df_chunk['product_name'].str.replace(r'[\r\n]+', ' ', regex=True)
    
    # Extract Nutrient Information
    def parse_nutriments(entry):
        if entry is None or (isinstance(entry, float) and np.isnan(entry)):
            return {}

        out = {}
        try:
            for d in entry:
                if isinstance(d, dict):
                    name = d.get("name")
                    val = d.get("100g")
                    if name and val is not None:
                        col = name.replace("-", "_") + "_100g"
                        out[col] = val
        except TypeError:
            if isinstance(entry, dict):
                 for k, v in entry.items():
                    out[k] = v
            pass # Return empty dict if it's not a recognized format
            
        return out
    
    # Apply parsing
    nutri_df = pd.json_normalize(df_chunk["nutriments"].apply(parse_nutriments))

    # Convert to numeric
    for c in nutri_df.columns:
        nutri_df[c] = pd.to_numeric(nutri_df[c], errors="coerce")

    # If sodium is present but salt missing, estimate salt (g) = sodium (g) * 2.5
    if "sodium_100g" in nutri_df.columns and "salt_100g" in nutri_df.columns:
        need_salt = nutri_df["salt_100g"].isna() & nutri_df["sodium_100g"].notna()
        nutri_df.loc[need_salt, "salt_100g"] = nutri_df.loc[need_salt, "sodium_100g"] * 2.5

    # If both energy_kcal and energy_kJ exist, prefer kcal; if only kJ is present:
    # kcal = kJ / 4.184
    nutri_df.rename(columns={"energy-kcal_100g": "energy_kcal_100g"}, inplace=True)
    
    # Re-assemble the final feature table
    id_cols = ["code", "product_name"]
    raw_meta_cols = ["nova_group", "additives_n", "ingredients_n"]
    target_col = "nutriscore_score"

    features_df = pd.concat(
        [
            df_chunk[id_cols + raw_meta_cols + [target_col]].reset_index(drop=True),
            nutri_df.reset_index(drop=True),
        ],
        axis=1
    )
    
    # Enforce numeric types on meta columns that should be numeric
    for c in ["nova_group", "additives_n", "ingredients_n", target_col]:
        features_df[c] = pd.to_numeric(features_df[c], errors="coerce")

    return features_df

if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True)
    args, _ = parser.parse_known_args()
    
    base_dir = "/opt/ml/processing"
    input_data_path = args.input_path

    # Load raw input data parquet files
    all_files = [
        os.path.join(input_data_path, f) 
        for f in os.listdir(input_data_path)
        if os.path.isfile(os.path.join(input_data_path, f)) # ignore utils
    ]

    # Feature engineering in chunks
    chunk_size = 10000
    all_processed_chunks = []
    for file_path in all_files:
        chunk_df = pd.read_parquet(file_path)
        processed_chunk = process_data_chunk(chunk_df)
        all_processed_chunks.append(processed_chunk)
        print(f"Processed {len(all_processed_chunks) * chunk_size} rows...")

    # Concatenate all processed chunks into a final DataFrame
    df = pd.concat(all_processed_chunks, ignore_index=True)
    print(f"Initial data shape after chunk processing: {df.shape}")

    # Clean and impute data
    print("Cleaning and imputing data...")
    null_rates = df.isna().mean()
    # Drop columns that are mostly missing (>50% NaN)
    mostly_missing = null_rates[null_rates > 0.50].index.tolist()
    
    # Impute missing product_names
    df['product_name'] = df['product_name'].fillna('nan_product_name')
    
    if mostly_missing:
        print("Dropping:", mostly_missing)
        print(f"Dropping a total of {len(mostly_missing)} columns")
        df.drop(columns=mostly_missing, inplace=True)

    # Drop extra target col
    df.drop('nutrition_score_fr_100g', axis=1)

    # Impute core nutrient values using the median
    median_impute_cols = [
        'energy_100g', 'sodium_100g', 'proteins_100g',
        'salt_100g', 'carbohydrates_100g', 'energy_kcal_100g',
        'sugars_100g', 'fat_100g', 'saturated_fat_100g',
        'fiber_100g', 'cholesterol_100g', 'calcium_100g',
        'iron_100g', 'vitamin_c_100g', 'vitamin_a_100g'
    ]
    
    for col in median_impute_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Impute other columns with 0, as they are not present
    zero_impute_cols = [
        'trans_fat_100g',
        'fruits_vegetables_legumes_estimate_from_ingredients_100g',
        'fruits_vegetables_nuts_estimate_from_ingredients_100g'
    ]
    
    for col in zero_impute_cols:
        df[col] = df[col].fillna(0)
    print(f"Data shape after cleaning: {df.shape}")

    # Split data into train/test/validation
    print("Splitting data...")
    train_data, validation_data, test_data = np.split(
        df.sample(frac=1, random_state=42),
        [int(0.7 * len(df)), int(0.85 * len(df))],
    )

    # Scale features
    print("Scaling features...")
    target_column = 'nutriscore_score'
    cols_to_exclude = [
        'nutriscore_score', 'code', 'product_name', 'nutrition_score_fr_100g',
        'eventtime', 'write_time', 'api_invocation_time', 'is_deleted'
    ]
    feature_columns = [
        col for col in df.columns 
        if pd.api.types.is_numeric_dtype(df[col]) and col not in cols_to_exclude
    ]

    scaler = StandardScaler()
    
    train_features = train_data[feature_columns]
    train_features_scaled = scaler.fit_transform(train_features)
    
    validation_features = validation_data[feature_columns]
    validation_features_scaled = scaler.transform(validation_features)
    
    test_features = test_data[feature_columns]
    test_features_scaled = scaler.transform(test_features)

    # Set final datasets
    print("Preparing final datasets...")
    train_final = pd.DataFrame(train_features_scaled, columns=feature_columns)
    train_final.insert(0, target_column, train_data[target_column].values)

    validation_final = pd.DataFrame(validation_features_scaled, columns=feature_columns)
    validation_final.insert(0, target_column, validation_data[target_column].values)

    test_final = pd.DataFrame(test_features_scaled, columns=feature_columns)
    test_final.insert(0, target_column, test_data[target_column].values)
    
    # Save outputs
    print("Saving processed data...")
    train_final.to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    validation_final.to_csv(f"{base_dir}/validation/validation.csv", header=False, index=False)
    test_final.to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
    
    print("Preprocessing script finished successfully.")