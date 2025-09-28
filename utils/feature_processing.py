import pandas as pd
import numpy as np
import json

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
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:
                return {}
        out = {}
        if isinstance(entry, list):
            for d in entry:
                name = d.get("name")
                val = d.get("100g")
                if name and val is not None:
                    col = name.replace("-", "_") + "_100g"
                    out[col] = val
        elif isinstance(entry, dict):
            for k, v in entry.items():
                out[k] = v
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