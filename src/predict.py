import argparse
import pandas as pd
import joblib
from train import extract_num 

def preprocess(df):
    # 1) Numeric casts
    for c in ["bedroom_count","bathroom_count","parking_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["building_m2"] = df["building_size"].apply(extract_num)

    # 2) One-hot encode
    cat_cols = ["property_type","city","listing_agency"]
    df[cat_cols] = df[cat_cols].astype(str)
    return pd.get_dummies(df, columns=cat_cols, dummy_na=True)

def main(model_path, input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df_proc = preprocess(df)

    feat_cols = joblib.load("models/feature_columns.pkl")  
    df_proc = df_proc.reindex(columns=feat_cols, fill_value=0)

    model = joblib.load(model_path)
    df["predicted_price"] = model.predict(df_proc)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    required=True)
    p.add_argument("--input",    required=True)
    p.add_argument("--output",   required=True)
    args = p.parse_args()
    main(args.model, args.input, args.output)
