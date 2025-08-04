import pandas as pd
import requests
from datetime import datetime
import os
from functools import reduce

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RAW_DIR      = os.path.join(PROJECT_ROOT, "raw")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")


files_and_names = {
    "historical_country_australia_indicator_average_house_prices_.csv": "avg_house_price",
    "historical_country_australia_indicator_corelogic_dwelling_prices_mom_ (1).csv": "corelogic_mom",
    "historical_country_australia_indicator_mortgage_rate_.csv": "mortgage_rate",
    "historical_country_australia_indicator_home_loans_.csv": "home_loans",
    "historical_country_australia_indicator_investment_lending_for_homes_.csv": "investment_loans",
    "historical_country_australia_indicator_building_permits_.csv": "building_permits"
}




def load_and_agg_macros():
    dfs = []
    for fname, new_col in files_and_names.items():
        path = os.path.join(RAW_DIR, fname)
        tmp = pd.read_csv(path, nrows=5)
        date_col = next(c for c in tmp.columns if "date" in c.lower())
        df = pd.read_csv(path, parse_dates=[date_col])
        val_col = next(c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c]))
        df = df.rename(columns={date_col: "date", val_col: new_col})
        df["quarter"] = df["date"].dt.to_period("Q")
        qdf = df.groupby("quarter", as_index=False)[new_col].mean()
        dfs.append(qdf)
    df_all = reduce(lambda left, right: pd.merge(left, right, on="quarter", how="outer"), dfs)
    return df_all

def load_and_merge_props(df_macros):
    prop_path = os.path.join(RAW_DIR, "RealEstateAU_1000_Samples.csv")
    dfp = pd.read_csv(prop_path, parse_dates=["RunDate"])
    dfp["quarter"] = dfp["RunDate"].dt.to_period("Q")
    df_merged = dfp.merge(df_macros, on="quarter", how="left")
    return df_merged

def main():
    df_macros = load_and_agg_macros()
    print(df_macros.head())

    df_feat = load_and_merge_props(df_macros)
    print(df_feat.columns.tolist())


    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "feature_table.parquet")
    df_feat.to_parquet(out_path, index=False)

if __name__ == "__main__":
    main()
