import os
import argparse
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from sklearn.metrics import r2_score, mean_absolute_error
import logging
import datetime

def extract_num(x):
    if pd.isnull(x):
        return None
    m = re.search(r'(\d+(?:\.\d+)?)', str(x))
    return float(m.group(1)) if m else None

# === Added: Simple model monitoring function ===
def model_monitor(mae, r2, threshold=100000):
    with open("metrics.log", "a") as f:
        f.write(f"{datetime.datetime.now()}, mae={mae:.4f}, r2={r2:.4f}\n")
    if mae > threshold:
        logging.warning(f"MAE {mae:.2f} exceeded threshold {threshold}!")
        # Optionally, you can uncomment this line to make the Prefect flow fail:
        # raise Exception(f"MAE {mae:.2f} exceeded threshold {threshold}!")

def main(feature_path, target_arg, model_out, val_out,
         test_size=0.2, random_seed=42):
    # 0) Configure MLflow experiment
    mlflow.set_experiment("AUSHousingML")

    with mlflow.start_run() as run:
        # 1) Load features
        df = pd.read_parquet(feature_path)

        # 2) Numeric conversions
        for col in ["bedroom_count", "bathroom_count", "parking_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "building_size" in df.columns:
            df["building_m2"] = df["building_size"].apply(extract_num)

        # 3) One-hot encode categorical columns
        cat_cols = [c for c in ["property_type", "city", "listing_agency"] if c in df.columns]
        if cat_cols:
            df[cat_cols] = df[cat_cols].astype(str)
            df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

        # 4) Determine target column
        if target_arg:
            if target_arg not in df.columns:
                matches = [c for c in df.columns if c.lower() == target_arg.lower()]
                if not matches:
                    raise KeyError(f"Target column '{target_arg}' not found.")
                target_col = matches[0]
            else:
                target_col = target_arg
        else:
            price_cols = [c for c in df.columns if "price" in c.lower()]
            if not price_cols:
                raise KeyError("No column containing 'price' found.")
            if len(price_cols) > 1:
                raise KeyError(f"Multiple price-like columns found: {price_cols}. Use --target-col.")
            target_col = price_cols[0]

        # 5) Clean & rename target
        df = df.rename(columns={target_col: "price_num"})
        df["price_num"] = (
            df["price_num"].astype(str)
            .replace(r'[\$,]', '', regex=True)
        )
        df["price_num"] = pd.to_numeric(df["price_num"], errors="coerce")
        df = df.dropna(subset=["price_num"]).reset_index(drop=True)

        # 6) Select numeric features
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if "price_num" in numeric_cols:
            numeric_cols.remove("price_num")
        if not numeric_cols:
            raise ValueError("No numeric features available for training.")

        # Persist feature list
        os.makedirs("models", exist_ok=True)
        joblib.dump(numeric_cols, "models/feature_columns.pkl")
        mlflow.log_artifact("models/feature_columns.pkl", artifact_path="features")

        X = df[numeric_cols]
        y = df["price_num"]
        if X.shape[0] == 0:
            raise ValueError("No rows remain after dropping NaNs in 'price_num'.")

        # 7) Train/validation split
        mlflow.log_params({
            "test_size": test_size,
            "random_seed": random_seed,
            "n_features": len(numeric_cols)
        })
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_seed
        )
        os.makedirs(os.path.dirname(val_out), exist_ok=True)
        pd.concat([X_val, y_val], axis=1).to_parquet(val_out, index=False)

        # 8) Train XGBoost
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        }
        mlflow.log_params(params)
        model = XGBRegressor(
            **params,
            random_state=random_seed,
            n_jobs=-1,
            eval_metric="mae",
            early_stopping_rounds=20,
            enable_categorical=False
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 9) Log metrics
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        mlflow.log_metric("val_R2", r2)
        mlflow.log_metric("val_MAE", mae)

        # === Added: record monitoring log and warning ===
        model_monitor(mae, r2, threshold=100000)

        # 10) Save model locally
        joblib.dump(model, model_out)

        # 10b) Save model to minio S3 bucket, with timestamped file name
        try:
            import boto3
            run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            minio_client = boto3.client(
                "s3",
                endpoint_url="http://minio:9000",
                aws_access_key_id="minio",
                aws_secret_access_key="minio123",
                region_name="us-east-1"
            )
            try:
                minio_client.create_bucket(Bucket="housing-model-artifacts")
            except minio_client.exceptions.BucketAlreadyOwnedByYou:
                pass
            except minio_client.exceptions.BucketAlreadyExists:
                pass

            minio_client.upload_file(
                model_out,
                "housing-model-artifacts",
                f"best_model_{run_id}.pkl"
            )
            print(f"[train] Model uploaded to minio S3 as best_model_{run_id}.pkl")
        except Exception as e:
            print(f"[train] WARNING: Failed to upload to minio S3: {e}")

        # 11) Infer signature & log model to MLflow
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)
        mlflow.xgboost.log_model(
            model,
            name="model",
            signature=signature,
            input_example=input_example
        )

        # 12) Register in Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "AUSHousingModel")

        print(f"[train] Feature columns saved to models/feature_columns.pkl")
        print(f"[train] Model saved to: {model_out}")
        print(f"[train] Validation set saved to: {val_out}")
        print(f"[train] R²={r2:.4f}, MAE={mae:.2f}, run_id={run.info.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an XGBoost regression model with MLflow"
    )
    parser.add_argument(
        "--features", required=True,
        help="Path to feature_table.parquet"
    )
    parser.add_argument(
        "--target-col", default="",
        help="Original price column name; case-insensitive"
    )
    parser.add_argument(
        "--model-out", required=True,
        help="Where to save the trained model (.pkl)"
    )
    parser.add_argument(
        "--val-out", required=True,
        help="Where to save the validation set (.parquet)"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data for validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    args = parser.parse_args()

    main(
        args.features,
        args.target_col.strip() or None,
        args.model_out,
        args.val_out,
        test_size=args.test_size,
        random_seed=args.seed
    )
