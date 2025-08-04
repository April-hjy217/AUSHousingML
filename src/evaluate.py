import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import mean_absolute_error, r2_score

def main(model_path, val_parquet):
    # 1) Load model and validation data
    model = joblib.load(model_path)
    df_val = pd.read_parquet(val_parquet)

    # 2) Split into X and y
    X_val = df_val.drop("price_num", axis=1)
    y_val = df_val["price_num"]

    # 3) Predict & print metrics
    y_pred = model.predict(X_val)
    print(f"R²  : {r2_score(y_val, y_pred):.4f}")
    print(f"MAE : {mean_absolute_error(y_val, y_pred):.4f}")

    # 4) Plot residuals distribution
    residuals = y_val - y_pred
    plt.figure(figsize=(8,5))
    plt.hist(residuals, bins=50, edgecolor='k')
    plt.title("Residuals Distribution")
    plt.xlabel("y_true – y_pred")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("residuals.png")
    plt.show()

    # 5) Plot top-10 feature importance
    plt.figure(figsize=(8,5))
    plot_importance(model.get_booster(), max_num_features=10)
    plt.title("Top 10 Feature Importance")
    plt.savefig("feature_importance.png")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained XGBoost model on the validation set"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the trained model (.pkl), e.g. models/best_model.pkl"
    )
    parser.add_argument(
        "--val", required=True,
        help="Path to the validation set (.parquet), e.g. data/processed/val.parquet"
    )
    args = parser.parse_args()
    main(args.model, args.val)
