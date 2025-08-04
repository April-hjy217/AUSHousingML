from prefect import flow, task
import subprocess

@task(retries=2, retry_delay_seconds=300)
def prepare_data():
    subprocess.run(["python", "src/prepare_data.py"], check=True)

@task(retries=3, retry_delay_seconds=600)
def train_model():
    subprocess.run([
        "python", "src/train.py",
        "--features", "data/feature_table.parquet",
        "--target-col", "price",
        "--model-out", "models/best_model.pkl",
        "--val-out", "data/processed/val.parquet"
    ], check=True)

@task
def evaluate_model():
    subprocess.run([
        "python", "src/evaluate.py",
        "--model", "models/best_model.pkl",
        "--val", "data/processed/val.parquet"
    ], check=True)

@task
def batch_predict():
    subprocess.run([
        "python", "src/predict.py",
        "--model", "models/best_model.pkl",
        "--input", "raw/Dummy_Listings_CSV.csv",
        "--output", "data/processed/dummy_predictions.csv"
    ], check=True)

@flow(name="AUSHousingML Pipeline")
def housing_pipeline():
    prepare_data()
    train_model()
    evaluate_model()
    

if __name__ == "__main__":
    housing_pipeline()
