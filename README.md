🏡 AU Housing Price Prediction — End-to-End MLOps Project
1. Overview
This project demonstrates an end-to-end MLOps workflow for predicting residential house prices in Australia. The goal is to empower financial analysts, real estate professionals, and buyers with reliable, up-to-date property value estimates using a robust, automated, and monitorable pipeline.

Key Features:

🧠 XGBoost-based regression model trained on real estate data

🚀 Prefect 2.0 for workflow orchestration

📈 MLflow for experiment tracking and model registry

⚙️ Flask API for batch & real-time model serving

🔍 Python-based model monitoring and threshold alerts

📦 Docker (Compose) for local containerization

☁️ MinIO for S3-compatible artifact storage

↻ CI/CD-ready (Makefile, GitHub Actions compatible)


2. Problem Description
Business Context
Property prices in Australia are dynamic, region-specific, and sensitive to market changes. Accurate price prediction is critical for banks, buyers, and sellers to inform decisions, assess loan risks, and plan investments.

Problem Statement:
Can we predict a property’s sale price from its features (location, size, property type, etc.) using recent and historical data?

ML Objective:
Train a regression model to predict house prices with high accuracy, automate the entire workflow, and monitor model health for production-readiness.

3. Dataset Information
Source: [https://www.kaggle.com/datasets/thedevastator/australian-housing-data-1000-properties-sampled; https://tradingeconomics.com/api/?source=footer]

Format: Processed to feature_table.parquet (for modeling) and val.parquet (for validation)

Features: Includes location, size, bedroom/bathroom counts, property type, etc.

Storage:

Raw data: data/raw/

Processed data: data/processed/

Model artifacts: models/

Metrics log: metrics.log

MLflow experiments: mlruns/

4. Technical Structure
Module/Folder	Description
data/	Raw, processed, and validation datasets
models/	Trained model files, feature lists
src/	All core Python code (pipeline, training, API)
api.py	Flask API for prediction
train.py	Training pipeline with MLflow logging
evaluate.py	Evaluation metrics & monitoring
predict.py	CLI & batch prediction script
test_train.py	Pytest unit/integration tests
docker-compose.yaml	Full multi-service stack
Makefile	Build, test, lint, run automation
requirements.txt	Full dependencies, pinned versions

MLflow UI: Experiment and model registry at http://localhost:5000 (if enabled)
MinIO: S3 artifact store UI at http://localhost:9001 (default login: minio/minio123)

5. Database & Storage Setup
MLflow Tracking DB: Local SQLite (mlruns/)

Artifacts: Saved in models/ and uploaded to MinIO (simulates S3)

Data: CSV/Parquet files in data/ (can scale to cloud storage)

6. ML Pipeline Breakdown
⚙️ Data Preprocessing (src/prepare_data.py)
Loads and cleans raw housing data

Feature engineering (numeric, categorical encoding)

Outputs processed features to feature_table.parquet

📈 Model Training with MLflow (src/train.py)
Trains XGBoost regressor

Evaluates MAE, R², logs metrics/artifacts to MLflow and metrics.log

Saves best model (locally and to MinIO)

MLflow UI for experiment comparison
mlflow ui --backend-store-uri mlruns/
# Visit http://127.0.0.1:5000

🎺 Prediction (src/predict.py)
Loads latest model from models/ or MLflow registry

Predicts on new data or batch CSV

Writes results to CSV or prints to console
python src/predict.py --input data/feature_table.parquet

↻ Orchestration with Prefect (orchestrate.py)
End-to-end @flow: prepare → train → evaluate → monitor
prefect deployment run 'AUSHousingML Pipeline/AUSHousingML Deployment'
Or via Docker Compose.

🌐 API Deployment (src/api.py)
/predict endpoint for batch/single predictions via JSON

Fully containerized (see Dockerfile.api)
make api
# Or run docker-compose up api
# Visit http://localhost:8000/docs for Swagger UI

🟢 Model Monitoring (metrics.log)
Every run logs R² and MAE

If MAE exceeds threshold, triggers warning (potential for alerts)

7. Deployment & CI/CD
Docker Compose: Local stack with Prefect, worker, API, and MinIO

Makefile: All workflows (make build, make lint, make test, make up, make down, etc.)


8. Monitoring and Reproducibility
Metrics Logging: All evaluation metrics go to metrics.log for post-hoc analysis and production alerting

MLflow: Tracks all params, runs, and artifacts for model reproducibility

MinIO: S3 storage ensures deployment-ready artifact management
