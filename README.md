# 🏡 AU Housing Price Prediction — End-to-End MLOps Project

## Overview

This project demonstrates a MLOps pipeline for predicting Australian residential house prices. 

This project empowers analysts, real estate professionals, and buyers with **automated, and monitorable house price predictions** using up-to-date property data.

---

## 🚀 Key Features

- **XGBoost regression model** trained on real estate features  
- **Prefect 2.0** for workflow orchestration (ETL → train → evaluate → monitor)
- **MLflow** for experiment tracking & model registry
- **Flask API** for real-time model serving
- **MinIO** for S3-compatible artifact storage
- **Python model monitoring** (log + threshold alerts)
- **Docker Compose** for local, multi-service stack
- **CI/CD-ready:** Makefile, pytest, flake8, GitHub Actions compatible

---

## 🏦 Business Context & Problem

Australian property prices are dynamic and region-specific.  
Accurate price predictions are vital for banks, buyers, and sellers to inform investment, lending, and pricing decisions.

**ML Objective:**  
Predict a property’s sale price from features (location, size, type, etc.) with accuracy, automated workflow, and monitoring.

---

## 📊 Dataset

- **Source:**  
  [Kaggle: Australian Housing Data (1,000+ properties)](https://www.kaggle.com/datasets/thedevastator/australian-housing-data-1000-properties-sampled)  
  [Trading Economics API](https://tradingeconomics.com/api/?source=footer)
- **Data Storage:**  
  - Raw data: `data/raw/`
  - Processed features: `data/processed/feature_table.parquet`, `val.parquet`
  - Model artifacts: `models/`
  - Metrics log: `metrics.log`
  - MLflow runs: `mlruns/`

---
## 🔬 Core Pipeline

1. **Data Preprocessing** (`src/prepare_data.py`)
    - Loads, cleans, and engineers features
    - Output: `data/processed/feature_table.parquet`
2. **Model Training** (`src/train.py`)
    - XGBoost regression training
    - MLflow logging, metrics logging, saves best model locally & to MinIO S3
3. **Prediction** (`src/predict.py`)
    - Loads latest model from `models/`
    - Predicts on new/batch data; outputs results to CSV or console
4. **Orchestration** (`orchestrate.py`)
    - ETL → train → evaluate → monitor as a Prefect flow
    - Run via Prefect or `make pipeline`
5. **API Deployment** (`src/api.py`)
    - `/predict` endpoint for JSON batch/real-time prediction
6. **Model Monitoring** (`metrics.log`)
    - Each run logs MAE & R²
    - Alerts if MAE exceeds threshold (extensible for production alerts)

---

## 🎛️ Experiment Tracking & Storage

- **MLflow UI:**  
  Run `mlflow ui --backend-store-uri mlruns/`  
  View at [http://localhost:5000](http://localhost:5000)
- **MinIO (S3):**  
  View uploaded model files at [http://localhost:9001](http://localhost:9001)  
  Login: minio / minio123

---

## 🐳 Deployment & CI/CD

- **Docker Compose:**  
  Spin up full stack (API, Prefect, worker, MinIO) in one command
- **Makefile:**  
  Unified automation: `make build`, `make lint`, `make test`, `make up`, `make down`
- **Testing:**  
  Pytest suite for key modules
- **Linting:**  
  Flake8 for PEP8 code style compliance
- **CI/CD:**  
  GitHub Actions compatible

---

## 🏁 How to Run (Locally)

```bash
# 1. Build and start everything (API, Prefect, MinIO)
docker-compose up --build

# 2. Run the full ML pipeline
make pipeline

# 3. Test prediction API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [485.0, 1351473.0, 139208727.0, 832.0, null, null, 3.0, 2.0, 2.0, 921.2, -0.56, 2.83, 56097.6, 29258.7, 16232.0, 212.0]}'

# 4. View dashboards:
# MLflow:  http://localhost:5000
# MinIO:   http://localhost:9001
# Prefect: http://localhost:4200
