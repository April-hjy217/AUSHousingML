init:            # 安装依赖
  pip install -r requirements.txt

data:            # 生成特征表
  python src/prepare_data.py

train:           # 训练并注册模型
  python src/train.py --config config/train.yaml

evaluate:        # 评估并输出报告
  python src/evaluate.py --model outputs/model.joblib --data data/processed/feature_table.parquet
