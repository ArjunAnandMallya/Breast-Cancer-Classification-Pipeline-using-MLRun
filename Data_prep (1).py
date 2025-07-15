import mlrun
import pandas as pd
from sklearn.datasets import load_breast_cancer

@mlrun.handler()
def fetch_data(context: mlrun.MLClientCtx, dataset_key: str = "cancer_dataset"):
    context.logger.info(f"Fetching the breast cancer dataset")
    cancer = load_breast_cancer(as_frame=True)

    df = cancer.data.copy()
    df['target'] = cancer.target
    df['target_name'] = df['target'].map({0: cancer.target_names[0], 1: cancer.target_names[1]})

    context.logger.info(f"Dataset shape: {df.shape}")
    context.logger.info(f"Dataset columns: {df.columns.tolist()}")
    context.logger.info(f"Target distribution:\n{df['target_name'].value_counts()}")
    context.log_dataset(dataset_key, df=df, format="parquet", index=False)
    context.logger.info(f"Dataset logged with key: {dataset_key}")