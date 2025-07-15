import mlrun
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlrun.frameworks.sklearn import apply_mlrun

@mlrun.handler()
def train_model(
    context: mlrun.MLClientCtx, # used for automatic logging and tracking
    dataset: mlrun.DataItem, # this is the dataset coming from the previous step
    model_name: str = "cancer_rf_model"
):
    df = dataset.as_df()
    X = df.drop(['target', 'target_name'], axis=1, errors='ignore')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )
    model = RandomForestClassifier(
        random_state=42)
    apply_mlrun(
        model=model,
        model_name=model_name,
        x_test=X_test, 
        y_test=y_test,
    )
    model.fit(X_train, y_train)
