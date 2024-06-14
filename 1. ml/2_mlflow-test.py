# NOTE: Start the server first "mlflow ui" in the command line

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
from mlflow.models import infer_signature

# Machine learning models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# 2. Setup the experiment
df = pd.read_csv('../data/adult.csv')
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype.kind in 'fi':
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col] = encoder.fit_transform(df[col])

y = df["income"]
X = df.drop("income", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Income Classification on US Census Data")


def train_and_log_model(model):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy for {model.__class__.__name__}: {accuracy}")

        mlflow.log_params(
            {param: value for param, value in model.get_params().items()})
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Model Type", model.__class__.__name__)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="models",
            signature=signature,
            input_example=X_train.iloc[0:1],
            registered_model_name=f"{model.__class__.__name__}-Adult-Income"
        )


# List of models to train
models = [
    KNeighborsClassifier(n_neighbors=5),
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(),
    SVC(probability=True),
]

# Train and log each model
for model in models:
    train_and_log_model(model)
