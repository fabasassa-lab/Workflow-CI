import dagshub
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import numpy as np

dagshub.init(repo_owner='fabasassa-lab', repo_name='Air_quality', mlflow=True)

mlflow.set_experiment("Air Quality_Model_Tuning")

data = pd.read_csv("ispu_preprocessing.csv")

X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Category", axis=1),
    data["Category"],
    random_state=42,
    test_size=0.2
)

input_example = X_train[0:5]

with mlflow.start_run():
    # Log parameters
    n_estimators = 505
    max_depth = 37
    mlflow.autolog(log_models=False)
    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
    model.fit(X_train, y_train)
    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

# Mendefinisikan Metode Random Search
n_estimators_range = np.linspace(10, 1000, 5, dtype=int)  # 5 evenly spaced values
max_depth_range = np.linspace(1, 50, 5, dtype=int)  # 5 evenly spaced values

best_accuracy = 0
best_params = {}

for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        with mlflow.start_run(run_name=f"elastic_search_{n_estimators}_{max_depth}"):
            mlflow.autolog(log_models=False)

            # Train model
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate model
            accuracy = model.score(X_test, y_test)
            mlflow.log_metric("accuracy", accuracy)

            # Save the best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                    )
