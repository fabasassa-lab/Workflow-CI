import mlflow
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
base_dir = os.path.dirname(__file__)
csv_path = os.path.join(base_dir, "ispu_preprocessing.csv")
data = pd.read_csv(csv_path)

# Pastikan nama kolom sesuai
print("📌 Kolom data:", data.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data.drop("Category", axis=1),
    data["Category"],
    test_size=0.2,
    random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name="rf_autolog"):
    mlflow.sklearn.autolog()

    # Buat model dengan parameter tetap
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, artifact_path="model")
    
    # Evaluasi model
    acc = model.score(X_test, y_test)
    print(f"✅ Akurasi: {acc:.4f}")
