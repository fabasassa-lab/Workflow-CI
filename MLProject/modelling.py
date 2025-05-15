import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set experiment name
mlflow.set_experiment("Air Quality_Modelling")

# Load dataset
data = pd.read_csv("ispu_preprocessing.csv")

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

    # Evaluasi model
    acc = model.score(X_test, y_test)
    print(f"✅ Akurasi: {acc:.4f}")
