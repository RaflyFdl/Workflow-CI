import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Load dataset
train = pd.read_csv("ai4i2020_preprocessing/train_clean.csv")
test = pd.read_csv("ai4i2020_preprocessing/test_clean.csv")

X_train = train.drop("Machine failure", axis=1)
y_train = train["Machine failure"]

X_test = test.drop("Machine failure", axis=1)
y_test = test["Machine failure"]

print("Data berhasil dimuat!")

# 2. Set experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Predictive_Maintenance_Basic")

# 4. Jalankan training
print("\nMemulai training model...")

# Cek apakah ini di GitHub Actions
is_github_ci = os.environ.get("GITHUB_ACTIONS") == "true"

if is_github_ci:
    # Di GitHub CI, jangan pakai start_run() karena mlflow run . sudah membuatkannya secara otomatis
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Model:", accuracy)
    mlflow.log_metric("testing_accuracy", accuracy)
else:
    # Di laptop lokal kamu, tetap berjalan normal pakai start_run()
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy Model:", accuracy)
        mlflow.log_metric("testing_accuracy", accuracy)

print("\n[SUKSES] Model selesai dilatih dan dicatat oleh MLflow Project!")