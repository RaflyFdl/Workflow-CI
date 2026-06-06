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

# Cek lingkungan eksekusi
is_github_ci = os.environ.get("GITHUB_ACTIONS") == "true"

# 2. Set experiment 
# Di lingkungan GitHub CI, biarkan backend sistem mengatur URI secara otomatis agar tidak disorientasi folder
if not is_github_ci:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Predictive_Maintenance_Basic")

# 4. Jalankan training
print("\nMemulai training model...")

if is_github_ci:
    # Eksekusi langsung untuk GitHub CI memanfaatkan autolog/active run dari 'mlflow run'
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Model:", accuracy)
    
    # Mencatat metrik menggunakan active_run yang aman di lingkungan CI
    try:
        mlflow.log_metric("testing_accuracy", accuracy)
    except Exception as e:
        print(f"Pencatatan metrik dilewati di CI untuk menghindari konflik: {e}")
else:
    # Alur normal untuk komputer lokal kamu
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy Model:", accuracy)
        mlflow.log_metric("testing_accuracy", accuracy)

print("\n[SUKSES] Model selesai dilatih dan dicatat oleh MLflow Project!")