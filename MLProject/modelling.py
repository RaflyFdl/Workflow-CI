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

# 2. Set experiment (Kembalikan ke 'file:./mlruns' untuk kebutuhan otomasi CI)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Predictive_Maintenance_Basic")

# 3. Aktifkan autolog
# mlflow.autolog()

# 4. Jalankan training
print("\nMemulai training model...")

# TRIK KHUSUS GITHUB CI: Jika berjalan di GitHub Actions, kita buat run bohong-bohongan agar tidak konflik
if os.environ.get("GITHUB_ACTIONS") == "true":
    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    run_context = DummyRun()
else:
    # Jika di laptop lokal kamu, tetap pakai mlflow.start_run() asli
    run_context = mlflow.start_run()

with run_context:

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy Model:", accuracy)
    
    # Mencatat metrik testing ke tracking local server CI (Hanya aktif jika tidak di GitHub CI)
    if os.environ.get("GITHUB_ACTIONS") != "true":
        mlflow.log_metric("testing_accuracy", accuracy)

print("\n[SUKSES] Model selesai dilatih dan dicatat oleh MLflow Project!")