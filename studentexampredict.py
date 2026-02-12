"""
Student Final Exam Score Predictor
Interactive version – no command line arguments needed
Just run: python this_file.py
"""

import json
import os
import sys
import numpy as np


class MiniStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class LinearRegressionGD:
    def __init__(self, learning_rate=0.03, max_iter=10000):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.scaler = MiniStandardScaler()

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.max_iter):
            y_pred = np.dot(X_scaled, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X_scaled.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 2000 == 0 and i > 0:
                loss = np.mean((y - y_pred) ** 2)
                if loss < 0.05:
                    break

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return np.dot(X_scaled, self.weights) + self.bias


# ────────────────────────────────────────────────
#  File handling
# ────────────────────────────────────────────────

DATA_FILE = "my_student_scores.csv"
MODEL_FILE = "my_trained_model.json"


def load_records():
    if not os.path.exists(DATA_FILE):
        return []
    records = []
    with open(DATA_FILE, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            try:
                records.append({
                    "mid1": float(parts[0]),
                    "mid2": float(parts[1]),
                    "final": float(parts[2])
                })
            except ValueError:
                pass
    return records


def save_records(records):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write("mid1,mid2,final\n")
        for r in records:
            f.write(f"{r['mid1']},{r['mid2']},{r['final']}\n")


def save_model(model):
    data = {
        "weights": model.weights.tolist(),
        "bias": float(model.bias),
        "scaler_mean": model.scaler.mean_.tolist(),
        "scaler_std": model.scaler.std_.tolist()
    }
    with open(MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    with open(MODEL_FILE, encoding="utf-8") as f:
        data = json.load(f)
    model = LinearRegressionGD()
    model.weights = np.array(data["weights"])
    model.bias = data["bias"]
    model.scaler = MiniStandardScaler()
    model.scaler.mean_ = np.array(data["scaler_mean"])
    model.scaler.std_ = np.array(data["scaler_std"])
    return model


# ────────────────────────────────────────────────
#  Metrics
# ────────────────────────────────────────────────

def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0.0
    return mae, rmse, r2


# ────────────────────────────────────────────────
#  Menu actions
# ────────────────────────────────────────────────

def action_add_record():
    print("\nAdd a past student result")
    try:
        m1 = float(input("Mid-term 1 score: "))
        m2 = float(input("Mid-term 2 score: "))
        fn = float(input("Final exam score:   "))
    except ValueError:
        print("Invalid number. Try again.\n")
        return

    records = load_records()
    records.append({"mid1": m1, "mid2": m2, "final": fn})
    save_records(records)
    print(f"Added → Total records now: {len(records)}\n")


def action_show_records():
    records = load_records()
    if not records:
        print("\nNo records yet. Add some first!\n")
        return

    print("\nYour records:")
    print(" Mid1  Mid2  Final")
    print("---------------------")
    for r in records:
        print(f"{r['mid1']:5.1f}  {r['mid2']:5.1f}  {r['final']:5.1f}")
    print(f"\nTotal: {len(records)} records\n")


def action_train():
    records = load_records()
    n = len(records)

    if n < 6:
        print(f"\nToo few records ({n}). Add at least 6–8 real results first.\n")
        return

    X = np.array([[r["mid1"], r["mid2"]] for r in records])
    y = np.array([r["final"] for r in records])

    model = LinearRegressionGD(learning_rate=0.03, max_iter=12000)
    model.fit(X, y)

    y_pred = model.predict(X)
    mae, rmse, r2 = calculate_metrics(y, y_pred)

    print("\nModel trained on your data!")
    print(f"  MAE  = {mae:.2f} points")
    print(f"  RMSE = {rmse:.2f} points")
    print(f"  R²   = {r2:.3f}")
    print("   (shown on all data – small dataset)")

    save_model(model)
    print(f"Model saved → {MODEL_FILE}\n")


def action_predict():
    model = load_model()
    if model is None:
        print("\nNo trained model found. Train first!\n")
        return

    print("\nPredict final score")
    try:
        m1 = float(input("Mid-term 1 score: "))
        m2 = float(input("Mid-term 2 score: "))
    except ValueError:
        print("Invalid input.\n")
        return

    X_new = np.array([[m1, m2]])
    pred = model.predict(X_new)[0]
    print(f"\n→ Predicted final score: {pred:.1f}\n")


def action_clear_data():
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    print("\nAll data and model cleared.\n")


def show_menu():
    print("╔════════════════════════════════════╗")
    print("║   Student Score Predictor          ║")
    print("╚════════════════════════════════════╝")
    print(" 1. Add past student result")
    print(" 2. Show all records")
    print(" 3. Train model")
    print(" 4. Predict final score")
    print(" 5. Clear all data & model")
    print(" 0. Exit")
    print("══════════════════════════════════════")
    choice = input("Enter choice (0–5): ").strip()
    return choice


def main():
    while True:
        ch = show_menu()

        if ch == "1":
            action_add_record()
        elif ch == "2":
            action_show_records()
        elif ch == "3":
            action_train()
        elif ch == "4":
            action_predict()
        elif ch == "5":
            action_clear_data()
        elif ch in ("0", "q", "exit"):
            print("\nGoodbye!\n")
            sys.exit(0)
        else:
            print("Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()