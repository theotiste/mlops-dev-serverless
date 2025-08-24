# -*- coding: utf-8 -*-
"""
Script d'entraînement RandomForest.
- Charge le CSV (pandas si dispo, sinon fallback CSV natif)
- Entraîne sur un NUMPY ARRAY (pas DataFrame) → plus de warning en inference
- Sauvegarde: RandomForest.pkl
"""

import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("Cancer_Data_Clean.csv")
MODEL_OUT = Path("RandomForest.pkl")

FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst",
]

def _load_csv() -> tuple[np.ndarray, np.ndarray]:
    """Charge le CSV. Essaie pandas si dispo, sinon csv natif."""
    try:
        import pandas as pd  # facultatif côté training
        df = pd.read_csv(DATA_PATH)
        # cible: 'diagnosis' (M/B ou 1/0)
        if df["diagnosis"].dtype == "O":
            y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int).to_numpy()
        else:
            y = df["diagnosis"].astype(int).to_numpy()
        X = df[FEATURES].to_numpy(dtype=float)  # <= clé : on passe en array
        return X, y
    except Exception:
        # fallback lecteur CSV
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            X_list, y_list = [], []
            for row in reader:
                diag = row["diagnosis"]
                if diag in ("M", "B"):
                    y_list.append(1 if diag == "M" else 0)
                else:
                    y_list.append(int(diag))
                X_list.append([float(row[c]) for c in FEATURES])
        return np.array(X_list, dtype=float), np.array(y_list, dtype=int)

def main():
    X, y = _load_csv()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, n_jobs=-1, random_state=42
    )
    clf.fit(X_train, y_train)

    y_tr = clf.predict(X_train)
    y_te = clf.predict(X_test)
    print(f"Train score: {accuracy_score(y_train, y_tr):.4f} "
          f"Test score: {accuracy_score(y_test, y_te):.4f}")

    joblib.dump(clf, MODEL_OUT)
    print(f"Model saved -> {MODEL_OUT}")

if __name__ == "__main__":
    main()
