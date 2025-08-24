# train_rf.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

FEATURES = [
 "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean","compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
 "radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
 "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst"
]

df = pd.read_csv("Cancer_Data_Clean.csv")
X = df[FEATURES]
y = (df["diagnosis"].astype(str).str.upper().map({"M":1,"B":0})).astype(int)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(Xtr, ytr)
print("Train score:", clf.score(Xtr, ytr), "Test score:", clf.score(Xte, yte))

joblib.dump(clf, "RandomForest.pkl")
print("Model saved -> RandomForest.pkl")
