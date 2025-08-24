# -*- coding: utf-8 -*-
"""
Lambda handler: endpoints 'health' et 'predict' pour le modèle RandomForest.
Dépendances (via layers ou packaging) : numpy, scikit-learn, joblib, boto3.
AUCUN besoin de pandas côté Lambda.
"""

import json, os, boto3
from botocore.exceptions import ClientError
from typing import List

S3_BUCKET = os.environ.get("S3_BUCKET")
MODEL_KEY = os.environ.get("MODEL_KEY")
_s3 = boto3.client("s3")
_model = None

EXPECTED_FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean",
    "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se",
    "smoothness_se","compactness_se","concavity_se","concave points_se","symmetry_se",
    "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst",
    "smoothness_worst","compactness_worst","concavity_worst","concave points_worst",
    "symmetry_worst","fractal_dimension_worst"
]

# ============================== Config ===============================

# Liste de FEATURES attendues (ordre du dataset Breast Cancer Wisconsin)
FEATURES: List[str] = [
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

def _ensure_model_loaded():
    global _model
    if _model is not None:
        return
    # imports lourds ici (pas en haut de fichier)
    import io, joblib
    from sklearn.ensemble import RandomForestClassifier  # si nécessaire
    # télécharger puis charger
    buf = io.BytesIO()
    _s3.download_fileobj(S3_BUCKET, MODEL_KEY, buf)
    buf.seek(0)
    _model = joblib.load(buf)

def health(event, context):
    # aucune dépendance lourde ici
    # Optionnel : HEAD sur l’objet pour confirmer la présence
    ok = True
    try:
        _s3.head_object(Bucket=S3_BUCKET, Key=MODEL_KEY)
    except ClientError:
        ok = False
    body = {
        "ok": ok,
        "expected_features": EXPECTED_FEATURES[:5] + ["…"]  # résumé
    }
    return {
        "statusCode": 200 if ok else 500,
        "headers": _cors_json(),
        "body": json.dumps(body),
    }

def predict(event, context):
    import numpy as np  # import ici
    _ensure_model_loaded()
    payload = _get_json(event)
    feats = payload.get("features")
    if isinstance(feats, dict):
        x = [feats[f] for f in EXPECTED_FEATURES]
        X = np.array([x])
    elif isinstance(feats, list) and all(isinstance(v, (int,float)) for v in feats):
        X = np.array([feats])
    else:
        raise ValueError("`features` doit être un dict clé->valeur ou une liste de nombres")
    proba = getattr(_model, "predict_proba")(X).tolist()
    pred  = _model.predict(X).tolist()
    return {"statusCode": 200, "headers": _cors_json(),
            "body": json.dumps({"predictions": pred, "probabilities": proba})}

def _get_json(event):
    if isinstance(event, dict) and "body" in event:
        return json.loads(event["body"] or "{}") if isinstance(event["body"], str) else (event["body"] or {})
    return event or {}

def _cors_json():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
        "Content-Type": "application/json",
    }








