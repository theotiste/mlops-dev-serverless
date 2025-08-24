# handler.py
import os
import json
import re
import base64
import tempfile
from typing import Any, Dict, List, Tuple

import boto3
import joblib
import numpy as np
import sklearn


# -------------------------------
# Configuration & feature schema
# -------------------------------

# Ordre attendu par le modèle (30 colonnes)
EXPECTED_FEATURES: List[str] = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]

# Quelques alias (illustratifs) → tous les autres cas sont gérés par la normalisation générique
ALIASES: Dict[str, str] = {
    # variantes avec espace + underscore (ton fichier CSV)
    "concave points_mean": "concave_points_mean",
    "concave points_se": "concave_points_se",
    "concave points_worst": "concave_points_worst",
}


# -------------------------------
# Modèle (chargé à la demande)
# -------------------------------

_MODEL = None
_S3 = boto3.client("s3")

def _get_env(var_main: str, var_alt: str, default: str = "") -> str:
    """Permet d'accepter deux conventions de variables d'env."""
    return os.getenv(var_main) or os.getenv(var_alt) or default

def _resolve_model_source() -> Tuple[str, str, str]:
    """
    Retourne (mode, bucket, key_or_path)
    mode ∈ {'local', 's3'}.
    """
    # Compatibilité double convention :
    #   - MODEL_S3_BUCKET / MODEL_S3_KEY (préférée)
    #   - S3_BUCKET / MODEL_KEY         (ancienne)
    bucket = _get_env("MODEL_S3_BUCKET", "S3_BUCKET")
    key    = _get_env("MODEL_S3_KEY", "MODEL_KEY")
    local  = os.getenv("MODEL_LOCAL_PATH", "RandomForest.pkl")

    if bucket and key:
        return ("s3", bucket, key)
    # Fallback local (embarqué / téléchargé au préalable)
    return ("local", "", local)

def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    mode, bucket, key_or_path = _resolve_model_source()

    if mode == "s3":
        # Téléchargement dans /tmp/ (écriture autorisée en Lambda)
        tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(key_or_path))
        _S3.download_file(bucket, key_or_path, tmp_path)
        _MODEL = joblib.load(tmp_path)
    else:
        # local
        _MODEL = joblib.load(key_or_path)

    return _MODEL


# -------------------------------
# Utilitaires d'E/S
# -------------------------------

def _response(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }

def _normalize_key(k: str) -> str:
    """
    Normalise les clés : minuscules, séquences d'espaces/tirets/points → underscore.
    Garde les underscores déjà présents.
    """
    if k in ALIASES:
        return ALIASES[k]
    kk = k.strip().lower()
    # remplace tout ce qui n'est pas alphanumérique par underscore,
    # mais garde les underscores existants
    kk = re.sub(r"[^a-z0-9_]+", "_", kk)
    kk = re.sub(r"_+", "_", kk).strip("_")
    return kk

def _coerce_float_list(values: List[Any]) -> List[float]:
    try:
        return [float(v) for v in values]
    except Exception as e:
        raise ValueError("`features` list must contain only numbers.") from e

def _extract_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrait un dict Python à partir d'un event (HTTP API v2 ou invoke local).
    - event.get("body") string JSON (éventuellement Base64)
    - ou event lui-même déjà dict (invoke local serverless)
    """
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode("utf-8")
        if isinstance(body, str):
            return json.loads(body)
        elif isinstance(body, dict):
            return body
    # fallback : event déjà est la charge utile
    if isinstance(event, dict):
        return event
    raise ValueError("Unable to parse request body as JSON.")


# -------------------------------
# Préparation des features
# -------------------------------

def _prepare_features(payload: Dict[str, Any]) -> np.ndarray:
    """
    Accepte:
      - {"features": [30 valeurs]}
      - {"features": {"radius_mean": ..., ...}}  (clés tolérantes aux variantes)
    Retourne un np.ndarray shape (1, 30)
    """
    if "features" not in payload:
        raise ValueError("Missing `features` in request body.")

    feats = payload["features"]

    # Cas list/tuple → on vérifie la longueur
    if isinstance(feats, (list, tuple)):
        values = list(feats)
        if len(values) != len(EXPECTED_FEATURES):
            raise ValueError(
                f"`features` list must contain exactly {len(EXPECTED_FEATURES)} numbers "
                f"(got {len(values)})."
            )
        values = _coerce_float_list(values)
        return np.array([values], dtype=float)

    # Cas dict → normalisation des clés & réordonnancement
    if isinstance(feats, dict):
        normalized: Dict[str, Any] = {_normalize_key(k): v for k, v in feats.items()}
        missing = [f for f in EXPECTED_FEATURES if f not in normalized]
        if missing:
            raise ValueError(f"Missing keys in `features`: {missing}")
        ordered = [_coerce_float_list([normalized[f]])[0] for f in EXPECTED_FEATURES]
        return np.array([ordered], dtype=float)

    raise ValueError("`features` must be a list of numbers or a dict of name->value.")


# -------------------------------
# Handlers
# -------------------------------

def health(event, context):
    """
    GET /health
    Retourne l'état, versions et les noms de features attendus (et exemples d’alias).
    """
    try:
        # Test léger de chargement (sans forcer si déjà chargé)
        model_loaded = _MODEL is not None
        if not model_loaded:
            # on essaye prudemment sans échouer si S3 indispo
            try:
                _load_model()
                model_loaded = True
            except Exception:
                model_loaded = False

        body = {
            "ok": True,
            "numpy_version": np.__version__,
            "sklearn_version": sklearn.__version__,
            "model_loaded": model_loaded,
            "expected_features": EXPECTED_FEATURES,
            "aliases_examples": [
                "concave points_mean → concave_points_mean",
                "concave points_se → concave_points_se",
                "concave points_worst → concave_points_worst",
            ],
        }
        return _response(200, body)
    except Exception as e:
        return _response(500, {"error_type": "HealthError", "error_message": str(e)})

def predict(event, context):
    """
    POST /predict
    Body JSON:
      - {"features": [v1, v2, ..., v30]}
      - ou {"features": {"radius_mean": ..., "texture_mean": ..., ...}}
        (les clés sont normalisées: espaces/tirets/points → underscore)
    """
    try:
        payload = _extract_payload(event)
        X = _prepare_features(payload)

        model = _load_model()
        y_pred = model.predict(X).tolist()

        # Si proba dispo : retourne proba classe 0/1
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X).tolist()
            # arrondi lisible (optionnel)
            proba = [[round(p, 2) for p in row] for row in proba]
        else:
            proba = None

        body = {"predictions": y_pred}
        if proba is not None:
            body["probabilities"] = proba

        return _response(200, body)

    except ValueError as ve:
        return _response(400, {"error_type": "ValueError", "error_message": str(ve)})
    except Exception as e:
        return _response(500, {"error_type": "RuntimeError", "error_message": str(e)})
