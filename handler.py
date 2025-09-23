import os
import json
import boto3
import joblib
import numpy as np
from botocore.client import Config

# --------- Configuration / cache ----------
S3_BUCKET = os.environ.get("S3_BUCKET", "").strip()
S3_KEY = os.environ.get("S3_KEY", "").strip()
CACHE_BUSTER = os.environ.get("MODEL_CACHE_BUSTER", "").strip()
LOCAL_PATH = "/tmp/model.pkl"
LOCAL_MARK = "/tmp/model.cache_buster"

_s3 = boto3.client("s3", config=Config(signature_version="s3v4"))
_model = None

def _ensure_model():
    """
    Télécharge le modèle depuis S3 dans /tmp la première fois ou
    quand MODEL_CACHE_BUSTER change, puis charge en mémoire.
    """
    global _model
    needs_download = True

    # Si déjà présent et cache_buster identique, on ne retélécharge pas
    if os.path.exists(LOCAL_PATH) and os.path.exists(LOCAL_MARK):
        try:
            with open(LOCAL_MARK, "r", encoding="utf-8") as f:
                current_buster = f.read().strip()
            if current_buster == CACHE_BUSTER:
                needs_download = False
        except Exception:
            needs_download = True

    if needs_download:
        if not S3_BUCKET or not S3_KEY:
            raise RuntimeError("S3_BUCKET/S3_KEY non définis")
        _s3.download_file(S3_BUCKET, S3_KEY, LOCAL_PATH)
        with open(LOCAL_MARK, "w", encoding="utf-8") as f:
            f.write(CACHE_BUSTER or "")

    if _model is None:
        _model = joblib.load(LOCAL_PATH)

    return _model


# --------- Utilitaires ----------
def _json_response(status: int, body: dict):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, default=_to_jsonable, ensure_ascii=False),
    }

def _to_jsonable(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    return x

def _parse_payload(event):
    """
    Accepte :
      - { "features": [30 floats] }
      - { "features": [[30 floats], [30 floats], ...] }
    """
    if event is None:
        raise ValueError("Pas d'event")

    body = event.get("body")
    if isinstance(body, str):
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}
    elif isinstance(body, dict):
        payload = body
    else:
        payload = event if isinstance(event, dict) else {}

    feats = payload.get("features")
    if feats is None:
        raise ValueError("Champ 'features' manquant")

    # Normalise en 2D
    if isinstance(feats, list) and feats and isinstance(feats[0], list):
        X = np.array(feats, dtype=float)
    else:
        X = np.array([feats], dtype=float)

    return X


# --------- Lambdas ----------
def predict(event, context):
    try:
        model = _ensure_model()
        X = _parse_payload(event)

        y = model.predict(X)
        # proba classe 1 si dispo
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)

        resp = {"predictions": y, "probabilities": proba}
        return _json_response(200, resp)

    except Exception as e:
        return _json_response(500, {"error": str(e)})


def health(event, context):
    return _json_response(200, {"status": "ok", "bucket": S3_BUCKET, "key": S3_KEY})
