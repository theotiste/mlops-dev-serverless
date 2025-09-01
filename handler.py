import json, os, math, traceback, base64
from typing import Any, Dict, List

# ---- CORS ----
ALLOWED_ORIGINS = {
    "https://form.mlopstheotiste.fr",
    os.getenv("AMPLIFY_URL", "https://main.d1s8fkj9in84k5.amplifyapp.com"),
}

def _cors_headers(origin: str) -> Dict[str, str]:
    allow = origin if origin in ALLOWED_ORIGINS else list(ALLOWED_ORIGINS)[0]
    return {
        "Access-Control-Allow-Origin": allow,
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
    }

def _to_py(x: Any) -> Any:
    try:
        if hasattr(x, "item"):  # numpy scalar
            return x.item()
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x

def _resp(status: int, body: Dict[str, Any], origin: str) -> Dict[str, Any]:
    return {
        "statusCode": status,
        "headers": _cors_headers(origin),
        "body": json.dumps(_to_py(body)),
    }

# ---- Modèle (chargement paresseux, fallback si absent) ----
_MODEL = None
def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import joblib
        model_path = os.path.join(os.path.dirname(__file__), "model", "clf.joblib")
        if os.path.exists(model_path):
            _MODEL = joblib.load(model_path)
            return _MODEL
    except Exception:
        traceback.print_exc()
    _MODEL = "fallback"  # pas de modèle, on utilisera une heuristique simple
    return _MODEL

def _predict_proba(features: List[float]) -> List[float]:
    m = _load_model()
    try:
        if m != "fallback":
            proba = m.predict_proba([features])[0]  # ndarray (2,)
            return [float(proba[0]), float(proba[1])]
    except Exception:
        traceback.print_exc()
    # Fallback : petite heuristique stable (évite 500)
    # ici: pondération grossière -> logistic sur la somme normalisée
    s = sum(features) / max(len(features), 1)
    p1 = 1.0 / (1.0 + math.exp(-0.01 * (s - 20.0)))
    p0 = 1.0 - p1
    return [float(p0), float(p1)]

# ---- Handlers ----
def predict(event, context):
    headers = event.get("headers") or {}
    origin = headers.get("origin") or headers.get("Origin") or "*"

    # Préflight
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 204, "headers": _cors_headers(origin), "body": ""}

    try:
        body_raw = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            body_raw = base64.b64decode(body_raw).decode("utf-8")
        data = json.loads(body_raw)

        features = data.get("features")
        if not isinstance(features, list) or len(features) != 30 or not all(isinstance(x, (int, float)) for x in features):
            return _resp(400, {"error": "payload must be {'features': [30 numbers]}."}, origin)

        proba = _predict_proba(features)  # [p0, p1]
        if not isinstance(proba, (list, tuple)) or len(proba) != 2:
            raise ValueError("invalid model output")

        pred = 1 if float(proba[1]) >= 0.5 else 0
        return _resp(200, {
            "predictions": [pred],
            "probabilities": [[float(proba[0]), float(proba[1])]],
        }, origin)

    except Exception as e:
        traceback.print_exc()
        return _resp(500, {"error": "internal_error", "detail": str(e)}, origin)

def health(event, context):
    headers = event.get("headers") or {}
    origin = headers.get("origin") or headers.get("Origin") or "*"
    return _resp(200, {"ok": True}, origin)
