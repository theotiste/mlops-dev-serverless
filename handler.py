import json
import os
import base64
import traceback

try:
    # charge ici ton modèle si présent
    from joblib import load
    _MODEL = None
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
    if os.path.exists(model_path):
        _MODEL = load(model_path)
except Exception:
    _MODEL = None

def _resp(status, body):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": os.getenv("ALLOWED_ORIGIN", "*"),
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET"
        },
        "body": json.dumps(body, ensure_ascii=False)
    }

def _coerce_features(x):
    """
    Essaie de convertir chaque élément en float (accepte int, float, str numériques).
    """
    out = []
    for v in x:
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out.append(float(v.strip()))
        else:
            raise ValueError(f"Unsupported item type: {type(v).__name__}")
    return out

def health(event, context):
    return _resp(200, {"ok": True, "stage": os.getenv("STAGE", "dev")})

def predict(event, context):
    try:
        raw = event.get("body") or ""
        # log minimal pour aider au debug (tronqué)
        print(f"[predict] isBase64Encoded={event.get('isBase64Encoded')} len(body)={len(raw)}")

        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode("utf-8", "ignore")

        try:
            data = json.loads(raw)
        except Exception:
            print("[predict] invalid JSON body:", raw[:500])
            return _resp(400, {"error": "invalid JSON"})

        if not isinstance(data, dict) or "features" not in data:
            return _resp(400, {"error": "payload must be {'features': [30 numbers]}."})

        feats = data["features"]
        if not isinstance(feats, list):
            return _resp(400, {"error": "features must be an array"})

        try:
            feats = _coerce_features(feats)
        except Exception as e:
            return _resp(400, {"error": f"features must be numbers (convertible): {e}"} )

        if len(feats) != 30:
            return _resp(400, {"error": f"features must have length 30, got {len(feats)}"})

        # Inference
        if _MODEL is None:
            # fallback déterministe pour ne pas casser le front/pipeline
            # (ex: seuil sur une somme simple)
            s = sum(feats)
            pred = 1 if s % 2 > 1 else 0
            proba = [1 - (s % 1), (s % 1)]
        else:
            import numpy as np
            X = np.array(feats, dtype=float).reshape(1, -1)
            pred = int(_MODEL.predict(X)[0])
            if hasattr(_MODEL, "predict_proba"):
                p = _MODEL.predict_proba(X)[0]
                proba = [float(p[0]), float(p[1])]
            else:
                proba = [float(1 - pred), float(pred)]

        return _resp(200, {"predictions": [pred], "probabilities": [proba]})

    except Exception as e:
        print("[predict] unexpected error:", e)
        traceback.print_exc()
        return _resp(500, {"error": "internal_error"})
