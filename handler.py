# handler.py
import os
import re
import json
import math
import base64
import traceback
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────────────
# Chargement optionnel du modèle (si présent). L'API reste fonctionnelle même
# sans modèle : un fallback simple calcule une proba et une prédiction.
# ──────────────────────────────────────────────────────────────────────────────
_MODEL = None
try:
    from joblib import load  # type: ignore

    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model.joblib")
    if os.path.exists(_MODEL_PATH):
        _MODEL = load(_MODEL_PATH)
        print(f"[boot] Model loaded from {_MODEL_PATH}")
    else:
        print(f"[boot] No model found at {_MODEL_PATH} (fallback will be used)")
except Exception as e:
    print("[boot] Could not import/load model:", e)
    _MODEL = None


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def _resp(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Réponse HTTP JSON + CORS."""
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": os.getenv("ALLOWED_ORIGIN", "*"),
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST,GET",
            "Access-Control-Allow-Credentials": "false",
        },
        "body": json.dumps(body, ensure_ascii=False),
        "isBase64Encoded": False,
    }


def _parse_body_from_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Récupère le body JSON depuis l'event (API GW v1/v2).
    Gère isBase64Encoded.
    """
    raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        try:
            raw = base64.b64decode(raw).decode("utf-8", "ignore")
        except Exception:
            # on laisse raw tel quel si échec de décodage
            pass

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        print("[parse] invalid JSON body (truncated):", str(raw)[:500])
        raise ValueError("invalid JSON")

    return data


def _coerce_numbers(seq: List[Any]) -> List[float]:
    """Convertit chaque élément en float (int/float/str numériques acceptés)."""
    out: List[float] = []
    for v in seq:
        if isinstance(v, (int, float)):
            out.append(float(v))
        elif isinstance(v, str):
            out.append(float(v.strip()))
        else:
            raise ValueError(f"Unsupported item type: {type(v).__name__}")
    return out


def _extract_features(data: Dict[str, Any]) -> List[float]:
    """
    Accepte différents formats pour data['features'] :
      - list: [n, n, ...]
      - string JSON: "[n, n, ...]"
      - CSV / espaces: "n,n,..." ou "n n ..."
      - dict indexé: {"0": n, "1": n, ...}
    Retourne une liste de floats.
    """
    if "features" not in data:
        raise ValueError("payload must be {'features': [30 numbers]}.")

    feats = data["features"]

    # 1) déjà une liste
    if isinstance(feats, list):
        return _coerce_numbers(feats)

    # 2) chaîne → tenter JSON puis CSV/espace
    if isinstance(feats, str):
        s = feats.strip()
        # JSON list ?
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return _coerce_numbers(parsed)
        except Exception:
            pass
        # CSV / espaces
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        return _coerce_numbers(parts)

    # 3) dict indexé
    if isinstance(feats, dict):
        try:
            keys = sorted(feats.keys(), key=lambda k: int(k))
            arr = [feats[k] for k in keys]
        except Exception:
            arr = list(feats.values())
        return _coerce_numbers(arr)

    raise ValueError("features must be an array")


# ──────────────────────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────────────────────
def health(event, context):
    """Endpoint de santé simple."""
    return _resp(200, {"ok": True, "stage": os.getenv("STAGE", "dev")})


def predict(event, context):
    try:
        data = _parse_body_from_event(event)
        feats = _extract_features(data)

        if len(feats) != 30:
            return _resp(400, {"error": f"features must have length 30, got {len(feats)}"})

        # --- Inference ---
        if _MODEL is None:
            # Fallback déterministe : proba basée sur une sigmoïde du score global
            s = sum(feats)
            p1 = 1.0 / (1.0 + math.exp(-0.01 * (s - 150.0)))
            pred = 1 if p1 >= 0.5 else 0
            proba = [float(1.0 - p1), float(p1)]
        else:
            # Modèle scikit-learn typique
            import numpy as np  # type: ignore

            X = np.array(feats, dtype=float).reshape(1, -1)
            y = _MODEL.predict(X)[0]
            pred = int(y)
            if hasattr(_MODEL, "predict_proba"):
                p = _MODEL.predict_proba(X)[0]
                proba = [float(p[0]), float(p[1])]
            else:
                proba = [float(1 - pred), float(pred)]

        return _resp(200, {"predictions": [pred], "probabilities": [proba]})

    except ValueError as ve:
        # Erreurs "utilisateur" explicites
        return _resp(400, {"error": str(ve)})
    except Exception as e:
        # Erreurs inattendues → 500 + trace dans CloudWatch
        print("[predict] unexpected error:", e)
        traceback.print_exc()
        return _resp(500, {"error": "internal_error"})
