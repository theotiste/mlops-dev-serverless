#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-eu-west-3}"
STAGE="${STAGE:-dev}"
STACK="${STACK:-cancer-prediction-api-${STAGE}}"

jq --version >/dev/null 2>&1 || { echo "jq manquant"; exit 1; }
aws --version >/dev/null 2>&1 || { echo "awscli manquant"; exit 1; }

echo "Stack  : $STACK"
echo "Region : $REGION"
echo "Stage  : $STAGE"

HEALTH_URL="$(aws cloudformation describe-stacks \
  --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='HealthUrl'].OutputValue" \
  --output text)"

PREDICT_URL="$(aws cloudformation describe-stacks \
  --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='PredictUrl'].OutputValue" \
  --output text)"

echo "Health : $HEALTH_URL"
echo "Predict: $PREDICT_URL"

# petit retry pour la propagation API Gateway/CloudFront
for i in {1..8}; do
  if out="$(curl -fsS "$HEALTH_URL")"; then
    echo "$out" | jq .
    echo "$out" | jq -e '.ok==true and .stage=="dev"' >/dev/null
    break
  fi
  echo "Health non disponible (tentative $i/8), on attend…"
  sleep 3
done

# payload
if [[ -f features_input.json ]]; then
  jq '{features: (.features | map(tonumber))}' features_input.json > payload.json
else
  # fallback : 30 zéros si le fichier n’est pas là
  jq -n '{features: (range(30)|[inputs])}' <<< "$(yes 0 | head -n 30)"
fi

resp="$(curl -fsS -H "Content-Type: application/json" --data-binary @payload.json "$PREDICT_URL")"
echo "$resp" | jq .

# validations minimales
echo "$resp" | jq -e '.predictions|length==1' >/dev/null
echo "$resp" | jq -e '.probabilities|length==1' >/devnull

echo "✅ Smoke test OK"
