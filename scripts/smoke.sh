#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-eu-west-3}"
STACK="${STACK:-cancer-prediction-api-dev}"

json=$(aws cloudformation describe-stacks \
  --region "$REGION" --stack-name "$STACK" --output json)

# Récupère directement les URLs finales
HEALTH_URL=$(jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="HealthUrl")  | .OutputValue' <<<"$json")
PREDICT_URL=$(jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="PredictUrl") | .OutputValue' <<<"$json")

# Fallback au cas où (très rarement utile)
if [[ -z "$HEALTH_URL" || "$HEALTH_URL" == "null" ]]; then
  BASE_URL=$(jq -r '.Stacks[0].Outputs[] | select(.OutputKey=="ServiceEndpoint") | .OutputValue' <<<"$json")
  BASE_URL="${BASE_URL%/}"           # retire un éventuel slash final
  HEALTH_URL="${BASE_URL}/health"
  PREDICT_URL="${BASE_URL}/predict"
fi

echo "Health:  $HEALTH_URL"
echo "Predict: $PREDICT_URL"

# --- /health : GET simple
curl --fail --show-error --silent --location --header "Accept: application/json" \
     "$HEALTH_URL" | jq -e '.ok == true' >/dev/null
echo "Health OK"

# --- /predict : exemple avec features_input.json
jq '{features: (.features | map(tonumber))}' features_input.json > payload.json

curl --fail --show-error --silent --location \
     -H "Content-Type: application/json" \
     --data-binary @payload.json \
     "$PREDICT_URL" | jq .
echo "Predict OK"
