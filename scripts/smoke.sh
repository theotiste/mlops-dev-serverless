#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-eu-west-3}"
STACK="${STACK:-cancer-prediction-api-dev}"

BASE_URL=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceEndpoint'].OutputValue" --output text)

HEALTH_URL="${BASE_URL}/health"
PREDICT_URL="${BASE_URL}/predict"

echo "Health: $HEALTH_URL"
curl -fsS "$HEALTH_URL" | jq -e '.ok == true' >/dev/null

# construit un payload valide à partir d'un fichier local
jq '{features: (.features | map(tonumber))}' features_input.json > payload.json
curl -fsS -H "Content-Type: application/json" --data-binary @payload.json "$PREDICT_URL" | jq .

echo "Smoke test OK"
