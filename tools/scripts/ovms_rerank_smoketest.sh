#!/usr/bin/env bash
set -euo pipefail

PORT=8001
MODEL="bge-reranker-large-int8-ov"
QUERY="什么是RAG"
DOCS=()

usage() {
  cat <<'EOF'
Usage:
  bash tools/scripts/ovms_rerank_smoketest.sh [options]

Options:
  --port <port>          OVMS rerank port (default: 8001)
  --model <model_name>   Rerank model name (default: bge-reranker-large-int8-ov)
  --query <text>         Query text (default: 什么是RAG)
  --doc <text>           Document text (repeatable, at least 1)
  -h, --help             Show help

Example:
  bash tools/scripts/ovms_rerank_smoketest.sh \
    --port 8011 \
    --model bge-reranker-large-int8-ov \
    --query "什么是RAG" \
    --doc "RAG是检索增强生成" \
    --doc "天气很好"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --query)
      QUERY="$2"
      shift 2
      ;;
    --doc)
      DOCS+=("$2")
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ${#DOCS[@]} -eq 0 ]]; then
  DOCS+=("RAG是检索增强生成" "天气很好")
fi

base_url="http://127.0.0.1:${PORT}/v3"

echo "[1/3] Checking model endpoint: ${base_url}/models/${MODEL}"
model_resp="$(curl -sS --max-time 10 "${base_url}/models/${MODEL}")"
echo "$model_resp"

echo "[2/3] Building rerank request payload"
json_escape() {
  local s="$1"
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  printf '%s' "$s"
}

docs_json=""
for d in "${DOCS[@]}"; do
  escaped="$(json_escape "$d")"
  if [[ -n "$docs_json" ]]; then
    docs_json+="," 
  fi
  docs_json+="\"${escaped}\""
done

query_escaped="$(json_escape "$QUERY")"
model_escaped="$(json_escape "$MODEL")"
payload="{\"model\":\"${model_escaped}\",\"query\":\"${query_escaped}\",\"documents\":[${docs_json}]}"

echo "[3/3] Calling rerank endpoint: ${base_url}/rerank"
rerank_resp="$(curl -sS --max-time 20 -H 'Content-Type: application/json' -X POST "${base_url}/rerank" -d "$payload")"
echo "$rerank_resp"

if echo "$rerank_resp" | grep -q '"results"'; then
  echo "PASS: rerank response contains results"
  exit 0
fi

echo "FAIL: rerank response does not contain results" >&2
if echo "$rerank_resp" | grep -q 'logits'; then
  echo "Hint: logits not found -> current model export is likely incompatible with OVMS RerankCalculatorOV" >&2
fi
exit 1
