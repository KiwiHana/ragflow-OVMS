#!/usr/bin/env bash
set -euo pipefail

HOST_PORT=8001
MODEL="bge-reranker-large-int8-ov"
RAGFLOW_CONTAINER="docker-ragflow-cpu-1"
RERANK_DIR="/home/intel/models/BAAI/bge-reranker-large-from-remote-bak-ov-int8"
QUERY="什么是RAG"
DOC1="RAG是检索增强生成"
DOC2="天气很好"

usage() {
  cat <<'EOF'
Usage:
  bash tools/scripts/ovms_rerank_deploy_check.sh [options]

Options:
  --port <port>            OVMS rerank port on host (default: 8001)
  --model <name>           Rerank model name (default: bge-reranker-large-int8-ov)
  --container <name>       RAGFlow container name (default: docker-ragflow-cpu-1)
  --rerank-dir <path>      Rerank model directory on host
                           (default: /home/intel/models/BAAI/bge-reranker-large-from-remote-bak-ov-int8)
  -h, --help               Show help

Output:
  - Prints PASS/FAIL for each check.
  - Exits 0 only if all checks pass.

Example:
  bash tools/scripts/ovms_rerank_deploy_check.sh \
    --port 8001 \
    --model bge-reranker-large-int8-ov \
    --container docker-ragflow-cpu-1 \
    --rerank-dir /home/intel/models/BAAI/bge-reranker-large-from-remote-bak-ov-int8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      HOST_PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --container)
      RAGFLOW_CONTAINER="$2"
      shift 2
      ;;
    --rerank-dir)
      RERANK_DIR="$2"
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

base_url="http://127.0.0.1:${HOST_PORT}/v3"
payload="{\"model\":\"${MODEL}\",\"query\":\"${QUERY}\",\"documents\":[\"${DOC1}\",\"${DOC2}\"]}"

pass_count=0
fail_count=0

pass() {
  pass_count=$((pass_count + 1))
  echo "PASS: $1"
}

fail() {
  fail_count=$((fail_count + 1))
  echo "FAIL: $1"
}

echo "== OVMS rerank deploy check =="
echo "base_url=${base_url}"
echo "model=${MODEL}"
echo "container=${RAGFLOW_CONTAINER}"
echo "rerank_dir=${RERANK_DIR}"
echo

if [[ -f "${RERANK_DIR}/graph.pbtxt" ]]; then
  pass "graph.pbtxt exists"
else
  fail "graph.pbtxt missing at ${RERANK_DIR}/graph.pbtxt"
fi

if [[ -f "${RERANK_DIR}/openvino_model.xml" ]] && grep -q 'logits' "${RERANK_DIR}/openvino_model.xml"; then
  pass "openvino_model.xml contains logits"
else
  fail "openvino_model.xml missing logits output"
fi

model_list_resp=""
for _ in $(seq 1 60); do
  if model_list_resp="$(curl -sS --max-time 2 "${base_url}/models" 2>/dev/null)"; then
    if echo "${model_list_resp}" | grep -q "${MODEL}"; then
      break
    fi
  fi
done

if echo "${model_list_resp}" | grep -q "${MODEL}"; then
  pass "host /v3/models contains ${MODEL}"
else
  fail "host /v3/models does not contain ${MODEL}; response=${model_list_resp:-<empty>}"
fi

host_rerank_resp=""
if host_rerank_resp="$(curl -sS --max-time 20 -X POST "${base_url}/rerank" -H 'Content-Type: application/json' -d "${payload}" 2>/dev/null)"; then
  if echo "${host_rerank_resp}" | grep -q '"results"'; then
    pass "host /v3/rerank returns results"
  else
    fail "host /v3/rerank did not return results; response=${host_rerank_resp}"
  fi
else
  fail "host /v3/rerank request failed"
fi

container_rerank_resp=""
if container_rerank_resp="$(sudo docker exec "${RAGFLOW_CONTAINER}" sh -lc "curl -sS --max-time 20 -X POST http://host.docker.internal:${HOST_PORT}/v3/rerank -H 'Content-Type: application/json' -d '${payload}'" 2>/dev/null)"; then
  if echo "${container_rerank_resp}" | grep -q '"results"'; then
    pass "container -> host.docker.internal:${HOST_PORT} /v3/rerank returns results"
  else
    fail "container /v3/rerank did not return results; response=${container_rerank_resp}"
  fi
else
  fail "container /v3/rerank request failed (container=${RAGFLOW_CONTAINER})"
fi

echo
echo "Summary: PASS=${pass_count}, FAIL=${fail_count}"

if [[ ${fail_count} -eq 0 ]]; then
  echo "OVERALL: PASS"
  exit 0
fi

echo "OVERALL: FAIL"
exit 1
