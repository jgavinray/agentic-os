#!/usr/bin/env bash
set -euo pipefail

API_KEY="${API_KEY:-sk-local-orchestrator}"
MODEL="${MODEL:-qwen36-35b-heretic}"
BASE="${BASE:-http://localhost:8088}"
LITELLM="${LITELLM:-http://localhost:4000}"

auth=(-H "Authorization: Bearer ${API_KEY}")
json=(-H "Content-Type: application/json")

wait_for_ready() {
  echo "== Waiting for orchestrator readiness =="
  for i in {1..30}; do
    if curl -fsS "$BASE/health/ready" >/dev/null 2>&1; then
      echo "orchestrator ready"
      return 0
    fi
    echo "waiting... ($i/30)"
    sleep 2
  done

  echo "FAIL: orchestrator did not become ready"
  curl -sS "$BASE/health/ready" || true
  docker compose logs --tail=100 orchestrator || true
  exit 1
}

echo "== Build and boot =="
docker compose down
docker compose up -d --build

wait_for_ready

echo "== Health =="
curl -fsS "$BASE/health" | jq .
curl -fsS "$BASE/health/live" | jq .
curl -fsS "$BASE/health/ready" | jq .

echo "== Models =="
curl -fsS "${auth[@]}" "$BASE/v1/models" | jq .

echo "== LiteLLM route check =="
curl -is "$LITELLM/v1/models" \
  -H "Authorization: Bearer sk-local-agentstack" | head -20

echo "== Non-streaming chat =="
curl -fsS "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"reply exactly: ok\"}]
  }" | tee /tmp/agentic-os-chat.json | jq .

echo "== Streaming chat =="
curl -fsS -N "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"stream\":true,
    \"messages\":[{\"role\":\"user\",\"content\":\"count from 1 to 5\"}]
  }" | tee /tmp/agentic-os-stream.out

grep -q '^data:' /tmp/agentic-os-stream.out || {
  echo "FAIL: streaming response did not contain SSE data frames"
  exit 1
}

echo "== Session =="
SESSION_ID="$(curl -fsS "$BASE/sessions/start" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "actor":"test-agent"
  }' | jq -r .session_id)"

test -n "$SESSION_ID"
test "$SESSION_ID" != "null"
echo "SESSION_ID=$SESSION_ID"

echo "== Append event =="
curl -fsS "$BASE/events/append" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"repo\":\"agentic-os-test\",
    \"actor\":\"test-agent\",
    \"event_type\":\"decision\",
    \"summary\":\"The MVP memory path works when this event appears in context packs.\",
    \"evidence\":\"Inserted by validation suite.\",
    \"metadata\":{\"test\":true}
  }" | jq .

echo "== Context pack =="
curl -fsS "$BASE/context/pack" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "limit":8
  }' | tee /tmp/agentic-os-context.json | jq .

grep -q "MVP memory path works" /tmp/agentic-os-context.json || {
  echo "FAIL: context pack did not include inserted memory"
  exit 1
}

echo "== Qdrant search =="
curl -fsS "$BASE/search" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "q":"MVP memory path works",
    "limit":5
  }' | tee /tmp/agentic-os-search.json | jq .

grep -q "event_type" /tmp/agentic-os-search.json || {
  echo "FAIL: search result missing event_type"
  exit 1
}

echo "== Context-injected chat =="
curl -fsS "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -H "x-agent-repo: agentic-os-test" \
  -H "x-agent-task: validate memory persistence" \
  -d "{
    \"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"What prior memory do you see for this repo?\"}]
  }" | jq .

echo "== Restart persistence =="
docker compose restart postgres qdrant orchestrator >/dev/null
wait_for_ready

curl -fsS "$BASE/context/pack" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "limit":8
  }' | tee /tmp/agentic-os-context-after-restart.json | jq .

grep -q "MVP memory path works" /tmp/agentic-os-context-after-restart.json || {
  echo "FAIL: memory did not survive restart"
  exit 1
}

echo "== OpenHands container =="
docker compose ps openhands

echo "PASS: agentic-os MVP integration test completed"
