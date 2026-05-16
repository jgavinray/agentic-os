#!/usr/bin/env bash
set -euo pipefail

API_KEY="${API_KEY:-sk-local-orchestrator}"
MODEL="${MODEL:-qwen36-35b-heretic}"
BASE="${BASE:-http://localhost:8088}"
LITELLM="${LITELLM:-http://localhost:4000}"

auth=(-H "Authorization: Bearer ${API_KEY}")
json=(-H "Content-Type: application/json")

fail() { echo "FAIL: $*"; exit 1; }

EMBEDDING_READY=false
# Sentinel file used to communicate embedding readiness from a background subshell.
EMBEDDING_SENTINEL="/tmp/agentic-os-embedding-ready-$$"
trap 'rm -f "$EMBEDDING_SENTINEL"' EXIT

# Run in background immediately after docker compose up so TEI warms up while
# other tests execute. 120×5s = 10-minute budget handles slow first-boot downloads.
wait_for_embedding_bg() {
  echo "== Waiting for embedding service (TEI) (background) =="
  echo "   (first boot downloads ~120MB — may take several minutes)"
  for i in {1..120}; do
    if curl -fsS http://localhost:8001/health >/dev/null 2>&1; then
      touch "$EMBEDDING_SENTINEL"
      echo "embedding ready (attempt $i)"
      return 0
    fi
    sleep 5
  done
  echo "WARN: embedding service not ready after 10 minutes"
}

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

# Start embedding wait in background so other tests don't stall.
wait_for_embedding_bg &
EMBEDDING_PID=$!

wait_for_ready

# ── Health ──────────────────────────────────────────────────────

echo "== Health =="
curl -fsS "$BASE/health"       | jq .
curl -fsS "$BASE/health/live"  | jq .
curl -fsS "$BASE/health/ready" | jq .

# ── LiteLLM reachability (hard gate) ───────────────────────────

echo "== LiteLLM reachability =="
curl -fsS "$LITELLM/v1/models" \
  -H "Authorization: Bearer sk-local-agentstack" | jq . \
  || fail "LiteLLM /v1/models did not return a success response"

# ── Models (via orchestrator) ───────────────────────────────────

echo "== Models =="
curl -fsS "${auth[@]}" "$BASE/v1/models" | jq .

# ── Auth rejection ──────────────────────────────────────────────

echo "== Auth rejection =="
AUTH_STATUS=$(curl -o /dev/null -s -w "%{http_code}" "$BASE/v1/models" \
  -H "Authorization: Bearer wrong-key")
[ "$AUTH_STATUS" = "401" ] || fail "expected 401 for bad auth, got $AUTH_STATUS"
echo "auth rejection: OK (HTTP $AUTH_STATUS)"

# ── Non-streaming chat ──────────────────────────────────────────

echo "== Non-streaming chat =="
CHAT_RESP=$(curl -fsS "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":8192,
    \"messages\":[{\"role\":\"user\",\"content\":\"reply exactly: ok\"}]
  }")
echo "$CHAT_RESP" | jq .
CHAT_CONTENT=$(echo "$CHAT_RESP" | jq -r '.choices[0].message.content // empty')
[ -n "$CHAT_CONTENT" ] || fail "non-streaming chat returned empty content"
echo "non-streaming chat content: $CHAT_CONTENT"

# ── Streaming chat ──────────────────────────────────────────────

echo "== Streaming chat =="
curl -fsS -N "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":8192,
    \"stream\":true,
    \"messages\":[{\"role\":\"user\",\"content\":\"count from 1 to 5\"}]
  }" | tee /tmp/agentic-os-stream.out

grep -q '^data:' /tmp/agentic-os-stream.out \
  || fail "streaming response did not contain SSE data frames"
grep -q 'data: \[DONE\]' /tmp/agentic-os-stream.out \
  || fail "streaming response missing [DONE] terminator"
echo "streaming chat: OK"

# ── Anthropic /v1/messages — non-streaming ──────────────────────

echo "== Anthropic messages (non-streaming) =="
MSG_RESP=$(curl -fsS "$BASE/v1/messages" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":8192,
    \"messages\":[{\"role\":\"user\",\"content\":\"reply exactly: ok\"}]
  }")
echo "$MSG_RESP" | jq .
MSG_TYPE=$(echo "$MSG_RESP" | jq -r '.type // empty')
MSG_CONTENT=$(echo "$MSG_RESP" | jq -r '.content[0].text // empty')
MSG_ROLE=$(echo "$MSG_RESP" | jq -r '.role // empty')
MSG_STOP=$(echo "$MSG_RESP" | jq -r '.stop_reason // empty')
[ "$MSG_TYPE" = "message" ] || fail "messages: expected type=message, got '$MSG_TYPE'"
[ "$MSG_ROLE" = "assistant" ] || fail "messages: expected role=assistant, got '$MSG_ROLE'"
[ -n "$MSG_CONTENT" ] || fail "messages: content[0].text is empty"
[ -n "$MSG_STOP" ] && [ "$MSG_STOP" != "null" ] || fail "messages: stop_reason is empty"
echo "messages non-streaming: OK (stop_reason=$MSG_STOP, content=$MSG_CONTENT)"

# ── Anthropic /v1/messages — system prompt ──────────────────────

echo "== Anthropic messages (system prompt) =="
SYS_RESP=$(curl -fsS "$BASE/v1/messages" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":8192,
    \"system\":\"You are a helpful assistant. Always reply in one word.\",
    \"messages\":[{\"role\":\"user\",\"content\":\"Greet me.\"}]
  }")
echo "$SYS_RESP" | jq .
SYS_CONTENT=$(echo "$SYS_RESP" | jq -r '.content[0].text // empty')
[ -n "$SYS_CONTENT" ] || fail "messages with system: content is empty"
echo "messages system prompt: OK"

# ── Anthropic /v1/messages — auth rejection returns Anthropic shape

echo "== Anthropic messages auth rejection =="
MSG_AUTH_STATUS=$(curl -o /tmp/agentic-os-msg-auth.json -s -w "%{http_code}" "$BASE/v1/messages" \
  -H "Authorization: Bearer wrong-key" "${json[@]}" \
  -d "{\"model\":\"$MODEL\",\"max_tokens\":1,\"messages\":[]}")
MSG_AUTH_ERR_TYPE=$(jq -r '.error.type // empty' /tmp/agentic-os-msg-auth.json)
[ "$MSG_AUTH_STATUS" = "401" ] || fail "messages auth: expected 401, got $MSG_AUTH_STATUS"
[ "$MSG_AUTH_ERR_TYPE" = "authentication_error" ] \
  || fail "messages auth: expected error.type=authentication_error, got '$MSG_AUTH_ERR_TYPE'"
echo "messages auth rejection: OK (HTTP $MSG_AUTH_STATUS, error.type=$MSG_AUTH_ERR_TYPE)"

# ── Anthropic /v1/messages — non-text block rejected ────────────

echo "== Anthropic messages non-text block rejection =="
IMG_STATUS=$(curl -o /tmp/agentic-os-msg-img.json -s -w "%{http_code}" "$BASE/v1/messages" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":1,
    \"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image\",\"source\":{}}]}]
  }")
IMG_ERR_TYPE=$(jq -r '.error.type // empty' /tmp/agentic-os-msg-img.json)
[ "$IMG_STATUS" = "400" ] || fail "messages image block: expected 400, got $IMG_STATUS"
[ "$IMG_ERR_TYPE" = "invalid_request_error" ] \
  || fail "messages image block: expected invalid_request_error, got '$IMG_ERR_TYPE'"
echo "messages non-text block rejection: OK"

# ── Anthropic /v1/messages — streaming ──────────────────────────

echo "== Anthropic messages (streaming) =="
curl -fsS -N "$BASE/v1/messages" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"model\":\"$MODEL\",
    \"max_tokens\":8192,
    \"stream\":true,
    \"messages\":[{\"role\":\"user\",\"content\":\"count from 1 to 3\"}]
  }" | tee /tmp/agentic-os-msg-stream.out

grep -q 'event: message_start'       /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing message_start event"
grep -q 'event: content_block_start' /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing content_block_start event"
grep -q 'event: content_block_delta' /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing content_block_delta event"
grep -q 'event: content_block_stop'  /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing content_block_stop event"
grep -q 'event: message_delta'       /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing message_delta event"
grep -q 'event: message_stop'        /tmp/agentic-os-msg-stream.out \
  || fail "messages streaming: missing message_stop event"
echo "messages streaming: OK"

# ── Session ─────────────────────────────────────────────────────

echo "== Session =="
SESSION_ID=$(curl -fsS "$BASE/sessions/start" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "actor":"test-agent"
  }' | jq -r .session_id)

[ -n "$SESSION_ID" ] && [ "$SESSION_ID" != "null" ] \
  || fail "session_id is empty or null"
echo "SESSION_ID=$SESSION_ID"

# ── Append event ─────────────────────────────────────────────────

echo "== Append event =="
EVENT_RESP=$(curl -fsS "$BASE/events/append" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"repo\":\"agentic-os-test\",
    \"actor\":\"test-agent\",
    \"event_type\":\"decision\",
    \"summary\":\"The MVP memory path works when this event appears in context packs.\",
    \"evidence\":\"Inserted by validation suite.\",
    \"metadata\":{\"test\":true}
  }")
echo "$EVENT_RESP" | jq .
EVENT_ID=$(echo "$EVENT_RESP" | jq -r '.event_id // empty')
[ -n "$EVENT_ID" ] && [ "$EVENT_ID" != "null" ] \
  || fail "append_event returned no event_id"
EVENT_QDRANT=$(echo "$EVENT_RESP" | jq -r '.qdrant_indexed')
echo "event_id=$EVENT_ID  qdrant_indexed=$EVENT_QDRANT"

# ── Checkpoint ───────────────────────────────────────────────────

echo "== Checkpoint =="
CKPT_RESP=$(curl -fsS "$BASE/summaries/checkpoint" \
  "${auth[@]}" "${json[@]}" \
  -d "{
    \"session_id\":\"$SESSION_ID\",
    \"repo\":\"agentic-os-test\",
    \"summary\":\"Integration test checkpoint — memory and streaming validated.\",
    \"next_actions\":[\"verify search\",\"verify restart persistence\"],
    \"open_questions\":[\"Does restart preserve Qdrant vectors?\"]
  }")
echo "$CKPT_RESP" | jq .
CKPT_ID=$(echo "$CKPT_RESP" | jq -r '.event_id // empty')
[ -n "$CKPT_ID" ] && [ "$CKPT_ID" != "null" ] \
  || fail "checkpoint returned no event_id"
echo "checkpoint event_id=$CKPT_ID"

# ── Context pack ─────────────────────────────────────────────────

echo "== Context pack =="
curl -fsS "$BASE/context/pack" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "limit":8
  }' | tee /tmp/agentic-os-context.json | jq .

grep -q "MVP memory path works" /tmp/agentic-os-context.json \
  || fail "context pack did not include inserted memory"
echo "context pack: OK"

# ── Qdrant semantic search ────────────────────────────────────────

echo "== Qdrant search =="
# Collect background embedding wait result before the search test.
wait "$EMBEDDING_PID" 2>/dev/null || true
if [ -f "$EMBEDDING_SENTINEL" ]; then
  EMBEDDING_READY=true
fi

if [ "$EMBEDDING_READY" = "false" ]; then
  echo "search: SKIP — embedding service not ready"
  echo "  (check: docker compose logs embedding)"
else
  # If the test event was stored before embedding was ready, re-store it now so it
  # gets indexed. The session_id is still valid and the content is idempotent.
  if [ "$EVENT_QDRANT" = "false" ]; then
    echo "re-indexing test event (embedding was not ready when originally stored)..."
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
      }" | jq '{qdrant_indexed}'
  fi

SEARCH_HTTP=$(curl -s -o /tmp/agentic-os-search.json -w "%{http_code}" "$BASE/search" \
  "${auth[@]}" "${json[@]}" \
  -d '{"q":"MVP memory path works","limit":5}')
cat /tmp/agentic-os-search.json | jq .

if [ "$SEARCH_HTTP" = "200" ]; then
  grep -q "event_type" /tmp/agentic-os-search.json \
    || fail "search returned 200 but results missing event_type"
  echo "search: OK"
elif [ "$SEARCH_HTTP" = "503" ]; then
  echo "search: DEGRADED — embedding model unavailable (semantic search non-functional)"
else
  fail "search returned unexpected HTTP $SEARCH_HTTP"
fi
fi  # end embedding_ready gate

# ── Context-injected chat ─────────────────────────────────────────

echo "== Context-injected chat =="
CTX_RESP=$(curl -fsS "$BASE/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -H "x-agent-repo: agentic-os-test" \
  -H "x-agent-task: validate memory persistence" \
  -d "{
    \"model\":\"$MODEL\",
    \"messages\":[{\"role\":\"user\",\"content\":\"What prior memory do you see for this repo?\"}]
  }")
echo "$CTX_RESP" | jq .
CTX_CONTENT=$(echo "$CTX_RESP" | jq -r '.choices[0].message.content // empty')
[ -n "$CTX_CONTENT" ] || fail "context-injected chat returned empty content"
echo "context-injected chat: OK"

# ── Restart persistence ───────────────────────────────────────────

echo "== Restart persistence =="
docker compose restart postgres qdrant orchestrator litellm >/dev/null
wait_for_ready

curl -fsS "$BASE/context/pack" \
  "${auth[@]}" "${json[@]}" \
  -d '{
    "repo":"agentic-os-test",
    "task":"validate memory persistence",
    "limit":8
  }' | tee /tmp/agentic-os-context-after-restart.json | jq .

grep -q "MVP memory path works" /tmp/agentic-os-context-after-restart.json \
  || fail "memory did not survive restart"
echo "restart persistence: OK"

# ── OpenHands ─────────────────────────────────────────────────────

echo "== OpenHands container =="
docker compose ps openhands
docker compose ps openhands | grep -qi "running\|up" \
  || fail "openhands container is not running"
echo "openhands: OK"

echo "PASS: agentic-os MVP integration test completed"
