#!/usr/bin/env bash
# litmus.sh — fast smoke test; assumes services are already running (no docker build)
set -euo pipefail

ORCH="${ORCH:-http://localhost:8088}"
LITELLM="${LITELLM:-http://localhost:4000}"
API_KEY="${API_KEY:-sk-local-orchestrator}"
MODEL="${MODEL:-qwen36-35b-heretic}"

PASS=0
FAIL=0

pass() { echo "  PASS  $1"; PASS=$((PASS+1)); }
fail() { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

# Assert a pattern is present in a response body; accumulates failures.
check() {
  local label="$1" pattern="$2" body="$3"
  if echo "$body" | grep -q "$pattern"; then
    pass "$label"
  else
    fail "$label — expected '$pattern' in: $body"
  fi
}

auth=(-H "Authorization: Bearer ${API_KEY}")
json=(-H "Content-Type: application/json")

echo "=== agentic-os litmus test ==="

# ── Health (unauthenticated) ────────────────────────────────────

echo "1. GET /health"
check "health" '"ok"' "$(curl -s "$ORCH/health")"

echo "2. GET /health/live"
check "health/live" '"ok"' "$(curl -s "$ORCH/health/live")"

echo "3. GET /health/ready"
check "health/ready" '"status"' "$(curl -s "$ORCH/health/ready")"

# ── LiteLLM reachability ────────────────────────────────────────

echo "4. GET LiteLLM /v1/models"
check "litellm/models" '"data"' \
  "$(curl -s "$LITELLM/v1/models" -H "Authorization: Bearer sk-local-agentstack")"

# ── Orchestrator model list ─────────────────────────────────────

echo "5. GET /v1/models"
check "models" '"data"' "$(curl -s "${auth[@]}" "$ORCH/v1/models")"

# ── Auth rejection ──────────────────────────────────────────────

echo "6. Auth rejection"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$ORCH/v1/models" \
  -H "Authorization: Bearer wrong-key")
[ "$STATUS" = "401" ] \
  && pass "auth-rejection (HTTP 401)" \
  || fail "auth-rejection — expected 401, got $STATUS"

# ── Session ─────────────────────────────────────────────────────

echo "7. POST /sessions/start"
SESS=$(curl -s -X POST "$ORCH/sessions/start" \
  "${auth[@]}" "${json[@]}" \
  -d '{"repo":"litmus-repo","task":"litmus-task","actor":"litmus"}')
SID=$(echo "$SESS" | jq -r '.session_id // empty')
[ -n "$SID" ] && [ "$SID" != "null" ] \
  && pass "session ($SID)" \
  || fail "session — got: $SESS"

# ── Event ───────────────────────────────────────────────────────

echo "8. POST /events/append"
if [ -n "${SID:-}" ]; then
  check "event" '"event_id"' "$(curl -s -X POST "$ORCH/events/append" \
    "${auth[@]}" "${json[@]}" \
    -d "{
      \"session_id\":\"$SID\",
      \"repo\":\"litmus-repo\",
      \"actor\":\"litmus\",
      \"event_type\":\"edit\",
      \"summary\":\"litmus smoke test event\"
    }")"
else
  fail "event (skipped — no session_id)"
fi

# ── Context pack ────────────────────────────────────────────────

echo "9. POST /context/pack"
check "context/pack" '"context"' "$(curl -s -X POST "$ORCH/context/pack" \
  "${auth[@]}" "${json[@]}" \
  -d '{"repo":"litmus-repo","task":"litmus-task","limit":4}')"

# ── Checkpoint ──────────────────────────────────────────────────

echo "10. POST /summaries/checkpoint"
if [ -n "${SID:-}" ]; then
  check "checkpoint" '"event_id"' "$(curl -s -X POST "$ORCH/summaries/checkpoint" \
    "${auth[@]}" "${json[@]}" \
    -d "{
      \"session_id\":\"$SID\",
      \"repo\":\"litmus-repo\",
      \"summary\":\"litmus checkpoint\"
    }")"
else
  fail "checkpoint (skipped — no session_id)"
fi

# ── Search ──────────────────────────────────────────────────────

echo "11. POST /search"
check "search" '"results"' "$(curl -s -X POST "$ORCH/search" \
  "${auth[@]}" "${json[@]}" \
  -d '{"q":"litmus smoke test event","limit":3}')"

# ── Chat completions ────────────────────────────────────────────

echo "12. POST /v1/chat/completions"
CHAT=$(curl -s -X POST "$ORCH/v1/chat/completions" \
  "${auth[@]}" "${json[@]}" \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"reply exactly: ok\"}]}")
CONTENT=$(echo "$CHAT" | jq -r '.choices[0].message.content // empty')
[ -n "$CONTENT" ] \
  && pass "chat ($CONTENT)" \
  || fail "chat — got: $CHAT"

# ── Results ─────────────────────────────────────────────────────

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] \
  && echo "All tests passed — system is ready" \
  || echo "$FAIL test(s) failed"
exit $FAIL
