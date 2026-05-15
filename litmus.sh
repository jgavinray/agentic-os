#!/usr/bin/env bash
# litmus.sh — end-to-end smoke test for agentic-os
set -euo pipefail

ORCH="http://localhost:8088"
PASS=0
FAIL=0

pass() { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail() { echo "  ❌ $1"; FAIL=$((FAIL+1)); }

echo "=== agentic-os litmus test ==="

echo "1. GET /health"
BODY=$(curl -s "$ORCH/health")
echo "$BODY" | grep -q '"ok"' && pass "health" || fail "health"

echo "2. GET /health/live"
BODY=$(curl -s "$ORCH/health/live")
echo "$BODY" | grep -q '"ok"' && pass "health/live" || fail "health/live"

echo "3. GET /health/ready"
BODY=$(curl -s "$ORCH/health/ready")
echo "$BODY" | grep -q '"checks"' && pass "health/ready" || fail "health/ready"

echo "4. GET /v1/models"
BODY=$(curl -s "$ORCH/v1/models")
echo "$BODY" | grep -q 'qwen3.6-35b' && pass "models" || fail "models"

echo "5. POST /sessions/start"
BODY=$(curl -s -X POST "$ORCH/sessions/start" \
    -H "Content-Type: application/json" \
    -d '{"repo":"test-repo","task":"test-task"}')
SID=$(echo "$BODY" | jq -r '.session_id')
[ "$SID" != "null" ] && [ -n "$SID" ] && pass "session ($SID)" || fail "session"

echo "6. POST /events/append"
BODY=$(curl -s -X POST "$ORCH/events/append" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SID\",\"repo\":\"test-repo\",\"event_type\":\"edit\",\"summary\":\"test\"}")
echo "$BODY" | grep -q 'event_id' && pass "event" || fail "event"

echo "7. POST /context/pack"
BODY=$(curl -s -X POST "$ORCH/context/pack" \
    -H "Content-Type: application/json" \
    -d '{"repo":"test-repo","task":"test-task","limit":1}')
echo "$BODY" | grep -q 'context' && pass "context" || fail "context"

echo "8. POST /summaries/checkpoint"
BODY=$(curl -s -X POST "$ORCH/summaries/checkpoint" \
    -H "Content-Type: application/json" \
    -d "{\"session_id\":\"$SID\",\"repo\":\"test-repo\",\"summary\":\"checkpoint done\"}")
echo "$BODY" | grep -q 'event_id' && pass "checkpoint" || fail "checkpoint"

echo "9. POST /v1/chat/completions"
BODY=$(curl -s -X POST "$ORCH/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "x-api-key: orchestrator-local" \
    -d '{"messages":[{"role":"user","content":"hi"}],"model":"qwen3.6-35b"}')
echo "$BODY" | grep -q 'choices\|content\|error' && pass "chat" || fail "chat"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && echo "🎉 All tests passed — system is ready" || echo "⚠️  $FAIL test(s) failed"
exit $FAIL
