#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "" ]; then
  echo "usage: $0 <backup.tar.gz>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$1"
COMPOSE="${COMPOSE:-docker compose}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-agent}"
POSTGRES_DB="${POSTGRES_DB:-agentstack}"

workdir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-os-restore.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

tar -C "$workdir" -xzf "$ARCHIVE"
test -f "$workdir/manifest.json"
test -f "$workdir/postgres/agentstack.sql"

collection="$(python3 - "$workdir/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
print(data["qdrant_collection"])
PY
)"
snapshot_rel="$(python3 - "$workdir/manifest.json" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
print(data["qdrant_snapshot"])
PY
)"
snapshot_path="$workdir/$snapshot_rel"
test -f "$snapshot_path"

echo "restoring Postgres"
$COMPOSE exec -T "$POSTGRES_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -v ON_ERROR_STOP=1 \
  -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
$COMPOSE exec -T "$POSTGRES_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -v ON_ERROR_STOP=1 < "$workdir/postgres/agentstack.sql"

echo "restoring Qdrant collection $collection"
curl -fsS -X PUT \
  "$QDRANT_URL/collections/$collection/snapshots/upload?priority=snapshot" \
  -F "snapshot=@$snapshot_path" >/dev/null

echo "verifying stores"
$COMPOSE exec -T "$POSTGRES_SERVICE" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  -v ON_ERROR_STOP=1 -c "SELECT count(*) FROM agent_events;" >/dev/null
curl -fsS "$QDRANT_URL/collections/$collection" >/dev/null

echo "restore complete"
