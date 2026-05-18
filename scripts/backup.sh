#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$ROOT/backups}"
COMPOSE="${COMPOSE:-docker compose}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
QDRANT_COLLECTION="${QDRANT_COLLECTION:-agent_events}"
POSTGRES_SERVICE="${POSTGRES_SERVICE:-postgres}"
POSTGRES_USER="${POSTGRES_USER:-agent}"
POSTGRES_DB="${POSTGRES_DB:-agentstack}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
workdir="$(mktemp -d "${TMPDIR:-/tmp}/agentic-os-backup.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

mkdir -p "$BACKUP_DIR"
mkdir -p "$workdir/postgres" "$workdir/qdrant"

echo "creating Postgres dump"
$COMPOSE exec -T "$POSTGRES_SERVICE" pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" \
  > "$workdir/postgres/agentstack.sql"

echo "creating Qdrant snapshot"
snapshot_json="$(curl -fsS -X POST "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots")"
snapshot_name="$(SNAPSHOT_JSON="$snapshot_json" python3 - <<'PY'
import json, os
payload = json.loads(os.environ["SNAPSHOT_JSON"])
print(payload.get("result", {}).get("name", ""))
PY
)"

if [ -z "$snapshot_name" ]; then
  echo "failed to parse Qdrant snapshot name" >&2
  exit 1
fi

curl -fsS \
  "$QDRANT_URL/collections/$QDRANT_COLLECTION/snapshots/$snapshot_name" \
  -o "$workdir/qdrant/$snapshot_name"

cat > "$workdir/manifest.json" <<EOF
{
  "created_at": "$timestamp",
  "postgres_dump": "postgres/agentstack.sql",
  "qdrant_collection": "$QDRANT_COLLECTION",
  "qdrant_snapshot": "qdrant/$snapshot_name"
}
EOF

archive="$BACKUP_DIR/agentic-os-backup-$timestamp.tar.gz"
tar -C "$workdir" -czf "$archive" manifest.json postgres qdrant
echo "$archive"
