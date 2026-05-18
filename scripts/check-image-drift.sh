#!/usr/bin/env bash
set -euo pipefail

images=(
  "postgres:16"
  "qdrant/qdrant:latest"
  "docker.litellm.ai/berriai/litellm:main-stable"
)

for ref in "${images[@]}"; do
  repo="${ref%%:*}"
  pinned="$(awk -v repo="$repo" '
    $1 == "image:" && index($2, repo "@sha256:") == 1 {
      split($2, parts, "@sha256:")
      print parts[2]
    }
  ' compose.yaml)"
  upstream="$(docker buildx imagetools inspect "$ref" --format '{{json .Manifest.Digest}}' | tr -d '"')"
  printf "%-48s pinned=%s upstream=%s\n" "$ref" "${pinned:-missing}" "${upstream:-unknown}"
done
