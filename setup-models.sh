#!/usr/bin/env bash
# Downloads embedding model weights to ./models/ using Python's huggingface_hub,
# which correctly handles HuggingFace CDN redirects (unlike hf-hub 0.3.x in TEI).
# Run once before 'docker compose up'. The model is ~125MB and persists on disk.
set -euo pipefail

MODEL="${1:-BAAI/bge-small-en-v1.5}"
MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

if [ -f "$MODELS_DIR/.hf_complete" ]; then
  echo "Model already downloaded ($MODEL). Delete $MODELS_DIR/.hf_complete to re-download."
  exit 0
fi

echo "Downloading $MODEL into $MODELS_DIR ..."
echo "(~125MB on first run — requires internet access)"

docker run --rm \
  -v "$MODELS_DIR:/data" \
  python:3.11-slim \
  bash -c "
    pip install -q huggingface_hub &&
    python -c \"
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', cache_dir='/data')
print('download complete')
\"
  "

touch "$MODELS_DIR/.hf_complete"
echo "Done. Run 'docker compose up -d' to start the stack."
