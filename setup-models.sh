#!/usr/bin/env bash
# Downloads and exports model weights to ./models/ for the orchestrator's
# in-process embedding and sentiment classifiers.
# Run once before 'docker compose up'.
set -euo pipefail

MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/models"

# Export snowflake-arctic-embed-l-v2.0 to ONNX for in-process CPU embedding.
EMBED_DIR="$MODELS_DIR/embed"
if [ ! -f "$EMBED_DIR/model.onnx" ]; then
  echo "Exporting Snowflake/snowflake-arctic-embed-l-v2.0 to ONNX into $EMBED_DIR ..."
  echo "(~1.3GB download + export — run once)"
  mkdir -p "$EMBED_DIR"
  docker run --rm \
    -v "$EMBED_DIR:/embed" \
    python:3.11-slim \
    bash -c "
      pip install -q 'optimum[onnxruntime]' transformers &&
      python -c \"
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
model_id = 'Snowflake/snowflake-arctic-embed-l-v2.0'
model = ORTModelForFeatureExtraction.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained('/embed')
tokenizer.save_pretrained('/embed')
print('embedding model exported to /embed')
\"
    "
  echo "Embedding model ready."
fi

# Download DistilBERT-SST2 ONNX model for the orchestrator's CPU sentiment classifier.
SENTIMENT_DIR="$MODELS_DIR/sentiment"
if [ ! -f "$SENTIMENT_DIR/model.onnx" ]; then
  echo "Downloading DistilBERT-SST2 ONNX model into $SENTIMENT_DIR ..."
  echo "(~65MB — requires internet access)"
  mkdir -p "$SENTIMENT_DIR"
  docker run --rm \
    -v "$SENTIMENT_DIR:/sentiment" \
    python:3.11-slim \
    bash -c "
      pip install -q 'optimum[onnxruntime]' transformers &&
      python -c \"
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
model_id = 'distilbert-base-uncased-finetuned-sst-2-english'
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained('/sentiment')
tokenizer.save_pretrained('/sentiment')
print('sentiment model exported to /sentiment')
\"
    "
  echo "Sentiment model ready."
fi

# Download a small CPU-friendly GGUF model for the dedicated local summarizer service.
SUMMARIZER_DIR="$MODELS_DIR/summarizer"
SUMMARIZER_FILE="$SUMMARIZER_DIR/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$SUMMARIZER_FILE" ]; then
  echo "Downloading Qwen2.5-3B-Instruct Q4_K_M GGUF into $SUMMARIZER_DIR ..."
  echo "(~2.0GB - requires internet access)"
  mkdir -p "$SUMMARIZER_DIR"
  docker run --rm \
    -v "$SUMMARIZER_DIR:/summarizer" \
    python:3.11-slim \
    bash -c "
      pip install -q huggingface_hub &&
      python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
    filename='qwen2.5-3b-instruct-q4_k_m.gguf',
    local_dir='/summarizer',
    local_dir_use_symlinks=False,
)
print(f'summarizer model downloaded to {path}')
PY
    "
  echo "Summarizer model ready."
fi

echo "Done. Run 'docker compose up -d' to start the stack."
