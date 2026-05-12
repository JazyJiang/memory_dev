#!/bin/bash
set -euo pipefail
# Download T5-small for offline use
# Usage: bash scripts/download_model.sh [output_dir]

OUTPUT_DIR=${1:-./models/t5-small}
mkdir -p "$OUTPUT_DIR"

echo "Downloading google-t5/t5-small to $OUTPUT_DIR ..."
python -c "
from transformers import T5ForConditionalGeneration, T5Tokenizer
model = T5ForConditionalGeneration.from_pretrained(google-t5/t5-small)
tokenizer = T5Tokenizer.from_pretrained(google-t5/t5-small)
model.save_pretrained()
tokenizer.save_pretrained()
print(Done.)
"
echo "Model saved to: $OUTPUT_DIR"
echo "Use --base_model $OUTPUT_DIR when running experiments offline."
