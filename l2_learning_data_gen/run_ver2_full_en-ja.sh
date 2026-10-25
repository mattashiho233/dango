#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment before running this script.}"

python run_responses.py run \
  --model="gpt-5.2-2025-12-11" \
  --system_path="data/system_prompt_ver2_en-ja.txt" \
  --user_path="data/user_prompts_full_vocab.txt" \
  --out_jsonl="runs/ver2_en-ja/output_system_ver2_user_full_vocab.jsonl" \
  --concurrency=20 \
  --reasoning_effort="none" \
  --max_output_tokens=8192 \
  --max_retries=6 \
  --timeout_s=90 \
  --base_backoff_s=0.5 \
  --max_backoff_s=20 \
  --show_prompts=true \
  --prompt_preview_chars=200

echo "Done. Results appended to runs/ver2_en-ja/output_system_ver2_user_full_vocab.jsonl"
