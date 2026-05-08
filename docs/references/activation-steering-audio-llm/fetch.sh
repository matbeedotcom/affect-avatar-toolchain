#!/usr/bin/env bash
# Download arXiv PDFs for the activation-steering-audio-llm bibliography.
# Idempotent: skips files that already exist.
# PDFs land in ./pdfs/ (gitignored).
#
# Usage:
#   ./fetch.sh             # download all 17 arXiv papers
#   ./fetch.sh --priority  # download only the 5 priority papers
#   ./fetch.sh --list      # list what would be downloaded, do not fetch
#
# arXiv asks downloaders to space requests; this script sleeps between fetches.

set -euo pipefail

cd "$(dirname "$0")"
mkdir -p pdfs

# Tuple format: <arxiv-id>|<short-slug>|<priority-bool>
ENTRIES=(
  # Workstream A — foundational
  "2308.10248|turner-activation-engineering|0"
  "2312.06681|panickssery-caa-llama2|1"
  "2507.21509|chen-persona-vectors-anthropic|1"
  "2410.12299|semantics-adaptive-dynamic-steering|0"
  "2501.09929|feature-guided-activation-additions|0"

  # Workstream A (cross-modal)
  "2603.14636|nudging-hidden-states-audio-llm|1"
  "2510.26769|steervlm|0"
  "2602.21704|dynamic-multimodal-steering-vlm|0"
  "2507.13255|autosteer-multimodal-safety|0"

  # Workstream C+D1 — Whisper for SER
  "2602.06000|whisper-ser-attentive-pooling|1"
  "2509.08454|whisper-lora-mech-interp-ser|0"
  "2306.05350|peft-ser|0"

  # Workstream D2/E — paralinguistic speech LLMs
  "2402.05706|usdm-paralinguistics-aware-llm|0"
  "2603.15981|pallm-multitask-rl|0"
  "2312.15316|paralingpt-spoken-dialogue|0"
  "2402.12786|varied-speaking-styles|0"
  "2510.18308|parastyletts|0"
  "2410.18908|speech-llm-survey|1"
)

PRIORITY_ONLY=0
LIST_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --priority) PRIORITY_ONLY=1 ;;
    --list)     LIST_ONLY=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

UA="remotemedia-sdk-references/1.0 (https://arxiv.org/help/user-agent)"
fetched=0
skipped=0
failed=0

for entry in "${ENTRIES[@]}"; do
  IFS='|' read -r id slug is_priority <<<"$entry"
  if [[ "$PRIORITY_ONLY" == "1" && "$is_priority" != "1" ]]; then
    continue
  fi

  out="pdfs/${id}-${slug}.pdf"
  url="https://arxiv.org/pdf/${id}.pdf"

  if [[ "$LIST_ONLY" == "1" ]]; then
    printf '%s  →  %s\n' "$url" "$out"
    continue
  fi

  if [[ -s "$out" ]]; then
    printf '[skip]   %s\n' "$out"
    skipped=$((skipped + 1))
    continue
  fi

  printf '[fetch]  %s ...\n' "$out"
  if curl --fail --silent --show-error --location \
          --user-agent "$UA" \
          --output "$out.partial" \
          "$url"; then
    mv "$out.partial" "$out"
    fetched=$((fetched + 1))
    sleep 1   # be polite to arXiv
  else
    rm -f "$out.partial"
    printf '[FAIL]   %s\n' "$url" >&2
    failed=$((failed + 1))
  fi
done

if [[ "$LIST_ONLY" == "0" ]]; then
  printf '\nfetched=%d  skipped=%d  failed=%d\n' "$fetched" "$skipped" "$failed"
  if [[ "$failed" -gt 0 ]]; then
    exit 1
  fi
fi
