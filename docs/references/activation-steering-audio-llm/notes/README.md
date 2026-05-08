# Phase 0 Spike Notes & ADRs

This directory holds the outputs of Phase 0 pre-work spikes and any
architecture decision records (ADRs) produced during implementation.

## Phase 0 Spike Memos

| File | Spike | Status |
|---|---|---|
| `spike-a-binding.md` | Verify `llama-cpp-4` exposes `set_adapter_cvec` | See file |
| `spike-b-existing-code-audit.md` | Audit existing emotion/steering code | See file |
| `spike-c-whisper-encoder.md` | Verify Whisper encoder hidden-state access | See file |
| `spike-c-whisper-encoder.py` | Throwaway script paired with spike-c memo | Reference |
| `spike-d-datasets.md` | Calibration dataset access and licensing | See file |
| `spike-f-persona-vectors-pipeline.md` | Adoption of Anthropic's persona-vector pipeline (Chen et al. 2025) for Phase 5 calibration | See file |
| `spike-g-mlx-target-llm.md` | Pivot Phase 5 target LLM from Qwen3-27B (llama.cpp) → LFM2-Audio-1.5B (MLX); accepted, 3 feasibility gates passed | See file |
| `spike-g-mlx-extraction.py` | Throwaway feasibility script paired with spike-g memo | Reference |

## ADRs (Architecture Decision Records)

ADRs land here when open questions from
[`PROJECT_PLAN.md` §12](../PROJECT_PLAN.md) and
[`IMPLEMENTATION_PLAN.md` §11](../IMPLEMENTATION_PLAN.md) are resolved.

Naming: `adr-NNN-<short-slug>.md` where `NNN` is monotonic. Each ADR records:

- **Context** — what was unknown before.
- **Decision** — what was decided.
- **Consequences** — what changes downstream because of this decision.
- **Date** — when the decision was made.

No ADRs yet.

## Conventions

- Spike memos are write-once during Phase 0; subsequent corrections go into
  follow-up ADRs.
- Throwaway scripts paired with memos (e.g. `spike-c-whisper-encoder.py`)
  are kept as reference artifacts even after Phase 0 closes — they
  document *exactly* what was tested.
- Memos are concise: enough for a developer joining the project later to
  understand what was tried, what worked, and what decision was made.
