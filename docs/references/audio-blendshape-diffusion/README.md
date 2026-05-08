# Audio → Blendshape Latent-Diffusion (`affect_avatar`)

Spec docs for the audio-driven blendshape diffusion model. Sibling
project to [`activation-steering-audio-llm/`](../activation-steering-audio-llm/);
re-uses its Whisper encoder and V/A/D regressor.

## Reading order

1. [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — what we're building, how, and
   why; full 12-section plan with phased delivery, risks, and
   evaluation criteria.
2. [`notes/`](notes/) — spike memos. Start with
   [`spike-b0-blendshape-extraction.md`](notes/spike-b0-blendshape-extraction.md).

## Code lives at

[`tools/affect_avatar/`](../../../tools/affect_avatar/) — Python
training toolchain. Phase B0 is paper-only; Phase B1+ fills in the
scripts.
