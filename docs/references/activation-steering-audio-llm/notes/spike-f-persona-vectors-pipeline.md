# Spike F — Anthropic Persona-Vector Pipeline Adoption

**Status:** Accepted in full (2026-05-05). Adoptions:
- Contrast-pair auto-generation (replaces hand-authored sentences).
- Response-token averaging for direction extraction.
- LLM-judge evaluator for `04_validate_pipeline.py`.
- Axis-as-trait (3 bipolar directions, V/A/D — not per-quadrant).
- ~~Qwen3-7B smoke test as Phase 5 prerequisite gating the 27B+ run~~ —
  **superseded** by [Spike G](spike-g-mlx-target-llm.md) (target-LLM pivot
  to LFM2-Audio-1.5B MLX); the gating concern moved from R9 to R10 and
  the gating run *is* the production calibration on the MLX target.

Deferred to v2: multi-layer incremental steering (`v_ℓ − v_{ℓ−1}`,
paper App. J.3).

**2026-05-05 post-script (Spike G addendum)**: the four pinned pipeline
decisions above all carry over to the MLX target unchanged — they are
architecture-agnostic. What changed is the *target model* (Qwen3-27B
GGUF → LFM2-Audio-1.5B MLX) and the *steering primitive* (`set_adapter_cvec`
in llama.cpp → wrapper-module substitution in `model.lfm.layers[L]` per
the verified pattern in
[`spike-g-mlx-extraction.py`](spike-g-mlx-extraction.py)). The contrast
pairs, the response-token pool, and the LLM-judge protocol are reused
verbatim.

Plan documents updated; this memo is now historical reference.
**Authored:** 2026-05-05.
**Owner:** Phase 5 (calibration) lead.
**Source:** Chen, Arditi, Sleight, Evans, Lindsey (Anthropic), 2025.
*Persona Vectors: Monitoring and Controlling Character Traits in Language Models.*
[arXiv:2507.21509](https://arxiv.org/abs/2507.21509).

---

## Question

The Phase 5 plan in
[`PROJECT_PLAN.md` §6](../PROJECT_PLAN.md) and
[`IMPLEMENTATION_PLAN.md` §4](../IMPLEMENTATION_PLAN.md) currently calls for
a hand-authored `data/contrast_sentences.json` (~200 VAD-rated sentences,
~1 developer-day) plus bespoke direction-extraction code. Anthropic's
"Persona Vectors" paper presents an end-to-end **automated** pipeline that
turns a natural-language trait description into a steering direction. Does
their methodology replace, augment, or compete with our current Phase 5
design? What concretely should change?

## TL;DR

**Adopt the paper's pipeline as the calibration backbone for Phase 5
LLM-side direction extraction.** It supplies ~80% of the Phase 5
calibration scaffolding off the shelf — the prompt generator, the
contrast-pair construction protocol, the response-token-averaging
extraction technique, and a validated automated evaluator. Three concrete
changes follow:

1. **Replace** the hand-authored `contrast_sentences.json` step with an
   auto-generated `contrast_pairs.jsonl` produced by the paper's
   trait-description → instruction-pair generator, run once per VAD axis.
2. **Adopt** response-token averaging for direction extraction (paper
   Appendix A.3) instead of any prompt-token-only or last-token strategy
   the current plan implies. This is a small but important change.
3. **Adopt** their LLM-judge evaluator pattern (paper Appendix B) for
   `04_validate_pipeline.py`: a separate model rates trait expression on a
   0–100 scale per response. This becomes the calibration regression
   metric.

The Whisper→VAD regressor (Workstream D1) is unaffected; that pipeline is
purely speech-side and doesn't intersect with the persona-vector method.
Datasets downloaded under Spike D (CREMA-D, JL-Corpus, EmoVoice-DB) remain
needed for the regressor, **not** for direction extraction.

## What the paper provides

The paper's central contribution is a fully automated pipeline:

```
trait name + 1-2 sentence description
    │
    ▼
elicitation LLM generates:
  - 5 "positive" system prompts that induce the trait
  - 5 "negative" system prompts that suppress it
  - 40 evaluation questions designed to surface the trait
    │
    ▼
target LLM generates responses for each (sys_prompt, question) pair
    │
    ▼
extract residual-stream activations at layer ℓ, averaged over response
tokens (NOT prompt tokens — see Appendix A.3)
    │
    ▼
direction v_ℓ = mean(positive activations) − mean(negative activations)
    │
    ▼
optional: incremental multi-layer v_inc = v_ℓ − v_{ℓ−1} (Appendix J.3)
```

Key empirical findings relevant to us:

- **Response-token averaging beats prompt-token averaging** for direction
  quality (Appendix A.3). Our current plan was vague on this; we should
  pin response-token averaging.
- The pipeline transfers across **trait types**: behavioral ("evil",
  "sycophancy"), tonal ("optimistic", "humorous"), and capability-like
  ("hallucination"). VAD axes are tonal — squarely in their tested
  envelope.
- An **automated LLM-judge evaluator** (Appendix B) achieves 94.7%
  agreement with human raters scoring 0–100 trait expression. This is the
  exact metric we need for `04_validate_pipeline.py`.
- **Cross-trait correlations** are non-trivial (Section 4.4): steering on
  "evil" partially activates "psychopathy" and "deception". Implication
  for us: V/A/D directions may not be orthogonal in hidden-state space;
  Phase 5 must measure post-hoc.
- Tested base models are **Llama-3.1-8B** and **Qwen2.5-7B-Instruct**.
  Our target is Qwen3-27B / Llama-3-70B. Method scales in principle; not
  verified at our scale.

## Mapping to our Phase 5 deliverables

| Phase 5 artifact (current plan) | Persona-Vectors source | Change |
|---|---|---|
| `data/contrast_sentences.json` (~200 hand-authored) | Auto-generated contrast pairs (per axis: 5 positive system prompts × 40 questions = 200 positive rows, plus 5 negative × 40 = 200 negative rows = 400 rows per axis; 1200 total across 3 axes) | **Replace.** Authored content becomes the trait *descriptions* (3 short paragraphs, one per V/A/D axis) instead. |
| `scripts/03_extract_llm_directions.py` direction-extraction logic | Appendix A.1–A.3 procedure | **Adapt.** Adopt response-token averaging + per-layer extraction. Multi-layer incremental steering optional. |
| `scripts/04_validate_pipeline.py` evaluation metric | Appendix B LLM-judge | **Adopt.** Score is 0–100 per response; ablate across α coefficients. |
| `scripts/02_train_vad_regressor.py` (Whisper → VAD) | — | **No change.** Speech-side; orthogonal to persona-vector pipeline. |
| `artifacts/llm_directions/{model}_layer{N}.npz` format | — | No change. Format is ours; the paper just provides what fills it. |

## Adopt / adapt / reject

| Paper component | Decision | Reason |
|---|---|---|
| Automated contrast-pair generation (§2) | **Adopt** | Replaces day-of hand-authoring; deterministic, regenerable. |
| Response-token averaging for extraction (App. A.3) | **Adopt** | Empirically beats alternatives; cost-equivalent. |
| LLM-judge evaluator (App. B) | **Adopt** | Reproducible calibration metric; 94.7% human-agreement. |
| Multi-layer incremental steering `v_ℓ − v_{ℓ−1}` (App. J.3) | **Adapt** | Phase 5 v1: single layer (matches IMPL_PLAN §4.4). v2: optional sweep. |
| α-coefficient grid sweep for control (App. tables) | **Adapt** | We have a different control surface — VAD scalars *from prosody* drive α at runtime, so no fixed grid. Sweep is needed only during calibration validation. |
| Preventative steering during finetuning (§5) | **Reject** | Out of scope. We don't finetune the LLM. |
| Pre-finetuning data screening (§6) | **Reject** | Out of scope. No SFT planned. |
| Cross-trait correlation analysis (§4.4) | **Adopt as validation step** | Phase 5 must verify V/A/D directions are sufficiently orthogonal in hidden-state space; if not, document and adjust runtime mixing. |

## Concrete changes to plan documents

If this memo is accepted, the following surgery in the plan documents
becomes the next step (Action 4 in the conversation):

- **`PROJECT_PLAN.md` §6 (Phase 5)** — replace the contrast-sentence
  authoring step with a reference to this memo and the auto-generation
  pipeline. The hand-authored artifact becomes 3 trait-description
  paragraphs instead of 200 sentences.
- **`PROJECT_PLAN.md` §10 R3** (risk: "calibration drift") — soften;
  automated pipeline is regenerable, reducing calibration as a one-shot
  risk.
- **`PROJECT_PLAN.md` §10** — add risk: "method developed for
  Llama-3.1-8B / Qwen2.5-7B; scaling to 27B/70B unverified. Mitigation:
  small Qwen3-7B smoke test before committing to 27B run."
- **`IMPLEMENTATION_PLAN.md` §4** — rewrite around the four stages:
  trait-description authoring (manual) → contrast-pair generation
  (automated) → response generation (automated) → direction extraction
  (automated). `03_extract_llm_directions.py` grows three internal
  phases.
- **`tools/affect_calibration/scripts/`** layout — add
  `02b_generate_contrast_pairs.py` (or fold into 03) and
  `lib/persona_pipeline.py`.

## Adaptations specific to VAD axes

The paper tests *categorical* traits ("evil", "sycophancy"). Our axes are
*continuous* and *bipolar* (V: negative ↔ positive valence; A: low ↔ high
arousal; D: submissive ↔ dominant). Two adaptation choices:

**Option A — Axis-as-trait (recommended).** Treat each axis as a single
bipolar trait. The "positive system prompts" induce one pole (e.g., high
arousal); the "negative system prompts" induce the other (low arousal).
Direction `v_arousal = mean(high) − mean(low)`. Three pipeline runs
total, one per axis.

- *Pro:* Direct mapping to runtime control (VAD scalars become α
  coefficients per direction).
- *Pro:* 3 trait descriptions is < ½ developer-day to author.
- *Con:* Untested in the paper; their bipolar coverage is implicit (a
  trait and its negation share the same direction up to sign).

**Option B — Quadrant-as-trait.** Author 4–6 emotion categories ("happy",
"angry", "sad", "calm", …); generate a direction per category; project to
VAD post-hoc using NRC-VAD-style category-to-VAD coordinates.

- *Pro:* Closer to the paper's tested setup.
- *Con:* Needs the NRC-VAD lexicon (commercial-licensing question we
  pivoted away from in Spike D), or a hand-authored category→VAD table.
- *Con:* More directions to extract (more compute), then a projection
  step; introduces an extra error source.

**Recommendation: Option A.** It's the simpler architecture and fits the
paper's bipolar-by-construction direction definition. Option B is the
fallback if Option A produces low-quality directions in a Qwen3-7B smoke
test.

## Closed questions

1. **Elicitation LLM choice.** ✅ Manifest schema records the model
   ([IMPLEMENTATION_PLAN §4.1](../IMPLEMENTATION_PLAN.md):
   `directions.elicitation_model`); default Claude-3.5-Sonnet.
2. **Layer selection.** ✅ Sweep at calibration time, record best layer
   in `manifest.json` ([IMPLEMENTATION_PLAN §4.5](../IMPLEMENTATION_PLAN.md)).
3. **Multi-layer steering.** ✅ Deferred to v2 ([IMPLEMENTATION_PLAN §4.5](../IMPLEMENTATION_PLAN.md)).
4. **Sign convention.** ✅ Positive = linguistically-positive pole
   (high V = pleasant, high A = aroused, high D = dominant). Documented
   in [IMPLEMENTATION_PLAN §4.4](../IMPLEMENTATION_PLAN.md) trait_descriptions
   schema.
5. **Smoke-test model.** ✅ Pinned as **Phase 5 prerequisite (gating)**:
   PROJECT_PLAN §10 R9 + IMPLEMENTATION_PLAN §4.7 PR D1.0.
6. **Trait modeling: axis-as-trait vs quadrant-as-trait.** ✅ Axis-as-trait
   (3 bipolar directions, V/A/D). Pinned [IMPLEMENTATION_PLAN §4.4](../IMPLEMENTATION_PLAN.md).

## Decision gate

- **(a) Adopt as proposed.** Phase 5 rewritten around persona-vectors
  pipeline; smoke test on 7B model gates the 27B run.
- **(b) Adopt with reservations.** Use response-token averaging and
  LLM-judge evaluator, but keep hand-authored contrast sentences as the
  prompt source. Hedge if generated prompts feel low-quality.
- **(c) Reject.** Stick to the original hand-authored pipeline. Only
  warranted if Phase 5 lead reads the paper and finds a structural
  incompatibility this memo missed.

## Cost / schedule impact

Adopting (Option a) reduces Phase 5 calibration setup cost: the
~1 developer-day authoring of 200 contrast sentences becomes
~½ developer-day authoring 3 trait descriptions plus running the
generator. The compute cost rises (1200 target-LLM evaluations across
3 axes, plus the elicitation-LLM and judge-LLM calls) but is < $100.

The validation metric becomes more rigorous (LLM-judge with documented
human-agreement) at no additional setup cost.

## Recommended next actions

If this memo is accepted:

1. **Action 4** in the conversation — surgery on `PROJECT_PLAN.md` §6,
   §10 and `IMPLEMENTATION_PLAN.md` §4 per the table above. Mechanical;
   ~1 hour.
2. **Action 5** in the conversation — restructure
   `tools/affect_calibration/`'s scripts/ layout (add contrast-pair
   generator, rename/refactor `03_extract_llm_directions.py`). Defer to
   Phase 5 kickoff; not Phase 0 work.
3. **Action 1** in the conversation — `lib/persona_pipeline.py` skeleton.
   Defer to Phase 5 kickoff.

This memo is the artifact to argue with; the plan-doc edits and code
follow once the memo is accepted.
