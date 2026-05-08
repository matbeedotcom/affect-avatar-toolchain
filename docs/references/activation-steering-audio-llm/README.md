# References — Activation Steering for Audio-Conditioned LLM Personas

Annotated bibliography for the proposed feature:
**Whisper encoder embeddings → VAD affect projection → frozen-LLM activation steering**, supporting per-turn persona modulation driven by user prosody.

> **See also**:
> - [`PROJECT_PLAN.md`](PROJECT_PLAN.md) — **start here.** Top-level
>   architecture: layered affect model (events → appraisal → dynamics →
>   regulation → steering), four steering channels, simulator-first phased
>   delivery. Supersedes the architecture in the older docs.
> - [`DESIGN.md`](DESIGN.md) — original feasibility study; deeper context
>   for the cross-modal projection mechanics and literature-grounded
>   novelty claim.
> - [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — PR-level technical
>   plan for steering injection and calibration tooling; authoritative for
>   `cvec.rs`, `tools/affect_calibration/`, and the related smoke tests.

PDFs are not committed to this repo (see [Licensing](#licensing-note)). Run [`./fetch.sh`](fetch.sh) to download all PDFs into `./pdfs/` (gitignored).

```bash
cd docs/references/activation-steering-audio-llm
./fetch.sh             # download all
./fetch.sh --priority  # download only the 4 priority papers
```

BibTeX entries for all sources are in [`bibliography.bib`](bibliography.bib).

---

## Priority reading (start here)

If you only have time for five papers, read these in order:

1. **Panickssery et al. 2024 — Contrastive Activation Addition on Llama 2** ([arXiv:2312.06681](https://arxiv.org/abs/2312.06681)). The technique your `LlamaCppSteerNode` already implements; concrete demonstration on the Llama family with thorough behavioral evaluation. *Cite as the technical foundation for Workstream A.*
2. **Chen et al. 2025 — Persona Vectors: Monitoring and Controlling Character Traits in LMs** ([arXiv:2507.21509](https://arxiv.org/abs/2507.21509)). Anthropic's automated pipeline turns a *natural-language trait description* into a steering direction via contrastive system-prompt pairs and response-token mean-difference. Provides ~80% of the Phase 5 calibration scaffolding off the shelf; the project's contrast-sentence authoring step gets replaced by their generator. *See [`notes/spike-f-persona-vectors-pipeline.md`](notes/spike-f-persona-vectors-pipeline.md) for the adoption plan.*
3. **Nudging Hidden States 2026 — Cross-modal steering of audio-LLMs** ([arXiv:2603.14636](https://arxiv.org/html/2603.14636)). The closest published precedent for the cross-modal-steering architecture you're proposing. Shows text-derived steering vectors transfer to speech inputs in audio-LLMs. *Read carefully before claiming novelty.*
4. **Whisper SER with Attentive Pooling 2026** ([arXiv:2602.06000](https://arxiv.org/abs/2602.06000)). Validates Whisper encoder + pooling head → emotion as the speech-side architecture for Workstream D1. Notes that intermediate layers often outperform final layers for SER — operationally relevant for `WhisperEmbeddingExtractorNode` design.
5. **Speech LLMs Survey 2024** ([arXiv:2410.18908](https://arxiv.org/html/2410.18908v6)). Field landscape; positions the proposed work relative to the input-layer-conditioning paradigm that dominates current Speech-LLM research.

---

## Workstream A — Activation Steering on a Frozen LLM

Foundational and follow-on work for `llama_apply_adapter_cvec`-style residual-stream interventions. These papers establish that linear directions in hidden-state space steer behavior with minimal capability degradation.

| arXiv ID | Short Title | Relevance |
|----------|-------------|-----------|
| [2308.10248](https://arxiv.org/abs/2308.10248) | Turner et al. — Steering Language Models with Activation Engineering | Foundational ActAdd technique; "Love − Hate" prompt-pair steering. |
| [2312.06681](https://arxiv.org/abs/2312.06681) | Panickssery et al. — Steering Llama 2 via Contrastive Activation Addition (CAA) | Concrete Llama 2 demonstration with behavioral benchmarks; ACL 2024. |
| [2507.21509](https://arxiv.org/abs/2507.21509) | Chen et al. (Anthropic) — Persona Vectors: Monitoring and Controlling Character Traits in LMs | **Adopted as the calibration backbone for Phase 5.** Automated generation of contrastive system-prompt pairs from a trait description; response-token mean-difference for direction extraction; multi-layer "incremental" steering. See [`notes/spike-f-persona-vectors-pipeline.md`](notes/spike-f-persona-vectors-pipeline.md). |
| [2410.12299](https://arxiv.org/pdf/2410.12299) | Semantics-Adaptive Activation Intervention via Dynamic Steering Vectors | ICLR 2025; argues static steering vectors are insufficient. Frames per-input dynamic vectors — your audio-derived per-utterance vector is a special case. |
| [2501.09929](https://arxiv.org/abs/2501.09929) | Interpretable Steering with Feature-Guided Activation Additions | SAE-based steering; orthogonal but useful if interpretable steering directions are later needed. |

---

## Workstream A (cross-modal extensions) — Multimodal Activation Steering

Recent papers establishing that activation steering generalizes beyond text-only LLMs. These collectively show the intervention pattern is becoming a small subfield.

| arXiv ID | Short Title | Relevance |
|----------|-------------|-----------|
| [2603.14636](https://arxiv.org/html/2603.14636) | Nudging Hidden States — Training-Free Steering of Audio-LLMs | **Closest precedent.** Steering of LALMs; cross-modal transfer of text-derived vectors. |
| [2510.26769](https://arxiv.org/html/2510.26769) | SteerVLM — Lightweight Activation Steering for Vision-LLMs | Vision-modality analogue; supports the general pattern. |
| [2602.21704](https://arxiv.org/html/2602.21704) | Dynamic Multimodal Activation Steering for Hallucination Mitigation in VLMs | Per-input dynamic steering applied to VLMs. |
| [2507.13255](https://arxiv.org/pdf/2507.13255) | AutoSteer — Automated Steering for Safe Multimodal LLMs | Inference-time intervention pipeline for MLLMs; frames steering as a safety control surface. |

---

## Workstream C+D1 — Whisper Encoder Embeddings as an Emotion Feature Space

Establishes that Whisper's encoder hidden states carry emotion-relevant prosodic information accessible via small probes/heads. Direct support for the Whisper-encoder-based affect regressor `g: ℝ^d_whisper → ℝ³`.

| arXiv ID | Short Title | Relevance |
|----------|-------------|-----------|
| [2602.06000](https://arxiv.org/abs/2602.06000) | Whisper SER with Attentive Pooling | Validates Whisper-encoder + attention pooling for SER; intermediate layers > final. |
| [2509.08454](https://arxiv.org/abs/2509.08454) | Mechanistic Interpretability of LoRA-adapted Whisper for SER | Mechanistic story for *why* Whisper carries emotion signal. |
| [2306.05350](https://arxiv.org/pdf/2306.05350) | PEFT-SER — Parameter-Efficient Transfer Learning for SER | Adapter/LoRA approaches to speech-encoder fine-tuning for emotion. |
| ScienceDirect (non-arXiv) | Whisper Embeddings + Hand-Crafted Descriptors for SER | [Article link](https://www.sciencedirect.com/science/article/pii/S2773186325001914). Useful for citation breadth. |

---

## Workstream D2/E (positioning) — Paralinguistic Speech LLMs

Adjacent prior art that conditions LLMs on prosody **at the input layer** (replacing text tokens with speech tokens, or training end-to-end). Useful for situating the proposed work, which intervenes at the **steering layer of a frozen LLM** instead.

| arXiv ID | Short Title | Relevance |
|----------|-------------|-----------|
| [2402.05706](https://arxiv.org/html/2402.05706v3) | USDM — Paralinguistics-Aware Speech-Empowered LLMs | End-to-end speech-token LLM; canonical input-layer-conditioning baseline. |
| [2603.15981](https://arxiv.org/html/2603.15981) | PALLM — Multi-Task RL for Paralinguistic Understanding/Generation | Two-stage SFT + RL on tone-conditioned data; harder/more expensive than the proposed approach. |
| [2312.15316](https://arxiv.org/html/2312.15316v2) | ParalinGPT — Paralinguistics-Enhanced LLM of Spoken Dialogue | Joint text+speech modeling for dialogue. |
| [2402.12786](https://arxiv.org/html/2402.12786v1) | Capturing Varied Speaking Styles in Spoken Conversations | LLM responding appropriately to speaking-style variation. |
| [2510.18308](https://arxiv.org/html/2510.18308) | ParaStyleTTS — Paralinguistic Style Control for Expressive TTS | Output-side analogue: paralinguistic control of TTS. Relevant if Workstream beyond steering eventually targets affect-matched speech synthesis. |
| [2410.18908](https://arxiv.org/html/2410.18908v6) | Speech LLMs Survey | Field landscape; useful for positioning. |

---

## Notable absence

**No paper found that does exactly what is proposed:** project audio-derived prosodic embeddings through a low-dimensional VAD bottleneck into the steering subspace of a frozen LLM. Closest precedents are (a) text-derived steering of audio-LLMs ([2603.14636](https://arxiv.org/html/2603.14636)) and (b) input-layer prosody conditioning of speech-LLMs (USDM, PALLM, ParalinGPT). The specific architecture — **audio prosody → VAD → activation steering of frozen LLM** — appears unpublished as of this survey (April 2026). This is opportunity for novelty in any future writeup; it is also a signal to read the closest precedent ([2603.14636](https://arxiv.org/html/2603.14636)) carefully for any reason it might already be implicit in that work.

---

## Licensing note

arXiv papers are posted under non-exclusive licenses that permit viewing and downloading but **do not uniformly permit redistribution via third-party repositories**. PDFs are therefore gitignored from this repo. Each paper's license (some are CC-BY) appears in the right sidebar of its `arxiv.org/abs/<id>` page; if a future use case requires committed PDFs, audit per-paper licensing first.

The ScienceDirect article in the Whisper-SER section is not on arXiv and is publisher-restricted; `fetch.sh` does not attempt to download it.

---

## Maintenance

When adding a new paper:

1. Append a BibTeX entry to [`bibliography.bib`](bibliography.bib).
2. Add a row in the appropriate workstream section above with arXiv ID, short title, and one-sentence relevance note.
3. Add the arXiv ID to the array in [`fetch.sh`](fetch.sh).

When a recalled arXiv ID turns out wrong: fix it in `bibliography.bib`, this `README.md`, and `fetch.sh` together.
