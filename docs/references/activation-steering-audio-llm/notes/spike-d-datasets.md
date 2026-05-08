# Spike D — Calibration Dataset Access & Licensing

**Status**: Inventory complete. The original recommendations (MSP-Podcast +
IEMOCAP) assumed institutional/research affiliation; that assumption was
revised on 2026-05-04 after the user clarified there is no edu/research
affiliation. **Recommendations have been re-tiered around commercial-clean
datasets** in §"Pivot — commercial-clean dataset shortlist (2026-05-04
revision)" below. The original MSP-Podcast / IEMOCAP / CREMA-D / ESD /
RAVDESS / NRC-VAD inventory is preserved unchanged for reference.

**Date**: 2026-05-04 (initial); revised 2026-05-04 with commercial-clean
re-tiering and HuggingFace dataset survey.

**Phase 0 deliverable**: this memo. **Not gated**: dataset acquisition is
asynchronous and does not block Phase 0 closure.

---

## Purpose

[`PROJECT_PLAN.md` §6 Phase 5](../PROJECT_PLAN.md) and
[`IMPLEMENTATION_PLAN.md` §4](../IMPLEMENTATION_PLAN.md) call for offline
calibration of:

1. A **Whisper → VAD regressor** trained on speaker-disjoint emotional
   speech with continuous valence/arousal/dominance labels.
2. **LLM hidden-direction matrices** projected from VAD coordinates,
   derived from contrast-pair sentences with text-side VAD via the
   NRC-VAD lexicon.

This requires:

- Audio-side training data with VAD annotations: **MSP-Podcast (primary),
  IEMOCAP, ESD, CREMA-D, RAVDESS**.
- Text-side affect lexicon: **NRC-VAD lexicon**.

This memo records, for each dataset: license terms, label format, scale
(speakers/hours), download path, and access request status.

Per [`PROJECT_PLAN.md` §6 Phase 5 acceptance criteria](../PROJECT_PLAN.md),
the recommended *primary* dataset is MSP-Podcast (spontaneous
conversational speech matches deployment distribution). Acted-only
datasets (RAVDESS, CREMA-D, ESD) overfit to performance prosody and
generalize less well to spontaneous user speech. The fallback if
MSP-Podcast access is delayed: IEMOCAP + ESD.

---

## Dataset inventory

### MSP-Podcast (primary)

| Attribute | Value |
|---|---|
| **Curator** | Multimodal Signal Processing Lab, UT Dallas |
| **Hours** | ~250+ (current release `v1.10` and later); released podcast segments |
| **Speakers** | 1,000+ |
| **Modality** | Spontaneous conversational speech (English) |
| **Labels** | Continuous valence, arousal, dominance (Likert 1–7 scale, multi-rater); also categorical emotion labels |
| **Speaker-disjoint splits?** | Yes, official train/val/test |
| **License** | Research-use license; institutional sign-off required (university or research-organization affiliation) |
| **Redistribution** | Not permitted — each user must request directly |
| **Why primary** | Spontaneous speech (not acted); large speaker pool (reduces speaker leakage); continuous VAD labels; recommended by [arXiv:2602.06000](https://arxiv.org/abs/2602.06000) (priority paper #3 in this project's [bibliography](../README.md)) |

**How to request**: visit https://msp.utdallas.edu/Database.html and
follow the access procedure (typically: download a license PDF, sign with
institutional or research affiliation, return via email to the lab
coordinator). UT Dallas approves and replies with a download URL.

**Lead time**: days to weeks, depending on UT Dallas response cadence.
This is the longest single lead-time item in Phase 0.

**Local storage path** (once granted):
`tools/affect_calibration/data/msp_podcast/` (gitignored).

---

### IEMOCAP

| Attribute | Value |
|---|---|
| **Curator** | SAIL Lab, USC |
| **Hours** | ~12 |
| **Speakers** | 10 (5 male, 5 female; acted dyadic interactions) |
| **Modality** | Audio + video, English; *acted* but improvised dialog |
| **Labels** | Continuous valence/activation/dominance (Likert 1–5); plus categorical emotion labels (happy/sad/angry/neutral/frustrated/excited/etc.) |
| **Speaker-disjoint splits?** | Possible (10 speakers; community convention is leave-one-session-out) |
| **License** | Research-use; **signed agreement** required (academic or research org) |
| **Redistribution** | Not permitted |

**How to request**: visit https://sail.usc.edu/iemocap/ and follow the
access procedure (download the EULA PDF, sign, return via email). SAIL
approves and provides download credentials.

**Lead time**: days to weeks.

**Why secondary**: smaller; acted speech less ideal than spontaneous.
Useful as a supplement and as a sanity-check dataset (the field has
canonical baselines on IEMOCAP that we can compare against).

**Local storage path** (once granted): `tools/affect_calibration/data/iemocap/`
(gitignored).

---

### CREMA-D (open license)

| Attribute | Value |
|---|---|
| **Curator** | Cao et al., U Penn |
| **Hours** | ~5 |
| **Speakers** | 91 (diverse demographics) |
| **Modality** | Audio + video, English; *acted* short utterances |
| **Labels** | Categorical emotion labels (anger/disgust/fear/happy/neutral/sad) and per-rater intensity scores |
| **Continuous VAD?** | Not directly; **derive via emotion → VAD lookup** (the standard practice in the SER literature is to map categorical labels to canonical VAD prototypes, e.g. via NRC-VAD or Russell circumplex coordinates) |
| **Speaker-disjoint splits?** | Yes (91 speakers; trivially split) |
| **License** | **Open-Database License (ODbL)** + Database Contents License — *redistributable* with attribution |
| **Redistribution** | Permitted under ODbL terms (must preserve attribution and license notice) |

**How to download**: directly from
https://github.com/CheyneyComputerScience/CREMA-D — public repository.

**Lead time**: minutes (clone repo or download release).

**Why secondary**: acted speech, only categorical labels. Useful as a
sanity-check dataset and for speaker-pool diversity (91 speakers >>
IEMOCAP's 10).

**Local storage path**: `tools/affect_calibration/data/crema_d/`
(gitignored, despite ODbL permitting redistribution — keeping it out of
the repo because the audio files are large; the project's `.gitignore`
already excludes the `data/` subtree).

---

### ESD — Emotional Speech Dataset (open license, commercial-friendly)

| Attribute | Value |
|---|---|
| **Curator** | Zhou et al., NUS |
| **Hours** | ~29 |
| **Speakers** | 20 (10 English, 10 Mandarin; balanced gender) |
| **Modality** | Audio, English + Mandarin; *acted* short utterances |
| **Labels** | Categorical emotion (neutral/happy/angry/sad/surprise) |
| **Continuous VAD?** | Not directly; derive as for CREMA-D |
| **Speaker-disjoint splits?** | Yes (20 speakers) |
| **License** | Public; non-commercial use stated in the original release; some derivatives are CC-BY |
| **Redistribution** | Verify per-distribution; the GitHub release is widely mirrored |

**How to download**: from the project page
https://github.com/HLTSingapore/Emotional-Speech-Data (or HuggingFace
mirror).

**Lead time**: minutes.

**Why secondary**: small per-language; acted; only categorical labels.
**Why useful**: most permissive license of the acted-corpus options; the
recommended commercial-friendly fallback per
[`PROJECT_PLAN.md` §10 R8](../PROJECT_PLAN.md).

**Local storage path**: `tools/affect_calibration/data/esd/` (gitignored).

---

### RAVDESS (open license, smallest)

| Attribute | Value |
|---|---|
| **Curator** | Livingstone & Russo, Ryerson U |
| **Hours** | ~1.5 audio (plus video) |
| **Speakers** | 24 (12 male, 12 female) |
| **Modality** | Audio + video, English; *highly acted* (theatrical) |
| **Labels** | Categorical (8 emotions including calm + neutral + 6 emotions × 2 intensities) |
| **License** | CC-BY-NC-SA-4.0 — non-commercial, redistributable with attribution |

**How to download**: https://zenodo.org/record/1188976 (Zenodo).

**Lead time**: minutes.

**Why optional**: theatrical acting style is even further from spontaneous
speech than IEMOCAP. Listed here for completeness; *not recommended* as
training data because of the theatrical-acting overfitting risk noted in
[`PROJECT_PLAN.md` §10 R2](../PROJECT_PLAN.md). May be useful as a
high-confidence test set (clear emotion expression, easy to evaluate).

**Local storage path** (if used): `tools/affect_calibration/data/ravdess/`
(gitignored).

---

### NRC-VAD lexicon (text-side; required)

| Attribute | Value |
|---|---|
| **Curator** | Saif Mohammad, NRC Canada |
| **Coverage** | ~20,000 English words and short phrases |
| **Labels** | Continuous valence, arousal, dominance (each in [0, 1]) |
| **Format** | Tab-separated values: `word\tvaluence\tarousal\tdominance` |
| **License** | **Free for research and personal use** with attribution; commercial use requires permission from NRC Canada |
| **Redistribution** | Permitted under license terms with attribution |

**How to download**: https://saifmohammad.com/WebPages/nrc-vad.html — direct
download (TSV file, ~1 MB).

**Lead time**: minutes.

**Why required**: Phase 5 step 3 ("extract LLM directions") needs
sentence-level VAD scores to fit `D ∈ ℝ^{3 × d_llm}` via least-squares.
The lexicon provides the per-word VAD that aggregates to sentence VAD
(simple mean over content words is the standard recipe). Without it,
emotion-name-only contrasts can be derived but the *dimensional*
projection cannot.

**Local storage path**: `tools/affect_calibration/data/nrc_vad_lexicon.tsv`
(gitignored, despite permissive license — same rationale as CREMA-D).

---

## Per-dataset access status

This is the live tracker for the **commercial-clean** Phase 5 stack. The
academic-restricted research path (MSP-Podcast / IEMOCAP) is documented
above for reference but is **not used** under the current commercial-
deployment posture. Update this table as downloads complete.

### Active (commercial-clean stack)

| Dataset | HF location | License | Status (2026-05-05) | Local size |
|---|---|---|---|---|
| EmoVoice-DB (synthetic primary) | [`yhaha/EmoVoice-DB`](https://huggingface.co/datasets/yhaha/EmoVoice-DB) | MIT | ✅ **Downloaded** to `tools/affect_calibration/data/emovoice_db/` | 5.7 GB |
| CREMA-D (human primary, faithful mirror) | [`myleslinder/crema-d`](https://huggingface.co/datasets/myleslinder/crema-d) | ODbL (attribution required) | ✅ **Downloaded** to `tools/affect_calibration/data/crema_d/` | 449 MB |
| JL-Corpus (held-out test) | [`CLAPv2/JL-Corpus`](https://huggingface.co/datasets/CLAPv2/JL-Corpus) | CC0 per CAMEO docs (verify card) | ✅ **Downloaded** to `tools/affect_calibration/data/jl_corpus/` | 396 MB |
| Trait descriptions (3 axis paragraphs, hand-authored) | n/a | Phase 5 deliverable (~½ developer-day) | **Author during Phase 5** | n/a |
| `contrast_pairs.jsonl` (auto-generated, persona-vectors pipeline) | n/a | Phase 5 deliverable (~1200 rows; LLM-API cost <$100) | **Generate during Phase 5 from trait descriptions** ([spike-f memo](spike-f-persona-vectors-pipeline.md)) | ~10 MB |
| SUBESCO (optional cross-lingual) | [`sajid73/SUBESCO-audio-dataset`](https://huggingface.co/datasets/sajid73/SUBESCO-audio-dataset) (verify license tag) | CC BY 4.0 per CAMEO | **Optional — defer to Phase 5 ablations** | — |
| Emozionalmente (optional cross-lingual) | (download from original source, not the CAMEO wrapper) | CC BY 4.0 | **Optional — defer to Phase 5 ablations** | — |

**Total local footprint**: 6.5 GB. Datasets are gitignored under
[`/tools/affect_calibration/data/*`](../../../tools/affect_calibration/) (verified
2026-05-05).

**Download notes** (for Phase 5 setup reference):

- **EmoVoice-DB layout**: 7 emotion zips under `audio/` (angry,
  disgusted, fearful, happy, neutral, sad, surprised) plus
  `train.jsonl` (~125 MB), `val.jsonl`, `test.jsonl`, and a 200 MB
  `laions_got_latent.jsonl`. JSONL rows include `key`, `source_text`,
  `emotion`, `emotion_text_prompt`, `target_wav`, and CosyVoice speech
  tokens. Phase 5 will need to extract the per-emotion zips before
  feeding audio to the Whisper encoder.
- **CREMA-D layout**: includes a `crema-d.py` loading script (HF
  legacy-style; will need `trust_remote_code=True` for the `datasets`
  library, or Phase 5 can read the underlying `data/` directory directly).
- **JL-Corpus layout**: parquet-native, 32 batches keyed by
  `<gender>{1,2}_<emotion>_<id>`. Emotions visible in filenames:
  angry, anxious, apologetic, assertive, concerned, encouraging,
  excited, happy, neutral, sad. 4 speakers (female1/2, male1/2).

### Reference-only (academic-restricted, NOT used)

| Dataset | Status |
|---|---|
| MSP-Podcast | **Not pursued** — research-only license; commercial deployment incompatible. Filing decision deferred unless deployment intent shifts. |
| IEMOCAP | **Not pursued** — same. |
| ESD | **Conditional** — open license but commercial-friendliness depends on per-mirror redistribution; verify before adoption if used. Tier B fallback. |
| RAVDESS | **Skipped** — CC-BY-NC-SA (non-commercial only). |
| NRC-VAD lexicon | **Replaced** by an auto-generated `contrast_pairs.jsonl` produced from 3 hand-authored axis trait descriptions via the persona-vectors pipeline of Chen et al. 2025 ([spike-f memo](spike-f-persona-vectors-pipeline.md)). Avoids the commercial-license question. Direct license: free for research/personal; commercial requires NRC Canada permission. |

---

## Action items for the user

These are tasks I (Claude) cannot execute on the user's behalf. They are
documented here so the user can perform them at their convenience without
re-deriving the procedure. **All revised for the commercial-clean stack.**

### Immediate (no lead time, no permissions needed)

```bash
cd tools/affect_calibration

# Tier A — primary training data
huggingface-cli download yhaha/EmoVoice-DB --repo-type dataset \
    --local-dir data/emovoice_db
huggingface-cli download myleslinder/crema-d --repo-type dataset \
    --local-dir data/crema_d

# Tier B — held-out test (required by Phase 5; verify the dataset card first)
huggingface-cli download CLAPv2/JL-Corpus --repo-type dataset \
    --local-dir data/jl_corpus
```

The `data/` subtree is already gitignored. Downloads are self-contained;
no signed agreements, no email loops, no institutional affiliation
required.

### Phase 5 (during calibration kickoff, not Phase 0)

- **Author the trait descriptions**: 3 short paragraphs (one per V/A/D
  axis) describing the positive and negative pole of each axis
  behaviorally. ~½ developer-day. Stored at
  `data/trait_descriptions.json`.
- **Generate `contrast_pairs.jsonl`**: run
  `scripts/02b_generate_contrast_pairs.py` against the trait descriptions
  with an elicitation LLM (Claude/GPT/Qwen). Produces ~1200 rows
  (3 axes × 400 rows = full cartesian product of 5 prompts × 40 questions
  per pole, both poles) per the persona-vectors pipeline of Chen et al.
  2025 ([spike-f memo](spike-f-persona-vectors-pipeline.md)). One-shot,
  deterministic given a seeded model. Cost: <$100 in LLM API calls.
- **(Optional) Cross-lingual augmentation**: if monolingual baseline
  saturates during Phase 5 ablation, download SUBESCO and Emozionalmente
  per the table above. Not a Phase 0 task.
- **(Optional) Internal evaluation set**: collect ~1 hour of internal
  spontaneous-conversational speech with informal VAD annotations to
  evaluate the trained regressor against deployment-distribution audio.
  Mitigates R8 distribution-shift risk; not a Phase 0 task.

### NOT action items (deliberately)

- ~~File MSP-Podcast access request~~ — **dropped** per commercial-deployment
  posture.
- ~~File IEMOCAP access request~~ — **dropped** per commercial-deployment
  posture.
- ~~Download NRC-VAD lexicon~~ — **dropped** in favor of an
  auto-generated `contrast_pairs.jsonl` from 3 hand-authored trait
  descriptions via the persona-vectors pipeline ([spike-f memo](spike-f-persona-vectors-pipeline.md)).
  Avoids commercial-license question.
- ~~Download RAVDESS~~ — **dropped** (CC-BY-NC-SA non-commercial).

---

## Decision gate evaluation

**Resolved (revised 2026-05-04)**: Phase 5 calibration uses the
commercial-clean stack documented in §"Pivot — commercial-clean dataset
shortlist" below. The academic-restricted MSP-Podcast / IEMOCAP path is
not pursued under the current commercial-deployment posture; both
branches of the original gate (timely access vs delayed access) are
moot. Phase 0 closure is unconditional with respect to dataset access:
all primary datasets are downloadable today without permissions.

The original two-branch gate is preserved here for traceability:

- ~~(a) MSP-Podcast access in time → Phase 5 starts with primary dataset.~~
- ~~(b) MSP-Podcast blocked or delayed beyond Phase 5 start → fallback to
  IEMOCAP + ESD as primary.~~

Both branches assumed research-only deployment; superseded by Q-D2's
resolution.

---

## Open questions

- **Q-D1**: ~~Does the project have institutional affiliation required by
  MSP-Podcast and IEMOCAP licenses?~~ **Resolved**: no edu/research
  affiliation; pivoted to commercial-clean stack. See §"Pivot — commercial-clean dataset shortlist".
- **Q-D2**: ~~Are there commercial-deployment plans that would require
  re-licensing of MSP-Podcast / IEMOCAP datasets?~~ **Resolved as yes**:
  commercial deployment is in scope, so research-only datasets are
  excluded by default. Phase 5 stack is fully commercial-clean (MIT, ODbL,
  CC0, Apache 2.0). Legal review is no longer required for the dataset
  axis specifically — but should still happen for the trained artifacts
  in aggregate (per-artifact attribution is the residual concern; see
  Q-D4).
- **Q-D3**: Is there a project-internal storage location for large datasets
  (e.g. a shared S3 bucket) where downloaded data should live, vs each
  developer's local `tools/affect_calibration/data/`? Out of Phase 0
  scope; revisit at Phase 5 kickoff.
- **Q-D4**: Per-artifact attribution metadata. The Phase 5 calibration
  manifest should include `dataset_attributions: [...]` per artifact,
  listing every dataset used and its license terms. For the commercial-
  clean stack: EmoVoice-DB (MIT, cite arXiv:2504.12867) + CREMA-D (ODbL,
  cite Cao et al. 2014) + JL-Corpus (CC0; no attribution required but
  recommended). NRC-VAD attribution is moot since the lexicon is not used.

---

## Pivot — commercial-clean dataset shortlist (2026-05-04 revision)

The original recommendations above assumed academic / institutional access.
The user clarified that there is no edu/research affiliation, and
commercial deployment may be in scope. Under that constraint, **the
academic-restricted datasets (MSP-Podcast, IEMOCAP, MSP-IMPROV) are off
the table regardless of access path**, because their licenses are
research-only and any artifacts trained on them inherit non-commercial
restrictions.

Below is the **revised** dataset strategy. All licenses verified against
the HuggingFace dataset cards on 2026-05-04 except where noted.

### Tier A — Verified commercial-clean, audio-bearing (use these)

| Dataset | HF location | License (verified) | Size | Labels | Recorded |
|---|---|---|---|---|---|
| **EmoVoice-DB** | [`yhaha/EmoVoice-DB`](https://huggingface.co/datasets/yhaha/EmoVoice-DB) | **MIT** | 22,100 samples / ~40 h | 7 categorical emotions + freestyle text prompts (rich) | **Synthetic (GPT-4o-audio TTS)** |
| **CREMA-D (faithful mirror)** | [`myleslinder/crema-d`](https://huggingface.co/datasets/myleslinder/crema-d) | **ODbL** (correctly declared, matches upstream) | 7,442 clips / ~5 h / 91 actors | 6 categorical emotions + intensity (Lo/Med/Hi) | Human |
| **EQ4You/Emotional_Speech** | [`EQ4You/Emotional_Speech`](https://huggingface.co/datasets/EQ4You/Emotional_Speech) | **Apache 2.0** | 870k FLAC / 169 GB | Free-text descriptions of arousal/dominance/etc. (NOT numeric VAD) | Human, "from publicly available videos" — **provenance flag, see §Caveats** |
| **NRC-VAD lexicon** | n/a (download direct) | Free for research/personal; **commercial requires permission from NRC Canada** | ~20k words | Continuous VAD per word | n/a (text lexicon) |

### Tier B — Verified commercial-clean, smaller / specific-purpose

The CAMEO collection ([`amu-cai/CAMEO`](https://huggingface.co/datasets/amu-cai/CAMEO))
is itself **CC BY-NC-SA 4.0 (non-commercial)**, but its component datasets
retain their original licenses. Download them from their original sources
(not the CAMEO wrapper) to use under the individual licenses below:

| Component | License (per CAMEO docs) | Language | Size | Notes |
|---|---|---|---|---|
| EMNS | Apache 2.0 | English | small | Acted; commercial-clean |
| Emozionalmente | CC BY 4.0 | Italian | ~6.9k samples | Acted |
| MESD | CC BY 4.0 | Mexican Spanish | small | Acted |
| Oréau | CC BY 4.0 | French | small | Acted |
| SUBESCO | CC BY 4.0 | Bangla | ~7k samples / 20 actors | Acted; useful for cross-lingual sanity |
| eNTERFACE | MIT | English | small | Acted; audio-visual |
| RESD | MIT | unknown | small | Verify per source |
| JL-Corpus | CC0 Public Domain | NZ English | ~2 h / 4 speakers | Smallest barrier; useful as held-out test set |

### Tier C — TTS-synthetic data path (license-clean by construction)

Generate emotional speech with explicit `(v, a, d)` targets using
emotion-controlled TTS. The synthetic distribution may diverge from real
human prosody; mitigate by validating the trained Whisper→VAD regressor
on held-out **CREMA-D + JL-Corpus** (real human speech, commercial-clean).

Candidate generators (verify each license before use):

- **XTTS-v2** (Coqui) — open-source, commercial conditions vary by version.
- **Bark** (Suno) — MIT but Suno has issued additional non-commercial
  guidance on weights; check current state.
- **Microsoft Edge TTS** — free per-call but ToS prohibits training
  derivative models from outputs.
- **ElevenLabs API** — commercial license depends on subscription tier;
  enterprise plans permit derivative training.
- **EmoVoice-DB** itself (Tier A) is *output* of this approach; you can
  treat it as pre-generated Tier C data.

### Tier D — Avoid (license issues for commercial deployment)

| Dataset | HF location | Issue |
|---|---|---|
| **MELD_audio** | [`ajyy/MELD_audio`](https://huggingface.co/datasets/ajyy/MELD_audio) | GPL-3.0 wrapper but uses Friends TV show audio (copyrighted). License tag does not grant rights the source material withholds. |
| **CAMEO** (whole) | [`amu-cai/CAMEO`](https://huggingface.co/datasets/amu-cai/CAMEO) | CC BY-NC-SA 4.0 — non-commercial only. Use individual components from their original sources (Tier B). |
| **SEIRDB** | [`GDGiangi/SEIRDB`](https://huggingface.co/datasets/GDGiangi/SEIRDB) | "Contact for commercial use." Also a meta-dataset that includes IEMOCAP, inheriting its restrictions. |
| **CLAPv2/MSP_podcast** | [`CLAPv2/MSP_podcast`](https://huggingface.co/datasets/CLAPv2/MSP_podcast) | License unstated; redistributes UTD's research-only data; audio replaced with spectrograms; dominance scores stripped. |
| **Ar4ikov/iemocap_audio_text_splitted** | [`Ar4ikov/iemocap_audio_text_splitted`](https://huggingface.co/datasets/Ar4ikov/iemocap_audio_text_splitted) | License unstated; redistributes USC SAIL's research-only audio. |
| **MahiA/CREMA-D** | [`MahiA/CREMA-D`](https://huggingface.co/datasets/MahiA/CREMA-D) | Tagged "MIT" but underlying CREMA-D is ODbL; the MIT tag is mis-applied. Use `myleslinder/crema-d` instead, which correctly declares ODbL. |
| **renumics/emodb-enriched** | [`renumics/emodb-enriched`](https://huggingface.co/datasets/renumics/emodb-enriched) | License unstated. EmoDB itself is "freely available" but no commercial-use clarity. |
| **AbstractTTS/CREMA-D** | [`AbstractTTS/CREMA-D`](https://huggingface.co/datasets/AbstractTTS/CREMA-D) | License not visible in card; has rich multi-emotion intensity scores but verify license before use. Prefer `myleslinder/crema-d`. |

### Caveats

- **EQ4You/Emotional_Speech "publicly available videos" provenance**: an
  Apache 2.0 license tag declared by the uploader does not grant rights
  the underlying source videos retain. If those videos include commercial
  content (movies, TV, copyrighted YouTube), the license is incoherent.
  Acceptable for low-stakes prototyping; legal review before commercial
  deployment.
- **NRC-VAD commercial license**: the "free for research and personal use"
  language is explicit; commercial use requires contacting NRC Canada.
  For a commercial pipeline, contact NRC for a commercial license, or
  auto-generate `contrast_pairs.jsonl` from 3 hand-authored trait
  descriptions via the persona-vectors pipeline ([spike-f memo](spike-f-persona-vectors-pipeline.md)) —
  sufficient for Phase 5's `D` matrix extraction step.
- **Non-VAD label limitations**: most commercial-clean datasets above ship
  categorical emotion labels only (anger/sad/happy/etc.), not continuous
  V/A/D. Phase 5's regressor target requires continuous VAD; the standard
  workaround is to map categorical labels to canonical VAD prototypes via
  NRC-VAD lookup of the emotion *name* (e.g., "anger" → mean VAD across
  NRC-VAD entries containing "anger" and synonyms). This is lossy but
  workable. Synthetic-data path (Tier C) sidesteps this by generating
  audio with known V/A/D targets.
- **Speaker disjointness**: PROJECT_PLAN §10 R5 requires speaker-disjoint
  splits to prevent identity leakage. CREMA-D (91 speakers) and SUBESCO
  (20 speakers) support this naturally. EmoVoice-DB (only 5 speaker
  timbres) has limited speaker diversity; combine with CREMA-D held-out
  speakers for validation.

### Revised primary recommendation

**Phase 5 calibration (commercial-deployable)**:

1. **Primary training data**: `EmoVoice-DB` (synthetic, MIT, large) +
   `myleslinder/crema-d` (real human, ODbL, diverse speakers).
2. **Held-out validation**: a speaker-disjoint slice of CREMA-D actors,
   plus JL-Corpus (CC0).
3. **Categorical → VAD mapping**: NRC-VAD lookup if commercial license is
   acquired; otherwise an auto-generated `contrast_pairs.jsonl` from 3
   hand-authored axis trait descriptions via the persona-vectors pipeline
   of Chen et al. 2025 ([spike-f memo](spike-f-persona-vectors-pipeline.md);
   ~½ developer-day for descriptions; <$50 LLM-API for generation).
4. **Optional augmentation**: cross-lingual via SUBESCO (Bangla) and
   Emozionalmente (Italian) for speaker-pool diversity. Use only if
   monolingual baseline saturates.

This combination is **fully commercial-clean**, requires no academic
affiliation, and is downloadable today.

The MSP-Podcast / IEMOCAP path is preserved above for reference in case
the project's deployment intent shifts to research-only or institutional
affiliation becomes available later.

### What this means for PROJECT_PLAN §10 R8

Risk R8 ("license/redistribution constraints on training data") is
**resolved as severe-but-bounded**: by sticking to Tier A + Tier B + Tier
C, the constraint never materializes. The cost is loss of MSP-Podcast's
spontaneous-conversational-speech training signal — distribution-shift
risk against deployment audio is now larger than it would be with
MSP-Podcast included. Mitigation: collect a small (~1 hour) internal
spontaneous-speech evaluation set during Phase 5 calibration, and report
metrics on it alongside the held-out CREMA-D / JL-Corpus numbers.

---

## Files referenced

**Created (Phase 0)**:
- This memo.

**Read (no edits)**:
- [`PROJECT_PLAN.md` §6 Phase 5 + §10 R2/R8](../PROJECT_PLAN.md) for
  acceptance criteria and risk register.
- [`IMPLEMENTATION_PLAN.md` §4](../IMPLEMENTATION_PLAN.md) for the four
  numbered calibration scripts these datasets feed into.
- [`README.md`](../README.md) priority-paper #3 ([arXiv:2602.06000](https://arxiv.org/abs/2602.06000))
  for the empirical recommendation of MSP-Podcast as primary.

**External resources cited**:
- MSP-Podcast: https://msp.utdallas.edu/Database.html
- IEMOCAP: https://sail.usc.edu/iemocap/
- CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D
- ESD: https://github.com/HLTSingapore/Emotional-Speech-Data
- RAVDESS: https://zenodo.org/record/1188976
- NRC-VAD lexicon: https://saifmohammad.com/WebPages/nrc-vad.html
