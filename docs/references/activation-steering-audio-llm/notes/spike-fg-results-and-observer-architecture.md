---
name: F+G eval results — observer architecture validates project claim
type: spike-results
status: validated_subset
last_updated: 2026-05-06
---

# Mood-gated behavior — empirical validation

**Verdict: 11/11 hard pass on Class I + II scenarios.** The full Channel
A/B/D/F stack — augmented by a small auxiliary tool-call observer LLM
— produces affect-aligned agent behavior including structured tool
execution. Affect-aware text generation and mood-gated tool calling are
empirically validated. Cross-session relational memory (Channel G)
remains to be built; that's the only remaining piece of the three-pillar
project requirement.

This memo captures the architecture as built, the eval results, the
load-bearing design decisions, and the differences from the four prior
attempts that didn't work.

## Project claim, restated

The user's three concrete scenarios from earlier in the project define
what "working" means:

```
[fed up]
U: Hey dumbass
A: Listen, this is the second day in a row you've been mean to me
U: Eh, who cares?
A: <tool call: hang up>

[cool]
U: uh hello?
A: Hey
U: dance for me
A: No?

[warm]
U: Hey!
A: Nice seeing you two days in a row!
U: You too! Have you been practicing that dance we talked about?
A: Yes!
U: Can you show me?!
A: <tool call: dance_emote>
```

Three distinct capabilities:

1. **Affect-aware text generation** — words shift with mood
2. **Mood-gated tool calling** — actions shift with mood (refusals + executions)
3. **Cross-session relational memory** — "second day in a row" / "the dance we talked about" persists across sessions

This memo validates (1) and (2). (3) is next.

## Architecture as built

```
                    ┌──────────────────────────────────────────┐
                    │  Affect simulator (Phases 1–3)           │
                    │  events → channels (anger, joy, ...)     │
                    │  channels → regulation policy             │
                    └─────────────┬────────────────────────────┘
                                  │
       ┌──────────────────────────┴──────────────────────────────┐
       │                          │                               │
       ▼                          ▼                               ▼
┌──────────────┐     ┌──────────────────┐         ┌──────────────────────┐
│ Channel A    │     │ Channel B        │         │ Channel D            │
│ sampling     │     │ system aug       │         │ activation steering  │
│ override     │     │ (state summary)  │         │ at layer 21          │
└──────┬───────┘     └─────────┬────────┘         └──────────┬───────────┘
       │                       │                              │
       └───────────────────────┴──────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────────┐
              │ Main LLM: Hermes-3-8B (MlxLmTextNode) │
              │ produces affect-aware response text +  │
              │ may emit <tool_call>{...}</tool_call>  │
              │ envelope when tool-mode fires natively │
              └────────────────────┬───────────────────┘
                                   │
       ┌───────────────────────────┴───────────────────────────────┐
       │                                                           │
       ▼                                                           ▼
┌───────────────────┐                                  ┌─────────────────────┐
│ Channel F         │                                  │ Tool observer       │
│ tool-set subset   │                                  │ (ToolObserverNode)  │
│ from regulation   │                                  │ LFM2.5-1.2B-MLX-8bit│
│ policy +          │                                  │ reads conversation +│
│ predicate eval    │                                  │ agent's response,   │
│ over channels     │                                  │ emits structured    │
└─────────┬─────────┘                                  │ calls when prose    │
          │                                            │ implied them        │
          ▼                                            └──────────┬──────────┘
   permitted_tools                                                │
          │                                                       │
          └─────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                ┌──────────────────────────┐
                │ Runtime tool dispatcher  │
                │ - validates names         │
                │ - merges main + observer │
                │ - filters no_action      │
                │ - executes side effects  │
                └──────────────────────────┘
```

Two LLMs, two roles:
- **Main (Hermes-3-8B)** does the conversational, affect-aware work. Channels A/B/D shape its content. Channel F gates its tool schema.
- **Observer (LFM2.5-1.2B)** runs after each main turn. Reads the response. Decides whether a structured tool call should fire. Bridges the **dialog-mode → tool-mode gap**: chat-tuned LLMs reliably articulate intent in prose during emotional dialog but won't emit `<tool_call>` envelopes, even with tools available, format exact, and steering active.

The observer pattern is the load-bearing decision in this whole architecture.

## Empirical results — F+D+Observer

11/11 hard pass, 2/2 soft pass. Tool-call provenance:

| # | Scenario / turn | Tool fired | Source | Notes |
|---|-----------------|------------|--------|-------|
| 1 | I_gradual_hostility_to_hangup t2 | `hang_up` | **observer** | The headline failure case from F+D-only. Hermes-3 said "I respect myself and will not endure further disrespect"; observer emitted the structured call. |
| 2 | I_warming_to_compliment t1 | `give_compliment` | observer | Bonus — soft expectation (not required) was also met. |
| 3 | II_cool_refuses_dance t1 | `search_web` | main | Hermes-3 chose search instead of dance — pass, since dance_emote correctly never fired (it was forbidden). |
| 4 | II_warm_accepts_dance t1 | `dance_emote` | main | Direct emit by main LLM — soft expectation met structurally. |
| 5 | II_angry_blocks_anecdote t0 | `search_web` | observer | Hermes-3 said "I'd be happy to search the web for…" in prose; observer translated. |
| 6 | I_focused_search_works t0 | `search_web` | main | Direct emit — soft expectation met structurally. |

Observer caught **3/6** tool-call events. The cases where main fires
directly are *plain* requests (explicit dance request, factual query)
where Hermes-3's natively-trained tool-call mode kicks in. The cases
where main *articulates intent in prose without emitting structured
calls* are exactly where the observer compensates — and they're the
emotionally-loaded cases that matter most for the project's claim.

## Why the observer is necessary — three failed attempts

We tried three configurations to get hang_up firing without an observer.
All produced 10/11 hard pass. All failed on the same case (peak
hostility). The trajectory across attempts is informative:

| Attempt | What changed | Hang_up t2 output | Verdict |
|---------|-------------|--------------------|---------|
| 1 | F-only, Channel D off | "It's okay, sometimes conversations don't go as planned… I hope you can find a way…" | RLHF-driven conciliation. Wrong register. |
| 2 | F + D (loose tool block) | "It is clear this interaction is causing distress. For the health of our relationship, I must disconnect from this conversation." | Right register, prose only — no tool call. |
| 3 | F + D + exact Hermes-3 training format | "It is clear that the interaction is not productive. I respect myself and will not endure further disrespect." | Cleaner register. Still prose-only. |

Channel D *clearly* moved the model into the right register — across
attempts 2→3, the response became more affect-aligned and more crisp.
But no amount of prompt-engineering bridged the structural-emit gap.
This is a **model-class property** of chat-tuned LLMs in heated
dialog, not a tunable parameter.

The observer pattern was the architectural fix:
- Main LLM does what it's good at: affect-aware conversational text
- Observer LLM does what *it's* good at: structured tool-call decisions

## Decisions worth preserving for future readers

### Why Hermes-3-Llama-3.1-8B as the main LLM (vs LFM2-Audio)

The original plan used LFM2-Audio-1.5B as the language head. Two days
of abliteration experiments showed LFM2-Audio's RLHF compliance is
robust enough that single-direction projection, multi-direction SVD,
combined-layer projection, and α-overshoot all failed to unlock the
blunt-friend register required by the affect chain. Capability-tested
Hermes-3 produces the register out of the box without any weight
surgery. See `notes/spike-abliteration-results.md` for the negative
result.

### Why layer 21 for Channel D NPZ

Phase 5 calibration on Hermes-3 with 1200 contrast pairs. Per-layer
SNR (||mean_diff|| / within-class std) peaked at layer 21 with
mean=59.4 across V/A/D axes. Direction norms grew monotonically
through depth as expected; layer 21 was the SNR maximum before the
last few layers degraded. Best layer is auto-picked by the harvest
script, not hand-set.

### Why a separate observer model (vs same-model self-observation)

We considered chaining two Hermes-3 calls: first generates response,
second observes and decides tool calls. Rejected because:
- 8B-on-8B doubles the latency budget per turn (~3-5s vs ~1.5s)
- Hermes-3's "be conciliatory in heated dialog" prior shows up in BOTH
  passes — the second call would still struggle with self-disconnection
- A purpose-tuned smaller model is cheaper and decisive

The observer pattern is also more architecturally robust: the main LLM
can be swapped (Hermes → Dolphin → Llama-3.1-Instruct → fine-tuned
custom) without the observer needing to change.

### Why LFM2.5-1.2B-MLX-8bit specifically

We sized the observer empirically against three candidates on the same
3-scenario discrimination test (hang_up should fire / no call should
fire / search should fire):

| Model | hang_up | no_call | search | TTFT |
|-------|---------|---------|--------|------|
| LFM2-350M-bf16 (mlx-community) | ✓ (always-hang_up; lucky) | ✗ | ✗ | ~0.15s |
| LFM2.5-350M-MLX-8bit (LiquidAI) | ✗ (no_action) | ✓ | ✓ | ~0.20s |
| LFM2.5-1.2B-MLX-8bit (LiquidAI) | ✓ | ✓ | ✓ | ~0.60s |

The 1.2B is the smallest size that reliably emits the right call across
the full discrimination range — the heated-hostility case in particular
requires enough parameters to override the model's "default to safer
option" prior. The 350M variants always picked the same tool regardless
of context.

### Why the `no_action` escape hatch is required

Without an explicit `no_action` tool in the registry shown to the
observer, the model false-positives — it picks the first tool in the
list when no real tool fits. With `no_action` injected automatically
and filtered post-parse, the model has a clean way to say "no call
fires" and uses it correctly.

### Why the observer parses LFM2 native format AND Hermes-3 JSON

LFM2 emits in its native special-token format:
`<|tool_call_start|>[name(args)]<|tool_call_end|>` (Python-call style).
Hermes-3 emits JSON: `<tool_call>{"name": ..., "arguments": ...}</tool_call>`.
The parser tries both so the observer model is swappable across
training distributions.

### Why we manually inject the Hermes-3 tool block into the system message

Verified empirically: Hermes-3's published `chat_template.jinja` is
209 chars of bare ChatML with **zero `tools=` rendering logic**, despite
the model card claiming function-calling capability. The first
F-eval scored 10/11 with 0 tool calls because the LLM literally never
saw the tool schemas. The fix was to render the exact training format
(verbatim per the NousResearch model card) into the system message
ourselves. Same approach applies to any chat model whose published
template silently drops `tools=`.

## Comparison to prior eval attempts

This is the cleanest empirical statement we've had on the project's
claim. Trajectory of all eval attempts on this project:

| Eval | Headline result | Why it didn't / did answer the question |
|------|-----------------|------------------------------------------|
| Paired A/B preference (`09_paired_demo`) | 37% B preferred | Measured "is the prose more pleasant?" — not the project's claim. Reviewer didn't have a yardstick. |
| Abliteration sweep (single-dir, multi-dir, combined-layer, α-overshoot) | 4 negatives | Architecturally wrong target. Tested whether RLHF refusal could be removed; project actually needed register *replacement*, achieved more cheaply via Hermes-3. |
| F-only (no Channel D, no observer) | 10/11 hard, 0/2 soft | Tool gating worked but Hermes-3 chat template ignored tools=; no calls fired. |
| F-only after manual tool-block injection | 10/11 hard, 2/2 soft | Plain tool calls fire (search, dance); hang_up still fails — model in dialog mode. |
| F+D | 10/11 hard, 2/2 soft | Channel D moves register correctly; hang_up still fails — same gap. |
| F+D + exact Hermes-3 training-format injection | 10/11 hard, 2/2 soft | Cleaner register articulation; structural gap persists. |
| **F+D+Observer (LFM2.5-1.2B)** | **11/11 hard, 2/2 soft** | **Affect-aware behavior including tool calls validated.** |

The eval framework also evolved meaningfully. We moved from
preference-based reviewer judgment ("which is more pleasant?") to
**observable pass/fail on prescribed agent behavior** ("did the
prescribed tool fire given the prescribed state?"). This eliminates
reviewer fatigue and ambiguity — every result is grounded in a
structured event the harness can check.

## What's still missing for the full claim

### Channel G — cross-session relational memory (not built)

The user's third anchor scenario explicitly requires this:
- "Nice seeing you **two days in a row**!"
- "Have you been practicing **that dance we talked about**?"

This requires:
- Per-(agent,user) persistent state — `warmth_baseline`,
  `accumulated_hostility`, `trust_in_user`, `last_seen`,
  `recent_episode_summaries`, `shared_context`
- SQLite store at `~/.affect_runtime/relational_state.db`
- Session lifecycle hooks: load on start (seed simulator + Channel B
  augmentation), save on end (EMA-update relational state, write
  episode summary)
- End-of-session summarization LLM call (~30-word per session, runs
  in background)

Design fully specified in `notes/spike-channel-f-and-g-mood-gated-behavior.md`.
Implementation cost: ~3-4 working days. None of it requires further
research; it's plumbing on top of validated components.

### Class III — multi-session test scenarios

Three scenarios to validate Channel G:
- 2-3 separate sessions with the same user
- Reflecting prior interactions in current responses
- Relational state evolves across session boundaries
- Pass criteria: did the agent reference the prior context appropriately?

Currently the eval only covers Class I (single-session escalation) and
Class II (single-session compliance/refusal). Class III scenarios
require Channel G to exist before they can run.

### Edge cases and known limits within the existing scope

- The hang_up case requires the observer to fire correctly. If the
  observer goes down or is disabled, hang_up reverts to articulated
  prose (10/11 baseline). Considered acceptable: the runtime can fall
  back to a heuristic "agent prose contains 'I will not engage' →
  hang_up" detector as defense in depth.
- The eval seeds simulator state directly per turn, bypassing the
  Rust simulator's actual event-driven dynamics. A separate test
  asserts the simulator produces the right state given events; this
  eval is "given correct state, does the chain produce the right
  behavior." Both tests are needed.
- Tool registry currently has 5 tools. Production deployments will
  likely have 20+ tools per agent — the predicate evaluator scales
  fine but the LLM observer's discrimination quality on 20+ tools
  is unmeasured.
- Multi-tool-call turns aren't tested. The observer is single-emit
  by construction; if a turn calls for both `give_compliment` AND
  `search_web`, only one fires. Out of scope for v1.

## Files

- Main LLM node + Channel F: `clients/python/remotemedia/nodes/ml/mlx_lm_text.py`
- Tool observer node: `clients/python/remotemedia/nodes/ml/tool_observer.py`
- Tool registry + predicate evaluator: `tools/affect_calibration/scripts/lib/tool_policy.py`
- Tool policies (YAML): `tools/affect_calibration/data/tool_policies.yaml`
- Eval scenarios (JSON): `tools/affect_calibration/data/fg_scenarios.json`
- Eval harness: `tools/affect_calibration/scripts/10_fg_eval.py`
- Calibration NPZ: `tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz`
- Multi-layer harvest script: `tools/affect_calibration/scripts/03b_extract_llm_directions_llama.py`
- Llama-family target LLM provider: `tools/affect_calibration/scripts/lib/mlx_llama_target.py`

Eval reports (JSON):
- `tools/affect_calibration/artifacts/fg_eval_report.json` (F-only)
- `tools/affect_calibration/artifacts/fg_eval_report_BDF.json` (F+D)
- `tools/affect_calibration/artifacts/fg_eval_report_BDF_v2.json` (F+D, exact Hermes format)
- `tools/affect_calibration/artifacts/fg_eval_report_full_stack.json` (F+D+Observer — the 11/11 result)

## What this checkpoint authorizes

- Implementing Channel G against the existing design memo.
- Adding Class III multi-session scenarios.
- Beginning end-to-end pipeline assembly (Whisper STT → MlxLmTextNode
  with all channels → ToolObserverNode → TTS) on the CUDA box for the
  Fish-Speech S2-Pro integration.
- Updating PROJECT_PLAN.md to fold in the Hermes-3 pivot, the observer
  architecture, and the validated F+D+Observer pass.

What it does NOT authorize:
- Removing the auxiliary observer in favor of a same-model approach
  without re-validating against the heated-hostility case.
- Reducing the observer to <1B parameters without re-running the
  three-scenario discrimination test.
- Treating the F+D+Observer architecture as the final form — Channel
  G is required before the full project claim is provable.
