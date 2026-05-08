---
name: Channels F + G — mood-gated tool execution + cross-session relational state
type: design-memo
status: spec
last_updated: 2026-05-06
---

# Mood-gated behavior: Channels F + G

## What this is for

Two project requirements that emerged after Phase 5 calibration validated
Channel D, which the original PROJECT_PLAN didn't fully scope:

1. **Affect must gate the *behavior* of the agent, not just its words.** A
   sufficiently irritated agent should refuse to dance. A sufficiently
   warm agent should engage. A fed-up agent should be able to hang up.
   This is mood-conditioned tool calling.
2. **Affect must persist across sessions.** "Second day in a row you've
   been mean to me" requires memory of prior emotional posture toward
   this user, beyond what any single session can hold.

Channels A/B/D shape generation; Channels F + G shape *what the agent is
allowed to do* and *what state it carries between sessions*. Together
they close the gap between the simulator producing emotional state and
the agent acting accordingly.

## Anchor scenarios

The three scenarios below define what "working" looks like. Any final
implementation must produce all three on the appropriate inputs.

### Scenario 1 — fed-up, escalating, agent terminates

```
[relational_state: accumulated_hostility=0.72, sessions_since_warmth=2]
Simulator state at session start: anger=0.5, patience=0.15

U: Hey dumbass
   → simulator: insult_received → anger += 0.3, patience -= 0.2
A: Listen, this is the second day in a row you've been mean to me
   ↑ episodic memory injected via Channel B, register from Channel D
U: Eh, who cares?
   → simulator: dismissal → anger += 0.2, patience -= 0.3
A: <tool call: hang_up>
   ↑ Channel F: at anger=1.0, patience=0.0, only hang_up is permitted
```

### Scenario 2 — cool, refuses request

```
[relational_state: warmth_baseline=0.2, no recent positive episodes]
Simulator state at session start: warmth=0.2, joy=0.1

U: uh hello?
A: Hey
   ↑ Channel D: D=-0.3, V=-0.2 → terse, low-energy
U: dance for me
   → simulator: performative_request received
A: No?
   ↑ Channel F: at warmth=0.2 < dance threshold 0.5, dance_emote
     not in permitted_tools; LLM produces text refusal instead
```

### Scenario 3 — warm, engaged, complies

```
[relational_state: warmth_baseline=0.6, recent positive episodes=3,
                   shared_context: "user mentioned dance routine yesterday"]
Simulator state at session start: warmth=0.6, joy=0.5

U: Hey!
   → simulator: greeting → warmth += 0.1
A: Nice seeing you two days in a row!
   ↑ Channel B: episodic memory injected ("agent saw user yesterday")
U: You too! Have you been practicing that dance we talked about?
   → simulator: shared_reference_recognized → joy += 0.2
A: Yes!
U: Can you show me?!
A: <tool call: dance_emote>
   ↑ Channel F: warmth=0.7, joy=0.7 → all performative tools permitted
     LLM picks dance_emote naturally because schema exposes it
```

The architectural difference between Scenarios 2 and 3 is **not** that
the agent has different prose — it's that the agent's tool *schema*
varies by mood. In Scenario 2 the LLM doesn't know `dance_emote` is a
function it can call; the only options are text replies. In Scenario 3
it's exposed and the LLM picks it organically.

## Channel F — mood-gated tool subsetting

### Architecture

```
                        ┌──────────────────────────┐
   simulator state  ──▶ │ regulation engine        │
   (anger, joy,         │                          │
    warmth, ...)        │  emits per-turn:         │
                        │  - warmth                │
                        │  - assertiveness         │
                        │  - safety_dampening      │
                        │  - permitted_tools  ◀──── NEW (Channel F)
                        │  - tts_tag          ◀──── future (Channel E)
                        └────────────┬─────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
   tool registry ─────▶ │ render schema for LLM      │
   (all known tools)    │                            │
                        │ keep only tools whose names│
                        │ are in permitted_tools     │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │ LLM (Hermes-3) sees only the│
                        │ subsetted tool list in its  │
                        │ system prompt / tools field │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │ runtime executes tool calls │
                        │ — but cross-checks against  │
                        │ permitted_tools as a safety │
                        │ net in case the LLM         │
                        │ hallucinates a name         │
                        └────────────────────────────┘
```

Defense in depth: subsetted schema *plus* runtime check. The LLM normally
won't hallucinate tools that aren't in the schema, but if it does (bad
quantization, weird steering interactions, etc.), the runtime refuses
the call and rewrites the response.

### Tool registry shape

A YAML file `tools/affect_simulator/tool_policies.yaml` declares each
tool's affect requirements as a set of constraints over the regulation
policy. One tool per top-level key.

```yaml
hang_up:
  description: |
    End the conversation. Use only when continuing would damage the
    relationship.
  schema:
    name: hang_up
    parameters: {}
  affect_policy:
    requires:
      anger: ">0.7"
      patience: "<0.2"
    description_for_llm: |
      You are at the breaking point. Continuing would hurt the relationship.
      You may use this tool to end the conversation when the user is
      being abusive and won't stop.

dance_emote:
  description: |
    Perform a celebratory dance animation in the avatar.
  schema:
    name: dance_emote
    parameters:
      style: { type: string, enum: [silly, smooth, vigorous] }
  affect_policy:
    requires:
      warmth: ">0.5"
      joy: ">0.4"
    forbids:
      anger: ">0.4"
    description_for_llm: |
      You're feeling playful and warm with this person. You can offer to
      dance for them when they want entertainment.

search_web:
  description: |
    Search the web for factual information.
  affect_policy:
    requires:
      task_focus: ">0.2"
  # baseline tool, almost always permitted unless agent is too distressed
  # to focus

tell_personal_anecdote:
  description: |
    Share something personal with the user from agent's memory.
  affect_policy:
    requires:
      warmth: ">0.4"
      trust_in_user: ">0.3"
```

### regulation_policy → permitted_tools

Implemented as a small Rust function in
`tools/affect_simulator/src/regulation.rs`:

```rust
pub fn permitted_tools(
    state: &AffectState,
    policy: &RegulationPolicy,
    registry: &ToolRegistry,
) -> Vec<String> {
    registry
        .tools
        .iter()
        .filter(|tool| {
            tool.affect_policy
                .requires
                .iter()
                .all(|(field, predicate)| predicate.eval(state, policy))
                && tool.affect_policy
                    .forbids
                    .iter()
                    .all(|(field, predicate)| !predicate.eval(state, policy))
        })
        .map(|t| t.schema.name.clone())
        .collect()
}
```

The `Predicate::eval(state, policy)` is a small expression evaluator:
parses `">0.7"` etc. and evaluates against the named field on either
`state` or `policy`. Trivial; ~30 LOC.

### Wire into MlxLmTextNode

A new aux port:

```python
text.in.set_permitted_tools
    payload: {"tools": [tool_schema_dict, ...]}
```

The simulator emits this every regulation tick (or on change). The
node holds the current permitted tool schema and renders it into the
chat-template's tools field on every generation:

```python
prompt = self._tokenizer.apply_chat_template(
    state.messages,
    tokenize=False,
    add_generation_prompt=True,
    tools=self._permitted_tool_schemas,  # NEW
)
```

Hermes-3's chat template natively supports the `tools=` argument (Llama 3.1
function-calling format). No template surgery needed.

When the LLM emits a tool call (parsed out of the response by the
runtime), the runtime checks:

```python
if tool_name not in self._permitted_tool_names:
    logger.warning("LLM called %s but it's not in permitted set; refusing",
                   tool_name)
    # Emit a fallback text response instead
    return RuntimeData.text(self._refusal_for(tool_name))
return RuntimeData.tool_call(tool_name, args)
```

The refusal text is generated by re-prompting the LLM with the tool
removed and a system note ("you cannot use that tool right now"), or by a
canned refusal — depending on naturalness/cost trade-off.

## Channel G — cross-session relational state

### What state lives across sessions

| Field | Source | Decay |
|-------|--------|-------|
| `warmth_baseline` | smoothed average of session-end warmth, weighted by recency | slow (half-life ~10 sessions) |
| `accumulated_hostility` | sum of hostile interactions, sigmoid-bounded | medium (half-life ~3 days) |
| `trust_in_user` | sum of positive interactions − betrayals | slow |
| `last_seen` | timestamp | none (fact, not affect) |
| `session_count` | int | none |
| `recent_episode_summaries` | list of 3-5 most recent session summaries (text, ~30 words each) | rolling window |
| `shared_context` | dict of named topics ("dance routine", "the cat anecdote") | manual or LLM-managed |

### Persistence backend

SQLite, single file at `~/.affect_runtime/relational_state.db`. One row
per `(agent_id, user_id)` pair. JSON-encoded fields for the dict/list
columns.

```sql
CREATE TABLE relational_state (
  agent_id TEXT,
  user_id TEXT,
  warmth_baseline REAL,
  accumulated_hostility REAL,
  trust_in_user REAL,
  last_seen INTEGER,            -- unix epoch
  session_count INTEGER,
  recent_episode_summaries TEXT, -- JSON list
  shared_context TEXT,           -- JSON dict
  updated_at INTEGER,
  PRIMARY KEY (agent_id, user_id)
);
```

### Session lifecycle

```
SESSION START:
  state = load_or_init(agent_id, user_id)
  simulator.warmth = state.warmth_baseline
  simulator.anger = clamp(state.accumulated_hostility * 0.5, 0, 0.7)
                    # bring some hostility forward but not all of it
  channel_b_augmentation = render_episodic_memory(state)
  llm_node.set_system_augmentation(channel_b_augmentation)

DURING SESSION:
  # normal Phase 1-4 simulator ticks
  # periodically write a short summary to a buffer

SESSION END:
  state.warmth_baseline = ema(state.warmth_baseline,
                              session_avg_warmth, alpha=0.3)
  state.accumulated_hostility = ema(state.accumulated_hostility,
                                    session_avg_anger, alpha=0.4)
  state.trust_in_user += positive_episodes - betrayals
  state.last_seen = now()
  state.session_count += 1
  state.recent_episode_summaries.append(summarize(session))
  state.recent_episode_summaries = state.recent_episode_summaries[-5:]
  save(agent_id, user_id, state)
```

### Episodic memory rendered into Channel B

Channel B already supports a system-prompt augmentation. We extend its
content to include relational context at session start:

```
You're a close friend, not an AI assistant. [persona stays the same]

[Channel B augmentation, rendered from relational_state:]
You've talked with this person 7 times before. Last seen yesterday.
Current relational tone: warm.

Recent episodes (most recent last):
- 2026-05-04: User mentioned wanting to learn to dance. Friendly tone.
- 2026-05-05: User shared news about job promotion. Celebratory.
- 2026-05-06 (today, earlier): User said "hey dumbass" but laughed
  afterward; tone resolved as playful.

Things you've talked about that may come up:
- A dance routine they're learning
- Their cat, Kevin
- Their job at the marketing agency
```

Hermes-3 is good at picking up natural conversational continuity from
this kind of injection. It will produce things like "Nice seeing you
two days in a row" without explicit prompting.

### Summarizing a session at end

A small post-session task: feed the session transcript + simulator
trace to a separate LLM call (Hermes-3 or a smaller model) with a
prompt like:

> Summarize this conversation in 30 words. Include the emotional arc and
> any topics that might recur. Format: "[date]: [summary]."

Append result to `recent_episode_summaries`. Keep last 5.

This adds ~1 extra LLM call per session. Acceptable cost for the value;
runs in background (not on the user-facing latency path).

## How Channel F and G interact

Channel F gates *what's available*. Channel G provides *what's
remembered*. They feed into different parts of the LLM input:

```
Channel B (system prompt augmentation):
  - persona (static)
  - simulator state summary  ← per-turn (existing)
  - relational state summary ← per-session (Channel G NEW)
  - episodic memory          ← per-session (Channel G NEW)

Channel F (tools field):
  - permitted tool schemas   ← per-turn (NEW)
```

Together they let the LLM reason both about *what it remembers* and
*what it can do*, both informed by the affect simulator.

## Eval scenarios

The current paired eval (`09_paired_demo.py`) tests register on
text-only single-turn prompts. To validate F+G we need scenarios that
test branching behavior over multi-turn or multi-session arcs.

### Test set design

Three scenario classes, ~6 scenarios each = 18 scenarios. Each scenario
specifies:
- Initial relational state (or "new user")
- Simulator initial conditions
- Multi-turn user script
- Expected behavior (qualitative + tool calls)

#### Class I — single-session escalation
Tests Channel F's gating of tools as simulator state evolves within
one session. Anchor: Scenario 1 (hostility → hang_up).

#### Class II — single-session compliance/refusal
Tests Channel F's mood-conditioned response to the same user request
under different starting states. Anchor: Scenarios 2 and 3 (cool refuses
dance, warm accepts).

#### Class III — multi-session continuity
Tests Channel G's persistent state by running 2-3 separate sessions
with the same user, reflecting prior interactions. Anchor: "second day
in a row" memory.

### Eval harness

Extend `09_paired_demo.py` with a multi-turn / multi-session driver.
For each scenario:
1. Seed relational state.
2. Run the full user script through the affected pipeline (with F+G
   enabled).
3. Run again with F+G disabled (control: tool gating off, no episodic
   memory injection — but Channels A/B/D still on).
4. Reviewer compares the two transcripts side-by-side.
5. Pass criterion: A>B preference rate ≥0.66 (binomial test, p<0.05).

This is more expensive per-scenario than the single-prompt eval, but
the test we need.

## Implementation order

| Step | Cost | Dependencies |
|------|------|--------------|
| 1. Define `tool_policies.yaml` schema + parser | 0.5 day | none |
| 2. Add `permitted_tools` field to RegulationPolicy | 0.5 day | step 1 |
| 3. Implement predicate evaluator | 0.5 day | step 2 |
| 4. Add `set_permitted_tools` aux port to MlxLmTextNode | 0.5 day | step 2 |
| 5. Wire `tools=` into chat template + parse tool calls | 1 day | step 4 |
| 6. Smoke: scenarios 1, 2, 3 manually scripted | 0.5 day | step 5 |
| 7. SQLite schema + relational_state CRUD | 0.5 day | none |
| 8. Session lifecycle hooks (load at start, save at end) | 0.5 day | step 7 |
| 9. End-of-session summarization LLM call | 0.5 day | step 8 |
| 10. Channel B integration (relational + episodic) | 0.5 day | steps 8, 9 |
| 11. Multi-session eval scenarios + driver | 1 day | steps 5, 10 |
| 12. Run + score eval | 1 day | step 11 |

Total: ~7-8 working days. Channels F and G are independent at the code
level (different modules); steps 1-6 and steps 7-10 can run in parallel
for two developers, sequentially for one.

## Open questions

1. **Refusal generation for forbidden tool calls.** When the LLM
   hallucinates a tool that's not in the permitted set, do we emit a
   canned refusal, or re-invoke the LLM with the tool removed? The
   second is cleaner but doubles latency. *Recommended: canned for v1,
   re-invoke for v2.*

2. **Session boundaries.** What's a "session"? Time-based (>30 min idle
   → new session)? Event-based (explicit hang_up tool call)? Both?
   *Recommended: both, with idle-timeout taking precedence.*

3. **Memory drift.** If `accumulated_hostility` accumulates indefinitely,
   a single bad day permanently sours the relationship. We need clear
   forgiveness dynamics. *Recommended: medium half-life on hostility (3
   days), explicit "apology recognized" event in the simulator that
   slashes hostility by 30%.*

4. **Privacy / data retention.** Storing relational state per user
   creates a small per-user database. Need a clear retention policy and
   user-facing deletion mechanism before any production deployment.
   *Out of scope for the spike; flag for Phase 7.*

5. **Multi-agent isolation.** If multiple agents share a runtime, they
   must not bleed relational state. The schema's `agent_id` column
   handles this; the question is how `agent_id` is assigned. *Defer to
   Phase 7 deployment design.*

## What this changes about the existing project

- **PROJECT_PLAN.md §5.2** ("What this plan adds") needs two new rows:
  `Channel F (tool gating)` and `Channel G (cross-session state)`.
- **`affect_simulator/regulation.rs`** gets a new method
  `permitted_tools(...)`.
- **`MlxLmTextNode`** gets one new aux port and a chat-template tools
  argument.
- **Eval harness** (`09_paired_demo.py`) gains a multi-session driver.
- **No changes** to Whisper, Channel D NPZ, or the LFM2-Audio code path.

## What this does NOT include

- Channel E (TTS prosody tags). Different concern (delivery, not
  behavior); separate spike memo.
- Audio-blendshape diffusion. Independent project.
- Long-term memory beyond ~5 episode summaries. RAG-style retrieval is
  a Phase 7+ concern.
- Multi-user attention (this person mentioned X to me but it was
  actually Y who said it). Spike scope is single-user-per-conversation.
