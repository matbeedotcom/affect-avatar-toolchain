# affect-coach — what to say and how to say it

Terminal companion that runs the full affect-aware-agent chain
(simulator → Channel D / Channel B → Hermes-3 with steering) and
prints, for any user-supplied context:

1. **What to say** — a suggested response from the affect-aware
   language head.
2. **How to say it** — the steering target translated into
   plain-English stage directions ("warm and direct", "yielding,
   low-arousal", "muted delivery, soften the blow"…).
3. **What to do with your face** — the same affect → ARKit-52
   blendshape mapping the live `AffectSimulatorNode` emits per
   tick. `--verbose` shows the top-5 active blendshapes inline;
   `bake_affect_face.py` exports the full per-frame trajectory as
   a canonical `{kind: "blendshapes", arkit_52, pts_ms}` JSONL for
   any avatar consumer.

This is the offline, terminal-only sibling of the
[`hermes3_affect_s2s_webrtc_server`](../../crates/transports/webrtc/examples/hermes3_affect_s2s_webrtc_server.rs)
demo. Same Hermes-3-Llama-3.1-8B-4bit + same calibrated Channel D
directions + same blunt-friend system prompt; the only thing missing
from the live audio loop is Whisper STT and Kokoro TTS.

## Usage

```bash
# REPL mode (default; loads model once, ~30s warmup, then fast turns)
./tools/affect_coach/coach.py

# One-shot
./tools/affect_coach/coach.py --once "I just got fired"

# Force a scenario (skip the keyword classifier)
./tools/affect_coach/coach.py --scenario user_distress

# Verbose — show steering V/A/D, felt state, multi-line stage directions
./tools/affect_coach/coach.py --verbose

# Stage-directions-only mode (no model load — ~instant per turn)
./tools/affect_coach/coach.py --no-llm
```

REPL slash-commands:

```
/scenario <name>   force a scenario for subsequent turns (or `clear`)
/scenarios         list available scenarios
/verbose           toggle verbose output
:q                 exit
```

## Example output

```
> I just got fired

  Scenario:    honest_concern (user_distress)
  How to say:  measured, yielding, calm

  What to say:
    Oh shit. When did this happen? You wanna talk through what's next
    or do you just need to vent for a minute first?
```

With `--verbose`:

```
> Hit me with my biggest blind spot

  Scenario:    no_holds_barred (roast_invitation)
  How to say:  friendly, direct, high-energy
               Tone: friendly. Posture: direct. Energy: high-energy.
               Warm + confident at once — you can be honest without being cold.
  Steering:    V=+0.36  A=+0.00  D=+0.59
  Felt:        calm 0.65, joy 0.30, curiosity 0.21

  What to say:
    You assume people who disagree with you just haven't thought about
    it as hard as you have. They have. They just landed somewhere else.
```

## Prerequisites

- **Hermes-3 directions NPZ** at
  `tools/affect_calibration/artifacts/llm_directions/hermes-3-8b/layer21.npz`
  (gitignored; produce via
  [`03b_extract_llm_directions_llama.py`](../affect_calibration/scripts/03b_extract_llm_directions_llama.py)).
- **Cargo + the workspace's `affect-simulator` binary** (the coach
  shells out to `cargo run -p affect-simulator -- run …` to build
  per-scenario traces; cached next to the scenario JSON).
- **Python deps:** `mlx-lm`, `transformers`, `numpy`. If the MLX
  stack isn't importable the coach falls back to a degraded
  "stage directions only" mode and still prints the recommended tone
  / posture / energy.

## How it picks a scenario

`coach.py` ships with a regex keyword classifier over six scenarios
(see `SCENARIOS` in [`coach.py`](coach.py)). Each scenario's
keywords are first-match-wins, ordered roughly affect-strongest-first
so `"I finished chemo"` lands on `honest_concern` rather than
`warm_admiration`.

| Coach scenario          | Simulator scenario             | Triggers on …                                |
|-------------------------|--------------------------------|----------------------------------------------|
| `warm_admiration`       | `task_success_after_struggle`  | finished/won/promoted/engaged/sober          |
| `honest_concern`        | `user_distress`                | fired/sick/crazy/should I/three months       |
| `amused_critique`       | `repeated_tool_failure`        | again/another/fourth/wing it/rage-quit       |
| `shared_distaste`       | `unfair_blame`                 | how bad/scale of/worst part/never sing       |
| `playful_disagreement`  | `novel_observation`            | hot take/defend/destroy/argue                |
| `no_holds_barred`       | `roast_invitation`             | brutal/unfiltered/biggest blind spot/roast   |

`--scenario <name>` overrides the classifier when you know what you
want. Unmatched lines fall through to `warm_admiration`.

## Baking a renderable face track

[`bake_affect_face.py`](bake_affect_face.py) drives the same simulator
the coach uses, runs every emitted frame through the same
affect → ARKit-52 mapping the live
[`AffectSimulatorNode`](../../crates/core/src/nodes/affect_sim.rs)
emits on every tick, and writes a canonical blendshape JSONL — one
`{kind: "blendshapes", arkit_52: [f32; 52], pts_ms}` envelope per
output frame:

```bash
# Bake one of the simulator scenarios:
./tools/affect_coach/bake_affect_face.py --scenario warm_admiration \
    --out out/warm_admiration.affect_face.jsonl
✓ wrote out/warm_admiration.affect_face.jsonl
  scenario:  warm_admiration (task_success_after_struggle)
  duration:  60.2s @ 30 fps (1825 frames)

# Free-form input — uses the coach's keyword classifier:
./tools/affect_coach/bake_affect_face.py --coach-input "I just got fired" \
    --out out/honest_concern.affect_face.jsonl

# Inspect the trajectory inline (no JSONL write):
./tools/affect_coach/bake_affect_face.py --scenario warm_admiration --diagnose
…
  t=25179ms  mouthSmileLeft 0.19, mouthSmileRight 0.19, cheekSquintLeft 0.13
  t=28182ms  mouthSmileLeft 0.20, mouthSmileRight 0.20, cheekSquintLeft 0.14
…
```

The JSONL is the same wire format every avatar consumer in the
workspace already speaks (`Audio2FaceLipSyncNode`,
`SyntheticLipSyncNode`, the running `Live2DRenderNode` /
`CcRenderNode`). Today the JSONL is the artifact you can hand to a
follow-up renderer; a direct render path (e.g. a sibling of
[`scripts/avatars/render_clip.sh`](../../scripts/avatars/render_clip.sh)
that pipes the JSONL into `cc_render`) is open work — see "Rendering
the baked face" below.

### Rendering the baked face

[`scripts/avatars/render_affect_face.sh`](../../scripts/avatars/render_affect_face.sh)
is a sibling of `render_clip.sh` that runs `bake_affect_face.py`
to produce the JSONL, then drives the
[`affect_face_smoke`](../../crates/core/examples/affect_face_smoke.rs)
example to render an MP4. The Bevy-based smoke binary loads the
JSONL and pushes each frame through the renderer's existing ARKit
input path (`CcRenderer::push_pose`) — no streaming-node / pipeline
involvement, just the renderer + ffmpeg.

```bash
# One-shot: bake the affect face for a scenario and render it.
scripts/avatars/render_affect_face.sh \
    --scenario warm_admiration \
    --out out/wa_face.mp4

# Free-form input — coach.py's keyword classifier picks the scenario.
scripts/avatars/render_affect_face.sh \
    --coach-input "I just got fired" \
    --out out/honest_concern_face.mp4

# Just bake the JSONL (skip the render). Useful for iterating on the
# affect→ARKit mapping without paying for Bevy / ffmpeg.
scripts/avatars/render_affect_face.sh \
    --scenario roast_invitation --bake-only \
    --out out/roast.affect_face.jsonl

# First-time build of the smoke binary (heavy — Bevy 0.15 cold build):
scripts/avatars/render_affect_face.sh \
    --scenario warm_admiration --build \
    --out out/wa_face.mp4
```

Output: an MP4 of a still avatar (T-pose body, no audio) with the
affect face evolving over time. Frame thumbnails get dumped to
`<out>.frames/` for visual QA.

**Camera defaults to `--focus face`** since this is a face-only
preview; pass `--focus none` (or `--focus head/torso/...`) to override.
Any of the env-knob flags from `render_clip.sh` apply
(`--camera-pos`, `--camera-look`, `--bg`, `--ambient`, `--fit`, etc.).

#### Limits of the face-only preview

- **No audio, no TTS, no body motion.** The avatar holds its bind /
  T-pose; the only thing animating is the face. The full demo (with
  Hermes-3 speech, Audio2Face lip-sync, kimodo body motion) needs a
  blendshape-merger node so Audio2Face owns the lip/jaw axes while
  the affect chain owns brow/cheek/eye. That's the v3 follow-up;
  this preview shows the affect-only contribution in isolation.
- **The renderer's own latest-wins watch** is the only synchronisation —
  fast renders / dropped frames may slightly stretch the perceived
  duration. Use `--fast` (sets `CC_RENDER_FAST=1`) for a
  pts-lockstep pass when timing matters.

#### How it composes with the live runtime

The `.affect_face.jsonl` baker produces the **same wire format**
the live `AffectSimulatorNode` emits on every tick. So the offline
preview and the live demo are running the same mapping over the
same simulator state — what you see in `wa_face.mp4` is what the
live avatar would do at the peak frame of `warm_admiration`, just
without audio and body motion. Once the merger node lands, the
live demo will produce the same face on top of TTS lip-sync.

## How it computes "how to say it"

For each turn the coach runs the simulator through the chosen scenario
to a peak-affect frame, then maps that frame's steering target +
regulation policy into three plain-English axes:

- **Tone** — from Channel D V (warmth-derived). Buckets: warm,
  friendly, neutral, measured, cool.
- **Posture** — from Channel D D (assertiveness, attenuated by
  safety_dampening). Buckets: direct, confident, balanced, yielding,
  deferential.
- **Energy** — from the regulator's `expressiveness` knob. We don't
  read Channel D arousal here because `channel_d.rs` deliberately
  holds A at 0 (CREMA-D's +arousal direction reads as terse-emphatic
  in the wrong way for warm conversational responding).

Plus contextual qualifiers when the regulator pushed the response
away from the default blunt-friend register (high `safety_dampening`
→ "soften the edges", high `social_dampening` → "reserved", high
`task_focus` → "stay on-topic", warmth + assertiveness both high
→ "warm + confident at once").

## Where it overlaps with the eval

Same simulator, same peak-frame selectors per scenario, same Channel
D alpha (1.0), same Channel B augmentation, same system prompt, same
greedy-decode (temp=0.0). The differences vs
[`09_paired_demo.py`](../affect_calibration/scripts/09_paired_demo.py):

- **No A/B split** — coach always runs the affect-aware path. There's
  no vanilla baseline; this tool exists to use the agent, not to
  evaluate it.
- **No conversation history** — the eval injects a 5–7-turn history
  per scenario to give the persona referents. The coach runs each
  user line cold against the system prompt; you get back what the
  agent would say without prior context. Add history support if your
  use case demands it.
- **No stored verdicts / CSV** — REPL output goes to stdout only.

## Acceptance

A 10-minute REPL session over the six scenarios should produce
register-appropriate responses (warm on praise, concerned on distress,
honest on roast invites) with stage directions that match the
simulator's expressed-state regulation. The tool surfaces the same
information a reviewer in the paired-eval CSV has access to, just
addressed to the user as advice rather than as evaluation prompts.
