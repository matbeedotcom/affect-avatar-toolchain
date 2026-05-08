#!/usr/bin/env python3
"""Bake a per-tick ARKit-52 blendshape JSONL from an affect scenario.

Companion to [`coach.py`](coach.py): runs the same affect simulator
the coach uses (via the workspace's `affect-simulator` Rust binary),
maps every emitted frame to the same ARKit-52 weights the live
`AffectSimulatorNode` produces (Python mirror of
`crates/core/src/nodes/affect_expression.rs`), and dumps one
canonical blendshape envelope per line:

```json
{"kind":"blendshapes","arkit_52":[…52 floats…],"pts_ms":0}
{"kind":"blendshapes","arkit_52":[…],"pts_ms":200}
…
```

This is the same wire shape every renderer in the workspace already
consumes — `Audio2FaceLipSyncNode`, `SyntheticLipSyncNode`,
`MotionPlayerNode` (poses), and the running
`Live2DRenderNode` / `CcRenderNode`. The JSONL is **the** thing to
hand to a renderer downstream; pipe it into a `BlendshapePlayer`-style
node when one exists, or render it via the future
`render_affect_face.sh` wrapper.

## Why an offline baker, not a live render

The simulator runs at 5 Hz, so a 30 s scenario produces ~150 frames —
small enough to bake to disk in a fraction of a second, big enough
that re-running the cargo `affect-simulator` binary on every render
iteration would be wasted. Baking once, rendering many times also
makes it trivial to inspect: `head -5 *.affect_face.jsonl` shows
exactly what the avatar would do, and `--diagnose` prints the
trajectory of top-active blendshapes per frame so you can sanity-
check the mapping before paying for an actual render.

## Usage

```bash
# Bake one of the simulator scenarios:
./tools/affect_coach/bake_affect_face.py --scenario warm_admiration \
    --out out/warm_admiration.affect_face.jsonl

# Free-form input (uses coach.py's keyword classifier):
./tools/affect_coach/bake_affect_face.py \
    --coach-input "I just got fired" \
    --out out/honest_concern.affect_face.jsonl

# Inspect what the face would look like over time, no render:
./tools/affect_coach/bake_affect_face.py --scenario roast_invitation --diagnose

# Override the renderer's frame rate for the JSONL (default 30 fps;
# we resample the 5 Hz simulator trace by holding the most recent
# frame, matching `MotionPlayerNode`'s `pace_realtime` semantics).
./tools/affect_coach/bake_affect_face.py --scenario warm_admiration --fps 60 \
    --out out/wa.affect_face.jsonl
```

## Resampling

The simulator emits at its own `tick_ms` (200 ms by default). Avatar
renderers typically run at 30 fps. We hold-and-emit so a 5 Hz tick
becomes 6 identical 30 Hz frames — same behaviour the runtime would
exhibit if you wired `affect_sim → BlendshapePlayer → CcRender` over
the live tick stream. `--fps` controls the output rate.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "tools/affect_simulator/scenarios"

# Reuse coach.py's mapping mirror (kept in sync with the Rust impl
# via the index-table tests on the Rust side). Keeping a single
# source of truth in the coach module avoids the inevitable drift
# between two near-identical hand-tuned tables.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coach import (  # noqa: E402
    SCENARIOS,
    classify_scenario,
    compute_blendshapes,
    ARKIT_NAMES,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("bake_affect_face")


def ensure_trace(simulator_scenario: str, *, force: bool = False) -> Path:
    """Build (or reuse) the trace JSON next to the scenario file.

    Mirrors `coach.ensure_trace`; duplicated here to keep the baker
    runnable even if `coach.py` evolves to load the LLM eagerly.
    Cache invalidates when either the scenario JSON or any simulator
    source file is newer than the trace.
    """
    scenario_path = SCENARIO_DIR / f"{simulator_scenario}.json"
    if not scenario_path.exists():
        raise FileNotFoundError(
            f"simulator scenario not found: {scenario_path}\n"
            f"Available: {sorted(p.stem for p in SCENARIO_DIR.glob('*.json'))}"
        )
    trace_path = scenario_path.with_suffix(".trace.json")
    sim_src_dir = REPO_ROOT / "tools/affect_simulator/src"
    sim_src_mtime = max(
        (p.stat().st_mtime for p in sim_src_dir.rglob("*.rs")),
        default=0.0,
    )
    if trace_path.exists() and not force:
        trace_mtime = trace_path.stat().st_mtime
        if (
            trace_mtime > scenario_path.stat().st_mtime
            and trace_mtime > sim_src_mtime
        ):
            return trace_path
    logger.info("rebuilding trace for %s", scenario_path.name)
    subprocess.run(
        [
            "cargo", "run", "-p", "affect-simulator", "--quiet", "--",
            "run", str(scenario_path), "--out", str(trace_path),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return trace_path


def bake_frames(
    trace: Dict,
    *,
    out_fps: int = 30,
    gain: float = 1.0,
) -> List[Tuple[int, List[float]]]:
    """Produce `(pts_ms, arkit_52)` pairs at the requested output rate.

    Hold-and-resample: the simulator emits at `trace["tick_ms"]`; we
    fill in the gaps by repeating the most recent simulator frame at
    the renderer rate. Same shape `MotionPlayerNode` would yield if
    you piped the simulator's stream through it at `pace_realtime=true`.
    """
    sim_tick_ms = int(trace.get("tick_ms", 200))
    frames = trace.get("frames", [])
    if not frames:
        return []

    # Sort just in case; simulator runner already emits in order.
    frames = sorted(frames, key=lambda f: int(f.get("timestamp_ms", 0)))
    last_ts = int(frames[-1]["timestamp_ms"])
    duration_ms = max(last_ts + sim_tick_ms, 0)

    out_period_ms = max(1, int(round(1000.0 / max(1, out_fps))))
    out: List[Tuple[int, List[float]]] = []
    sim_idx = 0
    t = 0
    while t <= duration_ms:
        # Advance the sim cursor while the next sim frame's timestamp
        # has already passed `t`. This is a step-and-hold sampler.
        while (
            sim_idx + 1 < len(frames)
            and int(frames[sim_idx + 1]["timestamp_ms"]) <= t
        ):
            sim_idx += 1
        weights = compute_blendshapes(frames[sim_idx], gain=gain)
        out.append((t, weights))
        t += out_period_ms
    return out


def write_jsonl(
    path: Path,
    samples: List[Tuple[int, List[float]]],
) -> None:
    """One blendshape envelope per line, canonical `{kind,arkit_52,pts_ms}`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for pts_ms, weights in samples:
            obj = {
                "kind": "blendshapes",
                "arkit_52": [round(w, 4) for w in weights],
                "pts_ms": int(pts_ms),
            }
            f.write(json.dumps(obj))
            f.write("\n")


def diagnose(
    samples: List[Tuple[int, List[float]]],
    *,
    every_ms: int = 200,
    top_k: int = 3,
    threshold: float = 0.05,
) -> None:
    """Print top-active blendshapes at a coarse cadence so a human can
    eyeball the trajectory without opening the JSONL."""
    last_dump = -10**9
    for pts_ms, weights in samples:
        if pts_ms - last_dump < every_ms:
            continue
        last_dump = pts_ms
        active = sorted(
            ((i, w) for i, w in enumerate(weights) if w >= threshold),
            key=lambda iw: iw[1],
            reverse=True,
        )[:top_k]
        if active:
            face = ", ".join(f"{ARKIT_NAMES[i]} {w:.2f}" for i, w in active)
        else:
            face = "(neutral)"
        print(f"  t={pts_ms:>5}ms  {face}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        help=f"Affect scenario name. One of: "
             f"{', '.join(s.name for s in SCENARIOS)}",
    )
    parser.add_argument(
        "--coach-input",
        help="Free-form line; coach.py's keyword classifier picks the "
             "scenario. Mutually exclusive with --scenario.",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Output `.affect_face.jsonl` path. If omitted, writes to "
             "out/<scenario>.affect_face.jsonl in the repo root.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output frame rate for the JSONL (default 30; the simulator "
             "ticks at 5 Hz, hold-and-repeat resamples to this rate).",
    )
    parser.add_argument(
        "--gain", type=float, default=1.0,
        help="Post-expressiveness gain on the affect→ARKit mapping. "
             "1.0 = baseline; raise for a more theatrical face, lower "
             "for muted readings (default 1.0).",
    )
    parser.add_argument(
        "--diagnose", action="store_true",
        help="Print top-active blendshapes every 200 ms instead of "
             "writing a JSONL file. Useful for sanity-checking the "
             "mapping before paying for a render.",
    )
    parser.add_argument(
        "--rebuild-trace", action="store_true",
        help="Force the simulator binary to rebuild the trace JSON "
             "(skip the cache check on the scenario / source mtime).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Bump logging to INFO so simulator-rebuild lines show.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if args.scenario and args.coach_input:
        print("ERROR: --scenario and --coach-input are mutually exclusive",
              file=sys.stderr)
        return 2

    # Resolve the simulator scenario.
    if args.coach_input:
        scenario = classify_scenario(args.coach_input)
        print(f"  classifier picked: {scenario.name} ({scenario.simulator})")
    else:
        scenario_name = args.scenario or "warm_admiration"
        scenario = next(
            (s for s in SCENARIOS if s.name == scenario_name),
            None,
        )
        if scenario is None:
            print(
                f"ERROR: unknown scenario {scenario_name!r}. Available: "
                f"{sorted(s.name for s in SCENARIOS)}",
                file=sys.stderr,
            )
            return 2

    # Build the trace + bake the frames.
    try:
        trace_path = ensure_trace(scenario.simulator, force=args.rebuild_trace)
        trace = json.loads(trace_path.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: simulator trace failed: {exc}", file=sys.stderr)
        return 1

    samples = bake_frames(trace, out_fps=args.fps, gain=args.gain)
    if not samples:
        print("ERROR: trace produced zero frames", file=sys.stderr)
        return 1

    duration_s = samples[-1][0] / 1000.0

    if args.diagnose:
        print(
            f"  scenario:  {scenario.name} ({scenario.simulator})\n"
            f"  duration:  {duration_s:.1f}s @ {args.fps} fps "
            f"({len(samples)} frames)\n"
            f"  trajectory:"
        )
        diagnose(samples)
        return 0

    out_path: Path
    if args.out is None:
        out_path = REPO_ROOT / "out" / f"{scenario.name}.affect_face.jsonl"
    else:
        out_path = args.out
    write_jsonl(out_path, samples)
    print(
        f"✓ wrote {out_path}\n"
        f"  scenario:  {scenario.name} ({scenario.simulator})\n"
        f"  duration:  {duration_s:.1f}s @ {args.fps} fps "
        f"({len(samples)} frames)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
