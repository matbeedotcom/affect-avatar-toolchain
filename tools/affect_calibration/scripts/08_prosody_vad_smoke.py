#!/usr/bin/env python3
"""Phase 3 prosody-VAD smoke test.

End-to-end check that the production Whisper-encoder → ridge → events
chain produces sensible signals on known-emotion clips. This script:

  1. Picks one CREMA-D clip per label (HAP, ANG, NEU, SAD), all from
     the same speaker so any cross-clip variance comes from prosody,
     not voice characteristics.
  2. Runs each through `remotemedia.nodes.affect.ProsodyVad` (Whisper
     encoder + ridge ONNX).
  3. Prints predicted V/A/D and the events emitted per clip.
  4. Verifies coarse alignment with labels:
       - HAP → positive valence + high arousal
       - ANG → negative valence + high arousal (fires both events)
       - SAD → negative valence + low arousal
       - NEU → low |VAD|, fires uncertain only
     Sanity check, not a quantitative gate (Phase 5 §4 already validated
     ridge metrics on the held-out test set).

Pass: at least 3 of 4 expected event-emission patterns hold. The
degraded path (regressor missing) returns a single `prosody_uncertain`
event per clip and the test reports that explicitly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "clients/python"))

from remotemedia.nodes.affect import ProsodyVad, ProsodyVadOptions  # noqa: E402

DEFAULT_AUDIO_DIR = REPO_ROOT / "tools/affect_calibration/data/crema_d/data/AudioWAV"
SPEAKER_ID = 1001  # Same speaker for all four labels — controls voice timbre.

# Each label's expectation expressed as a predicate over the predicted
# V/A/D and the events that fire. Tolerant: "fires HIGH arousal" means
# arousal > 0.4 (above threshold default 0.55 fires the event; 0.4 is
# the loose check).
LABEL_EXPECTATIONS = {
    "HAP": {
        "valence_min": 0.10,
        "arousal_min": 0.40,
        "events_must_include": [],  # HAP often reads as moderate-arousal
    },
    "ANG": {
        "valence_max": -0.05,
        "arousal_min": 0.40,
        "events_must_include": ["prosody_arousal_high", "prosody_valence_negative"],
    },
    "SAD": {
        "valence_max": -0.05,
        "arousal_max": 0.50,
        "events_must_include": ["prosody_valence_negative"],
    },
    "NEU": {
        "valence_min": -0.20,
        "valence_max": 0.30,
        "arousal_max": 0.50,
        "events_must_include": [],
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("phase3_smoke")


def _load_clip_16k(path: Path) -> np.ndarray:
    """Load a CREMA-D clip and ensure 16 kHz mono float32 (Whisper native)."""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32, copy=False)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32, copy=False)
    return audio


def _check(label: str, vad: np.ndarray, events: list) -> tuple[bool, list[str]]:
    """Verify the predicted V/A/D + events match the label's expectations."""
    exp = LABEL_EXPECTATIONS[label]
    v, a, d = float(vad[0]), float(vad[1]), float(vad[2])
    failures: list[str] = []

    if "valence_min" in exp and v < exp["valence_min"]:
        failures.append(f"valence={v:+.2f} < expected {exp['valence_min']:+.2f}")
    if "valence_max" in exp and v > exp["valence_max"]:
        failures.append(f"valence={v:+.2f} > expected {exp['valence_max']:+.2f}")
    if "arousal_min" in exp and a < exp["arousal_min"]:
        failures.append(f"arousal={a:.2f} < expected {exp['arousal_min']:.2f}")
    if "arousal_max" in exp and a > exp["arousal_max"]:
        failures.append(f"arousal={a:.2f} > expected {exp['arousal_max']:.2f}")

    fired = {e.kind for e in events}
    for required in exp.get("events_must_include", []):
        if required not in fired:
            failures.append(f"missing event {required!r}; got {sorted(fired)}")

    return len(failures) == 0, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR,
        help="CREMA-D AudioWAV directory.",
    )
    parser.add_argument(
        "--regressor",
        type=Path,
        default=Path(ProsodyVadOptions.regressor_path),
        help="Path to the whisper_to_vad ridge ONNX.",
    )
    parser.add_argument(
        "--sentence", default="DFA",
        help="CREMA-D sentence code; same sentence across labels controls content.",
    )
    parser.add_argument(
        "--speaker", type=int, default=SPEAKER_ID,
        help="CREMA-D speaker ID; same speaker across labels controls voice timbre.",
    )
    args = parser.parse_args()

    pv = ProsodyVad(ProsodyVadOptions(regressor_path=str(args.regressor)))

    if not pv._regressor_available:
        print(
            "REGRESSOR UNAVAILABLE — pipeline degraded to uncertain-only events. "
            "This is the Phase 3 graceful-degradation path; smoke test cannot "
            "validate label alignment without the trained ridge."
        )
        return 2

    expected_events_by_label: Dict[str, list] = {}
    pass_count = 0

    for label in ("HAP", "ANG", "SAD", "NEU"):
        clip = args.audio_dir / f"{args.speaker:04d}_{args.sentence}_{label}_XX.wav"
        if not clip.exists():
            print(f"\n[{label}] missing clip: {clip} — skipping")
            continue
        audio = _load_clip_16k(clip)

        vad = pv.predict_vad(audio)
        events = pv.process(audio, timestamp_ms=0)
        expected_events_by_label[label] = events

        v, a, d = float(vad[0]), float(vad[1]), float(vad[2])
        kinds = ", ".join(sorted({e.kind for e in events})) or "(none)"
        ok, failures = _check(label, vad, events)
        verdict = "OK" if ok else "MISMATCH"
        print(
            f"\n[{label}] {clip.name}\n"
            f"  V/A/D = ({v:+.2f}, {a:.2f}, {d:+.2f})  events: {kinds}\n"
            f"  → {verdict}"
        )
        if failures:
            for f in failures:
                print(f"     - {f}")
        else:
            pass_count += 1

    n_labels = len([l for l in ("HAP", "ANG", "SAD", "NEU")
                    if (args.audio_dir / f"{args.speaker:04d}_{args.sentence}_{l}_XX.wav").exists()])
    print(
        f"\n{pass_count}/{n_labels} labels pass alignment check "
        f"(threshold for PASS: ≥ {max(1, n_labels - 1)})."
    )

    # Bonus: dump the events as JSON in PerceivedEvent shape so a follow-on
    # script could feed them straight into the simulator.
    serialized = {
        label: [e.to_perceived_event_dict(f"{label}_{i}") for i, e in enumerate(events)]
        for label, events in expected_events_by_label.items()
    }
    print("\n--- Events as PerceivedEvent JSON ---")
    print(json.dumps(serialized, indent=2)[:1500] + "\n...")

    if pass_count >= max(1, n_labels - 1):
        print("\nPhase 3 smoke: PASS")
        return 0
    print("\nPhase 3 smoke: FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
