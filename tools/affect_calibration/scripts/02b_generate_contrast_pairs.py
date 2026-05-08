#!/usr/bin/env python3
"""Generate contrast_pairs.jsonl from trait_descriptions.json.

Per the persona-vectors pipeline (Chen et al. 2025, arXiv:2507.21509;
adoption recorded in
docs/references/activation-steering-audio-llm/notes/spike-f-persona-vectors-pipeline.md).

Three axes (V/A/D) -> 600 rows total:
  - 5 positive system prompts + 5 negative system prompts per axis
  - x 40 evaluation questions per axis
  - x 3 axes
  = 600 (axis, pole, system_prompt, question) rows.

Usage:
  ./02b_generate_contrast_pairs.py \\
      --trait-descriptions data/trait_descriptions.json \\
      --elicitation-model claude-3-5-sonnet@20250620 \\
      --output data/contrast_pairs.jsonl

  ./02b_generate_contrast_pairs.py --dry-run --output /tmp/pairs.jsonl
      # Emits structurally valid placeholder rows; useful for wiring the
      # downstream pipeline before D1.0 lands the elicitation LLM client.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make scripts/lib importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.persona_pipeline import (  # noqa: E402
    AXES,
    N_NEG_PROMPTS_PER_AXIS,
    N_POS_PROMPTS_PER_AXIS,
    N_QUESTIONS_PER_AXIS,
    Axis,
    ContrastPair,
    Pole,
    TraitDescription,
    dump_contrast_pairs,
    elicit_contrast_prompts_for_axis,
    load_trait_descriptions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trait-descriptions",
        type=Path,
        default=Path("data/trait_descriptions.json"),
        help="Path to trait_descriptions.json (3 axes: valence/arousal/dominance).",
    )
    parser.add_argument(
        "--elicitation-backend",
        choices=("anthropic", "openai", "mock"),
        default="anthropic",
        help="Elicitation LLM provider. `openai` uses any OpenAI-compatible "
             "/v1 endpoint via OPENAI_BASE_URL + OPENAI_API_KEY (good for "
             "local llama-cpp / vLLM / llama-swap servers). `mock` skips "
             "the API for wiring tests.",
    )
    parser.add_argument(
        "--elicitation-model",
        type=str,
        default=None,
        help="Override elicitation model id. Defaults to AnthropicElicitor.DEFAULT_MODEL.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for contrast_pairs.jsonl.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed passed to the elicitation LLM for reproducibility (where the provider supports it).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Bypass the elicitation LLM and emit placeholder rows. Useful for "
             "wiring 03_extract_llm_directions.py before D1.0 lands the LLM client.",
    )
    parser.add_argument(
        "--n-questions-per-axis",
        type=int,
        default=N_QUESTIONS_PER_AXIS,
        help=f"Override the per-axis question count (paper default: {N_QUESTIONS_PER_AXIS}).",
    )
    return parser.parse_args()


def _placeholder_pairs(trait: TraitDescription, n_questions: int) -> list[ContrastPair]:
    """Synthesize structurally valid (but content-meaningless) rows for --dry-run."""
    rows: list[ContrastPair] = []
    for pole, n_prompts, label in (
        (Pole.POSITIVE, N_POS_PROMPTS_PER_AXIS, "positive"),
        (Pole.NEGATIVE, N_NEG_PROMPTS_PER_AXIS, "negative"),
    ):
        for sp_idx in range(n_prompts):
            sp = f"[DRY-RUN] {trait.axis.value} {label} system prompt #{sp_idx + 1}"
            for q_idx in range(n_questions):
                rows.append(
                    ContrastPair(
                        axis=trait.axis,
                        pole=pole,
                        system_prompt=sp,
                        question=f"[DRY-RUN] {trait.axis.value} eval question #{q_idx + 1}",
                    )
                )
    return rows


def main() -> int:
    args = parse_args()

    if not args.trait_descriptions.exists():
        print(f"error: trait descriptions not found at {args.trait_descriptions}", file=sys.stderr)
        return 2

    descriptions = load_trait_descriptions(args.trait_descriptions)
    print(f"loaded {len(descriptions)} axis descriptions: {[a.value for a in descriptions]}", file=sys.stderr)

    all_pairs: list[ContrastPair] = []
    elicitation_llm = (
        None if args.dry_run
        else _build_elicitation_llm(args.elicitation_backend, args.elicitation_model)
    )
    for axis in AXES:
        trait = descriptions[axis]
        if args.dry_run:
            rows = _placeholder_pairs(trait, args.n_questions_per_axis)
        else:
            rows = list(
                elicit_contrast_prompts_for_axis(
                    trait,
                    elicitation_llm,
                    n_pos=N_POS_PROMPTS_PER_AXIS,
                    n_neg=N_NEG_PROMPTS_PER_AXIS,
                    n_questions=args.n_questions_per_axis,
                )
            )

        n_pos = sum(1 for r in rows if r.pole == Pole.POSITIVE)
        n_neg = sum(1 for r in rows if r.pole == Pole.NEGATIVE)
        print(f"  {axis.value:9s}  positive={n_pos:4d}  negative={n_neg:4d}", file=sys.stderr)
        all_pairs.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = dump_contrast_pairs(all_pairs, args.output)
    print(f"wrote {n_written} rows -> {args.output}", file=sys.stderr)
    return 0


def _build_elicitation_llm(backend: str, model_id):
    """Lazy import — keeps --dry-run free of the anthropic SDK."""
    from lib.elicitation import build_elicitor  # noqa: E402
    return build_elicitor(backend, model_id)


if __name__ == "__main__":
    sys.exit(main())
