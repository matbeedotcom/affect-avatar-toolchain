#!/usr/bin/env python3
"""Validate extracted directions via the LLM-judge protocol.

Per IMPLEMENTATION_PLAN §4.6 + Spike F memo. For each VAD axis:

    Run target LLM with vs without steering on a fixed prompt set.
    Judge LLM (different model than target) scores each response on the
    positive pole and the negative pole, both 0-100. Net = pos - neg.

    gap_pos = mean(net_steered@α=+1) - mean(net_unsteered)
    gap_neg = mean(net_steered@α=-1) - mean(net_unsteered)

    PASS iff gap_pos >= +15 AND gap_neg <= -15 AND sign(gap) matches sign(α)

A failure is a hard stop: the calibration is not useful and Phase 5 must
revisit (likely a different layer, or a rewrite of trait descriptions).

Usage:
  ./04_validate_pipeline.py \\
      --directions artifacts/llm_directions/qwen3-27b_layer18.npz \\
      --trait-descriptions data/trait_descriptions.json \\
      --model "$LLAMA_MODEL_PATH" \\
      --judge-model "$JUDGE_MODEL_ID" \\
      --probes-per-axis 50 \\
      --report artifacts/calibration_report.md

  ./04_validate_pipeline.py --dry-run \\
      --directions /tmp/dryrun_directions.npz \\
      --trait-descriptions data/trait_descriptions.json \\
      --report /tmp/dryrun_report.md
      # Generates a structurally valid report with deterministic
      # random scores; useful for downstream wiring before D1.0.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.persona_pipeline import (  # noqa: E402
    AXES,
    JUDGE_PASS_THRESHOLD_POINTS,
    Axis,
    SmokeTestResult,
    judge_steered_responses,
    load_trait_descriptions,
)


@dataclass
class AxisValidation:
    axis: Axis
    n_probes: int
    net_unsteered: float
    net_steered_pos: float
    net_steered_neg: float

    @property
    def gap_pos(self) -> float:
        return self.net_steered_pos - self.net_unsteered

    @property
    def gap_neg(self) -> float:
        return self.net_steered_neg - self.net_unsteered

    @property
    def passed(self) -> bool:
        return (
            self.gap_pos >= JUDGE_PASS_THRESHOLD_POINTS
            and self.gap_neg <= -JUDGE_PASS_THRESHOLD_POINTS
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--directions", type=Path, required=True, help="Path to .npz from 03_extract_llm_directions.py.")
    parser.add_argument("--trait-descriptions", type=Path, default=Path("data/trait_descriptions.json"))
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/LFM2.5-Audio-1.5B-4bit",
        help="HF repo id (or local path) for the MLX-loaded target LLM. "
             "Default per Spike G; ignored under --dry-run.",
    )
    parser.add_argument(
        "--probes",
        type=Path,
        default=Path("data/probe_prompts.json"),
        help="Probe-prompts JSON. Schema: {system_prompt, questions: [...]}.",
    )
    parser.add_argument(
        "--probes-per-axis",
        type=int,
        default=50,
        help="Cap on probes (slice from --probes file). Same set used per axis in v1.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=("anthropic", "openai", "mock"),
        default="anthropic",
        help="Judge LLM provider. `openai` uses any OpenAI-compatible /v1 "
             "endpoint via OPENAI_BASE_URL + OPENAI_API_KEY (good for local "
             "llama-cpp / vLLM / llama-swap servers — but the judge model "
             "MUST differ from the target per Spike F). `mock` skips the API.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model id (Anthropic). Defaults to AnthropicJudge.DEFAULT_MODEL.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering coefficient at α=±this for the gap test.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Cap per-probe response length.")
    parser.add_argument("--report", type=Path, required=True, help="Output calibration_report.md path.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _dry_run_axis(axis: Axis, n_probes: int, rng: np.random.Generator) -> AxisValidation:
    """Synthesize plausible-shape per-axis numbers for --dry-run.

    Skewed so that the positive gap exceeds the threshold and the negative
    gap clears it in the opposite direction; reflects what a successful
    real run would look like, modulo noise.
    """
    base = float(rng.uniform(-3.0, 3.0))  # near-zero baseline net score
    pos_offset = float(rng.uniform(18.0, 30.0))
    neg_offset = float(rng.uniform(-30.0, -18.0))
    return AxisValidation(
        axis=axis,
        n_probes=n_probes,
        net_unsteered=base,
        net_steered_pos=base + pos_offset,
        net_steered_neg=base + neg_offset,
    )


def _write_report(path: Path, validations: list[AxisValidation], dry_run: bool) -> None:
    lines: list[str] = []
    lines.append("# Calibration validation report\n")
    if dry_run:
        lines.append("> **DRY-RUN report** — random scores, NOT a real validation. "
                     "Re-run without `--dry-run` once D1.0 wires the target LLM and judge LLM.\n")
    overall_pass = all(v.passed for v in validations)
    status = "PASS" if overall_pass else "FAIL"
    lines.append(f"\n**Overall**: {status} (threshold: |gap| >= {JUDGE_PASS_THRESHOLD_POINTS:.0f} per axis, signs must oppose)\n")
    lines.append("\n## Per-axis results\n")
    lines.append("\n| Axis | n_probes | net_unsteered | net_steered@α=+1 | net_steered@α=-1 | gap_pos | gap_neg | result |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|\n")
    for v in validations:
        result = "✅ PASS" if v.passed else "❌ FAIL"
        lines.append(
            f"| {v.axis.value} | {v.n_probes} | "
            f"{v.net_unsteered:+.1f} | {v.net_steered_pos:+.1f} | {v.net_steered_neg:+.1f} | "
            f"{v.gap_pos:+.1f} | {v.gap_neg:+.1f} | {result} |\n"
        )
    lines.append("\n## Notes\n\n")
    lines.append("- Net score = positive_pole_judge - negative_pole_judge, each on a 0–100 scale.\n")
    lines.append("- Pass criteria from PROJECT_PLAN.md §10 R9 / IMPLEMENTATION_PLAN.md §4.6.\n")
    lines.append("- Judge model must differ from target model (cross-model judging).\n")
    if not overall_pass and not dry_run:
        lines.append("\n**Failure response**: revisit Phase 5 — likely candidates are layer choice "
                     "(re-run 03 across a different layer sweep) and trait description prose "
                     "(rewrite `*_pole` fields, regenerate contrast_pairs, re-extract).\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    if not args.directions.exists():
        print(f"error: directions not found at {args.directions}", file=sys.stderr)
        return 2

    descriptions = load_trait_descriptions(args.trait_descriptions)
    npz = np.load(args.directions, allow_pickle=True)
    D = np.asarray(npz["D"])
    if D.shape[0] != len(AXES):
        print(f"error: directions D has {D.shape[0]} rows; expected {len(AXES)}", file=sys.stderr)
        return 2
    layer = int(npz["layer"])
    print(f"loaded D shape={D.shape} layer={layer} pooling={str(npz['pooling'])}", file=sys.stderr)

    if args.dry_run:
        rng = np.random.default_rng(args.seed)
        validations = [
            _dry_run_axis(axis, args.probes_per_axis, rng) for axis in AXES
        ]
    else:
        target = _build_target(args.model)
        if target.n_embd != D.shape[1]:
            print(
                f"error: target n_embd={target.n_embd} != directions d_llm={D.shape[1]}; "
                f"the .npz was extracted from a different model",
                file=sys.stderr,
            )
            return 2
        judge = _build_judge(args.judge_backend, args.judge_model)
        probes_system, probes = _load_probes(args.probes, args.probes_per_axis)
        print(
            f"loaded {len(probes)} probes  target={target.model_id}  "
            f"judge_backend={args.judge_backend}",
            file=sys.stderr,
        )

        validations = []
        for axis_idx, axis in enumerate(AXES):
            print(f"  validating axis={axis.value}  n_probes={len(probes)}", file=sys.stderr)
            print(f"  [gen unsteered] {axis.value}", file=sys.stderr, flush=True)
            r_unsteered = _run_target(target, probes_system, probes, layer, None, 0.0,
                                      args.max_new_tokens)
            print(f"  [gen +α] {axis.value}", file=sys.stderr, flush=True)
            r_pos = _run_target(target, probes_system, probes, layer, D[axis_idx],
                                +args.alpha, args.max_new_tokens)
            print(f"  [gen -α] {axis.value}", file=sys.stderr, flush=True)
            r_neg = _run_target(target, probes_system, probes, layer, D[axis_idx],
                                -args.alpha, args.max_new_tokens)
            print(f"  [judge {axis.value}] {len(probes)} probes × 3 conditions × 2 poles "
                  f"= {len(probes) * 6} judge calls", file=sys.stderr, flush=True)
            scores = judge_steered_responses(
                responses_unsteered=r_unsteered,
                responses_steered_pos=r_pos,
                responses_steered_neg=r_neg,
                axis=axis,
                trait_description=descriptions[axis],
                judge_llm=judge,
            )
            validations.append(AxisValidation(
                axis=axis,
                n_probes=len(probes),
                net_unsteered=scores["net_unsteered"],
                net_steered_pos=scores["net_steered_pos"],
                net_steered_neg=scores["net_steered_neg"],
            ))

    for v in validations:
        marker = "✅" if v.passed else "❌"
        print(
            f"  {marker} {v.axis.value:9s}  net_unst={v.net_unsteered:+6.1f}  "
            f"+α={v.net_steered_pos:+6.1f}  -α={v.net_steered_neg:+6.1f}  "
            f"gap_pos={v.gap_pos:+6.1f}  gap_neg={v.gap_neg:+6.1f}",
            file=sys.stderr,
        )

    _write_report(args.report, validations, dry_run=args.dry_run)
    print(f"wrote report -> {args.report}", file=sys.stderr)

    overall_pass = all(v.passed for v in validations)
    return 0 if overall_pass else 1


def _build_target(model_id: str):
    """Lazy import — keeps --dry-run free of mlx + mlx-audio + ~2 GB weights."""
    from lib.mlx_target import MLXTargetLLM  # noqa: E402
    return MLXTargetLLM(model_id)


def _build_judge(backend: str, model_id):
    """Lazy import — keeps --dry-run free of the anthropic SDK dependency."""
    from lib.judge import build_judge  # noqa: E402
    return build_judge(backend, model_id)


def _load_probes(path: Path, cap: int) -> tuple[str, list[str]]:
    """Load probes. Returns (system_prompt, list_of_questions[:cap])."""
    import json
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    system = str(data.get("system_prompt", ""))
    questions = list(data.get("questions", []))
    if not questions:
        raise ValueError(f"no probe questions in {path}")
    return system, questions[:cap]


def _run_target(
    target,
    system_prompt: str,
    probes: list[str],
    layer: int,
    direction,
    alpha: float,
    max_new_tokens: int,
) -> list[str]:
    """Run target LLM on each probe; return responses in order. Skips empties."""
    responses: list[str] = []
    for i, q in enumerate(probes):
        text = target.generate_text(
            system_prompt=system_prompt,
            question=q,
            max_new_tokens=max_new_tokens,
            layer=layer if alpha != 0.0 else None,
            direction=direction if alpha != 0.0 else None,
            alpha=alpha,
        )
        responses.append(text)
        if (i + 1) % 10 == 0:
            print(f"    target {i + 1}/{len(probes)}  α={alpha:+.2f}", file=sys.stderr)
    return responses


if __name__ == "__main__":
    sys.exit(main())
