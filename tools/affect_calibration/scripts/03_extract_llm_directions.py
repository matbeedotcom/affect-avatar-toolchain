#!/usr/bin/env python3
"""Extract per-axis steering directions from contrast_pairs.jsonl.

Per the persona-vectors pipeline (Chen et al. 2025, arXiv:2507.21509;
spike-f memo). For each VAD axis:

    v_axis = mean(positive response activations) - mean(negative response activations)

Activations are pooled over response tokens only (response_mean), at
the configured layer L of the target LLM.

Usage:
  ./03_extract_llm_directions.py \\
      --contrast-pairs data/contrast_pairs.jsonl \\
      --model "$LLAMA_MODEL_PATH" \\
      --layer 18 \\
      --output artifacts/llm_directions/qwen3-27b_layer18.npz

  ./03_extract_llm_directions.py \\
      --contrast-pairs data/contrast_pairs.jsonl \\
      --dry-run --d-llm 4096 --layer 18 \\
      --output /tmp/dryrun_directions.npz
      # Emits a structurally valid npz with random direction vectors;
      # useful for wiring 04_validate_pipeline.py before D1.0 lands the
      # Rust extractor binary.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.persona_pipeline import (  # noqa: E402
    AXES,
    DEFAULT_POOLING,
    Axis,
    Pole,
    extract_axis_direction,
    load_contrast_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--contrast-pairs",
        type=Path,
        default=Path("data/contrast_pairs.jsonl"),
        help="Input contrast_pairs.jsonl produced by 02b_generate_contrast_pairs.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/LFM2.5-Audio-1.5B-4bit",
        help="HF repo id (or local path) for the MLX-loaded target LLM. "
             "Default per Spike G; ignored under --dry-run.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Residual-stream layer index L for direction extraction.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path. Stored array `D` has shape (3, d_llm) with axes ordered (V, A, D).",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default=DEFAULT_POOLING,
        choices=[DEFAULT_POOLING],
        help="Pooling strategy. Pinned to response_mean per Spike F memo.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Bypass the target LLM and emit random direction vectors. Requires --d-llm.",
    )
    parser.add_argument(
        "--d-llm",
        type=int,
        default=None,
        help="Embedding dim of the target LLM (n_embd). Required for --dry-run; ignored otherwise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for --dry-run random vectors (deterministic across runs).",
    )
    return parser.parse_args()


def _dry_run_direction(
    axis: Axis,
    n_pos: int,
    n_neg: int,
    d_llm: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthesize a structurally valid direction vector for --dry-run.

    Uses the same shape and dtype as the real extraction would; values
    are random (Gaussian, unit-variance) so downstream code that
    normalizes by ||D|| sees a reasonable magnitude.
    """
    pos_mean = rng.standard_normal(d_llm, dtype=np.float32)
    neg_mean = rng.standard_normal(d_llm, dtype=np.float32)
    return pos_mean - neg_mean


def main() -> int:
    args = parse_args()

    if not args.contrast_pairs.exists():
        print(f"error: contrast_pairs not found at {args.contrast_pairs}", file=sys.stderr)
        return 2
    if args.dry_run and args.d_llm is None:
        print("error: --d-llm is required for --dry-run", file=sys.stderr)
        return 2

    pairs_by_axis: dict[Axis, list] = defaultdict(list)
    for pair in load_contrast_pairs(args.contrast_pairs):
        pairs_by_axis[pair.axis].append(pair)

    for axis in AXES:
        if axis not in pairs_by_axis:
            print(f"error: contrast_pairs.jsonl has no rows for axis '{axis.value}'", file=sys.stderr)
            return 2

    if args.dry_run:
        d_llm = args.d_llm
        rng = np.random.default_rng(args.seed)
    else:
        target_llm = _build_target_llm(args.model)
        d_llm = target_llm.n_embd  # type: ignore[attr-defined]
        rng = None  # unused

    D = np.zeros((len(AXES), d_llm), dtype=np.float32)
    for axis_idx, axis in enumerate(AXES):
        rows = pairs_by_axis[axis]
        n_pos = sum(1 for r in rows if r.pole == Pole.POSITIVE)
        n_neg = sum(1 for r in rows if r.pole == Pole.NEGATIVE)
        print(f"  {axis.value:9s}  n_pos={n_pos:4d}  n_neg={n_neg:4d}  layer={args.layer}", file=sys.stderr)

        if args.dry_run:
            D[axis_idx] = _dry_run_direction(axis, n_pos, n_neg, d_llm, rng)
        else:
            # Real path; raises NotImplementedError until D1.0 wires the
            # Rust extractor binary as a TargetLLM provider.
            v = extract_axis_direction(rows, target_llm, args.layer, axis, pooling=args.pooling)
            D[axis_idx] = np.asarray(v, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        D=D,
        n_embd=np.int32(d_llm),
        layer=np.int32(args.layer),
        axes=np.array([a.value for a in AXES], dtype=object),
        pooling=np.array(args.pooling, dtype=object),
    )
    print(f"wrote D shape={D.shape} layer={args.layer} -> {args.output}", file=sys.stderr)
    return 0


def _build_target_llm(model_path: str):
    """Construct a TargetLLM provider over the configured MLX model.

    Per Spike G (notes/spike-g-mlx-target-llm.md): we pivoted from
    Qwen3-27B GGUF + llama.cpp to LFM2-Audio-1.5B MLX. `model_path` is
    a HuggingFace repo id (or local snapshot dir), e.g.
    'mlx-community/LFM2.5-Audio-1.5B-4bit'.

    The MLX provider is imported lazily so the dry-run path stays
    free of the mlx + mlx-audio + ~2 GB model dependency.
    """
    from lib.mlx_target import MLXTargetLLM  # noqa: E402  (lazy import)

    return MLXTargetLLM(model_path)


if __name__ == "__main__":
    sys.exit(main())
