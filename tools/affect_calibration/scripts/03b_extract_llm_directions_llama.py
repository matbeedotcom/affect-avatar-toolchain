#!/usr/bin/env python3
"""Multi-layer extraction of V/A/D steering directions for Llama-family chat LLMs.

Runs every contrast pair once, capturing pooled hidden states at every
layer in a single forward. Computes per-axis directions per layer; saves
both the full 3D direction tensor and a single best-layer NPZ matching
the runtime-loadable format.

Usage:
    ./03b_extract_llm_directions_llama.py \\
        --contrast-pairs data/contrast_pairs.jsonl \\
        --model mlx-community/Hermes-3-Llama-3.1-8B-4bit \\
        --output-dir artifacts/llm_directions/hermes-3-8b/

    # Pick layer with best signal-to-noise, save runtime NPZ:
    ./03b_extract_llm_directions_llama.py --pick-layer ...
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast-pairs", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True,
                    help="HF repo id loadable via mlx_lm.load")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-response-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N pairs (smoke test)")
    ap.add_argument("--harvest-only", action="store_true",
                    help="Save raw activations; skip direction computation")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip harvest; recompute directions from saved NPZ")
    return ap.parse_args()


def load_pairs(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def harvest(model_id: str, pairs: List[Dict[str, Any]],
            max_tokens: int, out_path: Path) -> Dict[str, Any]:
    """Run every pair through the LLM, capture per-layer pooled activations."""
    from lib.mlx_llama_target import MLXLlamaTargetLLM

    log.info("loading %s ...", model_id)
    target = MLXLlamaTargetLLM(model_id, max_response_tokens=max_tokens)
    L = target.num_hidden_layers
    D = target.n_embd
    log.info("model loaded: %d layers, hidden=%d", L, D)

    # Per-pair pooled (L, D) — stacked into (N, L, D).
    activations = np.zeros((len(pairs), L, D), dtype=np.float32)
    axes = np.array([p["axis"] for p in pairs], dtype=object)
    poles = np.array([p["pole"] for p in pairs], dtype=object)

    t0 = time.time()
    skipped = 0
    for i, pair in enumerate(pairs):
        try:
            acts = target.harvest_all_layers(pair["system_prompt"], pair["question"])
            activations[i] = acts
        except RuntimeError as e:
            # Some prompts elicit too-short responses; skip them.
            log.warning("pair %d skipped: %s", i, e)
            activations[i, :, :] = np.nan
            skipped += 1
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed * (len(pairs) - i - 1) / (i + 1)
            log.info("  %d/%d pairs (skipped=%d); elapsed %.0fs eta %.0fs",
                     i + 1, len(pairs), skipped, elapsed, eta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        activations=activations,
        axes=axes,
        poles=poles,
        num_layers=L,
        hidden_size=D,
        model_id=model_id,
    )
    log.info("saved harvest to %s (%d skipped)", out_path, skipped)
    return {"L": L, "D": D, "skipped": skipped}


def compute_directions(harvest_path: Path, out_dir: Path) -> None:
    """Compute per-axis mean-difference directions per layer; print diagnostics."""
    data = np.load(harvest_path, allow_pickle=True)
    activations = data["activations"]  # (N, L, D)
    axes_col = np.asarray(data["axes"])
    poles_col = np.asarray(data["poles"])
    L = int(data["num_layers"])
    D = int(data["hidden_size"])

    # Drop NaN rows (skipped pairs).
    valid = ~np.isnan(activations[:, 0, 0])
    activations = activations[valid]
    axes_col = axes_col[valid]
    poles_col = poles_col[valid]
    log.info("loaded %d valid pairs, %d layers, hidden=%d",
             activations.shape[0], L, D)

    AXES = ("valence", "arousal", "dominance")

    # Per-layer per-axis direction tensor.
    directions = np.zeros((L, len(AXES), D), dtype=np.float32)
    norms = np.zeros((L, len(AXES)), dtype=np.float32)
    snr = np.zeros((L, len(AXES)), dtype=np.float32)

    for ai, axis in enumerate(AXES):
        mask = axes_col == axis
        pos_mask = mask & (poles_col == "positive")
        neg_mask = mask & (poles_col == "negative")
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            log.warning("axis %s missing pos or neg pairs", axis)
            continue
        pos = activations[pos_mask]  # (n_pos, L, D)
        neg = activations[neg_mask]  # (n_neg, L, D)
        for li in range(L):
            mean_pos = pos[:, li, :].mean(axis=0)
            mean_neg = neg[:, li, :].mean(axis=0)
            d = mean_pos - mean_neg
            directions[li, ai] = d
            norms[li, ai] = float(np.linalg.norm(d))
            # SNR: ||mean_diff|| / mean(within-class std)
            within = 0.5 * (pos[:, li, :].std(axis=0).mean()
                            + neg[:, li, :].std(axis=0).mean())
            snr[li, ai] = norms[li, ai] / max(within, 1e-8)

    log.info("=" * 78)
    log.info("PER-LAYER DIAGNOSTICS")
    log.info("%-7s %-12s %-12s %-12s %-12s",
             "layer", "v_norm/snr", "a_norm/snr", "d_norm/snr", "snr_mean")
    log.info("=" * 78)
    for li in range(L):
        snr_mean = float(snr[li].mean())
        log.info(
            "%-7d %-6.2f/%-5.2f %-6.2f/%-5.2f %-6.2f/%-5.2f %.2f",
            li,
            norms[li, 0], snr[li, 0],
            norms[li, 1], snr[li, 1],
            norms[li, 2], snr[li, 2],
            snr_mean,
        )

    best_layer = int(np.argmax(snr.mean(axis=1)))
    log.info("=" * 78)
    log.info("BEST LAYER (by mean SNR across axes): %d", best_layer)
    log.info("  snr_mean=%.3f, norms=%s", snr[best_layer].mean(), norms[best_layer].tolist())

    # Save full per-layer tensor for offline analysis.
    full_path = out_dir / "directions_all_layers.npz"
    np.savez(
        full_path,
        directions=directions,
        norms=norms,
        snr=snr,
        axes=np.array(AXES, dtype=object),
        n_embd=np.int32(D),
        num_layers=np.int32(L),
    )
    log.info("wrote %s", full_path)

    # Save runtime-loadable NPZ at best layer in the existing format.
    # Match the schema produced by 03_extract_llm_directions.py:
    #   D shape (3, n_embd), n_embd, layer, axes, pooling
    runtime_path = out_dir / f"layer{best_layer}.npz"
    np.savez(
        runtime_path,
        D=directions[best_layer].astype(np.float32),
        n_embd=np.int32(D),
        layer=np.int32(best_layer),
        axes=np.array(AXES, dtype=object),
        pooling=np.array("response_mean", dtype=object),
    )
    log.info("wrote runtime-loadable NPZ to %s (layer=%d, D shape=%s)",
             runtime_path, best_layer, directions[best_layer].shape)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    harvest_path = args.output_dir / "raw_activations.npz"

    if not args.analyze_only:
        if not args.contrast_pairs.exists():
            log.error("missing %s", args.contrast_pairs)
            return 2
        pairs = load_pairs(args.contrast_pairs)
        if args.limit:
            pairs = pairs[: args.limit]
        log.info("processing %d contrast pairs", len(pairs))
        harvest(args.model, pairs, args.max_response_tokens, harvest_path)

    if args.harvest_only:
        return 0

    if not harvest_path.exists():
        log.error("missing harvest at %s — run without --analyze-only first",
                  harvest_path)
        return 2

    compute_directions(harvest_path, args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
