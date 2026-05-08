#!/usr/bin/env python3
"""Cache Whisper-large-v3-turbo encoder hidden states for every MEAD_3D clip.

Per [PROJECT_PLAN.md §3 + §4.2]
(../../docs/references/audio-blendshape-diffusion/PROJECT_PLAN.md): Stage-2
DiT cross-attends into Whisper hidden states. Pre-computing the
encodings collapses ~1.7 sec/clip Whisper forward into a one-time pass;
DiT training then iterates over cached `.npz` files.

Output layout:
    artifacts/whisper_cache/{row_idx:04d}.npz
        - hidden:    float16 (T_audio, 1280)   layer −2 hidden states
        - audio_sec: float32 scalar            input duration
        - path:      string                    MEAD_3D path
        - actor:     string
        - emotion:   string
        - intensity: int

Per-clip files (rather than one bundle) so DiT can stream-load only the
clips needed for the current batch — keeps RAM bounded and lets us add
new clips without rebuilding the bundle.

Usage:
  ./03_cache_whisper_embeddings.py \\
      --parquet data/mead_3d/train-00000-of-00001.parquet \\
      --output-dir artifacts/whisper_cache \\
      --whisper-model openai/whisper-large-v3-turbo \\
      --layer -2

  ./03_cache_whisper_embeddings.py --max-clips 3 --output-dir /tmp/wcache
      # Smoke run on 3 clips; ~10 s wall-clock.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.audio_decoder import decode_audio_bytes  # noqa: E402
from lib.mead_3d_loader import (  # noqa: E402
    MeadParquet,
    discover_parquets,
    parse_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--parquet", type=Path, default=None,
        help="Single MEAD_3D parquet to cache. Mutually exclusive with "
             "--data-dir.",
    )
    p.add_argument(
        "--data-dir", type=Path, default=None,
        help="Directory of `mead_3d*` subdirectories. Caches every parquet "
             "discovered, keyed by `{source_id}__{local_idx:04d}.npz`. "
             "Mutually exclusive with --parquet.",
    )
    p.add_argument("--output-dir", type=Path,
                   default=Path("artifacts/whisper_cache"))
    p.add_argument("--whisper-model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--layer", type=int, default=-2,
                   help="Encoder layer (negative indexes from the top). "
                        "Default -2 matches affect_calibration Spike C.")
    p.add_argument("--device", default=None,
                   help="cpu / mps / cuda. Auto-pick if unset.")
    p.add_argument("--max-clips", type=int, default=0,
                   help="0 = all rows. Cap for smoke runs.")
    p.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Re-cache rows whose .npz already exists. Default is to skip.",
    )
    p.set_defaults(skip_existing=True)
    return p.parse_args()


def pick_device(arg: str | None) -> str:
    if arg:
        return arg
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = pick_device(args.device)
    print(f"device: {device}", file=sys.stderr)

    print(f"loading {args.whisper_model}...", file=sys.stderr)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.whisper_model)
    # Force float32 on MPS — large-v3-turbo defaults to a mixed dtype that
    # blows up at the first conv1d ("Input type (float) and bias type
    # (c10::Half) should be the same"). MPS performance is fine in fp32 at
    # this scale, and cache values are downcast to fp16 on save anyway.
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model, torch_dtype=torch.float32,
    )
    encoder = model.get_encoder().to(device).eval()
    n_layers = encoder.config.encoder_layers
    layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer
    print(
        f"  loaded in {time.time() - t0:.1f}s; "
        f"d_whisper={encoder.config.d_model}  layers={n_layers}  "
        f"target layer={layer_idx}",
        file=sys.stderr,
    )

    if args.parquet and args.data_dir:
        print("error: --parquet and --data-dir are mutually exclusive",
              file=sys.stderr)
        return 2
    if not args.parquet and not args.data_dir:
        args.parquet = Path("data/mead_3d/train-00000-of-00001.parquet")

    if args.data_dir:
        triples = discover_parquets(args.data_dir)
        if not triples:
            print(f"error: no `mead_3d*` parquets discovered under "
                  f"{args.data_dir}", file=sys.stderr)
            return 2
        sources: list[tuple[MeadParquet, str, str | None]] = [
            (MeadParquet(p, actor_override=a, source_id=s), s or p.parent.name,
             a)
            for p, a, s in triples
        ]
        print(f"caching {len(sources)} parquets:", file=sys.stderr)
        for pq, sid, override in sources:
            print(f"  {sid:30s}  rows={pq.num_rows:5d}  "
                  f"actor_override={override!r}", file=sys.stderr)
    else:
        pq = MeadParquet(args.parquet)
        sources = [(pq, pq.source_id, pq.actor_override)]
        print(f"caching single parquet: rows={pq.num_rows}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_total = sum(s[0].num_rows for s in sources)
    if args.max_clips:
        n_total = min(n_total, args.max_clips)

    n_done = 0
    n_skipped = 0
    t_start = time.time()
    seen = 0   # global counter across all sources
    for pq, source_id, actor_override in sources:
        local_paths = pq.all_paths()
        for rg in range(pq.num_row_groups):
            if seen >= n_total:
                break
            tbl = pq._pf.read_row_group(rg, columns=["audio", "path"])
            local_offset = sum(
                pq._pf.metadata.row_group(prev).num_rows for prev in range(rg)
            )
            for i in range(tbl.num_rows):
                if seen >= n_total:
                    break
                local_idx = local_offset + i
                # Cache key — stable across reorderings of `sources` list.
                out_path = args.output_dir / f"{source_id}__{local_idx:04d}.npz"
                if out_path.exists() and args.skip_existing:
                    seen += 1
                    n_skipped += 1
                    continue
                row = tbl["audio"][i].as_py()
                p = tbl["path"][i].as_py() or local_paths[local_idx]
                meta = parse_path(p, actor_override=actor_override) or {}

                audio, sr = decode_audio_bytes(row["bytes"])
                audio_sec = len(audio) / sr
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                input_features = inputs["input_features"].to(device)
                with torch.inference_mode():
                    out = encoder(input_features, output_hidden_states=True,
                                  return_dict=True)
                hidden_full = out.hidden_states[layer_idx][0]   # (1500, d_whisper)
                t_keep = max(1, int(np.ceil(audio_sec * 50)))
                hidden = hidden_full[:t_keep]

                np.savez_compressed(
                    out_path,
                    hidden=hidden.detach().to(torch.float16).cpu().numpy(),
                    audio_sec=np.float32(audio_sec),
                    path=p,
                    source_id=source_id,
                    local_idx=np.int32(local_idx),
                    actor=meta.get("actor", ""),
                    emotion=meta.get("emotion", ""),
                    intensity=np.int32(meta.get("intensity", 0)),
                )
                n_done += 1
                seen += 1
                if n_done % 25 == 0 or seen == n_total:
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    remain = (n_total - seen) / max(rate, 1e-6)
                    print(
                        f"  [{seen}/{n_total}] {source_id}: "
                        f"done={n_done} skip={n_skipped}  "
                        f"{rate:.2f} clip/s  elapsed={elapsed:.0f}s  "
                        f"ETA~{remain:.0f}s",
                        file=sys.stderr,
                    )
        if seen >= n_total:
            break

    print(
        f"\ndone. {n_done} written, {n_skipped} skipped, "
        f"{time.time() - t_start:.0f}s total.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
