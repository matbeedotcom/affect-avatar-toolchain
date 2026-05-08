#!/usr/bin/env python3
"""Train a Whisper-embedding → VAD regressor with speaker-disjoint splits.

Per IMPLEMENTATION_PLAN §4.2 + PROJECT_PLAN §10 R5. Consumes the .npz
written by 01_extract_whisper_embeddings.py:

    embeddings (n, d_whisper) + clip_ids + speaker_ids
        + (vad_labels OR emotion_categoricals → mapped via emotion_to_vad)
            ↓
        speaker-disjoint train/val/test split
            ↓
        regressor fit (linear ridge or small MLP)
            ↓
        ONNX export + metrics report

Output: an .onnx model consumable from Rust at runtime, plus a sidecar
JSON metrics file (per-axis RMSE, MAE, Pearson r on val and test).

Speaker-disjoint splitting (R5): no speaker_id appears in more than one
split. Prevents the model from memorizing speaker timbres as a shortcut
for emotion. Implementation note: GroupKFold over speaker_ids, not
clip_ids.

Usage:
  ./02_train_vad_regressor.py \\
      --embeddings data/crema_d/whisper_embeds.npz \\
      --output artifacts/vad_regressor/crema_d_ridge.onnx \\
      --metrics artifacts/vad_regressor/crema_d_metrics.json \\
      --regressor ridge

  ./02_train_vad_regressor.py \\
      --embeddings /tmp/dryrun_embeds.npz \\
      --dry-run \\
      --output /tmp/dryrun_regressor.onnx \\
      --metrics /tmp/dryrun_metrics.json
      # Synthesizes plausible per-axis metrics + writes a stub .onnx
      # placeholder; useful for wiring the consuming Rust runtime
      # before sklearn / torch / onnx land in pyproject.

Phase 5 D1.1 (training math) is the substantive work; this driver
fixes the I/O contract and split policy so the training body lands
without re-litigating data plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lib.datasets import DATASETS, DatasetName  # noqa: E402
from lib.emotion_to_vad import categorical_to_vad  # noqa: E402
from lib.persona_pipeline import AXES  # noqa: E402


# Phase 5 split ratios. Train/val/test are speaker-disjoint; clips
# from one speaker land entirely in one split.
DEFAULT_SPLIT = (0.70, 0.15, 0.15)

REGRESSORS = ("ridge", "mlp")


@dataclass
class SplitMetrics:
    """Per-axis regression metrics on a single split (val or test)."""
    split: str          # "val" | "test"
    rmse: tuple[float, float, float]    # (V, A, D)
    mae: tuple[float, float, float]
    pearson_r: tuple[float, float, float]
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Input .npz from 01_extract_whisper_embeddings.py.")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx model path.")
    parser.add_argument("--metrics", type=Path, required=True, help="Output metrics JSON path.")
    parser.add_argument("--regressor", choices=REGRESSORS, default="ridge",
                        help="Regressor family. ridge = closed-form, mlp = small torch MLP.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="L2 regularization strength for --regressor=ridge.")
    parser.add_argument("--mlp-hidden", type=int, default=256,
                        help="Hidden width for --regressor=mlp (single layer + ReLU).")
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--split", type=float, nargs=3, default=list(DEFAULT_SPLIT),
                        metavar=("TRAIN", "VAL", "TEST"),
                        help="Speaker-disjoint split ratios; must sum to 1.0.")
    parser.add_argument("--dataset", choices=DATASETS, default=None,
                        help="Override dataset for emotion_to_vad mapping. "
                             "Inferred from clip_ids prefix if omitted.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip real training; emit a stub .onnx + plausible metrics.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _infer_dataset(clip_ids: np.ndarray) -> DatasetName:
    """Fallback dataset inference for legacy NPZs without the `dataset` field.

    01_extract_whisper_embeddings.py stamps the dataset name into the
    NPZ directly; this is only used if that field is missing. Most real
    clip IDs (e.g. JL-Corpus 'female1_angry_10a_1', CREMA-D filenames)
    don't contain the dataset name as a substring, so this fallback only
    works for the synthetic dry-run IDs ('dryrun_<dataset>_NNNNN').
    """
    sample = str(clip_ids[0])
    for name in DATASETS:
        if name in sample:
            return name  # type: ignore[return-value]
    raise ValueError(f"could not infer dataset from clip_id sample '{sample}'; pass --dataset explicitly")


def _resolve_vad_targets(
    vad_labels: np.ndarray,
    emotion_categoricals: np.ndarray,
    dataset: DatasetName,
) -> np.ndarray:
    """Fill NaN rows in vad_labels via categorical_to_vad lookup.

    Datasets that ship continuous VAD labels (none in current stack;
    placeholder for future MSP-Podcast access) keep their explicit
    values. Datasets shipping only categorical labels (all 3 in the
    current stack) get mapped here.
    """
    targets = np.array(vad_labels, dtype=np.float32, copy=True)
    nan_mask = np.isnan(targets).any(axis=1)
    for i in np.where(nan_mask)[0]:
        cat = str(emotion_categoricals[i])
        if not cat:
            raise ValueError(f"row {i} has neither vad_label nor emotion_categorical")
        targets[i] = categorical_to_vad(dataset, cat)
    return targets


def _speaker_disjoint_split(
    speaker_ids: np.ndarray,
    ratios: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition row indices into (train, val, test) with no shared speaker.

    Cumulative-midpoint assignment: shuffle unique speakers, then place
    each speaker in the bucket whose cumulative-clip-count range covers
    that speaker's midpoint. Guarantees each non-zero-ratio split gets
    at least one speaker when n_speakers >= 3 (a greedy fill-quota
    splitter empties the last bucket on small-speaker datasets like
    JL-Corpus's 4 speakers).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0; got {ratios} (sum={sum(ratios)})")

    unique_speakers = np.array(sorted(set(speaker_ids.tolist())))
    rng.shuffle(unique_speakers)

    clips_per_speaker = {s: int(np.sum(speaker_ids == s)) for s in unique_speakers}
    counts = np.array([clips_per_speaker[s] for s in unique_speakers])
    cum_end = np.cumsum(counts)
    midpoints = cum_end - counts / 2.0
    n_total = int(cum_end[-1]) if len(cum_end) else 0

    train_cutoff = ratios[0] * n_total
    val_cutoff = (ratios[0] + ratios[1]) * n_total

    train_speakers, val_speakers, test_speakers = set(), set(), set()
    for i, spk in enumerate(unique_speakers):
        if midpoints[i] < train_cutoff:
            train_speakers.add(spk)
        elif midpoints[i] < val_cutoff:
            val_speakers.add(spk)
        else:
            test_speakers.add(spk)

    train_idx = np.array([i for i, s in enumerate(speaker_ids) if s in train_speakers], dtype=np.int64)
    val_idx = np.array([i for i, s in enumerate(speaker_ids) if s in val_speakers], dtype=np.int64)
    test_idx = np.array([i for i, s in enumerate(speaker_ids) if s in test_speakers], dtype=np.int64)

    # Bail loudly when a non-zero-ratio bucket got 0 clips — silently
    # producing a NaN regressor (empty mean of empty array → NaN
    # weights → NaN ONNX output) is a much worse failure mode.
    if ratios[0] > 0 and len(train_idx) == 0:
        raise ValueError(
            f"speaker-disjoint split produced empty train bucket; "
            f"n_speakers={len(unique_speakers)}, ratios={ratios}. "
            f"Consider lowering --split[0] or pulling more clips so "
            f"the train bucket gets at least one speaker."
        )
    if ratios[1] > 0 and len(val_idx) == 0:
        raise ValueError(
            f"speaker-disjoint split produced empty val bucket; "
            f"n_speakers={len(unique_speakers)}, ratios={ratios}."
        )
    if ratios[2] > 0 and len(test_idx) == 0:
        raise ValueError(
            f"speaker-disjoint split produced empty test bucket; "
            f"n_speakers={len(unique_speakers)}, ratios={ratios}."
        )
    return train_idx, val_idx, test_idx


def _per_axis_metrics(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> SplitMetrics:
    """Compute per-axis RMSE / MAE / Pearson r."""
    err = y_pred - y_true
    rmse = tuple(float(np.sqrt(np.mean(err[:, j] ** 2))) for j in range(3))
    mae = tuple(float(np.mean(np.abs(err[:, j]))) for j in range(3))
    r: list[float] = []
    for j in range(3):
        a, b = y_true[:, j], y_pred[:, j]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            r.append(float("nan"))
        else:
            r.append(float(np.corrcoef(a, b)[0, 1]))
    return SplitMetrics(split=split, rmse=rmse, mae=mae, pearson_r=tuple(r), n=len(y_true))


def _dry_run_predictions(y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthesize predictions: ground truth + small per-axis Gaussian noise.

    Produces realistic-looking metrics (Pearson r ~ 0.7-0.9, RMSE ~ 0.2)
    for downstream Rust runtime wiring without invoking sklearn / torch.
    """
    return y_true + rng.normal(0.0, 0.2, size=y_true.shape).astype(np.float32)


def main() -> int:
    args = parse_args()

    if not args.embeddings.exists():
        print(f"error: embeddings not found at {args.embeddings}", file=sys.stderr)
        return 2

    npz = np.load(args.embeddings, allow_pickle=True)
    embeddings = np.asarray(npz["embeddings"], dtype=np.float32)
    clip_ids = np.asarray(npz["clip_ids"])
    speaker_ids = np.asarray(npz["speaker_ids"])
    emotion_categoricals = np.asarray(npz["emotion_categoricals"])
    vad_labels = np.asarray(npz["vad_labels"], dtype=np.float32)
    whisper_model = str(npz["whisper_model"]) if "whisper_model" in npz.files else "unknown"
    layer = int(npz["layer"]) if "layer" in npz.files else -1

    # Prefer 01-stamped dataset name; fall back to legacy clip-ID inference.
    if args.dataset:
        dataset = args.dataset
    elif "dataset" in npz.files and str(npz["dataset"]):
        dataset = str(npz["dataset"])
    else:
        dataset = _infer_dataset(clip_ids)
    print(f"loaded {len(clip_ids)} clips  d_whisper={embeddings.shape[1]}  "
          f"dataset={dataset}  model={whisper_model}  layer={layer}", file=sys.stderr)

    targets = _resolve_vad_targets(vad_labels, emotion_categoricals, dataset)

    rng = np.random.default_rng(args.seed)
    train_idx, val_idx, test_idx = _speaker_disjoint_split(
        speaker_ids, tuple(args.split), rng
    )
    print(f"  splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"(speaker-disjoint)", file=sys.stderr)

    if args.dry_run:
        y_val_pred = _dry_run_predictions(targets[val_idx], rng)
        y_test_pred = _dry_run_predictions(targets[test_idx], rng)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(b"DRYRUN_ONNX_PLACEHOLDER")
        print(f"  wrote stub .onnx placeholder -> {args.output}", file=sys.stderr)
    else:
        model = _train_regressor(
            X_train=embeddings[train_idx],
            y_train=targets[train_idx],
            X_val=embeddings[val_idx],
            y_val=targets[val_idx],
            kind=args.regressor,
            ridge_alpha=args.ridge_alpha,
            mlp_hidden=args.mlp_hidden,
            mlp_epochs=args.mlp_epochs,
            seed=args.seed,
        )
        y_val_pred = model.predict(embeddings[val_idx])
        y_test_pred = model.predict(embeddings[test_idx])
        _export_onnx(model, d_whisper=embeddings.shape[1], output=args.output)
        print(f"  wrote model -> {args.output}", file=sys.stderr)

    val_metrics = _per_axis_metrics(targets[val_idx], y_val_pred, split="val")
    test_metrics = _per_axis_metrics(targets[test_idx], y_test_pred, split="test")

    metrics_payload = {
        "dataset": dataset,
        "whisper_model": whisper_model,
        "whisper_layer": layer,
        "regressor": args.regressor,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "speaker_disjoint": True,
        "axes": [a.value for a in AXES],
        "val": _split_payload(val_metrics),
        "test": _split_payload(test_metrics),
        "dry_run": bool(args.dry_run),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"  wrote metrics -> {args.metrics}", file=sys.stderr)

    for split, m in (("val", val_metrics), ("test", test_metrics)):
        for j, axis in enumerate(AXES):
            print(f"  {split:5s}  {axis.value:9s}  rmse={m.rmse[j]:.3f}  "
                  f"mae={m.mae[j]:.3f}  r={m.pearson_r[j]:+.3f}", file=sys.stderr)

    return 0


def _split_payload(m: SplitMetrics) -> dict:
    return {
        "n": m.n,
        "rmse": {axis.value: float(m.rmse[j]) for j, axis in enumerate(AXES)},
        "mae": {axis.value: float(m.mae[j]) for j, axis in enumerate(AXES)},
        "pearson_r": {axis.value: float(m.pearson_r[j]) for j, axis in enumerate(AXES)},
    }


class _Regressor:
    """Minimal interface the dry-run path doesn't exercise but real training must.

    D1.1 implementations may use sklearn (Ridge / MLPRegressor) or torch
    + skl2onnx / torch.onnx.export. The interface is intentionally
    narrow so 02 doesn't depend on the choice.
    """
    def predict(self, X: np.ndarray) -> np.ndarray: ...


def _train_regressor(*, X_train, y_train, X_val, y_val, kind: str,
                     ridge_alpha: float, mlp_hidden: int, mlp_epochs: int,
                     seed: int) -> _Regressor:
    """Fit a regressor on Whisper embeddings → (V, A, D)."""
    from lib.regressor import build_regressor  # noqa: E402  (lazy)
    model = build_regressor(
        kind,
        d_whisper=X_train.shape[1],
        ridge_alpha=ridge_alpha,
        mlp_hidden=mlp_hidden,
        mlp_epochs=mlp_epochs,
        seed=seed,
    )
    if kind == "mlp":
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)
    return model


def _export_onnx(model: _Regressor, *, d_whisper: int, output: Path) -> None:
    """Serialize model to ONNX for Rust-side inference (ort crate)."""
    model.export_onnx(output, d_whisper=d_whisper)


if __name__ == "__main__":
    sys.exit(main())
