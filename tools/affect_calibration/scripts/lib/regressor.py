"""VAD regressor implementations + ONNX export.

Used by 02_train_vad_regressor.py. Two backends:

- `RidgeRegressor` — closed-form `(X^T X + αI)^-1 X^T y` in numpy. No
  sklearn / skl2onnx dependency. Exports as a 2-node ONNX graph
  (MatMul + Add). Suitable when the relationship between Whisper
  embeddings and VAD coordinates is approximately linear, which is the
  Phase 5 v1 expectation per IMPLEMENTATION_PLAN §4.3.

- `MLPRegressor` — small torch MLP (Linear → GELU → Linear → Tanh).
  Exported via `torch.onnx.export`. Use this if ridge val-RMSE is too
  high; cost is the torch dependency at training time and the
  Tanh-bounded output range.

Both regressors expose a uniform interface:
    .predict(X: np.ndarray) -> np.ndarray            # (n, 3)
    .export_onnx(path: Path, d_whisper: int) -> None

The loaded model is consumed at runtime by Rust via the `ort` crate;
ONNX is the lingua franca.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ridge (closed-form, numpy-only)
# ---------------------------------------------------------------------------

class RidgeRegressor:
    """Multivariate ridge regression: y = X @ W + b.

    Closed-form fit minimizes ||X W + b - y||^2 + alpha * ||W||^2 over
    W ∈ R^(d_whisper × 3). The bias `b` is not regularized: we center
    X and y before solving and reconstruct `b` from the centroids.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        # Set by .fit(); shapes (d_whisper, 3) and (3,).
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or y.ndim != 2 or y.shape[1] != 3:
            raise ValueError(
                f"ridge fit expects X=(n,d) and y=(n,3); got X={X.shape} y={y.shape}"
            )

        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        Xc = X - x_mean
        yc = y - y_mean

        d = Xc.shape[1]
        gram = Xc.T @ Xc + self.alpha * np.eye(d)
        # Solve gram @ W = Xc.T @ yc
        W = np.linalg.solve(gram, Xc.T @ yc)  # (d, 3)
        b = y_mean - x_mean @ W                # (3,)

        self.W = W.astype(np.float32)
        self.b = b.astype(np.float32)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise RuntimeError("RidgeRegressor: call fit() before predict()")
        X = np.asarray(X, dtype=np.float32)
        return (X @ self.W + self.b).astype(np.float32)

    # ------------------------------------------------------------------
    # ONNX export — 2-node graph: MatMul(input, W) + Add(b)
    # ------------------------------------------------------------------

    def export_onnx(self, path: Path, d_whisper: int) -> None:
        if self.W is None or self.b is None:
            raise RuntimeError("RidgeRegressor: call fit() before export_onnx()")

        import onnx  # lazy
        from onnx import TensorProto, helper, numpy_helper

        if self.W.shape != (d_whisper, 3):
            raise ValueError(
                f"W shape {self.W.shape} != (d_whisper={d_whisper}, 3)"
            )

        # Initializers
        W_init = numpy_helper.from_array(self.W.astype(np.float32), name="W")
        b_init = numpy_helper.from_array(self.b.astype(np.float32), name="b")

        # I/O
        input_info = helper.make_tensor_value_info(
            "whisper_embed", TensorProto.FLOAT, [None, d_whisper]
        )
        output_info = helper.make_tensor_value_info(
            "vad", TensorProto.FLOAT, [None, 3]
        )

        # MatMul + Add
        matmul_node = helper.make_node("MatMul", ["whisper_embed", "W"], ["matmul_out"])
        add_node = helper.make_node("Add", ["matmul_out", "b"], ["vad"])

        graph = helper.make_graph(
            nodes=[matmul_node, add_node],
            name="WhisperToVADRidge",
            inputs=[input_info],
            outputs=[output_info],
            initializer=[W_init, b_init],
        )

        # opset 17 matches the project convention (capabilities resolution
        # spec uses opset 17 per CLAUDE.md).
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", 17)],
            producer_name="affect_calibration_ridge",
            ir_version=8,
        )
        onnx.checker.check_model(model)

        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))


# ---------------------------------------------------------------------------
# MLP (torch)
# ---------------------------------------------------------------------------

class MLPRegressor:
    """Tiny MLP: Linear(d → h) → GELU → Linear(h → 3) → Tanh.

    Trained with AdamW + cosine schedule. Exported via torch.onnx.export.
    Architecture mirrors IMPLEMENTATION_PLAN §4.3 sketch.
    """

    def __init__(
        self,
        d_whisper: int,
        hidden: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        seed: int = 0,
    ) -> None:
        self.d_whisper = d_whisper
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

        # Lazy torch import at fit() time keeps this constructor cheap.
        self._torch: Any = None
        self._model: Any = None

    # ------------------------------------------------------------------

    def _ensure_torch(self) -> None:
        if self._torch is not None:
            return
        import torch  # lazy
        import torch.nn as nn  # lazy
        self._torch = torch
        self._nn = nn

    def _build(self) -> None:
        self._ensure_torch()
        torch = self._torch
        torch.manual_seed(self.seed)
        nn = self._nn
        self._model = nn.Sequential(
            nn.Linear(self.d_whisper, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 3),
            nn.Tanh(),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MLPRegressor":
        self._ensure_torch()
        torch = self._torch

        if self._model is None:
            self._build()

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        n = X_t.shape[0]
        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Cosine schedule over total *steps*, not epochs — full-batch
        # training was producing one step per epoch and undertraining
        # the model relative to ridge. Mini-batch SGD gives O(n/bs)
        # steps per epoch and converges in a sane number of passes.
        n_steps = max(1, self.epochs * max(1, (n + self.batch_size - 1) // self.batch_size))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
        loss_fn = self._nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)

        self._model.train()
        step = 0
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                xb, yb = X_t[idx], y_t[idx]
                opt.zero_grad(set_to_none=True)
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                sched.step()
                step += 1

            if X_val is not None and y_val is not None and (epoch + 1) % max(1, self.epochs // 5) == 0:
                self._model.eval()
                with torch.no_grad():
                    Xv = torch.from_numpy(np.asarray(X_val, dtype=np.float32))
                    yv = torch.from_numpy(np.asarray(y_val, dtype=np.float32))
                    val_loss = loss_fn(self._model(Xv), yv).item()
                logger.info("epoch %d/%d  step=%d  train_loss=%.4f  val_loss=%.4f",
                            epoch + 1, self.epochs, step, loss.item(), val_loss)
                self._model.train()

        self._model.eval()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MLPRegressor: call fit() before predict()")
        torch = self._torch
        with torch.no_grad():
            Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
            out = self._model(Xt).cpu().numpy().astype(np.float32)
        return out

    def export_onnx(self, path: Path, d_whisper: int) -> None:
        if self._model is None:
            raise RuntimeError("MLPRegressor: call fit() before export_onnx()")
        torch = self._torch

        path.parent.mkdir(parents=True, exist_ok=True)
        dummy = torch.zeros(1, d_whisper, dtype=torch.float32)
        torch.onnx.export(
            self._model,
            dummy,
            str(path),
            input_names=["whisper_embed"],
            output_names=["vad"],
            dynamic_axes={"whisper_embed": {0: "batch"}, "vad": {0: "batch"}},
            opset_version=17,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

REGRESSOR_KINDS = ("ridge", "mlp")


def build_regressor(kind: str, **kwargs):
    if kind == "ridge":
        return RidgeRegressor(alpha=kwargs.get("ridge_alpha", 1.0))
    if kind == "mlp":
        return MLPRegressor(
            d_whisper=kwargs["d_whisper"],
            hidden=kwargs.get("mlp_hidden", 64),
            epochs=kwargs.get("mlp_epochs", 50),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            seed=kwargs.get("seed", 0),
        )
    raise ValueError(f"unknown regressor kind: {kind!r}; expected one of {REGRESSOR_KINDS}")
