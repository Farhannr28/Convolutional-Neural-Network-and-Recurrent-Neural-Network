from __future__ import annotations

import os
from typing import Any, List, Protocol, Tuple

import numpy as np
import numpy.typing as npt
from keras import models  # only used to load weights
from sklearn.metrics import f1_score


# --------------- for type checking --------------- #
class ForwardLayer(Protocol):
    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


# --------------- Helper: padding calculation --------------- #
def _compute_same_padding(kernel_size: int) -> Tuple[int, int]:
    total_pad = kernel_size - 1
    pad_before = total_pad // 2
    pad_after = total_pad - pad_before
    return pad_before, pad_after


# ------------------------------------------------------------
# CNN Layers (Conv2D, Pooling, Flatten, Dense)
# ------------------------------------------------------------
_EPSILON = 1e-7


class Conv2DLayer:
    """ Convolutional Layers """

    def __init__(
        self,
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32],
        activation: str = "relu",
    ) -> None:
        self.W = weight.astype(np.float32)
        self.b = bias.astype(np.float32)
        self.kh, self.kw, self.Cin, self.Cout = self.W.shape
        self.pad_h_bef, self.pad_h_aft = _compute_same_padding(self.kh)
        self.pad_w_bef, self.pad_w_aft = _compute_same_padding(self.kw)
        self.activation = activation.lower()
        if self.activation not in {
            "relu",
            "tanh",
            "sigmoid",
            "softmax",
            "linear",
            "log",
        }:
            raise ValueError(f"Unsupported activation '{activation}' in Conv2DLayer")

    def _apply_activation(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "tanh":
            return np.tanh(x)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        if self.activation == "softmax":
            exp = np.exp(x - x.max(axis=-1, keepdims=True))
            return exp / exp.sum(axis=-1, keepdims=True)
        if self.activation == "linear":
            return x
        if self.activation == "log":
            return np.log(np.clip(x, _EPSILON, None))
        raise ValueError(f"Unsupported activation '{self.activation}'")

    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        N, H, W, _ = x.shape  # type: ignore[unused-ignore]
        x_pad = np.pad(
            x,
            (
                (0, 0),
                (self.pad_h_bef, self.pad_h_aft),
                (self.pad_w_bef, self.pad_w_aft),
                (0, 0),
            ),
            mode="constant",
        )
        out = np.zeros((N, H, W, self.Cout), dtype=np.float32)

        w_reshaped = self.W.reshape(-1, self.Cout)  # (kh*kw*Cin, Cout)
        for i in range(H):
            for j in range(W):
                window = x_pad[:, i : i + self.kh, j : j + self.kw, :]
                window = window.reshape(N, -1)
                out[:, i, j, :] = window @ w_reshaped + self.b

        return self._apply_activation(out)


class PoolingLayer:
    """2×2 MaxPooling or AveragePooling (stride 2)."""

    def __init__(self, mode: str = "max") -> None:
        if mode not in {"max", "avg"}:
            raise ValueError("Pooling mode must be 'max' or 'avg'")
        self.mode = mode

    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        N, H, W, C = x.shape  # type: ignore[unused-ignore]
        assert H % 2 == 0 and W % 2 == 0, "Input dims must be even for 2×2 pooling"
        x_resh = x.reshape(N, H // 2, 2, W // 2, 2, C)
        if self.mode == "max":
            return x_resh.max(axis=(2, 4))
        return x_resh.mean(axis=(2, 4))


class FlattenLayer:
    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return x.reshape(x.shape[0], -1)


class DenseLayer:
    def __init__(
        self,
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32],
        activation: str | None,
    ) -> None:
        self.W = weight.astype(np.float32)
        self.b = bias.astype(np.float32)
        self.activation = (activation or "linear").lower()
        if self.activation not in {
            "relu",
            "tanh",
            "sigmoid",
            "softmax",
            "linear",
            "log",
        }:
            raise ValueError(f"Unsupported activation '{activation}' in DenseLayer")

    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        out = x @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(out, 0.0)
        if self.activation == "tanh":
            return np.tanh(out)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-out))
        if self.activation == "softmax":
            exp = np.exp(out - out.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        if self.activation == "linear":
            return out
        if self.activation == "log":
            return np.log(np.clip(out, _EPSILON, None))
        raise ValueError(f"Unsupported activation '{self.activation}'")


# ------------------------------------------------------------
# Scratch CNN (load weights + forward prop)
# ------------------------------------------------------------
class ScratchCNN:
    """Forward Propagation CNN using Keras weights."""

    def __init__(self, keras_model_path: str) -> None:
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(keras_model_path)
        self.layers: List[ForwardLayer] = []
        self._build_from_keras(keras_model_path)

    def _build_from_keras(self, model_path: str) -> None:
        keras_model = models.load_model(model_path, compile=False)
        for layer in keras_model.layers:
            cls_name = layer.__class__.__name__
            if cls_name == "Conv2D":
                w, b = layer.get_weights()
                activation = layer.activation.__name__.lower()
                self.layers.append(Conv2DLayer(w, b, activation))
            elif cls_name in {"MaxPooling2D", "AveragePooling2D"}:
                mode = "max" if cls_name.startswith("Max") else "avg"
                self.layers.append(PoolingLayer(mode))
            elif cls_name == "Flatten":
                self.layers.append(FlattenLayer())
            elif cls_name == "Dense":
                w, b = layer.get_weights()
                activation = layer.activation.__name__.lower()
                self.layers.append(DenseLayer(w, b, activation))

    def _forward_internal(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[np.float32] = x.astype(np.float32) / 255.0
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward(
        self, x: npt.NDArray[Any], batch_size: int | None = None
    ) -> npt.NDArray[Any]:
        if batch_size is None:
            return self._forward_internal(x)
        preds: List[npt.NDArray[Any]] = []
        for i in range(0, x.shape[0], batch_size):
            preds.append(self._forward_internal(x[i : i + batch_size]))
        return np.concatenate(preds, axis=0)

    def predict(
        self, x: npt.NDArray[Any], batch_size: int | None = None
    ) -> npt.NDArray[Any]:
        return np.argmax(self.forward(x, batch_size), axis=1)


# ------------------------------------------------------------
# Quick Test
# ------------------------------------------------------------
if __name__ == "__main__":
    from keras.datasets import cifar10

    model_path = "saved_models/cnn.keras"
    batch_size = 256

    (_, _), (x_test, y_test) = cifar10.load_data()
    y_test = y_test.flatten()

    scratch_net = ScratchCNN(model_path)
    y_pred = scratch_net.predict(x_test, batch_size=batch_size)

    score = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1-score (scratch): {score:.4f}")
