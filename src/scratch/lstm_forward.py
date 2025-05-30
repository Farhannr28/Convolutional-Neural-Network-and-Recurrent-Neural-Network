from __future__ import annotations

import os
from typing import Any, List, Protocol

import numpy as np
import numpy.typing as npt
from keras import models
from sklearn.metrics import f1_score


class ForwardLayer(Protocol):
    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


_EPSILON = 1e-7


# -------------------- Embedding -------------------- #
class EmbeddingLayer:
    def __init__(self, weight: npt.NDArray[np.float32]) -> None:
        self.weight = weight.astype(np.float32)

    def forward(self, x: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        return self.weight[x]


# -------------------- LSTM (Unidirectional) -------------------- #
class LSTMLayer:
    def __init__(
        self,
        W: npt.NDArray[np.float32],
        U: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
    ) -> None:
        self.W = W.astype(np.float32)  # (input_dim, 4*units)
        self.U = U.astype(np.float32)  # (units, 4*units)
        self.b = b.astype(np.float32)  # (4*units,)
        self.units = U.shape[0]

    def sigmoid(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1 / (1 + np.exp(-x))

    def tanh(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.tanh(x)

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        batch_size, time_steps, input_dim = x.shape
        h = np.zeros((batch_size, self.units), dtype=np.float32)
        c = np.zeros((batch_size, self.units), dtype=np.float32)

        for t in range(time_steps):
            xt = x[:, t, :]  # (batch, input_dim)
            z = xt @ self.W + h @ self.U + self.b  # (batch, 4*units)

            i = self.sigmoid(z[:, :self.units])
            f = self.sigmoid(z[:, self.units:2*self.units])
            c_hat = self.tanh(z[:, 2*self.units:3*self.units])
            o = self.sigmoid(z[:, 3*self.units:])

            c = f * c + i * c_hat
            h = o * self.tanh(c)

        return h  # output terakhir


# -------------------- Dense -------------------- #
class DenseLayer:
    def __init__(
        self,
        weight: npt.NDArray[np.float32],
        bias: npt.NDArray[np.float32],
        activation: str = "linear",
    ) -> None:
        self.W = weight.astype(np.float32)
        self.b = bias.astype(np.float32)
        self.activation = activation.lower()

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        z = x @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(0, z)
        if self.activation == "tanh":
            return np.tanh(z)
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        if self.activation == "softmax":
            exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)
        return z  # linear


# -------------------- Full Model -------------------- #
class ScratchLSTM:
    def __init__(self, keras_model_path: str) -> None:
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(keras_model_path)
        self.layers: List[ForwardLayer] = []
        self._build_from_keras(keras_model_path)

    def _build_from_keras(self, model_path: str) -> None:
        keras_model = models.load_model(model_path, compile=False)
        for layer in keras_model.layers:
            cls_name = layer.__class__.__name__
            if cls_name == "Embedding":
                w = layer.get_weights()[0]
                self.layers.append(EmbeddingLayer(w))
            elif cls_name == "LSTM":
                W, U, b = layer.get_weights()
                self.layers.append(LSTMLayer(W, U, b))
            elif cls_name == "Bidirectional":
                raise NotImplementedError("Bidirectional LSTM belum didukung.")
            elif cls_name == "Dense":
                w, b = layer.get_weights()
                activation = layer.activation.__name__.lower()
                self.layers.append(DenseLayer(w, b, activation))
            elif cls_name == "Dropout":
                continue  # Dropout dilewati saat inference

    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[Any] = x.astype(np.int32)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[np.int32]:
        return np.argmax(self.forward(x), axis=1)
