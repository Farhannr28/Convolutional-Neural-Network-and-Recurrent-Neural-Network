from __future__ import annotations

import os
from typing import Any, List, Protocol

import numpy as np
import numpy.typing as npt
from keras import models
from sklearn.metrics import f1_score


_EPSILON = 1e-7

class ForwardLayer(Protocol):
    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]: ...


class EmbeddingLayer:
    def __init__(self, weight: npt.NDArray[np.float32]) -> None:
        self.weight = weight.astype(np.float32)

    def forward(self, x: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        return self.weight[x]


class LSTMLayer:
    def __init__(
        self,
        W: npt.NDArray[np.float32],
        U: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
        return_sequences: bool = False,
    ) -> None:
        self.W = W.astype(np.float32)
        self.U = U.astype(np.float32)
        self.b = b.astype(np.float32)
        self.units = U.shape[0]
        self.return_sequences = return_sequences

    def sigmoid(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return 1 / (1 + np.exp(-x))

    def tanh(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.tanh(x)

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if x.ndim != 3:
             raise ValueError(f"LSTMLayer expects 3D input (batch, time_steps, input_dim), but got {x.ndim}D input with shape {x.shape}")

        batch_size, time_steps, input_dim = x.shape
        h = np.zeros((batch_size, self.units), dtype=np.float32)
        c = np.zeros((batch_size, self.units), dtype=np.float32)

        if self.return_sequences:
            output_sequence = np.zeros((batch_size, time_steps, self.units), dtype=np.float32)

        for t in range(time_steps):
            xt = x[:, t, :]
            if xt.ndim == 1:
                xt = np.expand_dims(xt, axis=1)

            if h.ndim == 1:
                 h = np.expand_dims(h, axis=1)
            if c.ndim == 1:
                 c = np.expand_dims(c, axis=1)

            z = xt @ self.W + h @ self.U + self.b

            i = self.sigmoid(z[:, :self.units])
            f = self.sigmoid(z[:, self.units:2*self.units])
            c_hat = self.tanh(z[:, 2*self.units:3*self.units])
            o = self.sigmoid(z[:, 3*self.units:])

            c = f * c + i * c_hat
            h = o * self.tanh(c)

            if self.return_sequences:
                output_sequence[:, t, :] = h

        if self.return_sequences:
            return output_sequence
        else:
            return h



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
        return z


class BidirectionalLSTMLayer:
    def __init__(
        self,
        forward_weights: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
        backward_weights: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]],
        return_sequences: bool = False,
    ) -> None:
        self.forward_lstm = LSTMLayer(*forward_weights, return_sequences)
        self.backward_lstm = LSTMLayer(*backward_weights, return_sequences)
        self.return_sequences = return_sequences

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        out_forward = self.forward_lstm.forward(x)

        out_backward = self.backward_lstm.forward(x[:, ::-1, :])
        if self.return_sequences:
            out_backward = out_backward[:, ::-1, :]

        return np.concatenate([out_forward, out_backward], axis=-1)


class ScratchLSTM:
    def __init__(self, keras_model_path: str) -> None:
        if not os.path.exists(keras_model_path):
            raise FileNotFoundError(keras_model_path)
        self.layers: List[ForwardLayer] = []
        self._build_from_keras(keras_model_path)

    def _build_from_keras(self, model_path: str) -> None:
        keras_model = models.load_model(model_path, compile=False)
        self.layers: List[ForwardLayer] = []
        lstm_count = 0
        total_lstm_layers = sum(1 for layer in keras_model.layers if layer.__class__.__name__ == "LSTM")
        print(f"[DEBUG] Total LSTM layers in Keras model: {total_lstm_layers}")


        for layer in keras_model.layers:
            cls_name = layer.__class__.__name__
            print(f"[DEBUG] Processing Keras layer: {cls_name} - {layer.name}")
            if cls_name == "Embedding":
                w = layer.get_weights()[0]
                self.layers.append(EmbeddingLayer(w))
                print(f"[DEBUG] Added EmbeddingLayer")
            elif cls_name == "LSTM":
                W, U, b = layer.get_weights()
                lstm_count += 1
                return_sequences = lstm_count < total_lstm_layers
                self.layers.append(LSTMLayer(W, U, b, return_sequences=return_sequences))
                print(f"[DEBUG] Added LSTMLayer with units: {U.shape[0]}, return_sequences: {return_sequences}")
            elif cls_name == "Bidirectional":
                    print(f"[DEBUG] Adding BidirectionalLSTMLayer")
                    fw_weights = layer.forward_layer.get_weights()
                    bw_weights = layer.backward_layer.get_weights()
                    return_sequences = layer.return_sequences
                    self.layers.append(BidirectionalLSTMLayer(fw_weights, bw_weights, return_sequences))
            elif cls_name == "Dense":
                w, b = layer.get_weights()
                activation = layer.activation.__name__.lower()
                self.layers.append(DenseLayer(w, b, activation))
                print(f"[DEBUG] Added DenseLayer with units: {w.shape[1]}, activation: {activation}")
            elif cls_name == "Dropout":
                print(f"[DEBUG] Skipping Dropout layer")
                continue
            else:
                print(f"[DEBUG] Warning: Unhandled Keras layer type: {cls_name}")

    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        out: npt.NDArray[Any] = x.astype(np.int32)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: npt.NDArray[Any]) -> npt.NDArray[np.int32]:
        return np.argmax(self.forward(x), axis=1)
