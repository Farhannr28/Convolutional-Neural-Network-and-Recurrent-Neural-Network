from __future__ import annotations

import os
from typing import Any, List, Protocol, Literal

import numpy as np
import numpy.typing as npt
from keras import models  # only used to load weights
from sklearn.metrics import f1_score


# --------------- for type checking --------------- #
class ForwardLayer(Protocol):
    def forward(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]: ...

# ------------------------------------------------------------
# RNN Layers (SimpleRNN, BidirectionalRNN, Embedding, Dropout, Dense)
# ------------------------------------------------------------
_EPSILON = 1e-7


class SimpleRNNScratch:
    """ Simple RNN layer """

    def __init__(self, weights: list[npt.NDArray[np.float32]], return_sequences: bool = False) -> None:
        if len(weights) != 3:
            raise ValueError("Expected 3 weight arrays: [kernel, recurrent_kernel, bias]")
        self.Wx, self.Wh, self.b = weights  # Wx: (input_dim, units), Wh: (units, units), b: (units,)
        self.return_sequences = return_sequences

    def _apply_tanh_activation(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.tanh(x)

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        batch_size, time_steps, _ = x.shape
        units = self.b.shape[0]

        h = np.zeros((batch_size, units), dtype=np.float32)
        if self.return_sequences:
            outputs = np.zeros((batch_size, time_steps, units), dtype=np.float32)
            for t in range(time_steps):
                x_t = x[:, t, :]
                h = self._apply_tanh_activation(x_t @ self.Wx + h @ self.Wh + self.b)
                outputs[:, t, :] = h
            return outputs
        else:
            for t in range(time_steps):
                x_t = x[:, t, :]
                h = self._apply_tanh_activation(x_t @ self.Wx + h @ self.Wh + self.b)
            return h  # Final hidden state

class BidirectionalRNNScratch:
    MergeMode = Literal["concat", "sum", "ave", "mul"]

    def __init__(
        self,
        forward_weights: list[npt.NDArray[np.float32]],
        backward_weights: list[npt.NDArray[np.float32]],
        merge_mode: MergeMode = "concat",
        return_sequences: bool = False,
    ) -> None:
        self.forward_rnn = SimpleRNNScratch(forward_weights, return_sequences=return_sequences)
        self.backward_rnn = SimpleRNNScratch(backward_weights, return_sequences=return_sequences)
        self.merge_mode = merge_mode
        self.return_sequences = return_sequences

    def forward(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        h_forward = self.forward_rnn.forward(x)  # shape: (B, T, U) or (B, U)
        h_backward = self.backward_rnn.forward(np.flip(x, axis=1))  # reverse time

        # If return_sequences, flip h_backward back
        if self.return_sequences:
            h_backward = np.flip(h_backward, axis=1)

        if self.merge_mode == "concat":
            return np.concatenate([h_forward, h_backward], axis=-1)
        elif self.merge_mode == "sum":
            return h_forward + h_backward
        elif self.merge_mode == "ave":
            return (h_forward + h_backward) / 2
        elif self.merge_mode == "mul":
            return h_forward * h_backward
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")


class EmbeddingLayerScratch:
    """Embedding layer using pre-trained weights."""

    def __init__(self, weights: npt.NDArray[np.float32]) -> None:
        self.weights = weights  # shape: (vocab_size, embedding_dim)

    def forward(self, x: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        """
        x: (batch_size, sequence_length) with integer indices
        returns: (batch_size, sequence_length, embedding_dim)
        """
        return self.weights[x]


class Dropout:
    # Not used during Inference
    pass


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
# Scratch RNN (load weights + forward prop)
# ------------------------------------------------------------
class ScratchRNN:
    """Forward Propagation RNN using Keras weights."""

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
                weights = layer.get_weights()[0]  # shape: (vocab_size, embedding_dim)
                self.layers.append(EmbeddingLayerScratch(weights))

            elif cls_name in {"SimpleRNN", "Bidirectional"}:
                if cls_name == "Bidirectional":
                    forward_weights = layer.forward_layer.get_weights()
                    backward_weights = layer.backward_layer.get_weights()
                    merge = layer.merge_mode.lower()
                    return_sequences = layer.forward_layer.return_sequences
                    self.layers.append(
                        BidirectionalRNNScratch(forward_weights, backward_weights, merge, return_sequences)
                    )
                else:
                    weights = layer.get_weights()  # [kernel, recurrent_kernel, bias]
                    return_sequences = layer.return_sequences
                    self.layers.append(SimpleRNNScratch(weights, return_sequences=return_sequences))

            elif cls_name == "Dropout":
                # Dropout does nothing in inference
                continue

            elif cls_name == "Dense":
                w, b = layer.get_weights()
                activation = layer.activation.__name__.lower()
                self.layers.append(DenseLayer(w, b, activation))


    def _forward_internal(self, x: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
        out = x 
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def forward(
        self, x: npt.NDArray[np.int32], batch_size: int | None = None
    ) -> npt.NDArray[np.float32]:
        if batch_size is None:
            return self._forward_internal(x)
        preds: List[npt.NDArray[np.float32]] = []
        for i in range(0, x.shape[0], batch_size):
            preds.append(self._forward_internal(x[i : i + batch_size]))
        return np.concatenate(preds, axis=0)

    def predict(
        self, x: npt.NDArray[np.int32], batch_size: int | None = None
    ) -> npt.NDArray[np.int32]:
        return np.argmax(self.forward(x, batch_size), axis=1)


# ------------------------------------------------------------
# Quick Test
# ------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import tensorflow as tf
    import pickle
    from sklearn.preprocessing import LabelEncoder

    # Constants
    model_path = "saved_models/base_rnn.keras"
    batch_size = 64

    # Load data from GitHub raw links
    test_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/test.csv"

    # Load into DataFrames
    df_test = pd.read_csv(test_url)

    with open("saved_models/rnn_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Convert texts to integer sequences
    x_test = vectorizer(df_test["text"].values).numpy()

    # Encode string labels to integers
    label_encoder = LabelEncoder()

    y_test = label_encoder.fit_transform(df_test["label"])

    scratch_model = ScratchRNN(model_path)

    y_pred = scratch_model.predict(x_test, batch_size=batch_size)

    score = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1-score (scratch): {score:.6f}")

    print(f"Final Macro F1 Score on test set: {score:.6f}")