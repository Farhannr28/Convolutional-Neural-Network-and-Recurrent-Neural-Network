from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
from keras import layers, losses, models, optimizers
from keras.callbacks import History
from sklearn.metrics import f1_score


# ------------------------------------------------------------
# RNN Model Wrapper -- stores data and contains TensorFlow model
# ------------------------------------------------------------
class RNNModel:
    """
    Recurrent Neural Network model wrapper for sentiment analysis.
    Supports configurable layers including Embedding, RNN, Dropout, and Dense.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        # rnn_type: str = "lstm",  # "lstm" or "gru"
        rnn_units: int = 64,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: float = [0.5],
        dense_layers: List[int] = [64],
        dense_activations: List[str] = ["relu"],
        num_classes: int = 3,
        input_length: int = 100,
    ) -> None:
        if len(dense_layers) != len(dense_activations):
            raise ValueError("dense_layers and dense_activations must have the same length")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # self.rnn_type = rnn_type.lower()
        self.rnn_units = rnn_units
        self.rnn_layers = rnn_layers
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.dense_layers = dense_layers
        self.dense_activations = dense_activations
        self.num_classes = num_classes
        self.input_length = input_length

        self.model: models.Model = self._build_model()

    # def _get_rnn_layer(self):
    #     if self.rnn_type == "lstm":
    #         return layers.LSTM
    #     elif self.rnn_type == "gru":
    #         return layers.GRU
    #     else:
    #         raise ValueError("rnn_type must be either 'lstm' or 'gru'")

    def _build_model(self) -> models.Model:
        model = models.Sequential(name="rnn_nusax")
        model.add(layers.Input(shape=(self.input_length,)))
        model.add(
            layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                name="embedding",
            )
        )

        # RNN = self._get_rnn_layer()
        for i in range(self.rnn_layers):
            return_seq = i < self.rnn_layers - 1
            rnn_layer = layers.SimpleRNN(
                self.rnn_units,
                return_sequences=return_seq,
                name=f"uni_{i+1}",
            )
            if self.bidirectional:
                rnn_layer = layers.Bidirectional(rnn_layer, name=f"bi_{i+1}")
            model.add(rnn_layer)

        model.add(layers.Dropout(self.dropout_rate, name=f"dropout"))

        for idx, (units, activation) in enumerate(
            zip(self.dense_layers, self.dense_activations)
        ):
            model.add(layers.Dense(units, activation=activation, name=f"dense_{idx+1}"))

        model.add(layers.Dense(self.num_classes, activation="softmax", name="output"))
        return model

    def compile(self) -> None:
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=optimizers.Adam(learning_rate=1e-3), # Best Learning Rate
            metrics=["accuracy"],
        )

    def train(
        self,
        x_train: npt.NDArray[Any],
        y_train: npt.NDArray[Any],
        x_val: npt.NDArray[Any],
        y_val: npt.NDArray[Any],
        epochs: int = 5,
        batch_size: int = 64,
    ) -> History:
        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

    def evaluate(self, x_test: npt.NDArray[Any], y_test: npt.NDArray[Any]) -> float:
        y_pred_logits = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_logits, axis=1)
        f1: float = f1_score(y_test, y_pred, average="macro")
        print("Test Macro F1 Score:", f1)
        return f1

    def save(self, filename: str = "rnn_model.keras") -> str:
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        self.model.save(full_path)
        print(f"Model saved to {full_path}")
        return full_path

    def load(self, filename: str = "rnn_model.keras") -> None:
        full_path = os.path.join(os.getcwd(), "saved_models", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No saved model found at {full_path}")
        self.model = models.load_model(full_path, compile=False)
        print(f"Model loaded from {full_path}")

    def save_history(
        self, history: Dict[str, List[float]], filename: str = "rnn_history.json"
    ) -> str:
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        with open(full_path, "w") as f:
            json.dump(history, f)
        print(f"Training history saved to {full_path}")
        return full_path

    def load_history(
        self, filename: str = "rnn_history.json"
    ) -> Dict[str, List[float]]:
        full_path = os.path.join(os.getcwd(), "saved_models", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No history file found at {full_path}")
        with open(full_path, "r") as f:
            history: Dict[str, List[float]] = json.load(f)
        print(f"Training history loaded from {full_path}")
        return history


# ------------------------------------------------------------
# Class to help train RNN Model
# ------------------------------------------------------------
class TrainRNN:
    def __init__(
        self,
        x_train: npt.NDArray[Any],
        y_train: npt.NDArray[Any],
        x_val: npt.NDArray[Any],
        y_val: npt.NDArray[Any],
        x_test: npt.NDArray[Any],
        y_test: npt.NDArray[Any],
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def run(
        self,
        vocab_size: int,
        input_length: int,
        embedding_dim: int = 128,
        rnn_units: int = 64,
        rnn_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: List[float] = [0.5],
        dense_layers: List[int] = [64],
        dense_activations: List[str] = ["relu"],
        num_classes: int = 3,
        save_name: str = "rnn.keras",
        history_name: str = "rnn_history.json",
        epochs: int = 5,
        batch_size: int = 64,
    ) -> Tuple[float, Dict[str, List[float]]]:
        model = RNNModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            rnn_units=rnn_units,
            rnn_layers=rnn_layers,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
            dense_layers=dense_layers,
            dense_activations=dense_activations,
            num_classes=num_classes,
            input_length=input_length,
        )
        model.compile()
        history: History = model.train(
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            epochs=epochs,
            batch_size=batch_size,
        )

        model.save(save_name)
        model.save_history(history.history, filename=history_name)
        f1: float = model.evaluate(self.x_test, self.y_test)
        return f1, history.history

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
    MAX_VOCAB_SIZE = 10000  # or you can use None to adapt to data size
    SEQUENCE_LENGTH = 128   # or adjust based on sentence length distribution

    # Load data from GitHub raw links
    train_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/train.csv"
    test_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/test.csv"
    valid_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/valid.csv"

    # Load into DataFrames
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)
    df_valid = pd.read_csv(valid_url)

    # TextVectorization layer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation", # best standardization
        split="whitespace"  # default, could also use custom tokenizer
    )

    # Adapt the vectorizer on training data
    vectorizer.adapt(df_train["text"].values)

    # Save Vectorizer
    with open("saved_models/rnn_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Convert texts to integer sequences
    x_train = vectorizer(df_train["text"].values).numpy()
    x_val = vectorizer(df_valid["text"].values).numpy()
    x_test = vectorizer(df_test["text"].values).numpy()

    # Encode string labels to integers
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(df_train["label"])
    y_val = label_encoder.transform(df_valid["label"])
    y_test = label_encoder.transform(df_test["label"])

    # Extract vocab size and sequence length for model config
    vocab_size = len(vectorizer.get_vocabulary())

    trainer = TrainRNN(x_train, y_train, x_val, y_val, x_test, y_test)

    f1, history = trainer.run( # Base Hyperparameter
        vocab_size = vocab_size,
        input_length = SEQUENCE_LENGTH,
        embedding_dim = 128,
        rnn_units = 72,
        rnn_layers = 2,
        bidirectional = True,
        dropout_rate = 0.5,
        dense_layers = [64, 32],
        dense_activations = ["relu", "relu"],
        num_classes = 3,
        save_name = "base_rnn.keras",
        history_name = "base_rnn_history.json",
        epochs = 200,
        batch_size = 64,
    )

    print(f"Final Macro F1 Score on test set: {f1:.6f}")

    try:
        from src.utils.plot_utils import plot_loss

        plot_loss(history)
    except ImportError:
        print("matplotlib not installed or plot_utils missing, skipping loss plot.")