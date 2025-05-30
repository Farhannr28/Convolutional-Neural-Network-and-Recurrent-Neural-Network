from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from keras.layers import TextVectorization

# Local imports
from src.models.lstm import LSTMModel
from src.scratch.lstm_forward import ScratchLSTM
from src.utils.plot_utils import plot_loss


def load_nusax_sentiment(
    train_path = "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/train.csv",
    valid_path = "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/valid.csv",
    test_path = "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/test.csv"
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    print("[INFO] Loading train/val/test from individual CSVs...")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(valid_path)
    df_test = pd.read_csv(test_path)

    all_text = pd.concat([df_train['text'], df_val['text'], df_test['text']]).astype(str).values
    vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
    vectorizer.adapt(all_text)

    le = LabelEncoder()
    le.fit(pd.concat([df_train['label'], df_val['label'], df_test['label']]))

    def prepare(df):
        X = vectorizer(df['text'].astype(str).values).numpy()
        y = le.transform(df['label'].values)
        return X, y

    return prepare(df_train), prepare(df_val), prepare(df_test)


class LSTMSuite:
    def __init__(
        self, save_dir: str = "saved_models", model_name: str = "lstm_model.keras"
    ) -> None:
        print(
            f"[INIT] Initializing LSTMSuite with model path '{save_dir}/{model_name}'"
        )
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_dir / model_name
        self.history_path = self.save_dir / "lstm_history.json"

        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.history: Dict[str, List[float]] | None = None
        self.vectorizer: TextVectorization | None = None
        self.label_encoder: LabelEncoder | None = None

    def load_data(self,
                  train_path="https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/train.csv",
                  val_path="https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/valid.csv",
                  test_path="https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/test.csv"
                  ) -> None:
        print("[START] Loading dataset...")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        all_text = (
            pd.concat([df_train["text"], df_val["text"], df_test["text"]])
            .astype(str)
            .values
        )
        vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=100)
        vectorizer.adapt(all_text)
        self.vectorizer = vectorizer

        le = LabelEncoder()
        le.fit(pd.concat([df_train["label"], df_val["label"], df_test["label"]]))
        self.label_encoder = le

        def prepare(df):
            X = vectorizer(df["text"].astype(str).values).numpy()
            y = le.transform(df["label"].values)
            return X, y

        train = prepare(df_train)
        val = prepare(df_val)
        test = prepare(df_test)
        self.data = {
            "train": train,
            "val": val,
            "test": test,
        }

        print("[DONE] Dataset loaded:")
        for k, (x, y) in self.data.items():
            print(f"  - {k}: {x.shape}, {y.shape}")

    def train(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        embedding_dim: int = 128,
        lstm_layers: int = 1,
        lstm_units: Union[int, List[int]] = 64,
        bidirectional: bool = False,
        dense_units: List[int] = [128],
        dense_activations: List[str] = ["relu"],
    ) -> None:
        if not self.data:
            raise RuntimeError("Call load_data() first")

        print(f"[START] Training LSTM model...")
        x_tr, y_tr = self.data["train"]
        x_val, y_val = self.data["val"]
        x_test, y_test = self.data["test"]

        model = LSTMModel(
            vocab_size=10000,
            embedding_dim=embedding_dim,
            lstm_layers=lstm_layers,
            lstm_units=lstm_units,
            bidirectional=bidirectional,
            dense_units=dense_units,
            dense_activations=dense_activations,
        )
        model.compile()
        history = model.train(
            x_tr, y_tr, x_val, y_val, epochs=epochs, batch_size=batch_size
        )
        f1 = model.evaluate(x_test, y_test)
        model.save(self.model_path.name)
        model.save_history(history.history, name=self.history_path.name)
        self.history = history.history
        print(f"[DONE] Training complete - macro F1 on test (Keras): {f1:.4f}")

    def evaluate_keras(self) -> float:
        from keras.models import load_model

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        x_test, y_test = self.data.get("test", (None, None))
        if x_test is None:
            raise RuntimeError("Test data not loaded")

        print("[START] Evaluating Keras model on test set...")
        model = load_model(self.model_path, compile=False)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"[DONE] Macro F1-score (Keras): {score:.4f}")
        return score

    def evaluate_scratch(self, test_path = "https://raw.githubusercontent.com/IndoNLP/nusax/refs/heads/main/datasets/sentiment/indonesian/test.csv") -> float:
        if self.vectorizer is None or self.label_encoder is None:
            raise RuntimeError(
                "Vectorizer or LabelEncoder not initialized. Call load_data() first."
            )

        print("[START] Loading and preparing test data from NusaX...")
        df_test = pd.read_csv(test_path)
        texts = df_test["text"].astype(str).values
        labels = self.label_encoder.transform(df_test["label"].values)

        x_test = self.vectorizer(texts).numpy()
        y_test = np.array(labels)

        print("[START] Evaluating scratch LSTM on test set...")
        scratch_model = ScratchLSTM(str(self.model_path))
        y_pred = scratch_model.predict(x_test)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"[DONE] Macro F1-score (scratch): {score:.4f}")
        return score

    def load_history(self) -> Dict[str, List[float]]:
        if self.history is not None:
            print("[INFO] Using cached training history.")
            return self.history
        if not self.history_path.exists():
            raise FileNotFoundError(self.history_path)
        print(f"[INFO] Loading training history from {self.history_path}...")
        with open(self.history_path, "r") as f:
            self.history = json.load(f)
        print("[INFO] Training history loaded.")
        return self.history

    def plot_history(self) -> None:
        print("[START] Plotting training loss and accuracy history...")
        history = self.load_history()
        plot_loss(history)
        print("[DONE] Plot displayed.")


# ------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------
if __name__ == "__main__":
    print("[MAIN] LSTMSuite demo started.")
    suite = LSTMSuite()
    suite.load_data()  # path ke file NusaX
    suite.train(epochs=5)
    suite.evaluate_keras()
    suite.evaluate_scratch()
    suite.plot_history()
    print("[MAIN] LSTMSuite demo finished.")
