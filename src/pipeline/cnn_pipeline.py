from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score

# Local imports
from src.models.cnn import TrainCNN
from src.scratch.cnn_forward import ScratchCNN
from src.utils.plot_utils import plot_loss


# ------------------------------------------------------------
# Data loader (train / val / test split)
# ------------------------------------------------------------
def load_cifar10_with_val(val_ratio: float = 0.2):
    """
    Download CIFAR-10 and split the original train set into train and validation.
    """
    print("[INFO] Downloading CIFAR-10 dataset...")
    from keras.datasets import cifar10

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
    print("[INFO] Dataset downloaded successfully.")

    # Normalise images to [0,1] float32
    print("[INFO] Normalizing image data...")
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    val_size = int(len(x_train_full) * val_ratio)
    print(
        f"[INFO] Splitting training data: {len(x_train_full)} samples into "
        f"{len(x_train_full) - val_size} train and {val_size} validation samples."
    )
    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]
    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    print("[INFO] Data loading and splitting complete.")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ------------------------------------------------------------
# Pipeline wrapper class
# ------------------------------------------------------------
class CNNSuite:
    """Combine All CNN functionality."""

    def __init__(
        self, save_dir: str = "saved_models", model_name: str = "cnn.keras"
    ) -> None:
        print(f"[INIT] Initializing CNNSuite with model path '{save_dir}/{model_name}'")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.save_dir / model_name
        self.history_path = self.save_dir / "cnn_history.json"

        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.history: Dict[str, List[float]] | None = None

    # --------------- Data handling --------------- #
    def load_data(self) -> None:
        print("[START] Loading dataset...")
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_with_val()
        self.data = {
            "train": (x_train, y_train),
            "val": (x_val, y_val),
            "test": (x_test, y_test),
        }
        print("[DONE] Dataset loaded:")
        for k, (x, y) in self.data.items():
            print(f"  - {k}: {x.shape}, {y.shape}")

    # --------------- Training --------------- #
    def train(
        self,
        epochs: int = 5,
        batch_size: int = 64,
        conv_layers: int = 2,
        filters: Union[int, List[int]] = 32,
        kernel_size: Union[int, List[int]] = 3,
        pooling: str = "max",
        pool_size: int = 2,
        conv_activation: str = "relu",
        dense_layers: List[int] | None = None,
        dense_activations: List[str] | None = None,
    ) -> None:
        if not self.data:
            raise RuntimeError("Call load_data() first")
        print(f"[START] Training with {filters} filters, {epochs} epochs...")

        x_tr, y_tr = self.data["train"]
        x_val, y_val = self.data["val"]
        x_test, y_test = self.data["test"]

        trainer = TrainCNN(x_tr, y_tr, x_val, y_val, x_test, y_test)
        f1, history = trainer.run(
            save_name=self.model_path.name,
            history_name=self.history_path.name,
            epochs=epochs,
            batch_size=batch_size,
            conv_layers=conv_layers,
            filters=filters,
            kernel_size=kernel_size,
            pooling=pooling,
            pool_size=pool_size,
            conv_activation=conv_activation,
            dense_layers=dense_layers,
            dense_activations=dense_activations,
        )
        self.history = history
        print(f"[DONE] Training finished - macro F1 on test (Keras): {f1:.4f}")

    # --------------- Evaluation utilities --------------- #
    def evaluate_keras(self) -> float:
        from keras.models import load_model

        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        (x_test, y_test) = self.data.get("test", (None, None))
        if x_test is None:
            raise RuntimeError("Test data not loaded")

        print("[START] Evaluating Keras model on test set...")
        model = load_model(self.model_path, compile=False)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"[DONE] Macro F1-score (Keras): {score:.4f}")
        return score

    def evaluate_scratch(self, batch_size: int = 256) -> float:
        if not self.model_path.exists():
            raise FileNotFoundError(self.model_path)
        (x_test, y_test) = self.data.get("test", (None, None))
        if x_test is None:
            raise RuntimeError("Test data not loaded")

        print("[START] Evaluating scratch CNN on test set...")
        scratch_net = ScratchCNN(str(self.model_path))
        y_pred = scratch_net.predict(x_test * 255.0, batch_size=batch_size)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"[DONE] Macro F1-score (scratch): {score:.4f}")
        return score

    # --------------- History handling --------------- #
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
    print("[MAIN] CNNSuite demo started.")
    suite = CNNSuite()
    suite.load_data()
    suite.train(epochs=5)
    suite.evaluate_keras()
    suite.evaluate_scratch()
    suite.plot_history()
    print("[MAIN] CNNSuite demo finished.")
