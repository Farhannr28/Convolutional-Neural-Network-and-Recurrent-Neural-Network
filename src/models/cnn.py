from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from keras import layers, losses, models, optimizers
from keras.callbacks import History
from sklearn.metrics import f1_score


# ------------------------------------------------------------
# CNN Model Wrapper -- stores data and contains TensorFlow model
# ------------------------------------------------------------
class CNNModel:
    """
    Convolutional Neural Network model wrapper.
    """

    def __init__(
        self,
        conv_layers: int = 2,
        filters: Union[int, List[int]] = 32,
        kernel_size: Union[int, List[int]] = 3,
        pooling: str = "max",
        pool_size: int = 2,  
        conv_activation: str = "relu",
        dense_layers: List[int] | None = None,
        dense_activations: List[str] | None = None,
    ) -> None:
        if isinstance(filters, int):
            filters = [filters] * conv_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * conv_layers

        dense_layers = dense_layers or [128]
        dense_activations = dense_activations or ["relu"]

        if len(dense_layers) != len(dense_activations):
            raise ValueError(
                "dense_layers and dense_activations must be the same length"
            )

        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.pooling = pooling.lower()
        self.pool_size = pool_size
        self.conv_activation = conv_activation
        self.dense_layers = dense_layers
        self.dense_activations = dense_activations

        self.model: models.Model = self._build_model()

    def _build_model(self) -> models.Model:
        model = models.Sequential(name="cnn_if3270")
        model.add(layers.Input(shape=(32, 32, 3)))

        for i in range(self.conv_layers):
            model.add(
                layers.Conv2D(
                    self.filters[i],
                    self.kernel_size[i],
                    activation=self.conv_activation,
                    padding="same",
                    name=f"conv_{i+1}",
                )
            )
            if self.pooling == "max":
                model.add(layers.MaxPooling2D(pool_size=self.pool_size, name=f"maxpool_{i+1}"))
            else:
                model.add(layers.AveragePooling2D(pool_size=self.pool_size, name=f"avgpool_{i+1}"))  

        model.add(layers.Flatten(name="flatten"))

        for idx, (units, activation) in enumerate(
            zip(self.dense_layers, self.dense_activations)
        ):
            model.add(layers.Dense(units, activation=activation, name=f"dense_{idx+1}"))

        model.add(layers.Dense(10, activation="softmax", name="logits"))
        return model

    def compile(self) -> None:
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=optimizers.Adam(),
            metrics=["accuracy"],
        )

    def train(
        self,
        x_train: npt.NDArray[Any],
        y_train: npt.NDArray[Any],
        x_val: npt.NDArray[Any],
        y_val: npt.NDArray[Any],
        epochs: int = 2,
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

    def save(self, filename: str = "cnn.keras") -> str:
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        self.model.save(full_path)
        print(f"Model saved to {full_path}")
        return full_path

    def load(self, filename: str = "cnn.keras") -> None:
        full_path = os.path.join(os.getcwd(), "saved_models", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No saved model found at {full_path}")
        self.model = models.load_model(full_path, compile=False)
        print(f"Model loaded from {full_path}")

    def save_history(
        self, history: Dict[str, List[float]], filename: str = "cnn_history.json"
    ) -> str:
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        with open(full_path, "w") as f:
            json.dump(history, f)
        print(f"Training history saved to {full_path}")
        return full_path

    def load_history(
        self, filename: str = "cnn_history.json"
    ) -> Dict[str, List[float]]:
        full_path = os.path.join(os.getcwd(), "saved_models", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No history file found at {full_path}")
        with open(full_path, "r") as f:
            history: Dict[str, List[float]] = json.load(f)
        print(f"Training history loaded from {full_path}")
        return history


# ------------------------------------------------------------
# Class to help train CNN Model
# ------------------------------------------------------------
class TrainCNN:
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
        save_name: str = "cnn.keras",
        history_name: str = "cnn_history.json",
        epochs: int = 2,
        batch_size: int = 64,
        conv_layers: int = 2,
        filters: Union[int, List[int]] = 32,
        kernel_size: Union[int, List[int]] = 3,
        pooling: str = "max",
        pool_size: int = 2,
        conv_activation: str = "relu",
        dense_layers: List[int] | None = None,
        dense_activations: List[str] | None = None,
    ) -> Tuple[float, Dict[str, List[float]]]:
        model = CNNModel(
            conv_layers=conv_layers,
            filters=filters,
            kernel_size=kernel_size,
            pooling=pooling,
            pool_size=pool_size, 
            conv_activation=conv_activation,
            dense_layers=dense_layers or [128],
            dense_activations=dense_activations or ["relu"],
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
    from keras.datasets import cifar10

    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    val_size = int(len(x_train_full) * 0.2)
    x_val, x_train = x_train_full[:val_size], x_train_full[val_size:]
    y_val, y_train = y_train_full[:val_size], y_train_full[val_size:]

    trainer = TrainCNN(x_train, y_train, x_val, y_val, x_test, y_test)

    f1, history = trainer.run(
        epochs=5,
        batch_size=64,
        conv_layers=2,
        filters=[32, 64],
        kernel_size=[3, 3],
        pooling="max",
        conv_activation="relu",
        dense_layers=[256, 128],
        dense_activations=["relu", "relu"],
    )

    print(f"Final Macro F1 Score on test set: {f1:.4f}")

    try:
        from src.utils.plot_utils import plot_loss

        plot_loss(history)
    except ImportError:
        print("matplotlib not installed or plot_utils missing, skipping loss plot.")
