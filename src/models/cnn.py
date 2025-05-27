import os
import json
from typing import Union, List, Tuple, Dict

import numpy as np
from dotenv import load_dotenv

# Import ENV to stop warning messages from TensorFlow 
load_dotenv()

from keras import layers, models, losses, optimizers
from keras.callbacks import History
from sklearn.metrics import f1_score

# ------------------------------------------------------------
# CNN Model Wrapper -- stores data and contains Tensoflow model
# ------------------------------------------------------------
class CNNModel:
    """
    Convolutional Neural Network model wrapper.

    Parameters
    ----------
    conv_layers : int, default 2
        Number of convolutional layers.
    filters : int | list[int], default 32
        Number of filters for each convolutional layer.
    kernel_size : int | list[int], default 3
        Size of the kernel for each convolutional layer.
    pooling : str, {"max", "avg"}, default "max"
        Type of pooling layer to use after each convolution.
    """

    # --------------- Constructor --------------- #
    def __init__(
        self,
        conv_layers: int = 2,
        filters: Union[int, List[int]] = 32,
        kernel_size: Union[int, List[int]] = 3,
        pooling: str = "max",
    ) -> None:
        if isinstance(filters, int):
            filters = [filters] * conv_layers
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * conv_layers

        self.conv_layers = conv_layers
        self.filters = filters
        self.kernel_size = kernel_size
        self.pooling = pooling.lower()
        self.model: models.Model = self._build_model()

    # --------------- Model definition --------------- #
    def _build_model(self) -> models.Model:
        """ Create a CNN architecture using TensorFlow. """
        model = models.Sequential(name="cnn_if3270")
        model.add(layers.Input(shape=(32, 32, 3)))

        for i in range(self.conv_layers):
            model.add(
                layers.Conv2D(
                    self.filters[i],
                    self.kernel_size[i],
                    activation="relu",
                    padding="same",
                    name=f"conv_{i+1}",
                )
            )
            if self.pooling == "max":
                model.add(layers.MaxPooling2D(name=f"maxpool_{i+1}"))
            else:
                model.add(layers.AveragePooling2D(name=f"avgpool_{i+1}"))

        model.add(layers.Flatten(name="flatten"))
        model.add(layers.Dense(128, activation="relu", name="dense_128"))
        model.add(layers.Dense(10, activation="softmax", name="logits"))
        return model

    # --------------- Compilation / Training / Evaluation --------------- #
    def compile(self) -> None:
        """ Configure the model. """
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=optimizers.Adam(),
            metrics=["accuracy"],
        )

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 2,
        batch_size: int = 64,
    ) -> History:
        """ Train the model. """
        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray) -> float:
        """ Evaluate the model using macro-F1 score. """
        y_pred_logits = self.model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_logits, axis=1)
        f1: float = f1_score(y_test, y_pred, average="macro")
        print("Test Macro F1 Score:", f1)
        return f1

    # --------------- Save / Load --------------- #
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

    # --------------- Save / Load History --------------- #
    def save_history(self, history: Dict[str, List[float]], filename: str = "cnn_history.json") -> str:
        """
        Save training history (loss, val_loss, etc.) to JSON file.
        """
        save_dir = os.path.join(os.getcwd(), "saved_models")
        os.makedirs(save_dir, exist_ok=True)
        full_path = os.path.join(save_dir, filename)
        with open(full_path, "w") as f:
            json.dump(history, f)
        print(f"Training history saved to {full_path}")
        return full_path

    def load_history(self, filename: str = "cnn_history.json") -> Dict[str, List[float]]:
        """
        Load training history from JSON file.
        """
        full_path = os.path.join(os.getcwd(), "saved_models", filename)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"No history file found at {full_path}")
        with open(full_path, "r") as f:
            history = json.load(f)
        print(f"Training history loaded from {full_path}")
        return history


class TrainCNN:
    """ Utility class for dataset handling and model training."""

    # --------------- Constructor --------------- #
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    # --------------- Run Model --------------- #
    def run(
        self,
        save_name: str = "cnn.keras",
        history_name: str = "cnn_history.json",
        epochs: int = 2,
        batch_size: int = 64,
    ) -> Tuple[float, Dict[str, List[float]]]:
        """ Build, train, save, and evaluate the CNN, and save history to file. """
        model = CNNModel()
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

    # Load CIFAR-10 dataset
    (x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

    # Normalize images to [0,1]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Flatten labels (from shape (n,1) to (n,))
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    # Split into train and validation sets (4:1 ratio)
    val_size = int(len(x_train_full) * 0.2)
    x_val = x_train_full[:val_size]
    y_val = y_train_full[:val_size]
    x_train = x_train_full[val_size:]
    y_train = y_train_full[val_size:]

    print("Train shape:", x_train.shape, y_train.shape)
    print("Validation shape:", x_val.shape, y_val.shape)
    print("Test shape:", x_test.shape, y_test.shape)

    trainer = TrainCNN(x_train, y_train, x_val, y_val, x_test, y_test)
    f1_score, history = trainer.run(epochs=5, batch_size=64)

    print(f"Final Macro F1 Score on test set: {f1_score:.4f}")

    try:
        from src.utils.plot_utils import plot_loss
        plot_loss(history)
    except ImportError:
        print("matplotlib not installed or plot_utils missing, skipping loss plot.")
