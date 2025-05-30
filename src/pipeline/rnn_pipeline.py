import pandas as pd
import numpy as np
import tensorflow as tf
import json
from typing import Dict, List, Tuple
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from src.models.rnn import TrainRNN
from src.scratch.rnn_forward import ScratchRNN
from src.utils.plot_utils import plot_loss
from sklearn.metrics import f1_score

# === CONSTANTS ===
MAX_VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 128
EMBEDDING_DIM = 128
RNN_UNITS = 72
RNN_LAYERS = 2
DROPOUT_RATE = 0.5
DENSE_LAYERS = [64, 32]
DENSE_ACTIVATIONS = ["relu", "relu"]
NUM_CLASSES = 3
BIDIRECTIONAL = True
BATCH_SIZE = 64
EPOCHS = 50

class RNNSuite:
    """Combine All RNN functionality."""
    def __init__(
        self, model_name: str = "base_rnn.keras", history_name: str = "base_rnn_history.json"
    ) -> None:
        print(f"[INIT] Initializing CNNSuite with model path '{"saved_models"}/{model_name}'")
        self.save_dir = Path("saved_models")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.history_name = history_name
        self.model_path = self.save_dir / model_name
        self.history_path = self.save_dir / history_name
        self.data: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.history: Dict[str, List[float]] | None = None

    # === DATA LOADING ===
    def load_dataset(self):
        base_url = "https://raw.githubusercontent.com/IndoNLP/nusax/main/datasets/sentiment/indonesian/"
        df_train = pd.read_csv(base_url + "train.csv")
        df_val   = pd.read_csv(base_url + "valid.csv")
        df_test  = pd.read_csv(base_url + "test.csv")
        return df_train, df_val, df_test

    # === TEXT ENCODING ===
    def preprocess_text(self, df_train, df_val, df_test):
        vectorizer = tf.keras.layers.TextVectorization(
            max_tokens=MAX_VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=SEQUENCE_LENGTH,
            standardize="lower_and_strip_punctuation",
            split="whitespace"
        )
        vectorizer.adapt(df_train["text"].values)

        # # Save vectorizer
        # with open(VECTORIZER_PATH, "wb") as f:
        #     pickle.dump(vectorizer, f)

        # Transform text to sequences
        x_train = vectorizer(df_train["text"].values).numpy()
        x_val   = vectorizer(df_val["text"].values).numpy()
        x_test  = vectorizer(df_test["text"].values).numpy()
        return vectorizer, x_train, x_val, x_test

    # === LABEL ENCODING ===
    def encode_labels(self, df_train, df_val, df_test):
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(df_train["label"])
        y_val   = label_encoder.transform(df_val["label"])
        y_test  = label_encoder.transform(df_test["label"])
        return label_encoder, y_train, y_val, y_test

    # === TRAINING ===
    def train_rnn(self, x_train, y_train, x_val, y_val, x_test, y_test, vocab_size):
        trainer = TrainRNN(x_train, y_train, x_val, y_val, x_test, y_test)
        f1, history = trainer.run(
            vocab_size=vocab_size,
            input_length=SEQUENCE_LENGTH,
            embedding_dim=EMBEDDING_DIM,
            rnn_units=RNN_UNITS,
            rnn_layers=RNN_LAYERS,
            bidirectional=BIDIRECTIONAL,
            dropout_rate=DROPOUT_RATE,
            dense_layers=DENSE_LAYERS,
            dense_activations=DENSE_ACTIVATIONS,
            num_classes=NUM_CLASSES,
            save_name=self.model_name,
            history_name=self.history_name,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE
        )
        print(f"[DONE] Final Macro F1 Score (Keras RNN): {f1:.6f}")
        try:
            plot_loss(history)
        except:
            print("[WARN] Plotting failed (matplotlib or utils missing)")
        return f1

    # === SCRATCH INFERENCE ===
    def evaluate_scratch_model(self, x_test, y_test):
        scratch_model = ScratchRNN(self.model_path)
        y_pred = scratch_model.predict(x_test, batch_size=BATCH_SIZE)
        score = f1_score(y_test, y_pred, average="macro")
        print(f"[DONE] Final Macro F1 Score (ScratchRNN): {score:.6f}")
        return score

    # === SCRATCH INFERENCE ===
    def load_history(self) -> Dict[str, List[float]]:
        if self.history is not None:
            return self.history
        if not self.history_path.exists():
            raise FileNotFoundError(self.history_path)
        with open(self.history_path, "r") as f:
            self.history = json.load(f)
        return self.history

    # === SCRATCH INFERENCE ===
    def plot_history(self) -> None:
        history = self.load_history()
        plot_loss(history)
        print("[DONE] Plot displayed.")

    # === PIPELINE EXECUTION ===
    def main(self):
        print("[INFO] Loading datasets...")
        df_train, df_val, df_test = self.load_dataset()

        print("[INFO] Preprocessing text...")
        vectorizer, x_train, x_val, x_test = self.preprocess_text(df_train, df_val, df_test)

        print("[INFO] Encoding labels...")
        label_encoder, y_train, y_val, y_test = self.encode_labels(df_train, df_val, df_test)

        vocab_size = len(vectorizer.get_vocabulary())
        print(f"[INFO] Vocab size: {vocab_size}")

        print("[INFO] Training Keras RNN model...")
        self.train_rnn(x_train, y_train, x_val, y_val, x_test, y_test, vocab_size)

        print("[INFO] Running ScratchRNN forward pass...")
        self.evaluate_scratch_model(x_test, y_test)

        print("[START] Plotting training history...")
        self.plot_history

if __name__ == "__main__":
    print("[MAIN] RNNSuite demo started.")
    suite = RNNSuite("test_rnn.keras", "test_rnn_history.json")
    suite.main()
    print("[MAIN] RNNSuite demo finished.")
