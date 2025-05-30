# lstm.py
import os
import json
import numpy as np
from typing import Any, List, Tuple, Dict, Union
from keras import layers, models, losses, optimizers
from keras.callbacks import History
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LSTMModel:
    def __init__(
        self,
        vocab_size: int,
        sequence_length: int = 100,
        embedding_dim: int = 128,
        lstm_layers: int = 1,
        lstm_units: Union[int, List[int]] = 64,
        bidirectional: bool = False,
        dense_units: List[int] = [128],
        dense_activations: List[str] = ["relu"],
    ):
        if isinstance(lstm_units, int):
            lstm_units = [lstm_units] * lstm_layers

        self.model = self._build_model(
            vocab_size,
            sequence_length,
            embedding_dim,
            lstm_layers,
            lstm_units,
            bidirectional,
            dense_units,
            dense_activations
        )

    def _build_model(
        self, vocab_size, seq_len, embed_dim, lstm_layers, lstm_units,
        bidirectional, dense_units, dense_activations
    ):
        model = models.Sequential(name="lstm_nusax")
        model.add(layers.Input(shape=(seq_len,), name="input"))
        model.add(layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, name="embedding"))

        for i in range(len(lstm_units)):
            return_seq = i < len(lstm_units) - 1
            lstm_layer = layers.LSTM(lstm_units[i], return_sequences=return_seq, name=f"lstm_{i+1}")
            if bidirectional:
                lstm_layer = layers.Bidirectional(lstm_layer, name=f"bilstm_{i+1}")
            model.add(lstm_layer)

        model.add(layers.Dropout(0.5, name="dropout"))

        for idx, (units, act) in enumerate(zip(dense_units, dense_activations)):
            model.add(layers.Dense(units, activation=act, name=f"dense_{idx+1}"))

        model.add(layers.Dense(3, activation="softmax", name="output"))  # 3 kelas
        return model

    def compile(self):
        self.model.compile(
            loss=losses.SparseCategoricalCrossentropy(),
            optimizer=optimizers.Adam(),
            metrics=["accuracy"],
        )

    def train(
        self, x_train, y_train, x_val, y_val,
        epochs: int = 5,
        batch_size: int = 32
    ) -> History:
        return self.model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    def evaluate(self, x_test, y_test) -> float:
        y_pred_prob = self.model.predict(x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        f1 = f1_score(y_test, y_pred, average="macro")
        print("Test Macro F1 Score:", f1)
        return f1

    def save(self, name="lstm_model.keras") -> str:
        path = os.path.join("saved_models", name)
        os.makedirs("saved_models", exist_ok=True)
        self.model.save(path)
        return path

    def save_history(self, history, name="lstm_history.json") -> str:
        path = os.path.join("saved_models", name)
        with open(path, "w") as f:
            json.dump(history, f)
        return path


# def load_nusax_sentiment(path: str) -> Tuple[np.ndarray, np.ndarray]:
#     df = pd.read_csv(path, sep="\t")
#     texts = df["text"].astype(str).values
#     labels = df["label"].values
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)
#     return texts, labels

# def run_training(path_to_data: str):
#     texts, labels = load_nusax_sentiment(path_to_data)

#     # Vectorization
#     max_tokens = 10000
#     max_len = 100
#     vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len)
#     vectorizer.adapt(texts)
#     X = vectorizer(texts)
#     y = np.array(labels)

#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     # Build and train model
#     model = LSTMModel(vocab_size=max_tokens)
#     model.compile()
#     history = model.train(X_train, y_train, X_val, y_val, epochs=5)
#     f1 = model.evaluate(X_test, y_test)
#     model.save()
#     model.save_history(history.history)

#     return f1, history.history

# if __name__ == "__main__":
#     # Ganti dengan path file TSV dari NusaX-Sentiment
#     run_training("indonesia.tsv")