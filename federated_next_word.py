# ...existing code...
"""
Federated Next Word Prediction (Client-Specific Predictions)
Author: GPT-5

Each client trains on its own text data locally.
After training, you can choose a client and get predictions
that reflect that client's own local model.
"""

import os
import random
import numpy as np
import tensorflow as tf
import flwr as fl

# Reduce TensorFlow logging noise (warnings about retracing)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

# ----------------------------------
# Define Client Datasets
# ----------------------------------
CLIENT_TEXTS = {
    "client_1": [
        "hello how are you doing today",
        "i am happy to see you again",
        "hope you have a good time",
        "what are your plans for the day",
    ],
    "client_2": [
        "good morning have a wonderful day",
        "have a nice breakfast and enjoy your day",
        "good evening hope you are doing great",
        "take care and stay safe",
    ],
    "client_3": [
        "see you later my friend",
        "looking forward to meeting you soon",
        "have a safe journey and take care",
        "hope to catch up with you tomorrow",
    ],
}

# ----------------------------------
# Build Vocabulary
# ----------------------------------
def build_vocab(client_texts):
    tokens = set()
    for texts in client_texts.values():
        for s in texts:
            tokens.update(s.split())
    vocab = ["<pad>", "<unk>"] + sorted(tokens)
    idx = {w: i for i, w in enumerate(vocab)}
    return vocab, idx

VOCAB, IDX = build_vocab(CLIENT_TEXTS)
VOCAB_SIZE = len(VOCAB)
SEQ_LEN = 3


# ----------------------------------
# Convert text → sequences
# ----------------------------------
def texts_to_sequences(texts, idx_map, seq_len=SEQ_LEN):
    X, y = [], []
    for s in texts:
        toks = s.split()
        for i in range(len(toks) - seq_len):
            context = toks[i : i + seq_len]
            target = toks[i + seq_len]
            X.append([idx_map.get(t, idx_map["<unk>"]) for t in context])
            y.append(idx_map.get(target, idx_map["<unk>"]))
    if len(X) == 0:
        # return empty arrays with correct shapes to avoid crashes
        return np.zeros((0, seq_len), dtype=np.int32), np.zeros((0,), dtype=np.int32)
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# ----------------------------------
# 4️⃣ Define model
# ----------------------------------
def create_model(vocab_size=VOCAB_SIZE, embed_dim=64, lstm_units=128):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True, input_length=SEQ_LEN),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------
# Flower Client Class
# ----------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, model, x_train, y_train, x_val, y_val):
        self.cid = cid
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def get_parameters(self):
        return [v.numpy() for v in self.model.weights]

    def set_parameters(self, parameters):
        for var, p in zip(self.model.weights, parameters):
            var.assign(tf.convert_to_tensor(p))

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if len(self.x_train) > 0:
            self.model.fit(self.x_train, self.y_train, epochs=25, batch_size=4, verbose=0)
        print(f"[{self.cid}] local training complete.")
        return self.get_parameters(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        if len(self.x_val) == 0:
            return float("nan"), 0, {"accuracy": float("nan")}
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=0)
        print(f"[{self.cid}] val_acc = {acc:.4f}")
        return float(loss), len(self.x_val), {"accuracy": float(acc)}


# ----------------------------------
# Make Client Instance
# ----------------------------------
def make_client(cid):
    texts = CLIENT_TEXTS[cid].copy()
    random.shuffle(texts)
    split = int(len(texts) * 0.75)
    train_texts = texts[:split]
    val_texts = texts[split:]
    x_train, y_train = texts_to_sequences(train_texts, IDX)
    x_val, y_val = texts_to_sequences(val_texts, IDX)
    model = create_model()
    return FlowerClient(cid, model, x_train, y_train, x_val, y_val)


# ----------------------------------
# Federated Training Simulation
# ----------------------------------
def main():
    client_ids = list(CLIENT_TEXTS.keys())

    def client_fn(cid: str):
        return make_client(cid)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=len(client_ids),
        min_available_clients=len(client_ids),
        initial_parameters=fl.common.ndarrays_to_parameters([v.numpy() for v in create_model().weights]),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=8),
        strategy=strategy,
    )

    print("\n=== Federated Training Complete ===")

    # Cache per-client models to avoid recreating/compiling them repeatedly
    client_cache = {}

    # Pre-create and locally fine-tune each client's model once,
    # then "warm up" predict function to avoid TF retracing messages.
    for cid in client_ids:
        client = make_client(cid)
        if len(client.x_train) > 0:
            client.model.fit(client.x_train, client.y_train, epochs=25, batch_size=4, verbose=0)
        # Warm up prediction function with a dummy input of the correct shape
        try:
            dummy = np.zeros((1, SEQ_LEN), dtype=np.int32)
            client.model.predict(dummy, verbose=0)
        except Exception:
            pass
        client_cache[cid] = client

    # Interactive loop: reuse cached client models for prediction (no repeated compilation)
    while True:
        cid = input("\nSelect a client to test (client_1/client_2/client_3 or 'exit'): ").strip()
        if cid.lower() == "exit":
            break
        if cid not in client_cache:
            print("Invalid client id.")
            continue

        client = client_cache[cid]

        context = input(f"Enter 3 words for {cid} context: ").strip().split()
        if len(context) != 3:
            print("❌ Please enter exactly 3 words.")
            continue

        x = np.array([[IDX.get(t, IDX["<unk>"]) for t in context]], dtype=np.int32)
        preds = client.model.predict(x, verbose=0)[0]
        topk = np.argsort(preds)[-5:][::-1]
        print(f"Top predicted next words for {cid}: {[VOCAB[i] for i in topk]}")


if __name__ == "__main__":
    main()

