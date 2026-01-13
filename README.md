
# Federated Next Word Prediction

A federated learning implementation for next-word prediction using TensorFlow and Flower framework. Each client trains on its own text data locally, and predictions reflect individual client models.

## Features

- **Federated Learning**: Clients train locally without sharing raw text data
- **Client-Specific Predictions**: Each client maintains its own trained model
- **LSTM-based Architecture**: Uses embedding + LSTM layers for sequence modeling
- **Interactive Testing**: Choose a client and get top-5 next word predictions
- **Optimized Performance**: Caches models to avoid TensorFlow retracing warnings

## Project Structure

```
federated_next_word.py
├── Client Datasets (CLIENT_TEXTS)
├── Vocabulary Building
├── Text-to-Sequence Conversion
├── LSTM Model Definition
├── Flower Client Class (FlowerClient)
├── Client Instance Factory (make_client)
└── Federated Training Simulation (main)
```

## Installation

```bash
pip install tensorflow flwr numpy
```

## Usage

Run the script:

```bash
python federated_next_word.py
```

### Interactive Mode

After federated training completes:

1. Select a client: `client_1`, `client_2`, or `client_3`
2. Enter 3 words as context (space-separated)
3. Get top 5 predicted next words for that client's model
4. Type `exit` to quit

**Example:**
```
Select a client to test: client_3
Enter 3 words for client_3 context: good evening hope
Top predicted next words for client_3: ['you', 'with', 'soon', 'meeting', 'up']
```

## Model Architecture

- **Input**: 3-word sequence (SEQ_LEN=3)
- **Embedding**: 64-dimensional embeddings
- **LSTM**: 128 hidden units with dropout (0.3)
- **Output**: Softmax over vocabulary (~50 tokens)

## Training Details

- **Framework**: Flower (Federated Learning)
- **Strategy**: FedAvg (Federated Averaging)
- **Rounds**: 8 communication rounds
- **Local Epochs**: 25 per client per round
- **Batch Size**: 4

## Dataset

Three clients with domain-specific text:
- **client_1**: Friendly greetings & plans
- **client_2**: Morning/evening wishes & care phrases
- **client_3**: Farewell & meeting arrangements

## Performance Notes

- Model caching prevents TensorFlow retracing warnings
- Dummy input "warm-up" stabilizes predictions
- Verbose logging disabled for clean output


