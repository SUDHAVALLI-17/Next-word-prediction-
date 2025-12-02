from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import Counter
import editdistance

app = Flask(__name__)

# Load and preprocess dataset
with open("dataset.txt", "r", encoding="utf-8") as f:
    sample_text = f.read().lower()

sample_text = re.sub(r"[^\w\s]", "", sample_text)
words = sample_text.split()
total_tokens = len(words)
unique_tokens = len(set(words))

print(f"\nTotal tokens (words): {total_tokens}")
print(f"Unique tokens: {unique_tokens}")

# Vocabulary
word_counts = Counter(words)
vocab = ["<pad>"] + sorted(word_counts, key=word_counts.get, reverse=True)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(vocab)

# Ask for training ratio
try:
    train_ratio = float(input("Enter training data ratio (between 0 and 1): "))
    if not (0 < train_ratio < 1):
        raise ValueError("Ratio must be between 0 and 1.")
except ValueError as e:
    print(f"Invalid input: {e}")
    exit(1)

# Prepare sequences
seq_length = 4
data = [(words[i:i + seq_length], words[i + seq_length]) for i in range(len(words) - seq_length)]

# Train/test split
split_idx = int(train_ratio * len(data))
train_data = data[:split_idx]
test_data = data[split_idx:]

print(f"\nTotal sequences: {len(data)}")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

# Prepare tensor input
def prepare_sequence(seq, target):
    x = torch.tensor([word_to_ix.get(w, 0) for w in seq], dtype=torch.long)
    y = torch.tensor(word_to_ix.get(target, 0), dtype=torch.long)
    return x, y

# WER calculation
def calculate_wer(reference, hypothesis):
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    distance = editdistance.eval(ref_words, hyp_words)
    return round(distance / len(ref_words), 4) if ref_words else 1.0

# Models
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=10, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        x = self.embed(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1])

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, num_layers=2, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, -1])

# Device
device = torch.device("cpu")

# Instantiate models
models = {
    "lstm": LSTMModel(vocab_size).to(device),
    "gru": GRUModel(vocab_size).to(device),
    "transformer": TransformerModel(vocab_size).to(device)
}

# Train model
def train_model(model, model_name):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    print(f"\n[{model_name.upper()}] Training started:")
    for epoch in range(1, 21):  # 5 epochs
        total_loss = 0
        print(f"\nEpoch {epoch}/20")
        for i, (seq, target) in enumerate(train_data):
            x, y = prepare_sequence(seq, target)
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Print training info
            if i % 100 == 0 or i == len(train_data) - 1:
                predicted = ix_to_word[torch.argmax(out, dim=1).item()]
                print(f"Sample {i+1}/{len(train_data)} | Input: {' '.join(seq)} | Target: {target} | Predicted: {predicted} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_data)
        print(f"\n[{model_name.upper()}] Epoch {epoch} Average Loss: {avg_loss:.4f}")

# Predict word (for WER)
def predict_word(model, seq):
    x = torch.tensor([word_to_ix.get(w, 0) for w in seq], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred_idx = torch.argmax(out, dim=1).item()
    return ix_to_word[pred_idx]

# Train and evaluate all models
print("\nTraining all models...\n")
for name, model in models.items():
    train_model(model, name)

print("\nCalculating WER on test data...")
for name, model in models.items():
    total_wer = 0
    for seq, actual in test_data:
        pred = predict_word(model, seq)
        total_wer += calculate_wer(actual, pred)
    avg_wer = total_wer / len(test_data)
    print(f"{name.upper()} - Avg WER: {avg_wer:.4f}")

# Flask routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").lower().split()
    model_type = data.get("model", "transformer").lower()
    while len(text) < 4:
        text.insert(0, "<pad>")
    seq = text[-4:]
    x = torch.tensor([word_to_ix.get(w, 0) for w in seq], dtype=torch.long).unsqueeze(0).to(device)
    model = models.get(model_type, models["transformer"])
    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        top_preds = torch.topk(probs, 5).indices.squeeze().tolist()
    predictions = [ix_to_word[i] for i in top_preds]
    return jsonify({"predictions": predictions})

if __name__ == "__main__":
    app.run(debug=True)
