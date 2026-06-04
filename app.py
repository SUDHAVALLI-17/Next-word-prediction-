from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import math
import os
import threading
from collections import Counter
import editdistance

app = Flask(__name__)

# ─────────────────────────────────────────────
# Load & preprocess dataset
# ─────────────────────────────────────────────
with open("dataset.txt", "r", encoding="utf-8") as f:
    sample_text = f.read().lower()

sample_text = re.sub(r"[^\w\s]", "", sample_text)
words = sample_text.split()
total_tokens = len(words)
unique_tokens = len(set(words))

print(f"\nTotal tokens (words): {total_tokens}")
print(f"Unique tokens:        {unique_tokens}")

# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────
word_counts = Counter(words)
vocab = ["<pad>"] + sorted(word_counts, key=word_counts.get, reverse=True)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for word, i in word_to_ix.items()}
vocab_size = len(vocab)

# ─────────────────────────────────────────────
# Train/test split
# ─────────────────────────────────────────────
train_ratio = 0.8


seq_length = 4
data = [(words[i:i + seq_length], words[i + seq_length]) for i in range(len(words) - seq_length)]
split_idx = int(train_ratio * len(data))
train_data = data[:split_idx]
test_data  = data[split_idx:]

print(f"\nTotal sequences:   {len(data)}")
print(f"Training samples:  {len(train_data)}")
print(f"Testing  samples:  {len(test_data)}")

device = torch.device("cpu")

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def prepare_sequence(seq, target):
    x = torch.tensor([word_to_ix.get(w, 0) for w in seq], dtype=torch.long)
    y = torch.tensor(word_to_ix.get(target, 0), dtype=torch.long)
    return x, y

def calculate_wer(reference, hypothesis):
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    dist = editdistance.eval(ref_words, hyp_words)
    return round(dist / len(ref_words), 4) if ref_words else 1.0


# ══════════════════════════════════════════════════════════════════
#  FROM-SCRATCH LSTM
#  Gates computed manually; no nn.LSTM used.
# ══════════════════════════════════════════════════════════════════
class LSTMCell(nn.Module):
    """Single LSTM cell built from linear layers (no nn.LSTM)."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Combined weight matrix for all 4 gates: [i, f, g, o]
        self.W_ih = nn.Parameter(torch.randn(4 * hidden_dim, input_dim) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(4 * hidden_dim, hidden_dim) * 0.01)
        self.b    = nn.Parameter(torch.zeros(4 * hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        # x:      (batch, input_dim)
        # h_prev: (batch, hidden_dim)
        gates = x @ self.W_ih.t() + h_prev @ self.W_hh.t() + self.b   # (batch, 4*H)

        H = self.hidden_dim
        i_gate = torch.sigmoid(gates[:, 0*H : 1*H])   # input gate
        f_gate = torch.sigmoid(gates[:, 1*H : 2*H])   # forget gate
        g_gate = torch.tanh   (gates[:, 2*H : 3*H])   # cell gate
        o_gate = torch.sigmoid(gates[:, 3*H : 4*H])   # output gate

        c = f_gate * c_prev + i_gate * g_gate          # cell state
        h = o_gate * torch.tanh(c)                     # hidden state
        return h, c


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.lstm_cell  = LSTMCell(embed_dim, hidden_dim)
        self.fc         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch = x.size(0)
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, device=x.device)
        emb = self.embed(x)                             # (batch, seq, embed)
        for t in range(emb.size(1)):
            h, c = self.lstm_cell(emb[:, t, :], h, c)
        return self.fc(h)                               # (batch, vocab_size)


# ══════════════════════════════════════════════════════════════════
#  FROM-SCRATCH GRU
#  Gates computed manually; no nn.GRU used.
# ══════════════════════════════════════════════════════════════════
class GRUCell(nn.Module):
    """Single GRU cell built from linear layers (no nn.GRU)."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Reset & update gate weights
        self.W_r = nn.Parameter(torch.randn(hidden_dim, input_dim  + hidden_dim) * 0.01)
        self.b_r = nn.Parameter(torch.zeros(hidden_dim))
        self.W_z = nn.Parameter(torch.randn(hidden_dim, input_dim  + hidden_dim) * 0.01)
        self.b_z = nn.Parameter(torch.zeros(hidden_dim))
        # Candidate hidden state weights
        self.W_n = nn.Parameter(torch.randn(hidden_dim, input_dim  + hidden_dim) * 0.01)
        self.b_n = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h_prev):
        # x:      (batch, input_dim)
        # h_prev: (batch, hidden_dim)
        xh = torch.cat([x, h_prev], dim=1)             # (batch, input+hidden)

        r = torch.sigmoid(xh @ self.W_r.t() + self.b_r)   # reset gate
        z = torch.sigmoid(xh @ self.W_z.t() + self.b_z)   # update gate

        xrh = torch.cat([x, r * h_prev], dim=1)
        n   = torch.tanh(xrh @ self.W_n.t() + self.b_n)   # candidate

        h   = (1 - z) * n + z * h_prev                    # new hidden
        return h


class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed      = nn.Embedding(vocab_size, embed_dim)
        self.gru_cell   = GRUCell(embed_dim, hidden_dim)
        self.fc         = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch = x.size(0)
        h = torch.zeros(batch, self.hidden_dim, device=x.device)
        emb = self.embed(x)
        for t in range(emb.size(1)):
            h = self.gru_cell(emb[:, t, :], h)
        return self.fc(h)


# ══════════════════════════════════════════════════════════════════
#  FROM-SCRATCH TRANSFORMER
#  Multi-head self-attention and feed-forward built manually.
#  No nn.TransformerEncoderLayer / nn.MultiheadAttention used.
# ══════════════════════════════════════════════════════════════════
class ScaledDotProductAttention(nn.Module):
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V"""
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, heads, seq, seq)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn   = F.softmax(scores, dim=-1)
        return attn @ V                                       # (B, heads, seq, d_k)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = embed_dim // num_heads

        # Projection matrices (no nn.MultiheadAttention)
        self.W_Q = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.W_K = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.W_V = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.W_O = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.b_Q = nn.Parameter(torch.zeros(embed_dim))
        self.b_K = nn.Parameter(torch.zeros(embed_dim))
        self.b_V = nn.Parameter(torch.zeros(embed_dim))
        self.b_O = nn.Parameter(torch.zeros(embed_dim))

        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch):
        # x: (B, seq, embed) → (B, heads, seq, d_k)
        return x.view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B, S, E = x.shape

        Q = x @ self.W_Q.t() + self.b_Q   # (B, S, E)
        K = x @ self.W_K.t() + self.b_K
        V = x @ self.W_V.t() + self.b_V

        Q = self.split_heads(Q, B)         # (B, heads, S, d_k)
        K = self.split_heads(K, B)
        V = self.split_heads(V, B)

        out = self.attention(Q, K, V)      # (B, heads, S, d_k)
        out = out.transpose(1, 2).contiguous().view(B, S, E)   # (B, S, E)
        out = out @ self.W_O.t() + self.b_O
        return out


class FeedForward(nn.Module):
    """Position-wise FFN: two linear layers with ReLU (built manually)."""
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, embed_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(embed_dim, hidden_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, x):
        # x: (B, S, E)
        h = F.relu(x @ self.W1.t() + self.b1)   # (B, S, hidden)
        return h @ self.W2.t() + self.b2          # (B, S, embed)


class LayerNorm(nn.Module):
    """Layer normalisation built from scratch (no nn.LayerNorm)."""
    def __init__(self, embed_dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta  = nn.Parameter(torch.zeros(embed_dim))
        self.eps   = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std  = x.std (dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class TransformerEncoderBlock(nn.Module):
    """One encoder block: MHSA + residual + LN + FFN + residual + LN"""
    def __init__(self, embed_dim, num_heads, ff_hidden):
        super().__init__()
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ff    = FeedForward(embed_dim, ff_hidden)
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4,
                 ff_hidden=128, num_layers=2, max_len=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Sinusoidal positional encoding (no learnable PE shortcut)
        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() *
                        (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))   # (1, max_len, embed)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ff_hidden)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x) + self.pe[:, :x.size(1), :]
        for layer in self.layers:
            emb = layer(emb)
        return self.fc(emb[:, -1, :])   # use last token's output


# ─────────────────────────────────────────────
# Instantiate models
# ─────────────────────────────────────────────
models = {
    "lstm":        LSTMModel       (vocab_size).to(device),
    "gru":         GRUModel        (vocab_size).to(device),
    "transformer": TransformerModel(vocab_size).to(device),
}


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_model(model, model_name, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()
    print(f"\n[{model_name.upper()}] Training started")

    for epoch in range(1, epochs + 1):
        total_loss = 0
        print(f"\nEpoch {epoch}/{epochs}")
        for i, (seq, target) in enumerate(train_data):
            x, y = prepare_sequence(seq, target)
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            optimizer.zero_grad()
            out  = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0 or i == len(train_data) - 1:
                predicted = ix_to_word[torch.argmax(out, dim=1).item()]
                print(f"  [{i+1}/{len(train_data)}] "
                      f"Input: {' '.join(seq)} | "
                      f"Target: {target} | "
                      f"Pred: {predicted} | "
                      f"Loss: {loss.item():.4f}")

        print(f"[{model_name.upper()}] Epoch {epoch} Avg Loss: {total_loss/len(train_data):.4f}")


def predict_word(model, seq):
    x = torch.tensor([word_to_ix.get(w, 0) for w in seq],
                      dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out      = model(x)
        pred_idx = torch.argmax(out, dim=1).item()
    return ix_to_word[pred_idx]


# ─────────────────────────────────────────────
# Background training
# ─────────────────────────────────────────────
def train_all():
    print("\nTraining all models...\n")
    for name, model in models.items():
        train_model(model, name)

    print("\nCalculating WER on test data...")
    for name, model in models.items():
        total_wer = 0
        for seq, actual in test_data:
            pred = predict_word(model, seq)
            total_wer += calculate_wer(actual, pred)
        avg_wer = total_wer / len(test_data) if test_data else 0
        print(f"  {name.upper()} — Avg WER: {avg_wer:.4f}")

t = threading.Thread(target=train_all)
t.daemon = True
t.start()

# ─────────────────────────────────────────────
# Flask routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json()
    text       = data.get("text", "").lower().split()
    model_type = data.get("model", "transformer").lower()

    while len(text) < 4:
        text.insert(0, "<pad>")
    seq = text[-4:]

    x     = torch.tensor([word_to_ix.get(w, 0) for w in seq],
                          dtype=torch.long).unsqueeze(0).to(device)
    model = models.get(model_type, models["transformer"])

    with torch.no_grad():
        out      = model(x)
        probs    = F.softmax(out, dim=1)
        top_preds = torch.topk(probs, 5).indices.squeeze().tolist()

    predictions = [ix_to_word[i] for i in top_preds]
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=True,
        use_reloader=False
    )