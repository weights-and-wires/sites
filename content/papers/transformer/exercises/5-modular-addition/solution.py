import torch
import torch.nn as nn
import torch.nn.functional as F

P = 17  # modulus
EQ = P  # the id of the "=" token; the vocabulary has P + 1 ids


class Block(nn.Module):
    """One pre-LN Transformer block: x + Attn(LN(x)), then x + FFN(LN(x)).

    forward(x): (batch, seq, d_model) -> (batch, seq, d_model). No causal mask.
    """

    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)  # W^O
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def attn(self, x):
        b, n, d = x.shape
        h = self.n_heads
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, h, d // h).transpose(1, 2) for t in (q, k, v))  # (b, h, n, d_k)
        w = torch.softmax(q @ k.transpose(-2, -1) / (d // h) ** 0.5, dim=-1)
        return self.proj((w @ v).transpose(1, 2).reshape(b, n, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff2(F.relu(self.ff1(self.ln2(x))))


class TinyTransformer(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.tok = nn.Embedding(P + 1, d_model)
        self.pos = nn.Embedding(3, d_model)  # learned positions
        self.block = Block(d_model, n_heads)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, P)

    def forward(self, tokens):  # tokens: (batch, 3)
        x = self.tok(tokens) + self.pos(torch.arange(tokens.shape[1]))
        x = self.block(x)
        return self.head(self.ln_f(x[:, -1]))  # predict from the "=" position


def make_data(seed=0, train_frac=0.9):
    """All P * P pairs as [a, b, =] with target (a + b) mod P, split at random."""
    a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij")
    a, b = a.reshape(-1), b.reshape(-1)
    tokens = torch.stack([a, b, torch.full_like(a, EQ)], dim=1)
    target = (a + b) % P
    perm = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed))
    cut = int(train_frac * len(a))
    tr, va = perm[:cut], perm[cut:]
    return tokens[tr], target[tr], tokens[va], target[va]


def train(seed=0, steps=5000):
    """Train with full-batch AdamW and return the best validation accuracy seen.

    The weight decay of 1.0 makes validation accuracy jump around, so the loop
    evaluates every 100 steps and keeps the best value.
    """
    torch.manual_seed(seed)
    x_tr, y_tr, x_va, y_va = make_data(seed)
    model = TinyTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    best = 0.0
    for step in range(1, steps + 1):
        loss = F.cross_entropy(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                acc = (model(x_va).argmax(-1) == y_va).float().mean().item()
            best = max(best, acc)
    return best
