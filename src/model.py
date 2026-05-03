import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # tell pytorch to never update padding (here idx 0)

    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, V)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.h = num_heads
        self.d_k = embed_dim // num_heads

        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

        self.attn = ScaledDotProductAttention()

    def split(self, x):
        b, t, _ = x.size()
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, mask=None):
        Q = self.split(self.Wq(x))
        K = self.split(self.Wk(x))
        V = self.split(self.Wv(x))

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)

        out = self.attn(Q, K, V, mask)
        b, _, t, _ = out.size()
        out = out.transpose(1, 2).contiguous().view(b, t, self.h * self.d_k)
        return self.Wo(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ff = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=64, num_heads=4,
                 ff_dim=128, num_layers=1, num_classes=2,
                 max_len=20, dropout=0.1, use_pos_enc=True):
        super().__init__()
        self.use_pos_enc = use_pos_enc

        self.tok_emb = TokenEmbedding(vocab_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout) if use_pos_enc else None

        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.clf = nn.Linear(embed_dim, num_classes)

    def forward(self, tokens, mask):
        x = self.tok_emb(tokens)

        if self.use_pos_enc:
            x = self.pos_enc(x)

        for block in self.blocks:
            x = block(x, mask)

        m = mask.unsqueeze(-1).float()
        x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)

        return self.clf(x)
