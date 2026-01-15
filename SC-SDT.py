"""
This file provides the complete PyTorch implementation of the SC-SDT (Spectral Convolution-Spatial Differential Transformer) model.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, embed_dim=64):
        super(InputEmbedding, self).__init__()
        first = 25
        second = 125
        self.cov1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=first, kernel_size=1, groups=5),
            nn.BatchNorm1d(first)
        )

        self.cov2 = nn.Sequential(
            nn.Conv1d(in_channels=first, out_channels=second, kernel_size=1, groups=5),
            nn.BatchNorm1d(second),
            nn.ELU(),
        )

        self.cov3 = nn.Sequential(
            nn.Conv1d(in_channels=second, out_channels=embed_dim, kernel_size=1),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cov1(x)
        x = self.cov2(x)
        x = self.cov3(x)
        x = x.permute(0, 2, 1)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=62):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)

        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, [Batch size, Seq Length, Embedding Dimension]
        """
        x = x + self.pe[:, :x.size(1)]
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x.to(torch.float32)
        temp = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        temp = temp.to(x.dtype)
        return self.weight * temp


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(SwiGLU, self).__init__()
        self.WG = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W2 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        g = F.silu(self.WG(x))
        g = self.dropout(g)
        z = self.W1(x)
        z = self.dropout(z)
        result = self.W2(g * z)
        result = self.dropout(result)
        return result


class MultiHeadDifferentialAttention(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init, dropout_rate):
        super(MultiHeadDifferentialAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads // 2
        self.scale = self.head_dim ** -0.5

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model, bias=False)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.lambda_init = lambda_init

        self.proj_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        b, n, d_model = x.shape
        q = self.linear_q(x).view(b, n, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        k = self.linear_k(x).view(b, n, self.num_heads, 2 * self.head_dim).transpose(1, 2)
        v = self.linear_v(x).view(b, n, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)

        lambda_1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        lambda_2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        a1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        a2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attention1 = torch.softmax(a1, dim=-1)
        attention2 = torch.softmax(a2, dim=-1)
        attn_weights = attention1 - lambda_full * attention2
        attn = torch.matmul(attn_weights, v)

        attn = attn.transpose(1, 2).contiguous().view(b, -1, self.d_model)

        attn = self.linear_out(attn)

        attn = self.proj_dropout(attn)
        return attn


class DiffTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, lambda_init, dim_feedforward, dropout_rate):
        super(DiffTransformerLayer, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadDifferentialAttention(d_model, num_heads, lambda_init, dropout_rate)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLU(d_model, dim_feedforward, dropout_rate)

    def forward(self, x):
        y = self.norm1(self.attn(x) + x)
        z = self.norm2(self.ff(y) + y)
        return z


class EmotionClassifier(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, seq_length, lambda_init):
        super(EmotionClassifier, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.embed = InputEmbedding(d_model)

        dropout = 0.5

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)
        self.pos_emb = PositionalEncoding(d_model, seq_length + 1)
        self.layers = nn.ModuleList([
            DiffTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                lambda_init=lambda_init,
                dim_feedforward=d_model * 3,
                dropout_rate=dropout
            )
            for _ in range(1, num_layers + 1)
        ])
        self.final_dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(d_model, 3, bias=False)
        nn.init.xavier_uniform_(self.head.weight)

    def batch(self, x):
        b, n, d = x.shape
        x = self.embed(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_emb(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, 0, :]
        x = self.final_dropout(x)
        return x

    def forward(self, x):
        hidden_vector = self.batch(x)
        x = self.head(hidden_vector)
        return x, hidden_vector

    def predict(self, x):
        x = self.batch(x)
        x = self.head(x)
        return x
