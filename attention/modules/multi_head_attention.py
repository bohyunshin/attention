import torch.nn as nn
import numpy as np
from modules.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 n_head,
                 d_model,
                 d_emb,
                 d_k,
                 d_v,
                 dropout=0.1):
        super().__init__()

        assert d_model == n_head * d_k

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_emb, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_emb, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_emb, n_head * d_k, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_emb, bias=False)

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # q.shape, k.shape, v.shape = (batch_size, seq_len, d_emb)
        # just before passing (n x d_emb) sequence to query / key / value projection matrix
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_head, d_k)
        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k)
        v = self.w_ks(v).view(batch_size, len_v, n_head, d_k)

        # transpose for attention dot product
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)
        # q.shape = (batch_size, n_head, seq_len, d_k)

        # transpose to revert dimension without n_heads
        q = q.transpose(1,2).contiguous().view(batch_size, len_q, -1)
        # q.shape = (batch_size, seq_len, d_model)

        q = self.fc(q)
        q = self.dropout(q)
        q += residual
        q = self.layer_norm(q)

        return q, attn