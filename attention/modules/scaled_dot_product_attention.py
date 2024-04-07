import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(p=attn_dropout)

    def forward(self, q, k, v, mask=None):
        # q.shape, k.shape, v.shape = (batch_size, n_head, seq_len, d_k)
        attn = torch.matmul(q, k.transpose(2,3)) / self.temperature
        # attn.shape = (batch_size, n_hear, seq_len, seq_len)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1) # row sum up to 1
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)
        # output.shape = (batch_size, n_head, seq_len, d_k)

        return output, attn