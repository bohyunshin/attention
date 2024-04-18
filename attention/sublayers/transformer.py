import torch.nn as nn
import torch.nn.functional as F
from attention.modules.multi_head_attention import MultiHeadAttention as MultiHeadAttentionBase
from attention.modules.positionwise_feed_forward import PositionwiseFeedForward as PositionwiseFeedForwardBase


class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self,
                 n_head,
                 d_model,
                 d_emb,
                 d_k,
                 d_v,
                 dropout=0.1):
        super().__init__(n_head,
                         d_model,
                         d_emb,
                         d_k,
                         d_v,
                         dropout=dropout)



class PositionwiseFeedForward(PositionwiseFeedForwardBase):
    def __init__(self,
                 d_in,
                 d_hid,
                 dropout=0.1):
        super().__init__(d_in,
                         d_hid,
                         dropout)

# class PositionwiseFeedForwrd(nn.Module):
#     def __init__(self,
#                  d_in,
#                  d_hid,
#                  dropout=0.1):
#         super().__init__()
#
#         self.fc1 = nn.Linear(d_in, d_hid)
#         self.fc2 = nn.Linear(d_hid, d_in)
#         self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, x):
#         residual = x
#         x = self.fc2(F.relu(self.fc1(x)))
#         x = self.dropout(x)
#         x = self.layer_norm(x + residual)
#
#         return x