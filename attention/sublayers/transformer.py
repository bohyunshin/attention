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
                 activation,
                 dropout=0.1):
        super().__init__(d_in,
                         d_hid,
                         activation,
                         dropout)
