import torch.nn as nn
from attention.sublayers.bert import MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self,
                 n_head,
                 d_model,
                 d_emb,
                 d_k,
                 d_v,
                 d_hid,
                 activation=nn.GELU,
                 dropout=0.1
                 ):
        super().__init__()

        self.slf_attention = MultiHeadAttention(n_head=n_head,
                                                d_model=d_model,
                                                d_emb=d_emb,
                                                d_k=d_k,
                                                d_v=d_v,
                                                dropout=dropout)
        self.ffw = PositionwiseFeedForward(d_in=d_model,
                                           d_hid=d_hid,
                                           activation=activation,
                                           dropout=dropout)

    def forward(self, enc_input, mask):
        x = self.slf_attention(enc_input, enc_input, enc_input, mask)
        attn = self.ffw(x)
        return attn