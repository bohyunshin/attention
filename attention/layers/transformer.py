import torch
import torch.nn as nn
from attention.sublayers.transformer import MultiHeadAttention, PositionwiseFeedForwrd


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForwrd()

    def forward(self):
        enc_output = self.slf_attn()
        enc_output = self.pos_ffn()
        return enc_output


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 d_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention()
        self.cross_attn = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForwrd()

    def forward(self):
        dec_output = self.slf_attn()
        dec_output = self.cross_attn()
        dec_output = self.pos_ffn()
        return dec_output