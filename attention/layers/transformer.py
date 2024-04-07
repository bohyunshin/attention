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
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_emb=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForwrd(
            d_in=d_model,
            d_hid=d_inner,
            dropout=dropout
        )

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            q=enc_input,
            k=enc_input,
            v=enc_input,
            mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_emb=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        self.cross_attn = MultiHeadAttention(
            n_head=n_head,
            d_model=d_model,
            d_emb=d_model,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout
        )
        self.pos_ffn = PositionwiseFeedForwrd(
            d_in=d_model,
            d_hid=d_inner,
            dropout=dropout
        )

    def forward(self,
                dec_input,
                enc_output,
                slf_attn_mask=None,
                dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            q=dec_input,
            k=dec_input,
            v=dec_input,
            mask=slf_attn_mask
        )
        dec_output, dec_enc_attn = self.cross_attn(
            q=dec_output,
            k=enc_output,
            v=enc_output,
            mask=dec_enc_attn_mask
        )
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn