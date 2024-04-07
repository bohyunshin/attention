import torch
import torch.nn as nn
import numpy as np
from attention.layers.transformer import EncoderLayer, DecoderLayer


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, position, d_hid):

        def get_position_angle_vec(position, d_hid):
            return [position / np.power(10000, 2*(i // 2) / d_hid) for i in range(d_hid)]

        sinusoid_table = np.array(
            [get_position_angle_vec(pos, d_hid) for pos in range(position)]
        )
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:,0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:,1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # self.pos_table.shape = (1, max_length, d_emb)
        # x.shape = (batch_size, seq_length, d_emb)
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):

    def __init__(self,
                 n_src_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx,
                 dropout=0.1,
                 n_position=200,
                 scale_emb=False):
        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer() for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model


class Decoder(nn.Module):

    def __init__(self,
                 n_trg_vocab,
                 d_word_vec,
                 n_layers,
                 n_head,
                 d_k,
                 d_v,
                 d_model,
                 d_inner,
                 pad_idx,
                 n_position,
                 dropout=0.1,
                 scale_emb=False):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [DecoderLayer() for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self):
        dec_output = self.trg_word_emb()

        for dec_layer in self.layer_stack:
            dec_output = dec_layer()

        return dec_output


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self):
        enc_output = self.encoder()
        dec_output = self.decoer()

if __name__ == "__main__":
    word = torch.tensor([0,1,2,3,4])
    n_src_vocab = 3000
    n = 512
    src_word_emb = nn.Embedding(n_src_vocab, n)
    emb = src_word_emb(word)
    print(src_word_emb(word))