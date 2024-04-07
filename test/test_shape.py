import torch
import numpy as np
from attention.models.transformer import PositionalEncoding
from attention.modules.scaled_dot_product_attention import ScaledDotProductAttention
from attention.modules.multi_head_attention import MultiHeadAttention

def test_positional_encoding_shape():
    d_hid = 512
    n_position = 200
    seq_len = 100
    batch_size = 1
    rand_word_vec = torch.rand((batch_size, seq_len, d_hid)) # randomize 1 batch size tensor
    position_enc = PositionalEncoding(
        d_hid = d_hid,
        n_position = n_position
    )

    asserting((1,n_position,d_hid), position_enc.pos_table.shape)
    asserting((1,seq_len,d_hid), position_enc(rand_word_vec).shape)

def test_scaled_dot_product_attention_shape():
    d_emb = 512
    seq_len = 30
    batch_size = 1
    n_head = 8
    d_k = int(d_emb / n_head)
    dropout = 0.1
    q = torch.rand((batch_size, n_head, seq_len, d_k))
    k = torch.rand((batch_size, n_head, seq_len, d_k))
    v = torch.rand((batch_size, n_head, seq_len, d_k))
    temperature = np.power(d_k, 0.5)

    sdpa = ScaledDotProductAttention(temperature, dropout)

    asserting((batch_size, n_head, seq_len, d_k), sdpa(q,k,v)[0].shape)

def test_multi_head_attention_shape():
    d_emb = 512
    d_model = d_emb
    seq_len = 30
    batch_size = 1
    n_head = 8
    d_k = int(d_emb / n_head)
    dropout = 0.1
    q = torch.rand((batch_size, seq_len, d_model))
    k = torch.rand((batch_size, seq_len, d_model))
    v = torch.rand((batch_size, seq_len, d_model))

    mha = MultiHeadAttention(
        n_head=n_head,
        d_model=d_model,
        d_emb=d_emb,
        d_k=d_k,
        d_v=d_k,
        dropout=dropout
    )

    asserting((batch_size, seq_len, d_emb), mha(q, k, v)[0].shape)

def asserting(expected, result):
    assert result == expected, f"Expected {expected}, got {result} instead"
