import torch
from attention.models.transformer import PositionalEncoding

def test_positional_encoding_shape():
    d_hid = 512
    n_position = 200
    seq_len = 100
    rand_word_vec = torch.rand((1, seq_len, d_hid)) # randomize 1 batch size tensor
    position_enc = PositionalEncoding(
        d_hid = d_hid,
        n_position = n_position
    )

    asserting((1,n_position,d_hid), position_enc.pos_table.shape)
    asserting((1,seq_len,d_hid), position_enc(rand_word_vec).shape)

def asserting(expected, result):
    assert result == expected, f"Expected {expected}, got {result} instead"
