import torch
import numpy as np
from attention.models.transformer import PositionalEncoding
from attention.modules.scaled_dot_product_attention import ScaledDotProductAttention
from attention.modules.multi_head_attention import MultiHeadAttention
from attention.sublayers.transformer import PositionwiseFeedForwrd
from attention.layers.transformer import EncoderLayer, DecoderLayer
from attention.models.transformer import Encoder, Decoder, get_pad_mask, get_subsequent_mask

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

def test_positionwise_feedfowrd_shape():
    d_in = 512
    d_hid = 256
    seq_len = 200
    dropout = 0.1
    x = torch.rand((seq_len, d_in))
    pff = PositionwiseFeedForwrd(
        d_in=d_in,
        d_hid=d_hid,
        dropout=dropout
    )
    output = pff(x)

    asserting((seq_len, d_in), output.shape)

def test_encoder_layer_shape():
    d_model = 512
    d_inner = 256
    n_head = 8
    d_k = int(d_model // n_head)
    d_v = d_k
    dropout = 0.1
    batch_size = 1
    seq_len = 200
    x = torch.rand((batch_size, seq_len, d_model))
    encoder_layer = EncoderLayer(
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout
    )
    enc_output, enc_slf_attn = encoder_layer(x)

    asserting((batch_size, seq_len, d_model), enc_output.shape)
    asserting((batch_size, n_head, seq_len, seq_len), enc_slf_attn.shape)

def test_decoder_layer_shape():
    d_model = 512
    d_inner = 256
    n_head = 8
    d_k = int(d_model // n_head)
    d_v = d_k
    dropout = 0.1
    batch_size = 1
    seq_len = 200
    dec_input = torch.rand((batch_size, seq_len, d_model))
    enc_output = torch.rand((batch_size, seq_len, d_model))
    decoder_layer = DecoderLayer(
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout
    )
    dec_output, dec_slf_attn, dec_enc_attn = decoder_layer(
        dec_input=dec_input,
        enc_output=enc_output
    )

    asserting((batch_size, seq_len, d_model), dec_output.shape)
    asserting((batch_size, n_head, seq_len, seq_len), dec_slf_attn.shape)
    asserting((batch_size, n_head, seq_len, seq_len), dec_enc_attn.shape)

def test_encoder_shape():
    n_src_vocab = 3000
    d_word_vec = 512
    n_layers = 6
    n_head = 8
    d_k = int(d_word_vec // n_head)
    d_v = d_k
    d_model = d_word_vec
    d_inner = 256
    pad_idx = 0
    dropout = 0.1
    n_positoin = 200
    scale_emb=False
    seq_len = 100
    src_seq = torch.rand((batch_size, n_head, seq_len, d_k))
    

    encoder = Encoder(
        n_src_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        dropout=dropout,
        n_position=n_positoin,
        scale_emb=scale_emb
    )
    
    

def asserting(expected, result):
    assert result == expected, f"Expected {expected}, got {result} instead"
