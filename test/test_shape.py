import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
from attention.models.transformer import PositionalEncoding
from attention.modules.scaled_dot_product_attention import ScaledDotProductAttention
from attention.modules.multi_head_attention import MultiHeadAttention
from attention.sublayers.transformer import PositionwiseFeedForwrd
from attention.layers.transformer import EncoderLayer, DecoderLayer
from attention.models.transformer import Model as Transformer, Encoder, Decoder, get_pad_mask, get_subsequent_mask

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
    src_seq_len = 33
    batch_size = 256
    src_seq = torch.randint(0, n_src_vocab-1, (batch_size, src_seq_len))
    src_mask = get_pad_mask(src_seq, pad_idx)

    encoder = Encoder(
        n_src_vocab=n_src_vocab,
        d_word_vec=d_word_vec,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        d_model=d_model,
        d_inner=d_inner,
        pad_idx=pad_idx,
        dropout=dropout,
        n_position=n_positoin,
        scale_emb=scale_emb
    )

    enc_output = encoder(src_seq, src_mask)

    asserting((batch_size, src_seq_len, d_model), enc_output[0].shape)

def test_decoder_shape():
    n_src_vocab = 2000
    n_trg_vocab = 3000
    d_word_vec = 512
    d_model = d_word_vec
    n_layers = 6
    n_head = 8
    d_k = int(d_model // n_head)
    d_v = d_k
    d_inner = 256
    pad_idx = 0
    n_position = 200
    dropout = 0.1
    scale_emb = False
    batch_size=256
    src_seq_len = 33
    trg_seq_len = 28

    decoder = Decoder(
        n_trg_vocab=n_trg_vocab,
        d_word_vec=d_word_vec,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        d_model=d_model,
        d_inner=d_inner,
        pad_idx=pad_idx,
        n_position=n_position,
        dropout=dropout,
        scale_emb=scale_emb
    )

    src_seq = torch.randint(0, n_src_vocab-1, (batch_size, src_seq_len))
    src_mask = get_pad_mask(src_seq, pad_idx)
    trg_seq = torch.randint(0, n_trg_vocab-1, (batch_size, trg_seq_len))
    trg_mask = get_pad_mask(trg_seq, pad_idx) & get_subsequent_mask(trg_seq)
    enc_output = torch.rand((batch_size, src_seq_len, d_model))

    dec_output = decoder(trg_seq, trg_mask, enc_output, src_mask)

    asserting((batch_size, trg_seq_len, d_word_vec), dec_output[0].shape)

def test_transformer_shape():
    n_src_vocab = 3000
    n_trg_vocab = 2000
    d_word_vec = 512
    n_layers = 6
    n_head = 8
    d_k = int(d_word_vec // n_head)
    d_v = d_k
    d_model = d_word_vec
    d_inner = 256
    pad_idx = 0
    dropout = 0.1
    src_seq_len = 33
    trg_seq_len = 28
    n_position = 200
    batch_size = 256

    # torch.randint()

    src_seq = torch.randint(low=0, high=n_src_vocab-1, size=(batch_size, src_seq_len))
    trg_seq = torch.randint(low=0, high=n_trg_vocab-1, size=(batch_size, trg_seq_len))

    transformer = Transformer(
        n_src_vocab=n_src_vocab,
        n_trg_vocab=n_trg_vocab,
        src_pad_idx=pad_idx,
        trg_pad_idx=pad_idx,
        d_word_vec=d_word_vec,
        d_model=d_model,
        d_inner=d_inner,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout,
        n_position=n_position,
        trg_emb_prj_weight_sharing=True,
        emb_src_trg_weight_sharing=False,
        scale_emb_or_prj='prj'
    )

    output = transformer(src_seq, trg_seq)

    asserting((batch_size * trg_seq_len, n_trg_vocab), output.shape)

def asserting(expected, result):
    assert result == expected, f"Expected {expected}, got {result} instead"
