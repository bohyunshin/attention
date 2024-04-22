import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from attention.embedding.bert import BERTEmbedding
from attention.layers.bert import EncoderLayer
from attention.tools.utils import asserting

def test_bert_embedding_shape():
    batch_size = 256
    embed_size = 512
    seq_len = 64
    t1_len = 30
    t2_len = seq_len-t1_len
    segment_label = torch.tensor([
        [1 for _ in range(t1_len)] + [2 for _ in range(t2_len)] for _ in range(256)
    ])
    rand_word_vec = torch.rand((batch_size, seq_len, embed_size))
    bert_embedding = BERTEmbedding(embed_size=embed_size,
                                   seq_len=seq_len)
    embedding = bert_embedding(rand_word_vec, segment_label)
    asserting((batch_size, seq_len, embed_size), embedding.shape)

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
        d_hid=d_inner,
        d_emb=d_model,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout
    )

    enc_output, enc_slf_attn = encoder_layer(x)

    asserting((batch_size, seq_len, d_model), enc_output.shape)
    asserting((batch_size, n_head, seq_len, seq_len), enc_slf_attn.shape)
