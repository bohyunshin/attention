import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from attention.modules.scaled_dot_product_attention import ScaledDotProductAttention
from attention.modules.multi_head_attention import MultiHeadAttention
from attention.modules.positionwise_feed_forward import PositionwiseFeedForward
from tools.utils import asserting

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

def test_positionwise_feedfoward_shape():
    d_in = 512
    d_hid = 256
    seq_len = 200
    dropout = 0.1
    x = torch.rand((seq_len, d_in))
    pff = PositionwiseFeedForward(
        d_in=d_in,
        d_hid=d_hid,
        activation=F.relu,
        dropout=dropout
    )
    output = pff(x)

    asserting((seq_len, d_in), output.shape)