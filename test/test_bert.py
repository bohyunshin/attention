import os
import sys
sys.path.append(os.getcwd())

import torch
from attention.embedding.bert import BERTEmbedding
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
