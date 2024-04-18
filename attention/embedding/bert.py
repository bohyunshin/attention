import torch.nn as nn
from attention.modules.position_encoding import PositionalEncoding


class BERTEmbedding(nn.Module):

    def __init__(self,
                 vocab_size,
                 embed_size,
                 seq_len=64,
                 dropout=0.1):

        super().__init__()
        self.embed_size = embed_size
        self.token = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEncoding(d_hid=embed_size, n_position=seq_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)