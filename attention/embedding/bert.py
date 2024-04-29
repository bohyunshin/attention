import torch.nn as nn
from attention.modules.position_encoding import PositionalEncoding


class BERTEmbedding(nn.Module):

    def __init__(self,
                 embed_size,
                 seq_len=64):

        super().__init__()
        self.embed_size = embed_size
        # (batch_size, seq_len) -> (batch_size, seq_len, embed_size)
        self.segment = nn.Embedding(3, embed_size, padding_idx=0)
        self.position = PositionalEncoding(d_hid=embed_size, n_position=seq_len)

    def forward(self, x, segment_label):
        x = self.position(x) + self.segment(segment_label)
        return x