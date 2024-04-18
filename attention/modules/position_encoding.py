import torch
import torch.nn as nn
import numpy as np


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

    def forward(self, x, add=True):
        # self.pos_table.shape = (1, max_length, d_emb)
        # x.shape = (batch_size, seq_length, d_emb)
        if add:
            return x + self.pos_table[:, :x.size(1)].clone().detach()
        return self.pos_table[:, :x.size(1)].clone().detach()