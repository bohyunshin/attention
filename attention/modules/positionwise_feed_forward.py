import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    def __init__(self,
                 d_in,
                 d_hid,
                 activation,
                 dropout=0.1):
        super().__init__()

        self.activation = activation
        self.fc1 = nn.Linear(d_in, d_hid)
        self.fc2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        residual = x
        x = self.fc2(self.activation(self.fc1(x)))
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x