import torch.nn as nn
import torch
from modules.multi_head_attention import MultiHeadAttention as MultiHeadAttentionBase


class MultiHeadAttention(MultiHeadAttentionBase):
    def __init__(self):
        super().__init__()


class PositionwiseFeedForwrd(nn.Module):
    def __init__(self):
        super().__init__()