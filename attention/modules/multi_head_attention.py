import torch.nn as nn
import torch


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()