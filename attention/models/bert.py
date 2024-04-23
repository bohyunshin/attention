import torch.nn as nn
import torch.nn.functional as F
from attention.embedding.bert import BERTEmbedding
from attention.layers.bert import EncoderLayer


class Encoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 d_emb,
                 d_model,
                 d_k,
                 d_v,
                 n_head,
                 d_hid,
                 n_position,
                 n_layers):
        super().__init__()

        self.embedding = nn.Embedding(n_vocab, d_emb, padding_idx=0)
        self.bert_embedding = BERTEmbedding(embed_size=d_emb,
                                            seq_len=n_position)
        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(n_head,
                             d_model,
                             d_emb,
                             d_k,
                             d_v,
                             d_hid,
                             activation=F.gelu,
                             dropout=0.1) for _ in range(n_layers)
            ]
        )

    def forward(self, seq, segment_label, mask, return_attns=False):

        enc_slf_attn_list = []

        enc_output = self.bert_embedding(self.embedding(seq), segment_label)
        for layer in self.layer_stack:
            enc_output, enc_slf_attn = layer(enc_output, mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Bert(nn.Module):
    def __init__(self,
                 n_vocab,
                 d_emb,
                 d_model,
                 d_k,
                 d_v,
                 n_head,
                 d_hid,
                 n_position,
                 n_layers):

        super().__init__()

        self.encoder = Encoder(
            n_vocab=n_vocab,
            d_emb=d_emb,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_head=n_head,
            d_hid=d_hid,
            n_position=n_position,
            n_layers=n_layers
        )

    def forward(self, seq, segment_label):

        mask = (seq > 0).unsqueeze(1)
        enc_output, *_ = self.encoder(seq, segment_label, mask)

        return enc_output


class NextSentencePrediction(nn.Module):
    def __init__(self, hidden):

        super().__init__()

        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:,0]))


class MaskedLanguageModel(nn.Module):

    def __init__(self, hidden, vocab_size):

        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))


class BERTLM(nn.Module):

    def __init__(self, bert, vocab_size):

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.d_model)
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)