import torch.nn as nn


class Model(nn.Module):
    def __init__(self):

        super().__init__()

        self.embedding = BERTEmbedding()

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderLayer() for _ in range(n_layers)
            ]
        )

    def forward(self, x, segment_info):
        mask = ()

        x = self.embedding(x, segment_info)

        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)

        return x


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